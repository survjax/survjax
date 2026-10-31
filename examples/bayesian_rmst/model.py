"""Synthetic Bayesian trial; optional NumPyro example, not library API."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as distributions
from numpyro.infer import Predictive

import survjax
from examples.weibull_simulation.study import calibrate_censoring, generate_weibull


class Trial(NamedTuple):
    """Fixed trial, arm indicators and known censoring limit in years."""

    data: survjax.SurvivalData
    arm: jax.Array
    censor_limit: float


def use_case_key(stream: int) -> jax.Array:
    """Return stream 0=data, 1=NUTS, 2=prior or 3=posterior predictive."""
    root = jax.random.key(20261025)
    return jax.random.fold_in(jax.random.fold_in(root, 2), stream)


def make_trial() -> Trial:
    """Generate the specified 150-control/150-treatment float64 trial.

    Returns:
        Valid exact/right-censored data, (300,) arm and calibrated c in years.

    Raises:
        RuntimeError: Float64 is disabled or calibration fails.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError("Bayesian demo requires JAX_ENABLE_X64=1")
    arm = jnp.repeat(jnp.array([0.0, 1.0]), 150)
    eta = jnp.log(0.7) * arm
    censor_limit = calibrate_censoring(1.38, 11.42, 0.55, log_hazard_ratio=eta)
    generated = generate_weibull(
        use_case_key(0), 1.38, 11.42, censor_limit, log_hazard_ratio=eta
    )
    data = survjax.SurvivalData(generated.time, generated.event)
    data.issues.raise_if_invalid()
    return Trial(data, arm, censor_limit)


def model(
    data: survjax.SurvivalData,
    arm: jax.Array,
    *,
    compute_rmst: bool = False,
    tau: float = 5.0,
    tau_grid: jax.Array | None = None,
) -> None:
    """Define priors and per-row factors; optionally fill per-arm RMST sites.

    Args:
        data: N exact/right-censored observations.
        arm: (N,) float control/treatment indicators.
        compute_rmst: Enable summaries for Predictive, never for NUTS.
        tau: Scalar restriction time in years.
        tau_grid: (m,) restriction times, required when computing RMST.

    Returns:
        None; NumPyro records sample, factor and optional deterministic sites.

    Raises:
        ValueError: RMST requested without a grid.
    """
    if compute_rmst and tau_grid is None:
        raise ValueError("tau_grid is required when compute_rmst=True")

    # sample parameters
    log_k = numpyro.sample("log_k", distributions.Normal(0, 5))
    log_b0 = numpyro.sample("log_b0", distributions.Normal(0, 5))
    # sample log-hazard ratio for treatment arm
    gamma = numpyro.sample("gamma", distributions.Normal(0, 5))

    # set up survival model and likelihood factor
    base = survjax.Weibull.from_log(log_shape=log_k, log_scale=log_b0)
    survival = survjax.ProportionalHazards(base, log_hazard_ratio=gamma * arm)
    with numpyro.plate("observations", len(data), dim=-1):
        # compute the survival log likelihood and add to sampler target
        # Spec 11.3: summing here would multiply the total likelihood by N.
        numpyro.factor("survival", survival.log_likelihood(data))
    if compute_rmst:
        per_arm = survjax.ProportionalHazards(
            base, log_hazard_ratio=gamma * jnp.array([0.0, 1.0])
        )
        numpyro.deterministic("rmst_tau", per_arm.rmst(tau))
        numpyro.deterministic("rmst_curve", per_arm.rmst(tau_grid[:, None]))


class PosteriorDraws(NamedTuple):
    """Draw-level quantities; curves use time first, then draw, then arm."""

    shape: jax.Array  # (S,)
    hazard_ratio: jax.Array  # (S,)
    rmst_tau: jax.Array  # (S, 2)
    rmst_curve: jax.Array  # (m, S, 2)
    survival: jax.Array  # (K, S, 2)


def time_grids(data: survjax.SurvivalData) -> tuple[jax.Array, jax.Array]:
    """Return survival/tau grids in years using observed follow-up.

    Args:
        data: Nonempty exact/right-censored trial; lower stores observed time.

    Returns:
        Survival (201,) on [0,20] and tau (101,) ending at rounded-down p90.

    Raises:
        ValueError: Rounded p90 is nonfinite or below the required tau=5 years.
    """
    tau_max = np.floor(np.percentile(np.asarray(data.lower), 90) * 2) / 2
    if not np.isfinite(tau_max) or tau_max < 5:
        raise ValueError("Observed follow-up must support tau=5 years")
    return jnp.linspace(0.0, 20.0, 201), jnp.linspace(0.0, tau_max, 101)


def posterior_draws(
    samples: dict[str, jax.Array], t_grid: jax.Array, tau_grid: jax.Array
) -> PosteriorDraws:
    """Evaluate route A on one (draw, arm) distribution before aggregating.

    Args:
        samples: Flattened (S,) log_k, log_b0 and gamma posterior draws.
        t_grid: (K,) survival times in years.
        tau_grid: (m,) restriction times in years.

    Returns:
        Shape/HR draws, RMST at 5 years, RMST curves and survival curves.
    """
    base = survjax.Weibull.from_log(
        log_shape=samples["log_k"][:, None],
        log_scale=samples["log_b0"][:, None],
    )
    posterior = survjax.ProportionalHazards(
        base, log_hazard_ratio=samples["gamma"][:, None] * jnp.array([0.0, 1.0])
    )
    return PosteriorDraws(
        jnp.exp(samples["log_k"]),
        jnp.exp(samples["gamma"]),
        posterior.rmst(5.0),
        posterior.rmst(tau_grid[:, None, None]),
        posterior.survival_grid(t_grid),
    )


def check_rmst_routes(
    samples: dict[str, jax.Array],
    trial: Trial,
    tau_grid: jax.Array,
    draws: PosteriorDraws,
) -> float:
    """Require Predictive route B to match route A after curve-axis alignment.

    Args:
        samples: Flattened (S,) posterior parameters used for route A.
        trial: Observations and arms passed to the NumPyro model.
        tau_grid: (m,) restriction times used for route A.
        draws: Route A outputs at tau=5 and tau_grid.

    Returns:
        Largest absolute discrepancy across both RMST arrays.

    Raises:
        AssertionError: Shapes differ, values are nonfinite or error >1e-12.
    """
    predictive = Predictive(
        model, posterior_samples=samples, return_sites=("rmst_tau", "rmst_curve")
    )
    # All parameters are conditioned; these deterministic sites need no RNG stream.
    sites = predictive(
        jax.random.key(0),
        trial.data,
        trial.arm,
        compute_rmst=True,
        tau=5.0,
        tau_grid=tau_grid,
    )
    aligned = (sites["rmst_tau"], jnp.swapaxes(sites["rmst_curve"], 0, 1))
    errors = []
    for actual, expected in zip(
        aligned, (draws.rmst_tau, draws.rmst_curve), strict=True
    ):
        assert actual.shape == expected.shape
        np.testing.assert_allclose(
            actual, expected, atol=1e-12, rtol=0, equal_nan=False
        )
        errors.append(float(jnp.max(jnp.abs(actual - expected))))
    return max(errors)


def summarize_draws(draws: PosteriorDraws) -> tuple[dict[str, dict[str, float]], float]:
    """Aggregate per-draw effects into a table and P(delta RMST(5)>0).

    Args:
        draws: At least two finite posterior draws; RMST arms are control/treatment.

    Returns:
        k, HR, arm RMSTs and paired delta RMST: mean, sample SD and equal-tail
        95% interval; separately, the fraction of strictly positive differences.
    """
    rmst = np.asarray(draws.rmst_tau)
    delta = rmst[:, 1] - rmst[:, 0]
    quantities = {
        "k": np.asarray(draws.shape),
        "HR": np.asarray(draws.hazard_ratio),
        "RMST0(5)": rmst[:, 0],
        "RMST1(5)": rmst[:, 1],
        "delta_RMST(5)": delta,
    }
    table = {}
    for name, values in quantities.items():
        low, high = np.quantile(values, [0.025, 0.975])
        table[name] = {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "lower_95": float(low),
            "upper_95": float(high),
        }
    return table, float(np.mean(delta > 0))
