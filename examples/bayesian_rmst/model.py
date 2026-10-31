"""Synthetic Bayesian trial; optional NumPyro example, not library API."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as distributions

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
