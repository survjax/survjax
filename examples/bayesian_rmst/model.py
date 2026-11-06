"""Synthetic two-arm trial and its NumPyro model; optional example, not library API."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as distributions

import survjax
from examples.weibull_simulation.study import calibrate_censoring, generate_weibull

ROOT_KEY = 20261025  # shared by all use cases; this one folds in 2 (spec 02-details)
TRUE_SHAPE = 1.38
TRUE_SCALE = 11.42  # control arm, years
TRUE_HAZARD_RATIO = 0.7


class Trial(NamedTuple):
    """Observations, (N,) float 0/1 arm indicators and the censoring limit c in years."""

    data: survjax.SurvivalData
    arm: jax.Array
    censor_limit: float


def use_case_key(stream: int) -> jax.Array:
    """Return the key of stream 0 (trial data) or 1 (NUTS) of this use case."""
    return jax.random.fold_in(jax.random.fold_in(jax.random.key(ROOT_KEY), 2), stream)


def make_trial() -> Trial:
    """Generate the 150-control/150-treatment trial with about 55% censoring.

    Returns:
        The trial; times in years.

    Raises:
        RuntimeError: Float64 is disabled; the committed trial is a float64 draw.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError("The Bayesian example requires JAX_ENABLE_X64=1")
    arm = jnp.repeat(jnp.array([0.0, 1.0]), 150)
    eta = jnp.log(TRUE_HAZARD_RATIO) * arm
    censor_limit = calibrate_censoring(
        TRUE_SHAPE, TRUE_SCALE, 0.55, log_hazard_ratio=eta
    )
    generated = generate_weibull(
        use_case_key(0), TRUE_SHAPE, TRUE_SCALE, censor_limit, log_hazard_ratio=eta
    )
    data = survjax.SurvivalData(generated.time, generated.event)
    data.issues.raise_if_invalid()
    return Trial(data, arm, censor_limit)


def model(
    data: survjax.SurvivalData,
    arm: jax.Array,
    compute_rmst: bool = False,
    tau: float = 5.0,
    tau_grid: jax.Array | None = None,
) -> None:
    """Weibull proportional hazards model with a survjax likelihood factor.

    Args:
        data: N exact/right-censored observations.
        arm: (N,) float control/treatment indicators.
        compute_rmst: Add per-arm RMST sites, for Predictive after sampling.
        tau: Restriction time in years.
        tau_grid: (m,) restriction times; required when compute_rmst is True.
    """
    log_k = numpyro.sample("log_k", distributions.Normal(0, 5))
    log_b0 = numpyro.sample("log_b0", distributions.Normal(0, 5))
    gamma = numpyro.sample("gamma", distributions.Normal(0, 5))

    base = survjax.Weibull.from_log(log_shape=log_k, log_scale=log_b0)
    survival = survjax.ProportionalHazards(baseline=base, log_hazard_ratio=gamma * arm)
    with numpyro.plate("observations", len(data)):
        # spec 11.3: the (N,) vector, not its sum, or the likelihood counts N times
        numpyro.factor("survival", survival.log_likelihood(data))

    if compute_rmst:
        if tau_grid is None:
            raise ValueError("tau_grid is required when compute_rmst=True")
        per_arm = survjax.ProportionalHazards(
            baseline=base, log_hazard_ratio=gamma * jnp.array([0.0, 1.0])
        )
        numpyro.deterministic("rmst_tau", per_arm.rmst(tau))  # (2,)
        numpyro.deterministic("rmst_curve", per_arm.rmst(tau_grid[:, None]))  # (m, 2)
