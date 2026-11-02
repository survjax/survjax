"""Scenario engine; performance summaries and the CLI are subsequent steps."""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

import survjax

from .study import FitResult, fit_replicate, generate_weibull

ROOT_SEED = 20261025
TRUE_LOG_HR = float(np.log(0.7))


def prepare_scenario(
    scenario_index: int, n: int, replicates: int, censor_limit: float
) -> tuple[survjax.SurvivalData, jax.Array]:
    """Generate and host-validate one fixed-allocation Weibull PH scenario.

    Args:
        scenario_index: Nonnegative fold_in index; caller orders n, then target.
        n: Positive even cohort size, half per arm.
        replicates: Positive number R of independently split replicate keys.
        censor_limit: Positive finite c, calibrated once per target by caller.
    Returns:
        Validated (R,N) exact/right data and (N,) binary arm indicators.
    Raises:
        ValueError: Invalid configuration or float64 is disabled.
        survjax.InvalidDataError: Generated observations fail host validation.
    """
    if not jax.config.x64_enabled:
        raise ValueError("simulation scenarios require JAX_ENABLE_X64=1")
    if scenario_index < 0 or n < 2 or n % 2 or replicates < 1:
        raise ValueError("require index>=0, even n>=2 and replicates>=1")
    if not np.isfinite(censor_limit) or censor_limit <= 0:
        raise ValueError("censor_limit must be positive and finite")
    arm = jnp.repeat(jnp.arange(2), n // 2)
    lp = TRUE_LOG_HR * arm
    key = jax.random.fold_in(jax.random.key(ROOT_SEED), scenario_index)
    keys = jax.random.split(key, replicates)
    draws = jax.vmap(
        lambda k: generate_weibull(k, 1.38, 11.42, censor_limit, log_hazard_ratio=lp)
    )(keys)
    data = survjax.SurvivalData(draws.time, event=draws.event)
    data.issues.raise_if_invalid()
    return data, arm


def _fit_batch(
    data: survjax.SurvivalData,
    arm: jax.Array,
    *,
    model: Literal["weibull", "exponential"],
) -> FitResult:
    """Fit a batch in one compiled vmap, sharing arm allocation across replicates.

    Args:
        data: Host-validated float64 (R,N) exact/right untruncated observations.
        arm: Matching (N,) binary allocation, shared across all R replicates.
        model: Static 'weibull' or 'exponential' model name.
    Returns:
        Reviewed FitResult fields, each with a leading R axis.
    Raises:
        ValueError: Unsupported model or precision, from fit_replicate.
    """
    return jax.vmap(lambda d: fit_replicate(d, arm, model=model))(data)


fit_batch = jax.jit(_fit_batch, static_argnames=("model",))


def failure_columns(fits: FitResult) -> dict[str, jax.Array]:
    """Return (R,) independent reason masks and the unique overall failed mask.

    Args:
        fits: Reviewed fit_replicate results stacked across R replicates.
    Returns:
        Boolean columns failed, solver, nonfinite, gradient, hessian and
        unestimable. Reasons overlap (D-033); unattempted optimizer checks are
        inapplicable to unestimable rows. Overall failed is authoritative.
    """
    eligible = fits.estimable
    finite = jnp.all(jnp.isfinite(fits.theta), axis=-1) & jnp.isfinite(fits.objective)
    score_ok = jnp.max(jnp.abs(fits.gradient), axis=-1) < 1e-5
    return {
        "failed": ~fits.accepted,
        "solver": eligible & ~fits.solver_success,
        "nonfinite": eligible & ~finite,
        "gradient": eligible & ~score_ok,
        "hessian": eligible & ~fits.hessian_ok,
        "unestimable": ~eligible,
    }


def failure_counts(fits: FitResult) -> dict[str, int]:
    """Count outcomes on the host, with accepted+failed=attempted=R.

    Args:
        fits: (R,...) result fields from one model, including unestimable rows.
    Returns:
        Python integer totals; reason counts may sum above the failed total.
        Attempted counts requested replicate assessments, including unestimable
        ones; it is not the number of physical optimizer executions under vmap.
    """
    columns = failure_columns(fits)
    counts = {name: int(jnp.sum(mask)) for name, mask in columns.items()}
    return {
        "attempted": fits.accepted.size,
        "accepted": int(jnp.sum(fits.accepted)),
        **counts,
    }
