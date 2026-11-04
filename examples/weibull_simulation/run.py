"""Scenario engine and CLI performance tables; detailed exports follow later."""

import argparse
import csv
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import survjax

if __package__:
    from .study import FitResult, calibrate_censoring, fit_replicate, generate_weibull
else:
    from study import FitResult, calibrate_censoring, fit_replicate, generate_weibull

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


def performance(fits: FitResult, truth: float = TRUE_LOG_HR) -> dict[str, jax.Array]:
    """Summarize accepted log-HR fits with the use-case 01 MCSE formulas.

    Args:
        fits: Reviewed (R,...) fit results; only accepted entries contribute.
        truth: True log hazard ratio, scalar.
    Returns:
        Scalar estimates/MCSEs. Empirical SD uses ddof=1, model SE is RMS.
        Zero accepted gives NaN statistics; fewer than two gives NaN SD and
        SD/bias MCSE. No model-SE MCSE is reported. Counts remain separate.
    """
    accepted = fits.accepted
    dtype = fits.theta.dtype
    count = jnp.sum(accepted.astype(dtype))
    denominator = jnp.where(count > 0, count, 1.0)
    # spec 7.2: replace rejected NaNs before subtracting, squaring or dividing.
    gamma = jnp.where(accepted, fits.theta[..., -1], 0.0)
    se = jnp.where(accepted, fits.log_hr_se, 1.0)
    mean = jnp.sum(gamma) / denominator
    centered = jnp.where(accepted, gamma - mean, 0.0)
    variance = jnp.sum(centered**2) / jnp.where(count > 1, count - 1, 1.0)
    sd = jnp.sqrt(jnp.where(count > 1, variance, 1.0))
    model_variance = jnp.sum(jnp.where(accepted, se**2, 0.0)) / denominator
    result = {
        "bias": jnp.where(count > 0, mean - truth, jnp.nan),
        "bias_mcse": jnp.where(count > 1, sd / jnp.sqrt(denominator), jnp.nan),
        "empirical_se": jnp.where(count > 1, sd, jnp.nan),
        "empirical_se_mcse": jnp.where(
            count > 1,
            sd / jnp.sqrt(jnp.where(count > 1, 2 * (count - 1), 1.0)),
            jnp.nan,
        ),
        "model_se": jnp.where(count > 0, jnp.sqrt(model_variance), jnp.nan),
    }
    covered = (gamma - 1.96 * se <= truth) & (truth <= gamma + 1.96 * se)
    rejects = jnp.abs(gamma / se) > 1.96
    for name, indicator in (("coverage", covered), ("power", rejects)):
        fraction = jnp.sum((accepted & indicator).astype(dtype)) / denominator
        result[name] = jnp.where(count > 0, fraction, jnp.nan)
        result[name + "_mcse"] = jnp.where(
            count > 0, jnp.sqrt(fraction * (1 - fraction) / denominator), jnp.nan
        )
    return result


class ModelRun(NamedTuple):
    """One model's retained fits, counts, performance and synchronized timings."""

    fits: FitResult
    failures: dict[str, int]
    performance: dict[str, jax.Array]
    first_call_seconds: float
    warm_seconds: float
    fits_per_second: float


class ScenarioRun(NamedTuple):
    """One scenario's calibrated/realized censoring and both models."""

    censor_limit: float
    realized_censoring: float
    models: dict[str, ModelRun]


def _time_model(
    data: survjax.SurvivalData, arm: jax.Array, model: Literal["weibull", "exponential"]
) -> tuple[FitResult, float, float]:
    """Synchronize preparation, first call, and each of three warm calls."""
    jax.block_until_ready((data, arm))
    elapsed = []
    for _ in range(4):
        start = perf_counter()
        fits = fit_batch(data, arm, model=model)
        jax.block_until_ready(fits)
        elapsed.append(perf_counter() - start)
    return fits, elapsed[0], float(np.median(elapsed[1:]))


def run_scenario(
    scenario_index: int, n: int, replicates: int, censor_limit: float
) -> ScenarioRun:
    """Prepare once, then fit, time and summarize both models on the same data.

    Args:
        scenario_index: Nonnegative index for the prescribed scenario key.
        n: Positive even cohort size N.
        replicates: Positive R, also the throughput numerator (not accepted R).
        censor_limit: Positive finite c calibrated once per censoring target.
    Returns:
        Censoring and model records, including raw (R,...) fits and all failures.
        First-call seconds include compilation if not already cached; warm time
        is the median of three calls. All measured results are synchronized.
    Raises:
        ValueError: Invalid configuration or precision (prepare_scenario).
        survjax.InvalidDataError: Generated observations fail host validation.
    """
    data, arm = prepare_scenario(scenario_index, n, replicates, censor_limit)
    realized = float(jnp.mean((~data.exact).astype(data.lower.dtype)))
    models = {}
    for model in ("weibull", "exponential"):
        fits, first, warm = _time_model(data, arm, model)
        models[model] = ModelRun(
            fits,
            failure_counts(fits),
            performance(fits),
            first,
            warm,
            replicates / warm,
        )
    return ScenarioRun(censor_limit, realized, models)


def run_config(config: Literal["smoke", "full"], out: Path) -> None:
    """Run the specified grid and write performance.csv (detailed exports later).

    Args:
        config: Smoke uses n=100,R=20; full uses n=100,300,1000,R=1000.
        out: Output directory, created if needed; performance.csv is replaced.
    Raises:
        ValueError: Unknown configuration or float64 disabled.
        OSError: Output cannot be written. Calibration/validation errors propagate.
    """
    if config not in ("smoke", "full"):
        raise ValueError("config must be smoke or full")
    if not jax.config.x64_enabled:
        raise ValueError("simulation configurations require JAX_ENABLE_X64=1")
    sizes, replicates = ((100,), 20) if config == "smoke" else ((100, 300, 1000), 1000)
    limits = {
        target: calibrate_censoring(
            1.38, 11.42, target, log_hazard_ratio=np.array([0.0, TRUE_LOG_HR])
        )
        for target in (0.25, 0.55)
    }
    rows = []
    for index, (n, target) in enumerate(product(sizes, limits)):
        scenario = run_scenario(index, n, replicates, limits[target])
        for model, record in scenario.models.items():
            rows.append(
                {
                    "scenario": index,
                    "n": n,
                    "target_censoring": target,
                    "censor_limit": scenario.censor_limit,
                    "realized_censoring": scenario.realized_censoring,
                    "model": model,
                    **record.failures,
                    **{
                        name: float(value) for name, value in record.performance.items()
                    },
                    "first_call_seconds": record.first_call_seconds,
                    "warm_seconds": record.warm_seconds,
                    "fits_per_second": record.fits_per_second,
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "performance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    """Run the --config {smoke,full} --out DIR command in float64."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    # Common demo convention: the runner selects x64; imports never change it.
    jax.config.update("jax_enable_x64", True)
    run_config(args.config, args.out)


if __name__ == "__main__":
    main()
