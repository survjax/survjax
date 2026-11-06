"""Batched Weibull PH simulations with summary and replicate CSV exports."""

import argparse
from pathlib import Path
from time import perf_counter
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

import survjax

if __package__:
    from . import study
    from ._evidence import write_evidence
    from ._results import capture_metadata, write_metadata, write_results
    from .study import FitResult, calibrate_censoring, fit_replicate, generate_weibull
else:
    import study
    from _evidence import write_evidence
    from _results import capture_metadata, write_metadata, write_results
    from study import FitResult, calibrate_censoring, fit_replicate, generate_weibull

ROOT_SEED = 20261025
TRUE_LOG_HR = float(np.log(0.7))


def prepare_scenario(
    scenario_index: ArrayLike, n: int, replicates: int, censor_limit: ArrayLike
) -> tuple[survjax.SurvivalData, jax.Array]:
    """Generate and host-validate scalar or same-size (C,R,N) scenarios.

    Args:
        scenario_index: Nonnegative integer index or (C,) indices in n/target order.
        n: Positive even cohort size, half per arm.
        replicates: Positive number R of independently split replicate keys.
        censor_limit: Positive finite c or matching (C,) limits, calibrated once.
    Returns:
        Validated (R,N) or (C,R,N) exact/right data and shared (N,) arm.
    Raises:
        ValueError: Invalid configuration or float64 is disabled.
        survjax.InvalidDataError: Generated observations fail host validation.
    """
    if not jax.config.x64_enabled:
        raise ValueError("simulation scenarios require JAX_ENABLE_X64=1")
    indices, limits = np.asarray(scenario_index), np.asarray(censor_limit)
    if indices.ndim > 1 or indices.shape != limits.shape or not indices.size:
        raise ValueError(
            "indices and limits must be matching scalars or nonempty vectors"
        )
    if n < 2 or n % 2 or replicates < 1:
        raise ValueError("require even n>=2 and replicates>=1")
    if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0):
        raise ValueError("require integer indices>=0")
    if not np.all(np.isfinite(limits) & (limits > 0)):
        raise ValueError("censor_limit must be positive and finite")
    arm = jnp.repeat(jnp.arange(2), n // 2)
    lp = TRUE_LOG_HR * arm

    def generate(index, limit):
        key = jax.random.fold_in(jax.random.key(ROOT_SEED), index)
        keys = jax.random.split(key, replicates)
        return jax.vmap(
            lambda k: generate_weibull(
                k, study.DGP_SHAPE, study.DGP_SCALE, limit, log_hazard_ratio=lp
            )
        )(keys)

    draws = jax.vmap(generate)(jnp.atleast_1d(indices), jnp.atleast_1d(limits))
    data = survjax.SurvivalData(draws.time, event=draws.event)
    if indices.ndim == 0:
        data = data[0]
    data.issues.raise_if_invalid()
    return data, arm


def _fit_batch(
    data: survjax.SurvivalData,
    arm: jax.Array,
    *,
    model: Literal["weibull", "exponential"],
) -> FitResult:
    """Fit prevalidated float64 (R,N) or (C,R,N) exact/right data.

    Shared arm has shape (N,), model is static, output has leading R or C/R.
    Model/precision errors propagate from fit_replicate.
    """
    fit = jax.vmap(lambda d: fit_replicate(d, arm, model=model))
    return jax.vmap(fit)(data) if len(data.shape) == 3 else fit(data)


fit_batch = jax.jit(_fit_batch, static_argnames=("model",))


def failure_columns(fits: FitResult) -> dict[str, jax.Array]:
    """Return (R,) or (C,R) independent reasons and the unique failed mask.

    Args:
        fits: Reviewed fit_replicate results stacked across R replicates.
    Returns:
        Boolean columns failed, solver, nonfinite, gradient, hessian and
        unestimable. Reasons overlap (D-033); unattempted optimizer checks are
        inapplicable to unestimable rows. Overall failed is authoritative.
    """
    eligible = fits.estimable
    finite = jnp.all(jnp.isfinite(fits.theta), axis=-1) & jnp.isfinite(fits.objective)
    score_ok = jnp.max(jnp.abs(fits.gradient), axis=-1) < study.GRADIENT_TOLERANCE
    return {
        "failed": ~fits.accepted,
        "solver": eligible & ~fits.solver_success,
        "nonfinite": eligible & ~finite,
        "gradient": eligible & ~score_ok,
        "hessian": eligible & ~fits.hessian_ok,
        "unestimable": ~eligible,
    }


def failure_counts(fits: FitResult) -> dict[str, jax.Array]:
    """Count along the last replicate axis, retaining any scenario axes.

    Args:
        fits: (R,...) result fields from one model, including unestimable rows.
    Returns:
        JAX scalar or (C,) totals; reasons may sum above the unique failed total.
        Attempted counts requested replicate assessments, including unestimable
        ones; it is not the number of physical optimizer executions under vmap.
    """
    columns = failure_columns(fits)
    # Fixed field schema is traced/unrolled; reductions batch all scenarios.
    counts = {name: jnp.sum(mask, axis=-1) for name, mask in columns.items()}
    return {
        "attempted": jnp.full(fits.accepted.shape[:-1], fits.accepted.shape[-1]),
        "accepted": jnp.sum(fits.accepted, axis=-1),
        **counts,
    }


def performance(fits: FitResult, truth: float = TRUE_LOG_HR) -> dict[str, jax.Array]:
    """Summarize accepted log-HR fits, reducing only the last replicate axis.

    Args:
        fits: Reviewed (R,...) or (C,R,...) fields; only accepted fits contribute.
        truth: True log hazard ratio, scalar.
    Returns:
        Scalar or (C,) estimates/MCSEs. SD uses ddof=1; model SE is RMS.
        Zero accepted gives NaN statistics; fewer than two gives NaN SD and
        SD/bias MCSE. No model-SE MCSE is reported. Counts remain separate.
    """
    accepted = fits.accepted
    dtype = fits.theta.dtype
    count = jnp.sum(accepted.astype(dtype), axis=-1)
    denominator = jnp.where(count > 0, count, 1.0)
    # spec 7.2: replace rejected NaNs before subtracting, squaring or dividing.
    gamma = jnp.where(accepted, fits.theta[..., -1], 0.0)
    se = jnp.where(accepted, fits.log_hr_se, 1.0)
    mean = jnp.sum(gamma, axis=-1) / denominator
    centered = jnp.where(accepted, gamma - mean[..., None], 0.0)
    variance = jnp.sum(centered**2, axis=-1) / jnp.where(count > 1, count - 1, 1.0)
    sd = jnp.sqrt(jnp.where(count > 1, variance, 1.0))
    model_variance = jnp.sum(jnp.where(accepted, se**2, 0.0), axis=-1) / denominator
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
    # Two static output fields are unrolled at trace time, not scenario dispatch.
    for name, indicator in (("coverage", covered), ("power", rejects)):
        fraction = jnp.sum((accepted & indicator).astype(dtype), axis=-1) / denominator
        result[name] = jnp.where(count > 0, fraction, jnp.nan)
        result[name + "_mcse"] = jnp.where(
            count > 0, jnp.sqrt(fraction * (1 - fraction) / denominator), jnp.nan
        )
    return result


_summarize = jax.jit(
    lambda fits: (failure_counts(fits), performance(fits), failure_columns(fits))
)


class ModelRun(NamedTuple):
    """One model's retained fits, counts, performance and synchronized timings."""

    fits: FitResult
    failures: dict[str, jax.Array]
    performance: dict[str, jax.Array]
    first_call_seconds: float
    warm_seconds: float
    fits_per_second: float
    failure_masks: dict[str, jax.Array]


class ScenarioRun(NamedTuple):
    """One scenario's calibrated/realized censoring and both models."""

    censor_limit: ArrayLike
    realized_censoring: jax.Array
    models: dict[str, ModelRun]


def _time_model(
    data: survjax.SurvivalData, arm: jax.Array, model: Literal["weibull", "exponential"]
) -> tuple[FitResult, float, float]:
    """Synchronize preparation, first call, and each of three warm calls."""
    jax.block_until_ready((data, arm))
    elapsed = []
    # Host wall-clock measurements require four synchronized dispatches.
    for _ in range(4):
        start = perf_counter()
        fits = fit_batch(data, arm, model=model)
        jax.block_until_ready(fits)
        elapsed.append(perf_counter() - start)
    return fits, elapsed[0], float(np.median(elapsed[1:]))


def run_scenario(
    scenario_index: ArrayLike, n: int, replicates: int, censor_limit: ArrayLike
) -> ScenarioRun:
    """Prepare, fit and summarize both models across all same-size scenarios.

    Args:
        scenario_index: Nonnegative scalar or (C,) prescribed scenario indices.
        n: Positive even cohort size N.
        replicates: Positive R per scenario; throughput numerator is C*R (or R).
        censor_limit: Scalar or (C,) positive limits calibrated once per target.
    Returns:
        Censoring/model records with (R,...) or (C,R,...) raw fit fields.
        First-call seconds include compilation if not already cached; warm time
        is median(3). Grouped calls share batch timings, not individual timings.
    Raises:
        ValueError: Invalid configuration or precision (prepare_scenario).
        survjax.InvalidDataError: Generated observations fail host validation.
    """
    data, arm = prepare_scenario(scenario_index, n, replicates, censor_limit)
    realized = jnp.mean((~data.exact).astype(data.lower.dtype), axis=(-2, -1))
    models = {}
    # Different parameter dimensions require distinct model graphs; both batch C/R.
    for model in ("weibull", "exponential"):
        fits, first, warm = _time_model(data, arm, model)
        counts, measures, masks = _summarize(fits)
        models[model] = ModelRun(
            fits, counts, measures, first, warm, fits.accepted.size / warm, masks
        )
    return ScenarioRun(censor_limit, realized, models)


def run_config(config: Literal["smoke", "full"], out: Path) -> None:
    """Run the specified grid and write summary and complete replicate CSVs.

    Args:
        config: Smoke uses n=100,R=20; full uses n=100,300,1000,R=1000.
        out: Output directory; CSVs and metadata.json are replaced.
    Raises:
        ValueError: Unknown configuration or float64 disabled.
        OSError: Output cannot be written. Calibration/validation errors propagate.
    """
    if config not in ("smoke", "full"):
        raise ValueError("config must be smoke or full")
    if not jax.config.x64_enabled:
        raise ValueError("simulation configurations require JAX_ENABLE_X64=1")
    out.mkdir(parents=True, exist_ok=True)
    snapshot = capture_metadata(config, ROOT_SEED, TRUE_LOG_HR)
    sizes, replicates = ((100,), 20) if config == "smoke" else ((100, 300, 1000), 1000)
    # Host-only root finding runs once per target before compiled generation/fitting.
    limits = {
        target: calibrate_censoring(
            study.DGP_SHAPE,
            study.DGP_SCALE,
            target,
            log_hazard_ratio=np.array([0.0, TRUE_LOG_HR]),
        )
        for target in (0.25, 0.55)
    }
    batches = []
    # Changing n changes array shapes and requires a separate compiled batch.
    for group, n in enumerate(sizes):
        indices = np.arange(group * len(limits), (group + 1) * len(limits))
        scenario = jax.device_get(
            run_scenario(indices, n, replicates, np.array(list(limits.values())))
        )
        batches.append((indices, n, scenario))
    write_results(out, limits, batches)
    write_metadata(out, snapshot, limits, batches)
    write_evidence(out, out)


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
