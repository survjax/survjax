"""Run the Weibull demo: simulate, fit in batches, summarize and plot."""

import argparse
import json
import os
import platform
import subprocess
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import survjax

if __package__:
    from . import study
    from ._plots import write_figures
else:
    import study
    from _plots import write_figures

ROOT_SEED = 20261025
TRUE_LOG_HR = float(np.log(0.7))


@partial(jax.jit, static_argnames=("model",))
def fit_batch(data, arm, *, model):
    """Fit data with shape (censoring limit, trial, patient)."""

    def fit(d):
        return study.fit_replicate(d, arm, model=model)

    return jax.vmap(jax.vmap(fit))(data)


def summarize(fits):
    """Summarize one scenario on the host, retaining all failure reasons."""
    finite = np.all(np.isfinite(fits.theta), axis=-1) & np.isfinite(fits.objective)
    reasons = {
        "solver": fits.estimable & ~fits.solver_success,
        "nonfinite": fits.estimable & ~finite,
        "gradient": fits.estimable
        & ~(np.max(np.abs(fits.gradient), axis=-1) < study.GRADIENT_TOLERANCE),
        "hessian": fits.estimable & ~fits.hessian_ok,
        "unestimable": ~fits.estimable,
    }
    gamma = fits.theta[fits.accepted, -1]
    se = fits.log_hr_se[fits.accepted]
    count = len(gamma)
    empirical_se = gamma.std(ddof=1) if count > 1 else np.nan
    row = {
        "attempted": len(fits.accepted),
        "accepted": count,
        "failed": int((~fits.accepted).sum()),
        **{name: int(mask.sum()) for name, mask in reasons.items()},
        "bias": gamma.mean() - TRUE_LOG_HR if count else np.nan,
        "bias_mcse": empirical_se / np.sqrt(count) if count else np.nan,
        "empirical_se": empirical_se,
        "empirical_se_mcse": empirical_se / np.sqrt(2 * (count - 1))
        if count > 1
        else np.nan,
        "model_se": np.sqrt(np.mean(se**2)) if count else np.nan,
    }
    for name, indicator in (
        ("coverage", np.abs(gamma - TRUE_LOG_HR) <= 1.96 * se),
        ("power", np.abs(gamma / se) > 1.96),
    ):
        fraction = indicator.mean() if count else np.nan
        row[name] = fraction
        row[name + "_mcse"] = (
            np.sqrt(fraction * (1 - fraction) / count) if count else np.nan
        )
    return row, reasons


def run_config(config: str, out: Path) -> None:
    """Run smoke (80 fits) or full (12,000 fits) and save tables and figures.

    Requires float64. The output directory must be new or empty. All numerical
    fitting is batched; Python loops vary sample size/model and format results.
    """
    if config not in ("smoke", "full"):
        raise ValueError("config must be smoke or full")
    if not jax.config.x64_enabled:
        raise ValueError("enable float64 before running this study")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("choose a new or empty output directory")
    sizes, replicates = ((100,), 20) if config == "smoke" else ((100, 300, 1000), 1000)
    limits = (50.0, 20.0)
    rows, failures, illustration, samples = [], [], [], []
    for group, n in enumerate(sizes):
        arm = jnp.repeat(jnp.arange(2), n // 2)

        def generate(index, limit, arm=arm):
            keys = jax.random.split(
                jax.random.fold_in(jax.random.key(ROOT_SEED), index), replicates
            )
            return jax.vmap(
                lambda key: study.generate_weibull(
                    key,
                    study.DGP_SHAPE,
                    study.DGP_SCALE,
                    limit,
                    log_hazard_ratio=TRUE_LOG_HR * arm,
                )
            )(keys)

        draws = jax.vmap(generate)(
            jnp.arange(2 * group, 2 * group + 2), jnp.array(limits)
        )
        data = survjax.SurvivalData(draws.time, event=draws.event)
        data.issues.raise_if_invalid()
        jax.block_until_ready(data)
        models = {}
        for model in ("weibull", "exponential"):
            elapsed = []
            for _ in range(4):
                start = perf_counter()
                fits = fit_batch(data, arm, model=model)
                jax.block_until_ready(fits)
                elapsed.append(perf_counter() - start)
            models[model] = jax.device_get(fits)
            for level, limit in enumerate(limits):
                fit = jax.tree.map(lambda x, level=level: x[level], models[model])
                measures, reasons = summarize(fit)
                scenario = 2 * group + level
                rows.append(
                    {
                        "scenario": scenario,
                        "n": n,
                        "censor_limit": limit,
                        "realized_censoring": float(
                            (~np.asarray(data.exact[level])).mean()
                        ),
                        "model": model,
                        **measures,
                        "timed_scenarios": 2,
                        "first_call_seconds": elapsed[0],
                        "warm_seconds": float(np.median(elapsed[1:])),
                        "fits_per_second": 2
                        * replicates
                        / float(np.median(elapsed[1:])),
                    }
                )
                samples.append((scenario, model, np.exp(fit.theta[fit.accepted, -1])))
                for i in np.flatnonzero(~fit.accepted):
                    failures.append(
                        {
                            "scenario": scenario,
                            "model": model,
                            "replicate": int(i),
                            "failed": 1,
                            **{name: int(mask[i]) for name, mask in reasons.items()},
                            "max_abs_gradient": float(np.max(np.abs(fit.gradient[i]))),
                            "log_hr": float(fit.theta[i, -1]),
                        }
                    )
        if n == (300 if config == "full" else 100):
            common = np.flatnonzero(
                models["weibull"].accepted[1] & models["exponential"].accepted[1]
            )
            if len(common):
                for model, fits in models.items():
                    theta = fits.theta[1, common[0]]
                    illustration.append(
                        {
                            "scenario": 2 * group + 1,
                            "model": model,
                            "replicate": int(common[0]),
                            "parameter_count": len(theta),
                            **{
                                f"theta_{i}": float(value)
                                for i, value in enumerate(theta)
                            },
                        }
                    )

    pd.DataFrame(rows).to_csv(out / "performance.csv", index=False)
    failure_fields = "scenario model replicate failed solver nonfinite gradient hessian unestimable max_abs_gradient log_hr".split()
    pd.DataFrame(failures, columns=failure_fields).to_csv(
        out / "failures.csv", index=False
    )
    pd.DataFrame(
        illustration,
        columns="scenario model replicate parameter_count theta_0 theta_1 theta_2".split(),
    ).to_csv(out / "illustration.csv", index=False)
    pooled = np.concatenate([values for _, _, values in samples])
    low = np.floor(pooled.min() / 0.025) * 0.025 if len(pooled) else 0.0
    high = (
        max(low + 0.025, np.ceil(pooled.max() / 0.025) * 0.025)
        if len(pooled)
        else 0.025
    )
    # Include extrema even if floating-point rounding places a bin edge inward.
    edges = np.arange(low - 0.025, high + 0.05, 0.025)
    histogram = {
        "edges": edges.tolist(),
        "series": [
            {
                "scenario": s,
                "model": m,
                "accepted": len(values),
                "counts": np.histogram(values, edges)[0].tolist(),
            }
            for s, m, values in samples
        ],
    }
    (out / "sampling_histograms.json").write_text(json.dumps(histogram))
    cpu = (
        subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        if platform.system() == "Darwin"
        else platform.processor()
    )
    metadata = {
        "configuration": config,
        "seed": ROOT_SEED,
        "seed_scheme": "fold_in(scenario), split(replicates), split(event,censor)",
        "sizes": sizes,
        "replicates": replicates,
        "censoring_limits": limits,
        "dgp": {
            "shape": study.DGP_SHAPE,
            "baseline_scale": study.DGP_SCALE,
            "log_hr": TRUE_LOG_HR,
        },
        "fitting": {
            "solver": "BFGS",
            "rtol": study.BFGS_RTOL,
            "atol": study.BFGS_ATOL,
            "max_steps": study.BFGS_MAX_STEPS,
            "gradient_tolerance": study.GRADIENT_TOLERANCE,
        },
        "runtime": {
            "cpu_model": cpu,
            "platform": platform.platform(),
            "x64_enabled": True,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "XLA_FLAGS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
            "packages": {
                name: version(name)
                for name in ("jax", "numpy", "optimistix", "survjax")
            },
        },
        "timing": "Both censoring limits together; first call and median of three warm calls, synchronized",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_figures(out, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("smoke", "full"), default="full")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    run_config(args.config, args.out)
