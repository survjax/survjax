"""Run from checkout: uv run --group examples examples/bayesian_rmst/run.py."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax

    from examples.bayesian_rmst.model import PosteriorDraws, Trial


def audit_posterior(
    samples: dict[str, jax.Array],
    derived: PosteriorDraws,
    table: dict[str, dict[str, float]],
    divergences: jax.Array,
) -> dict:
    """Audit chain-grouped parameters and flattened derived draws against truth.

    Args:
        samples: Three parameter arrays, each (chain, draw).
        derived: PosteriorDraws in the matching NumPyro flattened order.
        table: Per-quantity equal-tail intervals from summarize_draws.
        divergences: Boolean (chain, draw) NUTS flags.
    Returns:
        Diagnostics, true quantities, threshold settings and explicit failures.
    Raises:
        ValueError: Malformed axes or nonfinite draws (hard input errors, before
            an audit exists). Acceptance failures return evidence for saving.
    """
    import arviz as az
    import jax.numpy as jnp
    import numpy as np
    from numpyro.diagnostics import summary

    import survjax

    chain_shape = np.asarray(samples["log_k"]).shape
    if len(chain_shape) != 2 or chain_shape[0] != 4:
        raise ValueError("Expected four chains with explicit draw axes")
    if any(np.shape(value) != chain_shape for value in samples.values()):
        raise ValueError("Parameter chain/draw shapes differ")
    if np.shape(divergences) != chain_shape:
        raise ValueError("Divergence chain/draw shape differs")
    rmst = np.asarray(derived.rmst_tau)
    quantities = dict(samples)
    quantities.update(
        {
            "k": np.asarray(derived.shape).reshape(chain_shape),
            "HR": np.asarray(derived.hazard_ratio).reshape(chain_shape),
            "RMST0(5)": rmst[:, 0].reshape(chain_shape),
            "RMST1(5)": rmst[:, 1].reshape(chain_shape),
            "delta_RMST(5)": (rmst[:, 1] - rmst[:, 0]).reshape(chain_shape),
        }
    )
    diagnostics = {}
    failures = []
    for name, values in quantities.items():
        values = np.asarray(values)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite posterior draws for {name}")
        stats = summary({name: values}, group_by_chain=True)[name]
        metrics = {
            "r_hat": float(stats["r_hat"]),
            "numpyro_ess": float(stats["n_eff"]),
            "tail_ess": float(az.ess(values, method="tail")),
        }
        for metric, value in metrics.items():
            passed = np.isfinite(value) and (
                value < 1.01 if metric == "r_hat" else value > 400
            )
            if not passed:
                failures.append(f"{name}: {metric}={value}")
        diagnostics[name] = {
            key: value if np.isfinite(value) else None for key, value in metrics.items()
        }
    if np.any(divergences):
        failures.append(f"divergences={int(np.sum(divergences))}")
    base = survjax.Weibull(shape=1.38, scale=11.42)
    truth_rmst = survjax.ProportionalHazards(
        base, log_hazard_ratio=jnp.log(0.7) * jnp.array([0.0, 1.0])
    ).rmst(5.0)
    truth = {
        "k": 1.38,
        "HR": 0.7,
        "RMST0(5)": float(truth_rmst[0]),
        "RMST1(5)": float(truth_rmst[1]),
        "delta_RMST(5)": float(truth_rmst[1] - truth_rmst[0]),
    }
    coverage = {
        name: bool(table[name]["lower_95"] <= value <= table[name]["upper_95"])
        for name, value in truth.items()
    }
    for name in ("HR", "delta_RMST(5)"):
        if not coverage[name]:
            failures.append(f"{name}: truth outside 95% interval")
    return {
        "diagnostics": diagnostics,
        "truth": truth,
        "truth_in_95": coverage,
        "thresholds": {"r_hat_lt": 1.01, "ess_gt": 400, "divergences": 0},
        "failures": failures,
        "passed": not failures,
    }


def save_results(
    output: Path,
    record: dict,
    samples: dict[str, jax.Array],
    trial: Trial,
    grids: tuple[jax.Array, jax.Array],
    derived: PosteriorDraws,
    divergences: jax.Array,
) -> None:
    """Save JSON metadata and numerical results to a new output directory.

    Args:
        output: New directory; existing paths are rejected to protect prior runs.
        record: JSON-compatible run settings and audit, including failures.
        samples: Chain-grouped posterior parameters.
        trial: Synthetic observations, arm indicators and censoring calibration.
        grids: Survival and restriction grids in years.
        derived: Flattened draw-level summaries and curves.
        divergences: Chain-grouped boolean flags.
    Returns:
        None; writes metadata.json and arrays.npz for plotting/re-auditing.
    Raises:
        FileExistsError: Destination exists; no prior artifacts are overwritten.
    """
    import numpy as np

    output.mkdir(parents=True, exist_ok=False)
    arrays = {f"sample_{name}": np.asarray(value) for name, value in samples.items()}
    arrays.update(
        {
            "time": np.asarray(trial.data.lower),
            "event": np.asarray(trial.data.exact),
            "arm": np.asarray(trial.arm),
            "t_grid": np.asarray(grids[0]),
            "tau_grid": np.asarray(grids[1]),
            "divergences": np.asarray(divergences),
        }
    )
    arrays.update(
        {name: np.asarray(value) for name, value in derived._asdict().items()}
    )
    np.savez_compressed(output / "arrays.npz", **arrays)
    (output / "metadata.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )


def main() -> None:
    """Run approved smoke/full NUTS settings and print reproducibility metadata."""
    run_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--output", type=Path, help="New directory for metadata and numerical results"
    )
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("--output must name a new directory")

    import numpyro

    # Configure devices and precision before any JAX arrays/backends initialize.
    numpyro.set_host_device_count(4)
    import jax

    jax.config.update("jax_enable_x64", True)
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    # File execution adds this directory, rather than the checkout root.
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples.bayesian_rmst.model import (
        check_rmst_routes,
        make_trial,
        model,
        posterior_draws,
        summarize_draws,
        time_grids,
        use_case_key,
    )

    trial = make_trial()
    draws = 100 if args.config == "smoke" else 1000
    chain_method = "parallel" if jax.local_device_count() >= 4 else "vectorized"
    sampler = MCMC(
        NUTS(model),
        num_warmup=draws,
        num_samples=draws,
        num_chains=4,
        chain_method=chain_method,
        progress_bar=False,
    )
    start = time.perf_counter()
    sampler.run(use_case_key(1), trial.data, trial.arm, extra_fields=("diverging",))
    samples = sampler.get_samples(group_by_chain=True)
    extra = sampler.get_extra_fields(group_by_chain=True)
    jax.block_until_ready((samples, extra))
    elapsed = time.perf_counter() - start
    diagnostics = summary(samples, group_by_chain=True)
    post_start = time.perf_counter()
    t_grid, tau_grid = time_grids(trial.data)
    flat_samples = sampler.get_samples()
    derived = posterior_draws(flat_samples, t_grid, tau_grid)
    route_error = check_rmst_routes(flat_samples, trial, tau_grid, derived)
    table, positive_probability = summarize_draws(derived)
    jax.block_until_ready(derived)
    post_elapsed = time.perf_counter() - post_start
    audit = audit_posterior(samples, derived, table, extra["diverging"])
    record = {
        "config": args.config,
        "audit": audit,
        "run_seconds_before_output": time.perf_counter() - run_start,
        "posterior_table": table,
        "probability_delta_rmst_positive": positive_probability,
        "tau_max_years": float(tau_grid[-1]),
        "rmst_route_max_absolute_error": route_error,
        "postprocessing_seconds_including_compile": post_elapsed,
        "root_seed": 20261025,
        "use_case": 2,
        "nuts_stream": 1,
        "chains": 4,
        "warmup": draws,
        "draws_per_chain": draws,
        "chain_method": chain_method,
        "initialization": "NumPyro default init_to_uniform",
        "priors": {
            name: {"distribution": "Normal", "loc": 0.0, "scale": 5.0}
            for name in ("log_k", "log_b0", "gamma")
        },
        "x64": jax.config.x64_enabled,
        "backend": jax.default_backend(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("survjax", "jax", "jaxlib", "numpy", "numpyro", "arviz")
        },
        "censor_limit_years": trial.censor_limit,
        "censored_fraction": float(
            jax.numpy.mean((~trial.data.exact).astype(trial.data.lower.dtype))
        ),
        "sampling_seconds_including_compile": elapsed,
        "divergences_per_chain": extra["diverging"].sum(axis=1).tolist(),
        "numpyro_diagnostics": {
            name: {key: float(value) for key, value in values.items()}
            for name, values in diagnostics.items()
        },
        "diagnostic_scope": "full acceptance"
        if args.config == "full"
        else "exploratory smoke; thresholds not enforced",
    }
    if args.output is not None:
        save_results(
            args.output,
            record,
            samples,
            trial,
            (t_grid, tau_grid),
            derived,
            extra["diverging"],
        )
    print(json.dumps(record, indent=2, allow_nan=False))
    if args.config == "full" and not audit["passed"]:
        raise SystemExit(
            "Full posterior audit failed; investigate recorded failures without reseeding"
        )


if __name__ == "__main__":
    main()
