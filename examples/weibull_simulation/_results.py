"""Host-only result export and provenance for the JAX simulation batches."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import numpy as np

if TYPE_CHECKING:
    from .run import ScenarioRun

_BASE = "scenario n target_censoring censor_limit realized_censoring model".split()
_COUNTS = (
    "attempted accepted failed solver nonfinite gradient hessian unestimable".split()
)
_MEASURES = "bias bias_mcse empirical_se empirical_se_mcse model_se coverage coverage_mcse power power_mcse".split()
_TIMES = ("timed_scenarios", "first_call_seconds", "warm_seconds", "fits_per_second")
_FLAGS = ("solver_success", "hessian_ok", "estimable", "accepted")
_REASONS = ("failed", "solver", "nonfinite", "gradient", "hessian", "unestimable")
# Static CSV schema: Weibull has three parameters, Exponential two (unused cells blank).
_RAW = (
    *_BASE,
    "replicate",
    "parameter_count",
    "log_hr",
    "objective",
    "log_hr_se",
    "hr_lower",
    "hr_upper",
    *(f"{name}_{i}" for name in ("theta", "gradient") for i in range(3)),
    *(f"hessian_{i}_{j}" for i in range(3) for j in range(3)),
    *_FLAGS,
    *_REASONS,
)


def write_results(
    out: Path,
    limits: dict[float, float],
    batches: list[tuple[np.ndarray, int, ScenarioRun]],
) -> None:
    """Write performance.csv and every replicate, including rejected fits.

    Args:
        out: Existing output directory; both CSVs are replaced.
        limits: Ordered target-to-limit mapping shared by the batches.
        batches: (indices, n, ScenarioRun) records already transferred to NumPy.
    Raises:
        OSError: Output cannot be written. No fitting/reductions happen here.
    """
    # Once CSV replacement starts, old provenance must not certify a partial rerun.
    (out / "metadata.json").unlink(missing_ok=True)
    with (
        (out / "performance.csv").open("w", newline="") as summary_file,
        (out / "replicates.csv").open("w", newline="") as raw_file,
    ):
        summary_writer = csv.DictWriter(
            summary_file, fieldnames=(*_BASE, *_COUNTS, *_MEASURES, *_TIMES)
        )
        raw_writer = csv.DictWriter(raw_file, fieldnames=_RAW)
        summary_writer.writeheader()
        raw_writer.writeheader()
        # These loops only serialize computed host records; no numerical work is dispatched.
        for indices, n, scenario in batches:
            for level, target in enumerate(limits):
                for model in ("weibull", "exponential"):
                    record = scenario.models[model]
                    base = dict(
                        zip(
                            _BASE,
                            (
                                int(indices[level]),
                                n,
                                target,
                                scenario.censor_limit[level],
                                scenario.realized_censoring[level],
                                model,
                            ),
                            strict=True,
                        )
                    )
                    summary_writer.writerow(
                        {
                            **base,
                            **{
                                name: int(record.failures[name][level])
                                for name in _COUNTS
                            },
                            **{
                                name: float(record.performance[name][level])
                                for name in _MEASURES
                            },
                            "timed_scenarios": len(limits),
                            "first_call_seconds": record.first_call_seconds,
                            "warm_seconds": record.warm_seconds,
                            "fits_per_second": record.fits_per_second,
                        }
                    )
                    fits = record.fits
                    for replicate in range(fits.theta.shape[1]):
                        p = fits.theta.shape[-1]
                        row = {
                            **base,
                            "replicate": replicate,
                            "parameter_count": p,
                            "log_hr": fits.theta[level, replicate, -1],
                            "objective": fits.objective[level, replicate],
                            "log_hr_se": fits.log_hr_se[level, replicate],
                            "hr_lower": fits.hr_interval[level, replicate, 0],
                            "hr_upper": fits.hr_interval[level, replicate, 1],
                            **{
                                f"{name}_{i}": getattr(fits, name)[level, replicate, i]
                                for name in ("theta", "gradient")
                                for i in range(p)
                            },
                            **{
                                f"hessian_{i}_{j}": fits.hessian[level, replicate, i, j]
                                for i in range(p)
                                for j in range(p)
                            },
                            **{
                                name: int(getattr(fits, name)[level, replicate])
                                for name in _FLAGS
                            },
                            **{
                                name: int(record.failure_masks[name][level, replicate])
                                for name in _REASONS
                            },
                        }
                        raw_writer.writerow(row)


_ROOT = Path(__file__).resolve().parents[2]
_CPUINFO = Path("/proc/cpuinfo")


def _cpu_model() -> str | None:
    """Inspect host CPU identity once; None means unavailable."""
    if platform.system() == "Darwin":
        command = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.check_output(command, text=True).strip() or None
    if _CPUINFO.exists():
        # Host metadata probing precedes simulation; this is not a numerical loop.
        for line in _CPUINFO.read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() in ("model name", "Model"):
                return value.strip() or None
    return platform.processor() or None


def capture_metadata(config: str, root_seed: int, log_hr: float) -> dict[str, object]:
    """Snapshot config, root_seed, log_hr and host facts before work starts.

    Returns JSON-native facts; unset thread settings are null and effective
    worker count is unmeasured. OSError/CalledProcessError propagate when
    required host inspection fails.
    """
    # Fixed host provenance fields; no scenario/replicate computations occur here.
    versions = {
        name: version(name)
        for name in ("survjax", "jax", "jaxlib", "numpy", "optimistix", "matplotlib")
    }
    threads = {
        name: os.environ.get(name)
        for name in "XLA_FLAGS OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS TF_NUM_INTRAOP_THREADS TF_NUM_INTEROP_THREADS JAX_NUM_CPU_DEVICES".split()
    }
    return {
        "schema_version": 1,
        "configuration": config,
        "source": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
            ).strip(),
            "worktree_dirty_at_start": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=_ROOT, text=True
                ).strip()
            ),
            "lock_sha256": hashlib.sha256((_ROOT / "uv.lock").read_bytes()).hexdigest(),
            "spec_seen": (_ROOT / ".spec-seen").read_text().strip(),
        },
        "invocation": {
            "python_argv": list(sys.orig_argv),
            "cwd": str(Path.cwd()),
            "entry_point": "run_config",
        },
        "runtime": {
            "python": platform.python_version(),
            "packages": versions,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "x64_enabled": jax.config.x64_enabled,
            "prng_impl": jax.config.jax_default_prng_impl,
            "thread_environment": threads,
            "effective_worker_count": "not measured",
        },
        "seeds": {
            "root": root_seed,
            "scheme": "key(root) -> fold_in(scenario_index) -> split(R) -> split(U,C)",
        },
        "dgp": {
            "shape": 1.38,
            "baseline_scale": 11.42,
            "log_hr": log_hr,
            "allocation": "fixed n/2 per arm",
            "censoring": "independent Uniform(0,c)",
        },
        "parameters": {
            "weibull": ["log_shape", "log_scale", "log_hr"],
            "exponential": ["log_rate", "log_hr"],
        },
        "fitting": {
            "solver": "optimistix.BFGS",
            "rtol": 1e-8,
            "atol": 1e-8,
            "max_steps": 1000,
            "throw": False,
            "starts": "Weibull (0,log(mean(Y)),0); Exponential (-log(mean(Y)),0)",
            "acceptance": "solver success; finite theta/objective; max|gradient|<1e-5; finite positive-definite Hessian via Cholesky",
            "unestimable": "at least one arm has no observed event",
            "inference": "inverse exact Hessian; HR interval exp(log_hr +/-1.96 SE)",
        },
        "timing_policy": {
            "scope": "all censoring levels at fixed n, per model (D-034)",
            "first_call_seconds": "synchronized first call; includes compilation if uncached",
            "warm_seconds": "median of three subsequent synchronized calls",
            "fits_per_second": "timed_scenarios * replicates_per_scenario / warm_seconds",
        },
        "csv_conventions": "unused parameters blank; raw nan/inf/-inf retained; flags/reasons 0/1; reasons overlap; failed unique",
    }


def write_metadata(
    out: Path,
    snapshot: dict[str, object],
    limits: dict[float, float],
    batches: list[tuple[np.ndarray, int, ScenarioRun]],
) -> None:
    """Write strict metadata.json after successful CSV export.

    Args:
        out: Existing output directory.
        snapshot: Pre-run facts from capture_metadata.
        limits, batches: Completed host inputs as for write_results.
    Raises:
        OSError: Output cannot be written. ValueError: A metadata number is nonfinite.
    """
    groups = []
    # These host loops serialize batch/scenario metadata, never perform fitting.
    for indices, n, scenario in batches:
        groups.append(
            {
                "n": n,
                "replicates_per_scenario": scenario.models["weibull"].fits.theta.shape[
                    1
                ],
                "timed_scenarios": len(limits),
                "scenarios": [
                    {
                        "index": int(indices[level]),
                        "target_censoring": target,
                        "censor_limit": float(scenario.censor_limit[level]),
                        "realized_censoring": float(scenario.realized_censoring[level]),
                    }
                    for level, target in enumerate(limits)
                ],
                "models": {
                    model: {
                        "first_call_seconds": record.first_call_seconds,
                        "warm_seconds": record.warm_seconds,
                        "fits_per_second": record.fits_per_second,
                    }
                    for model, record in scenario.models.items()
                },
            }
        )
    metadata = {
        **snapshot,
        "output_directory": str(out.resolve()),
        "calibration": [
            {"target_censoring": target, "censor_limit": c}
            for target, c in limits.items()
        ],
        "batches": groups,
    }
    payload = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    (out / "metadata.json").write_text(payload)
