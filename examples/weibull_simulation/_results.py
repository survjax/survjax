"""Host-only CSV serialization of the completed JAX scenario batches."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

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
