# A5.5 review fix-up — explicit cumulative-hazard names and helper placement

Implements the approved plan after `9a8bb2a` on A5.5.

## Reading order

1. `_survival_methods.py`: `log_cdf_from_log_cumulative_hazard` moved here beside generic `log_cdf`, with explicit `log_cumulative_hazard`, `safe_log_cumulative_hazard` and `log_cdf_from_cumulative_hazard` identifiers.
2. `_log_math.py`: retains only general guarded log arithmetic. Exponential and Weibull import the conversion from `_survival_methods.py` and keep their analytic log-H overrides.
3. `tests/unit/test_log_cdf_tails.py`, `test_exponential.py`, `test_weibull.py`: cumulative-hazard quantities and derivative arrays have explicit names; instantaneous log-hazard derivatives remain distinct.

## Invariants and spec-to-test mapping

- Spec 4.1–4.2/7.2: helper formulas, thresholds, safe substitutions and derivative paths are unchanged. H denotes cumulative hazard and h instantaneous hazard in mathematical comments. The generic default still uses safe_log1mexp(log S); built-ins retain log H when H underflows.
- Spec 4.1/7.2/12.1: existing lower-tail underflow and switch-gradient tests exercise the relocated conversion through both public built-ins. No expected values, tolerances, cases or assertions changed.
- Spec 7.2–7.3/12.1: the tail module retains mpmath values/Jacobians/Hessians, both AD modes and jit/vmap, boundary debug_nans and finite mixed-objective Hessians. The renamed Weibull hand-derived gradient/Hessian arrays preserve their original algebra.
- Spec 4.2/4.5/5: generic direct/density/infinity tests and grid tests verify the import relocation and callers, including family dispatch, cure limits and time-first grids.
- The audit of src and own tests found no remaining abbreviated cumulative-hazard identifiers. Actual instantaneous-hazard APIs/bindings and mathematical H/h notation remain. Public exports and numerical behavior are unchanged; no new uncertainty or deviation was identified.

## Checks actually run

    JAX_ENABLE_X64=1 .venv/bin/pytest tests/unit/test_log_cdf_tails.py tests/unit/test_exponential.py tests/unit/test_weibull.py tests/unit/test_family_direct.py tests/unit/test_family_density.py tests/unit/test_family_infinity.py tests/unit/test_survival_grid.py -q
    JAX_ENABLE_X64=0 .venv/bin/pytest tests/unit/test_log_cdf_tails.py tests/unit/test_exponential.py tests/unit/test_weibull.py tests/unit/test_family_direct.py tests/unit/test_family_density.py tests/unit/test_family_infinity.py tests/unit/test_survival_grid.py -q

Both exited 0: x64 **249 passed, 1 skipped** (x32-only warning check); x32 **250 passed**.

    .venv/bin/ruff format --check .
    .venv/bin/ruff check .
    .venv/bin/mypy src
    git diff --check

All exited 0: **48 files already formatted**, **All checks passed!**, **Success: no issues found in 13 source files**, and no whitespace errors.

## Historical gate evidence and stop

No full-gate rerun for this relocation/naming-only fix-up, as planned. `7208c81` passed the full A5 gate: own x64 **451 passed, 7 skipped**, x32 **457 passed, 1 skipped**; independent x64 **1116 passed, 26 skipped**, x32 **1121 passed, 21 skipped**; source doctests **1 passed** per mode; guard, Ruff and mypy passed. These are historical results.

No docs build or R checks were run. Spec, marker, docs pages, independent tests, dependencies, tolerances and deviation statuses remain untouched. Docs to-do is recorded in PLAN.md. Stop after this fix-up commit for review.
