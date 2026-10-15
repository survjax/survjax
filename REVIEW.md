# A5.5 review follow-up — shared guarded log arithmetic

Implements the approved refactor after `0369331` on A5.5.

## Reading order

1. `src/survjax/distributions/_log_math.py`: `safe_log`, `safe_log1mexp` and `log_cdf_from_log_hazard`. The old `_log_probability.py` is removed.
2. `_survival_methods.py`: the three numerical call sites; distribution-aware helpers and policy wrappers stay here.
3. `exponential.py`, `weibull.py`: each log-CDF override now delegates in one line, retaining its family's analytic log H.
4. `tests/unit/test_log_cdf_tails.py`: deeper tail derivatives and the mixed-objective regression.

## Invariants and spec-to-test mapping

- Spec 4.1–4.2/7.2: existing guarded formulas are moved unchanged. Both built-ins share the lower-tail switch, overflow handling and safe exponentiation. No clipping, new threshold, custom derivative or public export.
- Spec 7.2: `safe_log` retains its zero substitution; `safe_log1mexp` retains safe operands in both branches. Stops apply only to substituted constants. Existing generic density/hazard tests cover the moved safe-log calls.
- Spec 7.3/12.1: `test_switch_and_upper_tail_parameter_derivatives` now checks H=54.6 and H=70 (x32) / H=600 (x64), including both Weibull parameters, against 100-digit mpmath. Eager and jit(vmap) values, Jacobians in both modes and Hessians use unchanged relative-only tolerances.
- Spec 7.2–7.3: `test_negligible_cdf_term_preserves_mixed_objective_hessian` checks L(eta)=exp(eta)-eta-log F(c) for all three family paths. Independent analytical derivatives retain the tiny CDF correction using mpmath. Value, gradient and Hessian are finite eagerly and under jit with debug_nans; useful curvature remains about 2. This is a synthetic composition, not a new likelihood API or optimizer test.
- Existing D-021 lower-tail, zero/infinity, NaN precedence, parameter-dependent cure limits, second-order finite differences and time-first grid regressions remain applicable.

## Checks actually run

    JAX_ENABLE_X64=1 .venv/bin/pytest tests/unit/test_log_cdf_tails.py -q
    JAX_ENABLE_X64=0 .venv/bin/pytest tests/unit/test_log_cdf_tails.py -q

Both exited 0: **36 passed** per mode, with no skips.

    JAX_ENABLE_X64=1 .venv/bin/pytest tests/unit/test_log_cdf_tails.py tests/unit/test_exponential.py tests/unit/test_weibull.py tests/unit/test_family_direct.py tests/unit/test_family_density.py tests/unit/test_family_infinity.py tests/unit/test_survival_grid.py -q
    JAX_ENABLE_X64=0 .venv/bin/pytest tests/unit/test_log_cdf_tails.py tests/unit/test_exponential.py tests/unit/test_weibull.py tests/unit/test_family_direct.py tests/unit/test_family_density.py tests/unit/test_family_infinity.py tests/unit/test_survival_grid.py -q

Both exited 0: x64 **249 passed, 1 skipped** (x32-only warning check); x32 **250 passed**.

    UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A5

Exited 0: **check: all gates passed through A5**. Own x64: **451 passed, 7 skipped**; own x32: **457 passed, 1 skipped**. Independent x64: **1116 passed, 26 skipped**; x32: **1121 passed, 21 skipped**. Source doctests: **1 passed** per mode. Guard passed; Ruff and mypy passed with the outputs below. Log: `/private/tmp/survjax-log-math-check.log`.

Standalone Ruff format/lint, mypy and `git diff --check` passed: **48 files already formatted / All checks passed! / Success: no issues found in 13 source files**. No docs build or R checks were run; neither is part of this refactor.

## Remaining limitation and stop

No new uncertainty was found. This step keeps the guarded implementation; adopting JAX values with a corrected custom JVP requires a separate numerical plan. The generic default still cannot recover a lower tail once its primitive rounds log S to zero; subnormal-tail accuracy is not claimed. Current spec revision 25 and the revision-24 `.spec-seen` marker remain untouched, as planned. Deviations remain open for the maintainer; docs to-do stays in PLAN.md. Stop after this commit for review.
