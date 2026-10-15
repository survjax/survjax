# A5.5 follow-up — D-022 log-CDF upper tails

Implemented the approved plan on A5.5, following planning commit `dbf58e4`.

## Reading order

1. `_log_probability.py`: dtype-preserving log1mexp switches at -log(2).
2. `_survival_methods.py`, `exponential.py`, `weibull.py`: the three call sites.
3. `tests/unit/test_log_cdf_tails.py` and `tests/impl_tolerances.py`: independent references and relative-only tail assertions.

## Invariants and spec-to-test mapping

- Spec 7.2: both inactive branches receive safe operands before operations; only substituted constants have stopped gradients. No floors or clipping.
- Spec 4.2/7.1/12.1: `test_upper_tail_values_match_mpmath` checks all three paths against 100-digit mpmath, including H=20.1, 33.1, 54.6 and deeper normal-range tails. Zero cannot pass.
- Spec 7.3: `test_switch_and_upper_tail_parameter_derivatives` checks values, both AD modes and Hessians at/beside the switch and in the upper tail, eagerly and through jit(vmap). `test_interior_second_order_finite_differences` uses order-2 check_grads in both modes.
- Spec 4.5/7.2: `test_mixed_boundaries_do_not_contaminate_derivatives` compares finite-row Jacobians and Hessians with independent mpmath under eager debug_nans. Infinite log F(0) has no derivative assertion. NaN/undefined inputs are checked separately.
- Spec 5: `test_equal_length_grid_retains_upper_tails` checks K=N against per-cell mpmath references. Existing D-021, cure-limit, policy and grid regressions pass; wrappers, exports and grid adapters are unchanged.

## Checks actually run

Before the fix, `JAX_ENABLE_X64=1 .venv/bin/pytest tests/unit/test_log_cdf_tails.py -q` and its `JAX_ENABLE_X64=0` counterpart each exited 1: **3 failed**, all returning premature zero tails. After implementation the initial expanded module passed **27 cases** per mode; the final module is included in both runs below.

    JAX_ENABLE_X64=1 .venv/bin/pytest tests/unit/test_log_cdf_tails.py tests/unit/test_exponential.py tests/unit/test_weibull.py tests/unit/test_family_direct.py tests/unit/test_family_infinity.py tests/unit/test_survival_grid.py -q
    JAX_ENABLE_X64=0 .venv/bin/pytest tests/unit/test_log_cdf_tails.py tests/unit/test_exponential.py tests/unit/test_weibull.py tests/unit/test_family_direct.py tests/unit/test_family_infinity.py tests/unit/test_survival_grid.py -q

Both exited 0: x64 **216 passed, 1 skipped** (x32-only warning check); x32 **217 passed**.

    UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A5

Exited 0: **check: all gates passed through A5**. Own x64: **442 passed, 7 skipped**; own x32: **448 passed, 1 skipped**. Independent x64: **1116 passed, 26 skipped**; x32: **1121 passed, 21 skipped**. Source doctests: **1 passed** per mode. Guard passed, Ruff reported **48 files already formatted / All checks passed!**, mypy **Success: no issues found in 13 source files**. Log: `/private/tmp/survjax-d022-check.log`.

Standalone `.venv/bin/ruff format --check .`, `.venv/bin/ruff check .`, `.venv/bin/mypy src` and `git diff --check` also passed. No docs build or R checks were run; neither is part of this step.

## Remaining limitation and stop

The generic method cannot recover a lower tail once its primitive rounds log S to zero. No subnormal-tail accuracy claim is added. No new uncertainty was found; D-022 remains open for maintainer closure. Docs to-do is recorded in PLAN.md; spec, independent tests and docs pages remain untouched. Stop after the implementation commit for review.
