# A5.4 review fix-up — assert log-CDF switch derivatives

This test-only follow-up is based on commit `c477d31` on branch `A5.4`. The
runtime implementation and public API are unchanged.

## Reading order

1. `tests/unit/test_weibull.py`,
   `test_log_cdf_switch_matches_analytic_shape_gradient`.
2. `tests/unit/test_exponential.py`,
   `test_log_cdf_switch_matches_analytic_rate_gradient`.
3. `tests/impl_tolerances.py`, where the existing dtype-specific `ATOL` and
   `RTOL` values are defined.
4. `PLAN.md` records the requested review follow-up. D-021 in `DEVIATIONS.md`
   remains the numerical-policy note for the implementation.

## Invariants and test mapping

- The threshold points still straddle `log(eps)`, and `jax.debug_nans` remains
  enabled around the jitted values and gradients.
- Values are compared with the shared `ATOL[x64_enabled]` and
  `RTOL[x64_enabled]`. No test-local tolerance or new tolerance scale was
  added; this avoids requiring bit-identical `log`/`expm1` rounding near -36
  in float64 and -16 in float32.
- The Weibull gradient expectation is derived from
  `d log(F)/dk = log(t) * H/expm1(H)` at shape=scale=1. Since
  `H/expm1(H) = 1 + O(H)`, it approaches `log(t)` around the switch.
- The Exponential gradient expectation at rate=1 is
  `d log(F)/dr = t/expm1(t) = 1 + O(H)`. Both tests assert these expected
  gradients on each side, so a silently zero tiny-branch gradient fails.
- The hand-derived underflow-tail tests remain unchanged and continue to cover
  the log-H branch when forming H gives zero in float32.

## Checks actually run

    JAX_ENABLE_X64=1 .venv/bin/pytest tests/unit/test_weibull.py tests/unit/test_exponential.py -q
    JAX_ENABLE_X64=0 .venv/bin/pytest tests/unit/test_weibull.py tests/unit/test_exponential.py -q

x64: **36 passed, 1 skipped**. x32: **37 passed**.

    .venv/bin/ruff format --check .
    .venv/bin/ruff check .
    .venv/bin/mypy src
    git diff --check

All passed: 44 files formatted, Ruff clean, mypy clean for 11 source files, and
no whitespace errors. The full A5 gate was not rerun because this follow-up
changes tests only. Commit `c477d31` records its prior result: the implementation
and non-grid A5 checks passed, while revision-24 grid tests await A5.5.

## Uncertainty and stop

No new uncertainty or runtime behavior. Commit this test fix-up on A5.4 and
stop before the next step.
