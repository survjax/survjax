# A5.3 review fix-up — explicit Exponential hazard broadcasting

Implemented on the maintainer-selected `a5.3-exponential` branch, starting at
`cc2f85d`, after the maintainer requested this reviewer suggestion. PLAN.md
records the same-step scope. Spec revision 23; numerical tests remain rev 22.

## Reading order and invariants

1. `src/survjax/distributions/exponential.py`, `log_hazard` and `hazard`: both
   compute `broadcast_shapes(shape(t), self.batch_shape)` before broadcasting
   log(rate) or rate. Their raw formulas no longer require the registration
   wrapper to pre-broadcast time. No numerical formula or public API changed.
2. `tests/unit/test_exponential.py`,
   `test_raw_hazard_methods_broadcast_scalar_and_multiaxis_times`: scalar time
   and time shape (2,1) against rates of shape (2,) give output shapes (2,)
   and (2,2). It checks raw and wrapped values, raw eager/jit and jit/vmap.
3. PLAN.md's reviewer broadcasting fix-up scope.

Expected values are analytically h(t)=r and log h(t)=log(r); the test does not
use the implementation as its oracle. It accesses the existing private raw
marker because the wrapper already hid the original shape dependency. No new
marker protocol is introduced. The raw calls use valid positive times; the
public wrapper still supplies negative/NaN/infinity time policies. Dtype,
rate gradients, analytic limits and pairing of equal-length vectors remain
unchanged. Existing parameter and boundary derivative tests are regressions.

## Checks actually run

Before the fix, the new test selection failed in both dtype modes:
**2 failed, 13 deselected** (x64 0.83s; x32 0.43s), on the scalar-time call
with batched rates. After the fix, the complete Exponential own module gave
**14 passed, 1 skipped** in x64 (2.38s) and **15 passed** in x32 (2.35s).
The skip is the existing x32-only float64-conversion-warning test.

The unchanged Exponential-only A5 acceptance selection was run in separate
`JAX_ENABLE_X64=1`/`0` processes with `SPEC_THROUGH=A5`:

```bash
.venv/bin/python -B -m pytest   tests/spec/test_a5_construction.py tests/spec/test_a5_from_log.py   tests/spec/test_a5_closed_forms.py tests/spec/test_a5_boundaries.py   tests/spec/test_a5_derivatives.py   -k '(exponential and not weibull) or nan_rate or boundary_values_agree_under_vmap_over_time'   -q -p no:cacheprovider
```

Results: x64 **95 passed, 227 deselected** (3.77s); x32 **91 passed, 4 skipped,
227 deselected** (2.36s). Existing x64-only finite differences account for
those skips. This is the existing selection for Exponential, not full A5
acceptance; Weibull, mixed-family construction and grids remain excluded.

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A4`
exited 0: **guard: ok (a5.3-exponential)**; Ruff **42 files already formatted**,
**All checks passed!**; mypy **no issues in 10 source files**. Own x64:
**351 passed, 7 skipped** (17.00s); x32: **357 passed, 1 skipped** (15.87s).
Source doctests: **1 passed** each (0.51s/0.54s). Independent through-A4 x64:
**679 passed, 25 skipped, 370 deselected** (15.03s); x32: **693 passed,
11 skipped, 370 deselected** (12.55s). Final output:
**check: all gates passed through A4**. Existing skip counts are unchanged.
`git diff --check` passed with no output.

## Deferred work and stop

Full documentation updates/builds remain deferred until the end of A5.
No docs files, independent tests, spec, alignment marker, exports, dependencies
or tolerances changed. No new benchmark, mutation, R fixture, wheel or rendered
page check was run. No claim is made about measured runtime cost. Existing
DEVIATIONS.md entries remain open for maintainer closure. Commit this fix-up
and stop before Weibull/grids or any further step.
