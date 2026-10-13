# A5.2b — Isolate finite-entry derivatives from infinity entries

Implemented on `A5`, starting at `8ab4c4d`, after the maintainer approved the
revised plan on 2026-10-13. Specification revision 22; independent tests
`7f1d6df`. This addresses A5.2 review F-1. D-016 remains open for maintainer
closure. Full documentation is deferred until the end of slice A5.

## Reading order and invariants

1. `tests/unit/test_family_mixed_infinity.py`: read the analytical finite-entry
   derivative/Hessian cases, then cure-limit and analytic-override regressions.
2. `src/survjax/distributions/_survival_methods.py`, `apply_policy`: the
   original primitive probe supplies Boolean conditions only. Density always
   substitutes positive infinity before raw evaluation, including overrides;
   its final -inf/NaN policy uses the original probe's NaN status. The NaNLimit
   test distinguishes the original infinity from its finite reference.
3. `src/survjax/distributions/base.py`: registration supplies a private primitive
   flag for log survival; arbitrary author function names are never inspected.
   Existing raw markers and generic function identities retain their roles.
4. Function docstrings, PLAN.md's approved scope and DEVIATIONS.md's D-016 note.

For log survival and the generic survival/log-CDF/cumulative-hazard/log-hazard,
substitution requires positive infinite time AND negative infinite raw log
survival. Selected boundary constants are -inf, 0, 0, +inf and NaN respectively.
Generic hazard gets the NaN guard only if its underlying log-hazard is generic;
analytic log-hazard/hazard and cumulative-hazard overrides remain active.
Finite cure limits retain parameter dependence. The one reference is reused
and only the substituted reference is stopped. No active value or real
parameter is stopped, and no floor, epsilon or clipping is introduced.

The new condition never substitutes finite times, even with an underflowing
tail; this restriction rests on the explicit isposinf condition. NaN primitive
limits remain NaN. Existing negative/NaN and zero-time behavior is unchanged.
No public API, exports, dependencies, tolerances, spec, independent tests or
alignment marker changed. Derivatives of infinite/NaN outputs, discontinuous
parameter limits and arbitrary unsafe custom overrides are not guaranteed.

## Tests mapped to requirements

- Spec 7.2/7.3: `test_finite_component_derivatives_beside_infinity` checks the
  finite component of [1,+inf] for shared scalar parameters and independent
  parameter batches, constant/quadratic primitives, all five affected methods,
  forward/reverse derivatives and Hessians, eager/jit/jit-vmap. Expected values
  and derivatives follow the algebra in the test; no nonfinite sum is used.
- Spec 4.5: the same cases pin existing infinity values (-inf/+inf/NaN), without
  treating their derivatives as guaranteed. `test_density_nan_status_comes_from_original_infinite_time`
  proves that the reference does not decide density's NaN precedence.
- Spec 4.2/7.3: `test_cure_infinite_log_limit_remains_parameter_dependent` checks
  log(p)/-log(p) values, first derivatives and Hessians; analytic/inherited
  override tests preserve hazard, log-hazard and cumulative-hazard derivatives.
- Spec 4.5/7.2: the density override regression isolates finite derivatives
  despite its infinity-sensitive formula. `test_mixed_call_reuses_one_reference`
  checks all five affected methods. Existing A5.1 NaN/finite density overrides,
  A5.2 survival/log-CDF, finite-tail and seven-method reference tests also pass.

## Checks actually run

Before the source fix, the new module gave **8 failed, 13 passed** in each
mode (2.32s/2.34s), reproducing the mixed-entry derivative leak. After the fix,
the focused three-module suite initially gave **68 passed** in each mode
(8.24s/7.93s). After adding the density override case and boundary assertions,
`JAX_ENABLE_X64=1` and `0` with `.venv/bin/pytest -q
 tests/unit/test_family_mixed_infinity.py tests/unit/test_family_infinity.py
 tests/unit/test_family_density.py` gave **69 passed** (8.23s/9.08s).

Mutation: removing only the new safe-time substitution while retaining final
value masks gave **8 failed, 2 passed, 12 deselected** in both modes
(2.52s/2.58s). The source was restored before final tests and the full gate.

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A4`
exited 0: **guard: ok (A5)**; Ruff **40 files already formatted**, **All checks
passed!**; mypy **no issues in 9 source files**. Own x64: **336 passed, 6 skipped**
(17.68s); x32: **341 passed, 1 skipped** (16.14s). Source doctests: **1 passed**
each (0.57s/0.55s). Independent x64: **679 passed, 25 skipped, 370 deselected**
(16.60s); x32: **693 passed, 11 skipped, 370 deselected** (14.48s).
Final output: **check: all gates passed through A4**. Existing skips unchanged.

Separate x64/x32 eager checks under `jax.debug_nans(True)` passed for quadratic
mixed-batch values and finite-component reverse derivatives of log survival,
cumulative hazard, density, survival and log-CDF. Generic hazard/log-hazard
were excluded because their intended infinity output is NaN; invalid-parameter
cases were excluded. `git diff --check` passed with no output.

## Deferred work and stop

Full docs and `scripts/build_docs.sh` were deliberately not changed/run, per
maintainer instruction. At the end of A5, explain mixed-entry isolation and
its limits in the distribution guide and execute documentation in both modes.
Function docstrings are updated now. R fixtures, installation, rendered-page
checks, the full A5 gate and performance measurement were not run. All 370 A5
acceptance cases remain deselected until built-ins/grids exist.

The raw primitive is evaluated once more for guarded generic methods and the
log-survival primitive, including finite calls. Density reuses its existing
probe before substitution rather than adding another one. Cost is unmeasured.
Source/test/docstring changes total about 280 lines; planning/review records
add documentation overhead. Commit this numerical step and stop before A5.3.
