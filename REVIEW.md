# A5.2 — Generic finite boundary gradients and extension recipe

Implemented on the maintainer-selected `A5` branch, starting at `3f50962`,
against specification revision 22 and independent tests `7f1d6df`.
The maintainer approved the detailed PLAN.md before implementation.

## Reading order and invariants

1. `tests/unit/test_family_infinity.py`: read the proper-family gradient tests,
   mixed finite/infinite Hessians, then the guarded cure primitive and its
   analytical fraction derivatives. Earlier A5.1 density cases remain intact.
2. `src/survjax/distributions/_survival_methods.py`: the existing policy wrapper
   recognizes only the generic survival/log-CDF by static function identity.
   It probes raw log survival at the actual infinity, uses that result only
   as a Boolean mask, substitutes the stopped reference before the default's
   operations, then selects zero. One reference call is reused throughout.
3. `docs/distributions.md`: the executable cure recipe and gradient assertions,
   exact-shape/dtype PiecewiseHazard reference, and Gompertz display correction.
   PLAN.md contains the approved scope and remaining separately planned steps.

The guard requires both positive infinite time and negative infinite raw log
survival. Finite cure limits remain active; their parameter derivatives survive.
NaN limits remain NaN. Finite times are never guarded by this new condition,
including tails where survival underflows. Overrides, density/hazard policy,
zero-time behavior, exports, dependencies and tolerances are unchanged.
Only the substituted reference is stopped; real parameters and active results
are not. Custom authors remain responsible for safe primitive operands and a
valid reference input. The guarantee does not extend to arbitrary unsafe cure
primitives, invalid parameters or discontinuous parameter boundaries.

## Requirements demonstrated by tests

- Spec 4.5/7.2: `test_proper_infinity_has_zero_parameter_gradient` tests constant
  and quadratic primitives, exact zero first parameter derivatives in forward
  and reverse modes, eager/jit and jit/vmap composition.
- Spec 7.2/7.3: `test_mixed_finite_infinite_gradient_and_hessian` checks sums
  against analytical values, gradients and Hessians, plus independent batched
  parameters. An inactive infinity entry cannot contaminate the finite entry.
- Spec 4.2/7.3: `test_cure_limit_preserves_fraction_derivative` checks finite
  and infinite values and forward/reverse fraction/rate derivatives under
  jit/vmap, including the nonzero derivative of log(1-p).
- Spec 4.5: `test_infinite_survival_methods_keep_nan_entries` preserves entrywise
  NaN precedence; `test_infinite_survival_overrides_are_honored` checks explicit
  and inherited dispatch. `test_infinite_default_uses_one_reference_call`
  pins reference reuse. Existing seven-method finite reference tests also pass.
- Spec 4.2/7.2: `test_finite_tail_and_lower_tail_remain_active` checks finite
  underflow values and the analytical lower-tail log-CDF/rate derivative.
  Existing zero-CDF, negative/NaN-time, interior second-order and density tests
  remain regressions. Spec 10.5: the extension recipe executes in both modes.

## Checks and exact results

Before the fix, the focused module gave **6 failed, 16 passed** in each dtype
mode: infinity parameter derivatives and mixed-batch derivatives were NaN.
After the fix and final test additions, `JAX_ENABLE_X64=1` and `0` with
`.venv/bin/pytest -q tests/unit/test_family_infinity.py` each gave **23 passed**
(2.34s/2.49s). Mutation: removing only the safe-time substitution while retaining
final output masking gave **4 failed, 19 deselected** in each mode (1.03s/1.04s).
The source was restored before all final checks.

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A4`
exited 0: **guard: ok (A5)**; Ruff **39 files already formatted**, **All checks
passed!**; mypy **no issues in 9 source files**. Own x64: **314 passed, 6 skipped**
(12.12s); x32: **319 passed, 1 skipped** (11.70s). Source doctests: **1 passed**
each (0.50s/0.49s). Independent x64: **679 passed, 25 skipped, 370 deselected**
(15.19s); x32: **693 passed, 11 skipped, 370 deselected** (12.61s).
Final output: **check: all gates passed through A4**. Existing skip counts are
unchanged; independent tests were not edited, skipped or xfailed by this step.

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/build_docs.sh` initially
failed with sandbox `PermissionError` during kernel startup. The permitted
rerun exited 0: **build succeeded** for x32 and x64; all three executable guides
ran afresh, including the new cure and infinity-gradient assertions. Kernels
printed the existing TCP transport warning; Sphinx reported no errors.
`git diff --check` passed with no output.

## Limits and stop

Generic survival/log-CDF now evaluate the raw primitive once more per call,
including finite calls; performance was not measured. Source/test/doc changes
are about 230 added lines; planning and review records add documentation overhead.
No R fixture, wheel installation or rendered-page visual check was run.
Built-ins and grids are future steps: all 370 A5 acceptance cases remain
unimplemented/deselected and the full A5 gate was not run. Spec, independent
tests, alignment marker and DEVIATIONS.md remain untouched; D-016 stays open
for maintainer closure. Commit this step and stop before A5.3.
