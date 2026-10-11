# A5.1 — NaN precedence for density at infinity

Implemented on `A5` after maintainer sync `89cb9f4`, against specification
revision 22. Approved planning record: `b85f2a1`. Commit this step, then stop.

## Reading order and invariants

1. `tests/unit/test_family_infinity.py`: constant/quadratic primitives and
   deliberately conflicting density overrides; all expected constants come
   from section 4.5's boundary policy, not the implementation.
2. `_survival_methods.py`: the density wrapper checks raw log survival on the
   already-substituted safe times. At +inf, only NaN log survival gives NaN
   density; otherwise density is -inf, even if its raw formula gives inf-inf.
3. `docs/distributions.md`: the updated infinity row and executable assertions.
   `PLAN.md` records the remaining separately approved steps.

Spec 4.2/4.5: the default and overrides follow the same infinity rule.
`test_nan_parameter_density_at_infinity_is_nan` covers scalar eager/jit/vmap;
`test_infinite_density_keeps_nan_and_valid_batch_entries_separate` covers
mixed batches and divergent valid density formulas. The two conflicting-override
tests prove that primitive NaN status, rather than density NaN status, decides.
The remaining tests pin finite and negative-time override behavior.

Spec 7.2: the additional raw call receives the existing safe_time; it never
calls the policy wrapper or obtains another reference. Existing seven-method
reference-call tests still pass. No parameter domain is inferred, gradients
are not stopped on real parameters, and no public API or tolerance changes.

## Checks and exact results

Before the source fix, `JAX_ENABLE_X64=1` and `0` with
`.venv/bin/python -m pytest tests/unit/test_family_infinity.py -q` each gave
**7 failed, 1 passed** (1.26s); a repeated pre-fix run gave the same counts
(0.81s). After the fix, each gave **8 passed** (1.09s).
Ruff formatting of the changed source/test files: **1 reformatted, 1 unchanged**.

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A4`
exited 0: **guard: ok (A5)**; Ruff **39 files already formatted**, **All checks
passed!**; mypy **no issues in 9 source files**. Own x64: **299 passed, 6 skipped**
(11.73s); own x32: **304 passed, 1 skipped** (10.73s). Source doctests:
**1 passed** each (0.48s/0.46s). Independent x64: **679 passed, 25 skipped,
370 deselected** (15.45s); x32: **693 passed, 11 skipped, 370 deselected**
(12.91s). Final output: **check: all gates passed through A4**.
All three new revision-22 A4 regressions pass in the synced suite (`7f1d6df`).
Existing skips are dtype-specific, optional pandas, or x64-only finite differences.

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/build_docs.sh` exited 0:
**build succeeded** for x32 and x64. The distribution guide, playground and
SurvivalData guide executed afresh in both modes, including both new assertions.
Kernels printed their existing TCP transport warning; Sphinx reported no errors.
`git diff --check` passed with no output.

## Limits and next stop

Density now evaluates the raw primitive once more per call, also at finite
times; performance was not measured. No custom-family boundary-gradient repair,
built-ins or grids were implemented. The 370 A5 acceptance cases remain
unimplemented/deselected; the full A5 gate was not run. No R, installation or
rendered-page visual check was run. Independent tests, spec, alignment marker,
exports and dependencies were untouched. D-016/D-019 remain open for maintainer
closure. Stop here before A5.2, as required by the approved plan and AGENTS.md.
