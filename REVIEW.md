# A5.2b review fix-up — cure density substitution coverage

Starting at `71403c3` on the maintainer-selected `A5` branch. Review:
`../review/slices/A5.2b.md` (the requested `../reviewer` path does not exist).
Verdict: merge, with a test gap and readability nits. The maintainer authorized
implementing these comments; PLAN.md records this same-step fix-up.

## Reading order and invariants

1. `tests/unit/test_family_mixed_infinity.py`: `CureDensityOverride` and
   `test_cure_density_override_infinity_preserves_finite_derivative` address F-1.
   The primitive has finite log S(inf)=log(p); its deliberate density override
   is -r*t. At r=2, p=0.25 and t=[1,+inf], policy gives [-2,-inf]. The finite
   component has rate derivative -1, independently of the cure limit.
2. `src/survjax/distributions/_survival_methods.py`, guard configuration:
   F-2 now has one static conditional chain that assigns both the guard flag
   and boundary value for each generic method/primitive role. Generic hazard
   still decides guarding from the family's raw log-hazard method, because
   analytic overrides need their active limits. Its local is now named
   `log_hazard_method`, avoiding shadowing of the wrapper's method parameter.
3. PLAN.md's review fix-up scope. Existing DEVIATIONS.md entries remain open;
   no spec, registration table or public API was changed.

The source edit preserves numerical policy and evaluation order. Density
substitutes every positive infinity, even when log survival has a finite cure
limit; the original primitive probe determines NaN status. Other guarded
methods require both positive infinity and negative infinite log survival.
Only the reference is stopped, and analytic overrides remain active.

F-3 remains the documented structural argument: `isposinf(time)` explicitly
excludes finite times from the zero-limit guard. No new test asserts NaN
parameter gradients as promised behavior. Arbitrary unsafe custom overrides
and derivatives of nonfinite outputs remain outside the guarantee.

## Tests and mutation evidence

Spec 4.5/7.2: the new regression pins eager/jit density values and exact
finite-component first rate derivatives in forward/reverse mode, eager/jit
and jit/vmap. Expected values are the stated -r*t algebra and the density
infinity policy, not outputs from the implementation. No tolerance change.
Existing cure-fraction derivatives, analytic/inherited overrides, reference
reuse, NaN precedence and mixed-component Hessians remain regressions.

The new test alone passed in x64/x32: **1 passed, 22 deselected**
(1.23s/0.81s). Reviewer mutant M1 replaces
`substitute = infinite if density else zero_limit` with
`substitute = zero_limit`. It now gives **1 failed, 22 deselected** in each
mode (0.82s/0.79s), detecting the finite-component reverse-gradient NaN.
The source was restored before focused regressions and the full gate.

Focused three-module suite (`test_family_mixed_infinity.py`,
`test_family_infinity.py`, `test_family_density.py`): **70 passed** in each
separate dtype process (7.65s/7.96s).

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A4`
exited 0: **guard: ok (A5)**; Ruff **40 files already formatted**, **All checks
passed!**; mypy **no issues in 9 source files**. Own x64: **337 passed, 6 skipped**
(15.40s); x32: **342 passed, 1 skipped** (13.71s). Source doctests: **1 passed**
each (0.48s/0.48s). Independent x64: **679 passed, 25 skipped, 370 deselected**
(14.05s); x32: **693 passed, 11 skipped, 370 deselected** (11.96s).
Final output: **check: all gates passed through A4**. Existing skip counts are
unchanged. `git diff --check` passed with no output.

## Deferred work and stop

Full docs and `scripts/build_docs.sh` remain deferred until the end of A5 by
maintainer instruction. No full-doc files changed. Performance, R fixtures,
installation, rendered-page checks and debug-nans checks were not rerun for
this test/readability fix-up. Independent tests and spec remain read-only and
unchanged. All 370 A5 acceptance cases remain outside the through-A4 gate.
D-016/D-019 remain open for maintainer closure. Commit this fix-up and stop;
no A5.3 work was started.
