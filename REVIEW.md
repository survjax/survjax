# A6.1 follow-up — ordinary JAX constraint comparisons

Status: approved follow-up implemented on `A6.1`, based on `b26a461`.
Focused tests and the through-A5 gate pass in both dtypes. Ready for review;
stop after this commit. No A6.2 work has begun.

The maintainer requested removing frexp, updated spec section 4.7 to revision
30 and resumed implementation after the independent tests were updated.
Marker: `fd150db`. No specification or protected-test edits in this step.

## Read first

1. `src/survjax/constraints.py`: nonnegative, _GreaterThan and _Interval.
2. `tests/unit/test_constraints.py`: signed-zero boundary test replacing the
   obsolete mathematical-subnormal-sign guarantee. The simplex exact-threshold
   and huge-bound regression tests are retained.
3. PLAN.md's approved plain-comparison follow-up and the revision-30 notes
   in DEVIATIONS.md D-024/D-025; formal entry closure stays with the maintainer.

## Invariants and changes

- nonnegative is finite x>=0. Both signed zeros remain valid; positive and
  open intervals starting at zero exclude them. Simplex still permits zero
  weights. The finite-bound TypeError policy and exact 4*n*eps simplex
  tolerance are unchanged.
- Every comparison uses ordinary JAX operators on the parameter array.
  Bound factories explicitly convert bounds to x.dtype before comparison.
  There is no zero-bound special case, frexp, decomposition or bit-level
  replacement. rg on src/survjax and the own constraint test finds no frexp.
- Spec 4.7 revision 30 permits backend-dependent subnormal classification.
  Neither values nor eager/jit/vmap agreement at subnormal boundaries are
  strengthened beyond ordinary JAX behavior. Signed-zero, normal-domain,
  endpoint and nonfinite rules are tested in all three execution modes.
- No signatures, exports, constructor checks, distribution PyTree leaves,
  direct-method policy, dependencies, docs or later-slice behavior changed.
  No probability floors, stopped parameters or global dtype changes added.
- Declarations remain metadata. Only the callable dataclass instances are
  frozen; ordinary functions' Python attributes remain mutable.

## Verification actually run

All uv commands use UV_CACHE_DIR=/private/tmp/survjax-uv-cache.

- Ruff format --check .: exit 0, 56 files already formatted.
- Ruff check .: exit 0, All checks passed!
- mypy src: exit 0, Success: no issues found in 14 source files.
- Focused suite: separately with JAX_ENABLE_X64=1 and 0, SPEC_THROUGH=A6,
  uv run --locked --group test pytest tests/unit/test_constraints.py
  tests/unit/test_parameter_constraints.py tests/unit/test_import.py
  ../spectest/tests/spec/test_a6_constraints.py -q --tb=short
  -o cache_dir=/private/tmp/survjax-a61-plain-<mode>.
  Each process: exit 0, 98 passed in 4.09s (36 own, 62 independent).
  Expected signed-zero masks follow the inequalities in section 4.7;
  exact simplex masks follow the specified 4*n*eps inequality.
- scripts/check.sh --through A5: exit 0, guard ok (A6.1), Ruff format/lint
  and mypy clean with the outputs above. Own tests: x64 501 passed, 7 skipped
  in 43.21s; x32 507 passed, 1 skipped in 43.02s. Source doctests: 4 passed
  per mode (0.91s/0.94s). Independent through-A5: x64 1256 passed, 26 skipped
  in 27.95s; x32 1261 passed, 21 skipped in 23.02s. Final line:
  "check: all gates passed through A5".
- git diff --check: clean after implementation.

## Test snapshot and scope limits

The revision-30 independent module is at tester HEAD `f527f75`, with its
.spec-seen matching dev. A6.1 still records tests/spec/SOURCE_COMMIT `1d87b72`;
the maintainer's spectests-A6r1 branch has not arrived on this branch. Executed
the new constraint module directly from ../spectest; did not merge, copy,
modify, skip or xfail any independent test. The through-A5 gate therefore
covers this branch's earlier A0–A5 snapshot, and the focused run separately
covers the updated A6 constraint vocabulary.

No docs build: docs and docstrings are unchanged. No complete A6 gate, R
fixtures, install check or benchmark run. The revision-29 likelihood plan
still needs revision before approval; unsupported-mode D-023 remains open.
Passing this follow-up is not evidence that later A6 steps are implemented.
