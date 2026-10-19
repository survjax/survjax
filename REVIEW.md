# A6.1 review fix-up — simplex threshold and bound errors

Status: approved fix-up implemented on `A6.1`, based on `98be832`.
Focused checks, M2 mutation evidence and the through-A5 gate pass.
Ready for review; stop after this fix-up commit. A6.2 has not started.

Review addressed: `../review/slices/A6.1.md`, S1 and N1/N2. The maintainer
approved the fix-up and chose JAX parameter-dtype/backend behavior for Q1.
Spec 4.7 is unchanged between revisions 28 and 29. The alignment marker stays
at revision 28 (`0683e35`); future-step replanning/alignment remains separate.

## Read first

1. `tests/unit/test_constraints.py`, the new exact simplex threshold test.
2. `src/survjax/constraints.py`, `_bound`, then the new huge-bound regression
   in the same own test module.
3. `DEVIATIONS.md` D-024 for the maintainer's precision decision and D-023 for
   the revision-29 note about the still-open unsupported-mode policy.
4. `PLAN.md`, the approved A6.1 fix-up and warning that the revision-28
   likelihood plans need updating before another step is approved.

## Changes and invariants

- S1: each length-10 vector has only x[0]=1+k*eps nonzero, so its sum is exact.
  Spec 4.7's inequality gives acceptance for k=30 and 40, rejection for 41
  and 50. In particular, 40 is the inclusive 4*n endpoint and 41 is its next
  representable neighbour. The expected masks are hand-derived from the
  domain, with no approximate assertion tolerance. Test eager/jit/vmap in
  each separate dtype process, reading the configured mode via the fixture.
- N1: overflow during finite-bound validation now gives the existing TypeError
  and message. Test 10**400 through greater_than and both endpoints of open
  and closed interval factories. Valid-bound comparisons are unchanged.
- N2: frozen applies only to the callable dataclass instances. The module
  functions real/nonnegative/unit_interval have mutable Python attributes;
  they were not converted to objects or otherwise changed by this fix-up.
- Q1/D-024: keep current JAX parameter-dtype/backend comparisons and the
  existing zero-bound frexp handling, following the maintainer's decision.
  No new mathematical comparison against Python bounds is introduced.
  Formal deviation closure remains with the maintainer; status stays open.
- N3: no registration refactor. No public signature/export, constructor,
  direct-method, constraint domain, simplex implementation, dependency, docs
  or independent-test changes. No probability floors or epsilons added to
  numerical calculations; eps here is the specified diagnostic domain.

## Checks actually run

All uv commands use UV_CACHE_DIR=/private/tmp/survjax-uv-cache.

- Ruff format --check .: exit 0, 56 files already formatted.
- Ruff check .: exit 0, All checks passed!
- mypy src: exit 0, Success: no issues found in 14 source files.
- Focused command, separately for JAX_ENABLE_X64=1 and 0, with SPEC_THROUGH=A6:
  uv run --locked --group test pytest tests/unit/test_constraints.py
  tests/unit/test_parameter_constraints.py tests/unit/test_import.py
  ../spectest/tests/spec/test_a6_constraints.py -q --tb=short
  -o cache_dir=/private/tmp/survjax-a61-fixup-<mode>.
  Each process: exit 0, 93 passed in 4.72s (36 own, 57 independent).
- M2: execute constraints.py's source in an isolated in-memory module after
  replacing only 4*x.shape[-1]*finfo(x.dtype).eps with 40*..., then temporarily
  assign that process's constraints.simplex to the mutant. Run pytest.main
  on the new exact-threshold test with -q --tb=short. Each process, x64/x32:
  exit 1 as expected, 1 failed in 0.22s; mutant mask [True, True, True, True]
  differs from expected [True, True, False, False] at indices 2 and 3. The
  original checkout source was never mutated and needs no restoration.
- scripts/check.sh --through A5: exit 0, guard ok (A6.1); Ruff format/lint
  and mypy pass with the same outputs above. Own tests: x64 501 passed,
  7 skipped in 43.84s; x32 507 passed, 1 skipped in 44.99s. Source doctests:
  4 passed per mode (1.02s/1.03s). Independent through-A5: x64 1256 passed,
  26 skipped in 28.32s; x32 1261 passed, 21 skipped in 23.33s. Final line:
  "check: all gates passed through A5".
- git diff --check: clean after implementation.

No docs build was run for this fix-up: docs and docstrings are unchanged.
No complete A6 gate, R fixture check, install check or benchmark was run.
Independent tests remain read-only and were executed directly from spectest,
without syncing or changing them. The previous main-step docs entry was
explicitly authorized by the maintainer's "ok make the minimal entry".

## Remaining questions

D-023 still needs a maintainer decision before A6.4. Revision 29 removes
LikelihoodIssues/_with_info and adds the base log_likelihood method, so the
later plan must be refreshed before approval. No later API is implemented
or asserted verified by this fix-up. There is no remaining A6.1 code question
from this review; the full gate passes.
