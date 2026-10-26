# A8.4a.2 — BC groups, grids and acceptance-export consistency (2026-10-26)

Approved by "good on A8.4a.1; get to the next bit". Parent: 92e54d6;
branch: a8-1-r-generator. One code step, then stop for review.

## Read in this order

1. tests/unit/test_bc_reference_fixtures.py: positive shape/precision checks
   and rehashed group, grid and export corruption cases with exact errors.
2. tests/conftest.py: _check_bc_fixture, called only after every required
   payload is hashed and the existing common schema checks have passed.
3. PLAN.md: completed A8.4a loader scope; distribution comparisons are next.

## Invariants and interface

The existing r_fixture callback and primary header/cases/grid/data structure
remain. BC grid times and each source/case grid are now NumPy float64 arrays;
no reconstruction of R seq or change to JAX configuration occurs.

Spec 10.4/11.1/12.1: require Good/Medium/Poor blocks of 229/229/228 and
matching group-size/event-count metadata. Grids have declared group order,
shape (141, 3), finite values, S(0)=1, and 141 strictly increasing finite
times with endpoints 0 and 7. Keep the actual stored time coordinates; the
independent A8 acceptance tests enforce canonical 0.05 spacing separately.

Parse the additional acceptance JSON only after integrity verification.
Require its exact top-level schema, pinned provenance, CSV observations,
grid times, three case labels, natural parameter vectors, both named sources'
per-row likelihood/survival arrays, and flexsurv grid curves to match the
primary R outputs. These are exact representation checks, not tolerant
numerical comparisons or independent recomputations of expected values.

All existing data/manifests, R generators, library API/source, dependencies,
tolerances, docs and independent tests remain unchanged. Temporary JSON/CSV
parsing is used only to build malformed test inputs. The protected independent
fixture is still absent; the maintainer handoff remains pending.

## Checks and spec mapping

With UV_CACHE_DIR=/private/tmp/survjax-uv-cache, ran
`uv run pytest -q tests/unit/test_reference_fixtures.py
 tests/unit/test_bc_reference_fixtures.py` in separate JAX_ENABLE_X64=1 and 0
processes: 58 passed in 0.58s per mode (28 common plus 30 BC tests).

- Spec 3.3/11.1: every case/source returns a (141, 3) float64 grid. Original
  R %a anchors for time index 51 and the fitted Medium curve match exactly
  in both modes (A8.3b audit). Time coordinates are never regenerated.
- Spec 10.4/11.1: validly rehashed inputs separately damage CSV group order,
  group sizes/event counts, grid labels/dimensions/times/finiteness/S(0),
  acceptance schema/meta/observations, all three case-label maps, parameters,
  source labels, each source's two row quantities and curve columns.
  Expected errors are escaped/anchored per case; CSV damage refreshes its
  header MD5 too, so it reaches the group check instead of earlier provenance.
- `python3 /private/tmp/survjax-a84a2-mutations.py`: exit 0. In temporary
  copies, remove each of the new helper's 15 rejection guards individually;
  run the full 30-case BC module in both dtypes. Every mutation yields
  assertion failures, never setup errors. Failure counts per mode: group
  order/sizes 2; group event counts 1; grid metadata 2; grid times 5; reference
  grid shape/finiteness 2; S(0) 1; export schema 1; export provenance 1;
  export rows 3; export grid times 1; case labels 3; parameters 1; source
  labels 1; source values 4; curve values 1. All remaining tests pass.
  Log: /private/tmp/survjax-a84a2-mutations.log; per-mutant logs/root:
  /private/tmp/survjax-a84a2-mutations.json.

The code/test diff is 276 lines (96 additions/1 deletion in conftest plus
179 new test lines). Module sizes: conftest 247, BC test module 179. Existing
common tests are unchanged. No new abstraction or dependency is introduced.

## Final gate

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`:
exit 0, check: all gates passed through A7.
Complete log: /private/tmp/survjax-a84a2-gate.log.

- Guard OK; Ruff: 75 files formatted / All checks passed!
- mypy: no issues in 17 source files.
- Own float64: 835 passed, 7 skipped, 145.91s.
- Own float32: 841 passed, 1 skipped, 140.52s.
- Source doctests: 4 passed per mode, 0.81s / 0.84s.
- Independent float64: 2123 passed, 28 skipped, 223 deselected, 111.61s.
- Independent float32: 2127 passed, 24 skipped, 223 deselected, 106.64s.

Only result records and plan status are updated after the gate. Final Ruff
formatting/lint and git diff --check pass on the finished records.

## Limits and next step

No through-A8 claim, R run, docs build or new deviation. This completes the
planned A8.4a loader work, subject to review. A8.4b adds exponential comparisons;
A8.5 adds Weibull comparisons and the through-A8 gate after protected fixture
handoff; A8.d1 documents the evidence. Do not start those in this commit.
