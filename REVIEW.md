# A8.4a.1 review fix-up — SF-1/N-1 (2026-10-25)

Authorized explicitly by the request to check the review and fix without
another approval stop. Parent: affaff6; branch: a8-1-r-generator.
Review source: ../review/slices/A8.4a.1.md. One fix-up commit, then stop.

## Read in this order

1. tests/unit/test_reference_fixtures.py: rehashed JSON and CSV damages with
   exact rejection messages; all three quantities checked on valid fixtures.
2. The mutation table below: each L6–L11 guard now has a sensitive test.
3. PLAN.md: review fix-up scope; A8.4a.2 is still separate and unstarted.

## Findings addressed

- SF-1: valid loads now require ndarray/float64/row-shape for log_likelihood,
  log_survival and survival from both sources. Rehashed bad JSON exercises
  finite-array rejection using the valid numeric token 1e999 (which bypasses
  parse_constant), non-positive exponential rate and bc scale_Good, stale CSV
  MD5 provenance, and short log_survival. Rehashed CSV tests exercise zero and
  infinite times and an event flip with stale event counts.
- CSV test inputs update the header's MD5 to the changed CSV bytes and then
  update SHA256SUMS for both files. They cannot pass by accidentally hitting
  the earlier provenance check; each must reach its exact semantic error.
- N-1: every JSON/CSV damage carries its own expected error message, escaped
  and anchored. A wrong validation failure no longer satisfies a shared regex.
- N-2: retain KeyError for missing header keys, explicitly accepted by the
  reviewer for this infrastructure. Missing data still fails and never skips.

## Invariants and scope

The loader itself is unchanged, as are all committed fixtures/manifests,
R generators, library source, tolerances, dependencies, independent tests,
spec marker and docs. All damage and rehash operations target temporary
copies; JSON/CSV parsing in those builders creates malformed test inputs,
not reference expectations. R-origin precision anchors remain unchanged.
No A8.4a.2 bc group/grid/export checks or distribution comparisons are added.
Spec mapping: 3.3 stored precision; 10.4 integrity/provenance/row alignment;
12.1 finite reference values and valid fixed parameters.

## Focused checks

With UV_CACHE_DIR=/private/tmp/survjax-uv-cache:
`uv run pytest -q tests/unit/test_reference_fixtures.py` in separate
JAX_ENABLE_X64=1 and 0 processes: 28 passed in 0.24s per mode.

`python3 /private/tmp/survjax-a84a1-review-mutations.py`: exit 0. It copies the
loader, tests and fixtures to a temporary tree and runs each mutant's full
focused suite in both dtypes. Child exit 1 and assertion failures are expected;
the driver verifies those counts and rejects setup errors.

| Mutant | Float64 | Float32 |
| --- | --- | --- |
| L6 drop per-row isfinite | 1 failed / 27 passed | 1 failed / 27 passed |
| L7 drop event-count check | 1 failed / 27 passed | 1 failed / 27 passed |
| L8 drop CSV MD5 check | 1 failed / 27 passed | 1 failed / 27 passed |
| L9 drop positive-time check | 1 failed / 27 passed | 1 failed / 27 passed |
| L10 drop parameter positivity | 2 failed / 26 passed | 2 failed / 26 passed |
| L11 omit log_survival validation | 3 failed / 25 passed | 3 failed / 25 passed |

Log: /private/tmp/survjax-a84a1-review-mutations.log; per-mutant logs and tree
location: /private/tmp/survjax-a84a1-review-mutations.json.
All six formerly surviving mutants are rejected in both dtypes.

## Final gate

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`:
exit 0, check: all gates passed through A7.
Log: /private/tmp/survjax-a84a1-fix-gate.log.

- Guard OK; Ruff: 74 files formatted / All checks passed!
- mypy: no issues in 17 source files.
- Own float64: 805 passed, 7 skipped, 149.54s.
- Own float32: 811 passed, 1 skipped, 148.92s.
- Source doctests: 4 passed per mode, 0.95s / 0.88s.
- Independent float64: 2123 passed, 28 skipped, 223 deselected, 100.04s.
- Independent float32: 2127 passed, 24 skipped, 223 deselected, 98.66s.

Only result records are updated after the gate. Final Ruff formatting/lint
and git diff --check pass; loader and protected/data paths are unchanged.

## Remaining work and stop

No through-A8 claim, R run, docs build or new deviation. The protected fixture
handoff and A8.4a.2 remain as previously recorded. Commit this test-only fix-up
and stop for review; do not start the next implementation step.
