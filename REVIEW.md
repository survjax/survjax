# Integrate independent A7/A8 acceptance tests — 2026-10-23

Authorized by the maintainer's request to bring the new main commits onto A8.
A8 parent: `16d0003`; incoming main: `2c030c5` (includes `5d105ab`).

## Read in this order

1. `tests/spec/fixtures/README.md` and `test_a8_bc_fixtures.py`: the independent
   A8 fixture contract, imported unchanged from main.
2. `PLAN.md`: current revision-34 baseline and the fixture-schema handoff.
3. The merge diff: incoming spec marker and tests; no library changes.

## Invariants and scope

- All independent tests and the spec marker match incoming main exactly.
- Existing generators, fixture payloads, manifests, library and docs remain
  unchanged. No test skips, xfails or assertion/loader edits were added.
- The A8 test expects `tests/spec/fixtures/bc_synthetic_weibull.json`, which
  our current CSV/JSON export does not provide. A separate planned generator
  adaptation and maintainer-coordinated delivery to that protected path are
  needed. This merge makes no A7/A8 completion claim.
- Previous A6 integration evidence is at `16d0003:REVIEW.md`.

## Verification

- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A6`:
  exit 0, `check: all gates passed through A6`.
  - Guard OK; Ruff: 69 files formatted, all checks passed.
  - mypy: no issues in 17 source files.
  - Own tests: float64 722 passed / 7 skipped; float32 728 passed / 1 skipped.
  - Source doctests: 4 passed per dtype mode.
  - Independent tests: float64 2034 passed / 28 skipped; float32 2038 passed /
    24 skipped. Each mode deselected 133 later-step tests using SPEC_THROUGH=A6.
- `JAX_ENABLE_X64=1 SPEC_THROUGH=A8 UV_CACHE_DIR=/private/tmp/survjax-uv-cache
  uv run pytest tests/spec/test_a8_bc_fixtures.py -q -x`: exit 1, one setup
  error: committed fixture `bc_synthetic_weibull.json` not found. This is the
  outstanding generator/schema task, not a numerical comparison result.
- `git diff --check` and `git diff --cached --check`: exit 0, no output.
- Verified the spec marker and independent tests match MERGE_HEAD exactly;
  source, docs, generators and fixture payloads match the A8 parent.

No R regeneration, docs build, through-A7 gate or full through-A8 gate was run.
No new code, dependencies, skips or deviations were introduced.
