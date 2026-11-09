# D.FAMILIES.MERGE.1 — combined family merge handoff

Independent review: round 1/3

## Read first

1. PLAN.md: maintainer-initiated merge, parents, scope and prior family records.
2. Conflict resolutions in the two package __init__.py files and
   tests/install/test_wheel.py: retain both family imports/exports.
3. tests/unit/test_import.py: auto-merged exact inventories retain both names.
4. Generated tests/slow_tests.txt and the validation record below.

This completes the merge already initiated by the maintainer on this branch.
Incoming LogNormal code/tests/docs/records are preserved without edits. No
numerical code, constraints, observation policy, dependencies or protected
paths changed during conflict resolution. Both families' original reviews and
implementation evidence remain in their own work directories.

## Invariants and validation

The existing registry still owns pytree/parameter handling and boundary policy.
Both package entry points and installed exports must expose LogNormal and
LogLogistic alongside the existing families. Importing both does not modify
JAX dtype configuration or require optional demo dependencies. The existing
LogLogistic API-reference docs exception remains; incoming LogNormal docs are
present. No hand-selected generated timing entries are retained.

- `uv run ruff format --check .`: 208 files already formatted.
- `uv run ruff check .`: All checks passed.
- `uv run mypy src`: Success: no issues found in 24 source files.
- `JAX_ENABLE_X64=1 uv run --group test pytest -n 6 tests/unit/test_log_logistic.py tests/unit/test_lognormal.py tests/unit/test_import.py tests/unit/test_api_reference.py tests/install/test_wheel.py -q`: 48 passed, 28.09s.
- Same command with `JAX_ENABLE_X64=0`: 48 passed, 25.66s.
- No remaining text conflict markers in src/, tests/, docs/ or work/.
- Timing regeneration command: `PYTEST_WORKERS=6 JAX_COMPILATION_CACHE_DIR=/Users/wamster3/dev/survjax/dev/.git/survjax-jax-cache JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 scripts/update_slow_tests.sh --through A12`: exit 0. The combined generated list has 98 tests (253s slow / 634s summed test time). No hand filtering or overlapping test runs from this agent.
- Captured full profiling collections: float64 implementer 1352 passed / 8 skipped (71.45s); float64 spec 3580 passed / 33 skipped / 2 warnings (50.97s); float32 implementer 1357 passed / 3 skipped (70.15s); float32 spec 3557 passed / 56 skipped / 2 warnings (40.39s).
- `git diff --check`: clean. Git reports no remaining unmerged paths after staging the four conflict resolutions.
- `scripts/check.sh --fast --through A12`: exit 0. Final static checks: 209 files formatted, lint passed, mypy passed on 24 source files. Implementer tests: float64 1272 passed / 8 skipped (24.75s), float32 1277 passed / 3 skipped (24.66s). Source doctests: 4 passed each mode. Spec tests: float64 3562 passed / 33 skipped (37.97s), float32 3546 passed / 49 skipped (37.28s), two existing NumPy warnings each mode at test_a11_ph_direct_methods.py:78. No failures.
- `git diff --cached --check`: clean. No full non-fast check.sh invocation; full profiling collections above did execute all selected tests in both modes.

The family tests cover analytical values, tails, boundaries, second parameter
derivatives, eager/jit/vmap, constructor/diagnostic and grid behavior. Existing
import/API-reference tests and fresh installed-wheel checks establish the
combined public surface; no new tests mirroring import statements are needed.

## Commits and remaining work

Worktree/step base and preceding LogLogistic step's final commit:
c31f73368c07786c95ed48371ba4f51e59485ad5.
Incoming LogNormal docs-step final commit:
3bb4e3dbdb68b3b9d30f0f37ce3ce1164636c7cb.
Original independent-family worktree base:
3abefc5683a3aee8169eb2deffe57602897a3af3.
Earlier A12 checkpoint: ff07cf9f7942db0ce0e5d3a27525e0c1ca3d3431.

No spec questions or deviations. No further integration substep is planned if
checks and independent review pass. LogLogistic docs and the maintainer's full
non-fast merge gate remain separate; no publishing or merge to main occurs.
No docs build has been run here: imported docs are unchanged from LogNormal's
reviewed docs step. The maintainer owns roadmap/integration status updates.
