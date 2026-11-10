# CI.WINDOWS.3 — current Git Bash path conversion

Independent review: round 1/3.

## Read first

1. PLAN.md and scripts/build_docs.sh's platform detection.
2. tests/unit/test_docs_build_script.py's platform cases.
3. tests/tooling/test_build_docs.py's fake uv platform detection.

## Diagnosis and invariants

Full-matrix run https://github.com/survjax/survjax/actions/runs/42781567128
tested b8454db. Linux/macOS passed, while both Windows jobs had seven failures:
two native checkout identity mismatches (/c/... versus C:/...) and five cache
path checks exiting with code 7. Bash discovery itself succeeded.

The script and fake uv recognized only OSTYPE=msys. MSYS2 now reports cygwin
(https://github.com/msys2/msys2.github.io/blob/main/web/news.md, 2025-04-18 entry).
Both values must select cygpath conversion and native Windows Python's semicolon
search-path separator. Both still use explicit cache directories and build
identity, and all failure/invalidation checks stay enabled.

The existing isolation test now runs under default, msys and cygwin OSTYPE for
default and custom output roots. POSIX hosts use an identity cygpath stub to
exercise platform selection; Windows retains real cygpath. The test checks the
separator as well as the paths, so reverting the script condition breaks the
cygwin case on POSIX too. Bash startup configuration applies to child shells.
No public API, numerical behavior or spec contract changes (spec 10.3 tooling).

## Validation

- `JAX_ENABLE_X64=1 uv run --group test pytest -n 6 tests/unit/test_docs_build_script.py tests/tooling/test_build_docs.py -q`: 17 passed, 4.39s.
- Same with `JAX_ENABLE_X64=0`: 17 passed, 4.09s.
- `bash -n scripts/build_docs.sh`: exit 0; `git diff --check`: exit 0.
- `scripts/check.sh --fast --through A12`: exit 0, all gates passed.
  Format: 219 files, lint clean, mypy: 24 source files with no issues.
  Implementer float64: 1281 passed / 8 skipped (55.51s); float32: 1286 passed /
  3 skipped (51.40s). Source doctests: 4 passed per mode. Spec float64:
  3562 passed / 33 skipped (40.92s); float32: 3546 passed / 49 skipped (40.14s).
  Both spec modes emitted the two existing NumPy warnings at
  test_a11_ph_direct_methods.py:78.

No full non-fast gate, actual Sphinx build or local Windows execution performed.
Full-matrix CI on the PR branch remains necessary for native verification.

## Commits and remaining work

Previous step final commit, step base and original worktree base:
b8454db475ae20465ac942346bd9d4922b097f76 (CI.WINDOWS.2).
No further implementation substeps planned. Push to PR #6 and run full-matrix
CI; merging remains a separate action. No shared progress file changed.
