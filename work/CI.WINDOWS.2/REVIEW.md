# CI.WINDOWS.2 — Git Bash discovery repair

Independent review: round 1/3.

## Read first

1. PLAN.md and tests/conftest.py's bash_executable fixture.
2. tests/tooling/test_bash_executable.py, then the existing docs shell tests.

## Diagnosis and invariants

Run https://github.com/survjax/survjax/actions/runs/42701729597 used commit
420b56c105da8e8a275337a1afcf1e6acebbe005. Linux and macOS passed both modes.
Windows float64: 1339 passed, 8 skipped, 13 setup errors; float32: 1344 passed,
3 skipped, 13 setup errors. All errors are the same missing Bash path.
Git on PATH resolves inside Git/ucrt64/bin; the old fixed parent calculation
looks for Git/ucrt64/bin/bash.exe instead of Git/bin/bash.exe.

The fixture now searches ancestor directories for bin/bash.exe and still fails
explicitly if none exists. It never selects the WSL shim through PATH. Tests
construct cmd, bin and ucrt64/bin layouts in temporary directories, independent
of the host platform, and check absence. Existing docs tests retain their checks.
Spec 10.3 verification tooling only: no library API or numerical change.

## Validation

- `JAX_ENABLE_X64=1 uv run --group test pytest -n 6 tests/tooling/test_bash_executable.py tests/unit/test_docs_build_script.py tests/tooling/test_build_docs.py -q`: 17 passed, 2.62s.
- Same command with `JAX_ENABLE_X64=0`: 17 passed, 2.57s.
- `git diff --check`: exit 0.
- `scripts/check.sh --fast --through A12`: exit 0, all gates passed.
  Formatting: 217 files; lint passed; mypy: 24 source files, no issues.
  Implementer float64: 1276 passed / 8 skipped (30.93s); float32: 1281 passed /
  3 skipped (22.60s). Source doctests: 4 passed in each mode.
  Spec float64: 3562 passed / 33 skipped (38.74s); float32: 3546 passed /
  49 skipped (37.92s). Each spec mode emitted the two existing NumPy warnings
  from test_a11_ph_direct_methods.py:78.

Actual Windows execution remains pending CI integration; simulated paths verify
discovery only. No full non-fast gate or docs build was run.

## Commits and remaining work

Previous step final commit, this step base and original worktree base:
420b56c105da8e8a275337a1afcf1e6acebbe005 (CI.WINDOWS.1).
No further implementation substeps planned. Maintainer integration and a full
Windows rerun remain required. No shared progress record was changed.
