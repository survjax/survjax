# CI.WINDOWS.1 — Windows CI tooling repair

Independent Claude review waived explicitly by the maintainer (token budget).

## Read first

1. PLAN.md, then tests/install/test_wheel.py's README decoding change.
2. tests/conftest.py's bash_executable fixture and the two docs shell test files.
3. scripts/build_docs.sh's Git Bash path handling.

## Diagnosis and invariants

Failed scheduled run: https://github.com/survjax/survjax/actions/runs/42611200442,
commit 197dd37d68768c3d2f641d54428e5b6fa61a17a4. Both Windows dtype jobs
failed with the same 14 failures: one README encoding comparison, eight direct
shell-script launches (WinError 193), and five calls to Windows' WSL bash
launcher (no distribution installed). Linux/macOS jobs passed. The Windows
float64 implementer suite otherwise reported 1298 passed and 8 skipped.

- The wheel README is UTF-8, independently of the host's default text encoding.
- Python launches an explicit Bash interpreter; on Windows it comes from the
  Git installation, never the WSL shim. Fake uv programs are shell launchers,
  with LF line endings and quoted native Python paths.
- Git Bash shell operations and native Python agree on drive-letter paths with
  forward slashes. Python resolves the output root, and the shell removes a
  trailing CR from Windows Python's captured stdout. Python's module search path
  uses a semicolon on Windows and a colon on POSIX.
- Existing isolation, dtype, dirty-tree, failure, interruption and invalid-argument
  assertions remain enabled. Native path comparisons allow either slash style;
  they still check the exact directories. No tests are skipped or weakened.
- No public API, numerical behavior, dependency, docs page, protected path or
  spec requirement changed. Requirements covered are spec 10.2–10.3.

## Validation

- `JAX_ENABLE_X64=1 uv run --group test pytest -n 6 tests/install/test_wheel.py tests/unit/test_docs_build_script.py tests/tooling/test_build_docs.py -q`: 17 passed, 27.18s.
- Same command with `JAX_ENABLE_X64=0`: 17 passed, 27.19s.
- `git diff --check`: clean.
- `bash -n scripts/build_docs.sh`: exit 0.
- `scripts/check.sh --fast --through A12`: formatting (214 files), lint and
  mypy (24 source files) passed. Implementer tests: float64 1272 passed /
  8 skipped (31.81s), float32 1277 passed / 3 skipped (22.18s). Source doctests:
  4 passed in each mode. Spec float64: 3562 passed / 33 skipped (40.61s).
  Spec float32: 3546 passed / 49 skipped (39.53s). Exit 0, ending with
  "check: all gates passed through A12 (fast: slow tests skipped)". Each spec
  mode emitted the two existing NumPy warnings at test_a11_ph_direct_methods.py:78.

Windows execution remains unverified locally: this checkout runs on macOS.
No actual Sphinx/notebook build, full non-fast gate, R run, remote rerun or
publication was performed. The shell tests use fake uv to exercise the builder
without Sphinx/JAX. A real Windows CI rerun must confirm the platform-specific
branches and reveal any failures beyond the implementer suite, where the
original jobs stopped.

## Commits and remaining work

Original worktree and step base: 0c494ad05376aa3407e98427cd2d11e995f95eb7.
Preceding documented step D.LOGLOGISTIC.2 ended at 54b242c; its integration is
included in the base commit. No additional repair substeps are planned. Merge,
push and a Windows full-matrix rerun remain maintainer actions. No shared
roadmap/progress record was edited.
