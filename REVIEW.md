Independent review: round 2/3

# CI.1 — CPU checks on GitHub-hosted Linux, macOS and Windows

Read PLAN.md's CI.1 section, .github/workflows/ci.yml, then the small
platform-path change in tests/install/test_wheel.py. Compare against
84d2943e788440db60504afb49a13360d43ad95e. Existing scripts/check.sh,
scripts/run_tests.sh and the installed-wheel fixtures explain the reused gate.
Spec 10.1/10.3 and 12.1/12.2 govern this infrastructure-only step.

Invariants:
- The unchanged uv.lock and Python 3.12 supply the CPU reference. Runtime,
  test, dev and demo dependencies are unchanged; no new public API or slice.
- Every OS runs the same through-A10 gate, with each dtype in a separate
  process: Ruff/mypy, implementer tests, source doctests, independent tests.
- Setup may download dependencies and build tooling. The gate runs with
  UV_OFFLINE=1, UV_PYTHON_DOWNLOADS=never and UV_LOCKED=true. A fresh wheel is
  built before tests; Python emits its native absolute path to GITHUB_ENV.
  The existing installation fixture exports locked runtime artifacts and
  installs them offline into an out-of-tree environment, without demo deps.
- JAX_PLATFORMS=cpu applies to all commands and children; log actual devices
  and versions. Windows uses Scripts/python.exe, Unix bin/python. Existing
  installation and custom-family typing/differentiation tests exercise this
  selection on each platform, rather than a new mock-path test.
- Read-only token, no saved checkout credential, no publishing, no GPU runner,
  no secrets. Matrix jobs fail independently; superseded runs are cancelled.
  Two pytest workers limit concurrent JAX compilation on hosted machines.
- Protected paths/spec/lock/docs/fixtures and numerical code stay untouched.
  Prior unfinished PLAN history is preserved verbatim under this new step.
  scripts/check.sh is unchanged. The workflow creates the local main baseline
  so its guard executes. PRs attach the source branch name at the existing
  merge commit, preserving its SHA/tree/parents and source-prefix exemptions.
  Source name main maps to ci-pr-main, retaining ordinary guard rules without
  overwriting the main baseline. Source names enter shell through an env var.

Checks completed:
- uv run pre-commit run check-yaml --files .github/workflows/ci.yml: Passed.
- YAML BaseLoader checks: expected push/pull_request/workflow_dispatch
  triggers and three OSes; CPU backend selected. ast.parse accepts the
  embedded Python. No actionlint or GitHub-hosted execution was run locally.
- uv run ruff format --check .: 125 files already formatted.
- uv run ruff check .: All checks passed!
- uv run mypy src: Success: no issues found in 21 source files.
- uv lock --check: Resolved 171 packages; no lock change.
- git diff --check: clean.

An initial gate attempt stopped at Ruff formatting for the new path expression;
ran uv run ruff format tests/install/test_wheel.py and restarted the gate.
Final scripts/check_wheel.sh --through A10 exited 0, ending
"check: all gates passed through A10". Fresh wheel:
/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.3upnOO/survjax-0.0.1.dev0-py3-none-any.whl
- Implementer x64: 1097 passed, 7 skipped in 105.64s.
- Implementer x32: 1103 passed, 1 skipped in 75.46s.
- Source doctests: 4 passed in each mode (1.00s / 0.95s).
- Independent x64: 2332 passed, 28 skipped in 67.68s.
- Independent x32: 2322 passed, 38 skipped in 60.63s.
These include existing out-of-tree installed-core and typed custom-family
checks, validating the unchanged Unix path branch on this host. Windows
path execution remains for the hosted Windows job.

Local configuration: macOS 26.6.2 arm64, Python 3.12.9, uv 0.11.19,
JAX/jaxlib 0.11.2, NumPy 2.5.3, CpuDevice(id=0). Hosted Linux/Windows and
macos-15 remain unexecuted until the maintainer pushes; no platform support
or GPU compatibility claim follows from writing the workflow. Remote setup,
pushing and branch-protection settings are outside this authorized step.

GPU assessment remains a PLAN.md proposal: manually dispatched Linux/NVIDIA,
GitHub T4 larger runner or ephemeral cloud/self-hosted hardware, separately
pinned CUDA environment matching CPU JAX, mandatory GPU device assertion,
serial numerical/transformation checks in both dtypes and recorded driver /
versions / failures. Select runner/account/billing and a compatible driver
before implementation. Metal is experimental float32 only. CPU remains the
correctness baseline; no accelerator dependencies or runner are enabled.

Previous implementation endpoint: aae68920769855aa08227f4b7c4818ce7a820923
(RMST.integration). Worktree base / immediate previous commit:
84d2943e788440db60504afb49a13360d43ad95e. CI.1 implementation commit is the
commit containing this review; report its hash after independent review.
Remaining: hosted CPU evidence after push; optional GPU provisioning/lock/job
in a later step. Earlier RMST/A9 work is not advanced by this CI change.

## Round 1 outcome and follow-up

Independent report: ../review/slices/CI.1.md, round 1 at 2166045,
VERDICT: fix-first. F-1: set git core.autocrlf=false and core.eol=lf before
checkout, preventing CRLF shell scripts on Windows. F-2/S-1: D-030 records
the skipped guard and pending PR branch policy; asked the maintainer before
dependent implementation. S-2 is settled by the user's explicit three-OS
request. No tests, spec or protected files are altered.

Reviewer reproduced the CRLF shell failure and caught an inverted interpreter
path mutant in both dtypes. Its unmutated install suite passed three tests.
Its plain check.sh run passed static checks, both own suites and doctests,
then had five x64 spec setup errors because SURVJAX_WHEEL was unset. It
classified those as invocation/environment errors; its check_wheel.sh attempt
was blocked by its permissions. Its x32 spec suite and hosted OSes were not
verified. My complete fresh-wheel gate above passed in both modes.

Nonblocking nits N-1/N-2/N-3 remain: extra pre-commit hooks are not added to
the existing gate, push+PR runs can duplicate jobs, checkout uses a version
tag while setup-uv uses a SHA. GPU remains a proposal only. Reviewer reported
an unremoved temporary mutation worktree registration at /tmp/survjax-mut-CI.1;
cleanup was not attempted here, preserving the review ownership boundary.

## Round 2 fixes and evidence

Maintainer explicitly agreed to the recommended S-1 policy in conversation:
use the PR source branch name for guard exemptions, keep testing the merge
commit, and ensure the guard actually runs. D-030 records that decision;
its status remains open for maintainer resolution. S-2 was already authorized.

F-2: the new workflow step creates local main from refs/remotes/origin/main
when absent and, for pull_request only, attaches the source branch name at
HEAD. git switch -c has no startpoint argument, so it retains the checked-out
merge commit. A source named main maps to ordinary ci-pr-main; neither gets
an exemption. This avoids replacing the baseline. There is no protected
script edit and no inline source-name interpolation into shell commands.
F-1 remains fixed by pre-checkout LF configuration in 0c5acdb.

Validation on the current fix:
- YAML check hook: Passed. bash -n accepts the new embedded shell.
- Ten throwaway Git repository scenarios execute the exact workflow step and
  the unchanged guard block extracted from scripts/check.sh. Each starts
  without local main; each preserves the two-parent merge SHA and restores
  main at the baseline. Ordinary PRs (including source main) pass clean paths
  and fail scripts/check.sh changes. meta-* allows scripts/check.sh but rejects
  tests/spec; spectests-* allows tests/spec but rejects scripts/check.sh.
  Ordinary branch pushes also pass clean paths / reject protected paths.
  All ten scenarios passed.
- Ruff format: 125 files already formatted. Ruff check: All checks passed!
  mypy: Success: no issues found in 21 source files. git diff --check: clean.
- No numerical/test implementation changed after the complete initial
  fresh-wheel A10 gate; no new numerical rerun is warranted. Hosted runs,
  Windows path execution and GPU evidence remain pending as stated above.

Round 2 reviews this policy implementation plus F-1 from 0c5acdb. Initial
implementation: 2166045; last pre-fix commit: 0c5acdb; original base: 84d2943;
previous implementation step endpoint: aae6892. Nonblocking N-1/N-2/N-3
remain deliberately unchanged; the scope is the two should-fix findings.
