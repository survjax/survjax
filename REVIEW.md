Independent review: round 1/3

# CI.integration — Merge current main into the CI branch

Read PLAN.md CI.integration first, DEVIATIONS.md's D-033 reconciliation,
then the two-parent diff for tests/install/test_wheel.py. Parent commits:
CI 1299a88fbf444fe1717a50f896ea024121f96f29;
main 95f453279ab9bffad1c67dc234a56d46b534a677.
Original worktree base: 84d2943e788440db60504afb49a13360d43ad95e.
This finishes the maintainer-started merge only. No rebase/push/main merge.

Invariants:
- Only PLAN.md, REVIEW.md and DEVIATIONS.md conflicted. Both parent plans
  are preserved verbatim below the current integration section. Previous
  complete reviews are in their respective parents, not discarded history.
- Main's deviation sequence, bodies and statuses are preserved; CI's guard
  entry changes ID only from D-030 to D-033 to avoid main's RMST collision.
  A mapping explains historical CI references. The guard policy remains the
  previously approved source-branch exemption policy; no new deviation or
  spec interpretation is introduced. Formal statuses stay maintainer-owned.
- Main's PH/RMST source, docs, appendices, exports and tests remain byte-exact.
  tests/install/test_wheel.py equals main plus the existing Windows/Unix
  interpreter selection from CI. Its PH export and eager/jit RMST smoke
  assertions remain unchanged. .github/workflows/ci.yml equals the CI parent.
- No new numerical implementation, test expectation/tolerance, dependency,
  lock, protected path or spec change. Incoming .spec-seen is preserved.
  No tests are added for concatenating records; the unchanged through-A10
  suite exercises the automatically combined wheel-install fixture.
- Existing A10 acceptance is not evidence of still-pending independent
  PH/staged-RMST acceptance; those delivery/ownership limits remain intact.

Validation complete:
- Both complete parent plans are exact substrings. Main's full deviation
  sequence is unchanged; CI's entry body/status is exact under D-033. IDs
  are unique. No conflict markers or unmerged index entries remain.
- Relative to main, only .github/workflows/ci.yml, the three conflict records
  and tests/install/test_wheel.py differ. Workflow equals the CI parent;
  wheel fixture equals main plus CI's exact interpreter-path selection.
  Thus all other incoming files, including source/docs/tests/spec marker,
  are byte-exact main. No protected path differs from main.
- uv run pre-commit run check-yaml --files .github/workflows/ci.yml: Passed.
  uv lock --check: Resolved 171 packages, lock unchanged.
  git diff --cached --check: clean.
- scripts/check_wheel.sh --through A10: exit 0, all gates passed.
  Ruff: 139 files already formatted / All checks passed!
  mypy: Success: no issues found in 22 source files.
  Own x64: 1282 passed, 8 skipped, 174.38s;
  own x32: 1287 passed, 3 skipped, 159.27s.
  Source doctests: 4 passed per dtype, 1.54s /1.55s.
  Existing independent A10 acceptance: x64 2332 passed, 28 skipped, 125.40s;
  x32 2322 passed, 38 skipped, 84.16s. No test/skip/tolerance was edited.
  Fresh wheel supplied to both installed-wheel suites:
  /var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.dpZks4/survjax-0.0.1.dev0-py3-none-any.whl
  This includes main's PH export/RMST checks and the retained Unix path.
  Windows path/checkout remains for the hosted run.
Imported docs are unedited from main; this is not a docs-authoring step and
no new docs build or hosted Linux/Windows/GPU evidence is claimed.

Previous CI step endpoint / immediate pre-merge commit: 1299a88.
Incoming main endpoint: 95f4532. Original worktree base: 84d2943.
Remaining: hosted CPU checks after push; maintainer ci-to-main integration;
optional GPU runner/environment/job. Main's PH/RMST handoff limits remain.
