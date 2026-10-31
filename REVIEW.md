Independent review: round 1/3

# MNIST.integration — Resolve incoming main

Read PLAN.md MNIST.integration, then DEVIATIONS.md and the merge parents
1d06cc9 (MNIST) and ae569e4 (incoming main with RMST). Review the merge
resolution against both parents rather than treating inherited code as new.

Invariants:
- Only the three workflow records conflicted. Source, tests, tolerances,
  appendix, README, .spec-seen and incoming CLAUDE.md retain the automatic
  merge result without local changes. No protected-path edit is introduced.
- Both complete parent plans are preserved verbatim under labelled headings.
- Distinct MNIST and RMST deviations are retained with unchanged statuses;
  shared D-027-and-earlier history is identical and retained once. Historical
  D-028/D-029 collisions are explained by topic; no IDs or decisions change.
- The previous MNIST.5s review is available at 1d06cc9:REVIEW.md; the incoming
  RMST.integration review is at ae569e4:REVIEW.md. This file follows the
  overwrite convention. No new MNIST implementation step begins here.

Checks and requirement evidence: complete. Static checks and cumulative
fresh-wheel gate through A10 cover inherited core behavior and packaging;
optional MNIST checks in separate dtype processes cover the existing demo
against the incoming numerical methods. No new behavioral test is needed
for the record-only conflict resolutions. No new spec interpretation.

Completed checks (all exit 0):
- scripts/check_wheel.sh --through A10: all gates passed; Ruff format 133
  files already formatted, Ruff all checks passed, mypy no issues in 21 source
  files. Own suites: x64 1119 passed/7 skipped (76.73s), x32 1125 passed/1
  skipped (79.41s). Source doctests: 4 passed each mode. Acceptance: x64
  2333 passed/27 skipped (68.79s), x32 2324 passed/36 skipped (67.15s).
- JAX_ENABLE_X64=1 and =0 uv run --group examples --group test pytest with
  check_mnist_model.py, check_mnist_reference.py, check_mnist_training.py,
  test_mnist_data.py, test_mnist_outcomes.py: respectively 58 passed/1 skipped
  (27.79s) and 38 passed/21 skipped (18.63s).
- Parent preservation script: both full parent plans and every deviation entry
  preserved exactly; all inherited staged files outside the three resolutions
  equal incoming main. git diff --check and git diff --cached --check clean;
  no unmerged entries or conflict markers in the resolved files.

No docs build, R generation, full MNIST training or new performance result is
claimed. This is a merge-resolution step, not new numerical or docs authorship.
Previous step: 1d06cc9. Worktree branch starting point: ee72c83. Incoming main:
ae569e4. Remaining MNIST work and docs ownership are unchanged by this merge.
