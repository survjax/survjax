# AFT.MERGE.1 — Integration handoff

Independent review: round 1/3

Read work/AFT.MERGE.1/PLAN.md and SPEC-DIFF.md; then compare the combined
source exports, installed-wheel inventory and docs with local main. Original
approved AFT tip 3e4cd40; integration plan 4978b31; original worktree base
cc0e30f; previous implementation step 607a11b. Main initially d8c85c7, then
advanced concurrently to a8ee092 (rev-58 independent tests).

Maintainer explicitly approved the complete AFT work and asked to merge main,
check the spec difference and ensure tests pass. This authorizes the local
integration/merge, but not remote publication. Continue the authorized merge
pipeline after review if it clears; do not begin another implementation step.

The command policy rejected git rebase main. Instead main was merged into
aft without rewriting approved history: bccfbff, then 6a1aee8 after main's
concurrent rev-58 update. No acceptance files edited: they are imported from
main unchanged. Protected-file diff against current main is empty.

Resolution invariants:
- AFT wrappers.py and both AFT test files equal 3e4cd40 exactly.
- Both new families plus AFT remain in root/distribution/wheel inventories.
- Main's root PLAN/REVIEW preserved; AFT legacy records saved verbatim in this
  integration directory before merging. Both archive index links preserved.
- Full timing-list regeneration is needed after combining the branches; no
  numerical assertion, tolerance or feature marker weakened.
- Spec rev 47→59 audit finds no change to the implemented AFT direct contract;
  summaries and Slice B fixture changes remain with their owners. No spec edit.

Validation is running: full scripts/check.sh --through A12, not --fast.
The first run at older main was interrupted after main advanced; its timings
are not reported as a passing gate. Timing regeneration was also stopped and
restarted for a stable latest-main snapshot. No failures triggered those stops.
Docs build and final test summaries will be recorded before review.

Final merge must verify main has not advanced unexpectedly, its checkout is
clean, and the merged tree matches the validated aft tree. No fetching/pushing.

Full merge gate at 6a1aee8 (main a8ee092 ancestor):
- scripts/check.sh --through A12 exited 0, all gates passed (no --fast).
- Ruff format 222 files already formatted; Ruff lint clean; mypy 24 source
  files clean. The gate built and used its wheel for installed-artifact checks.
- Implementation x64: 1406 passed, 8 skipped in 150.19s;
  x32: 1411 passed, 3 skipped in 145.57s.
- Source doctests: 4 passed per mode (0.73s/0.64s).
- Acceptance x64: 3581 passed, 32 skipped in 76.39s;
  x32: 3559 passed, 54 skipped in 58.10s.
- Both acceptance modes emitted the two existing NumPy expected-value warnings
  at test_a11_ph_direct_methods.py:78 (log zero and zero*infinity). No failures.
- Current registered post-release features are none, unchanged from main.
  Staged unimplemented feature suites are not being claimed as passing.
- Revision 59 (0ea22df) appeared during integration; its full diff was read.
  It only changes Slice B flexsurv fixture/doc conventions, with no AFT direct
  contract change. See SPEC-DIFF.md. Main's inherited checkpoint is revision 58.

Timing regeneration completed: scripts/update_slow_tests.sh --through A12
wrote 647 measured slow entries for the combined implementation/A12 suite.
The full gate above used no slow filter, so this metadata cannot weaken it.
Main then advanced to 4e40d48 with rev-59 Slice B fixture tests and marker;
no library code or existing A0–A12 test assertions changed. Integrate that
final author update and repeat the full gate plus docs on a stable snapshot.
