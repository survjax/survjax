# D.FAMILIES.MERGE.1 — resolve LogNormal integration conflicts

Feature: complete the maintainer-initiated merge of log-normal-family into
log-logistic-family. Current base: c31f73368c07786c95ed48371ba4f51e59485ad5.
Incoming merge head: 3bb4e3dbdb68b3b9d30f0f37ce3ce1164636c7cb.
Dependencies: [LogLogistic](../D.LOGLOGISTIC.1/REVIEW.md),
[LogNormal code](../D.LOGNORMAL.1/REVIEW.md) and
[LogNormal docs](../D.LOGNORMAL.2/REVIEW.md).
Cumulative gate target: A12; the two families are separately authorized Slice D
work, not completion of Slice D. Prior checkpoint: ff07cf9 (A12 acceptance).

## Resolution

- Preserve both family imports in src/survjax/__init__.py and
  src/survjax/distributions/__init__.py; the auto-merged __all__ sets already
  retain both names.
- Combine both exports in tests/install/test_wheel.py's exact inventory.
- Temporarily restore the local generated timing list to make collection valid,
  then replace it with script-generated timings for the combined tree, six
  workers and the standard JAX cache, without overlapping test commands.
- Preserve incoming LogNormal implementation, tests, docs and records without
  edits; preserve the existing pending LogLogistic docs entry. No public API
  beyond the two already-approved families, dependencies, numerical changes or
  protected-path edits. Manual conflict resolution is well below 300 lines;
  imported commits and generated metadata retain their established scope.
- Finish only this already-initiated merge on the assigned branch. No rebase,
  merge to main, push, publication or other worktree operations.

## Validation

Check conflict markers and exact import inventories. Run both family tests,
import/API-reference tests and installed-wheel smoke tests in both dtype modes,
in full. Regenerate timings through A12; run scripts/check.sh --fast --through
A12. Commit resolutions and this step's handoff, then independent review and stop.
Spec contract: revision 47 sections 4.1/4.2, 4.5–4.7, 5, 7, 10.2 and 12.1.

## Docs to-do

LogNormal docs arrive from the incoming branch. The separate LogLogistic docs
follow-up remains owned by D.LOGLOGISTIC.1/PLAN.md; no new docs scope here.
