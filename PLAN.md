# A12.GATE — Integrate PH and staged RMST acceptance

Start / previous step / original worktree base: cc0e30f.
Archive-only commit: e944567. Spec sections 4.4, 7.3, 8.7, 12.1–12.2.
Files: .github/workflows/ci.yml, .github/workflows/release.yml, PLAN.md,
REVIEW.md. No API, dependency, spec or protected-path changes.
Advance CI and release artifact checks from A10 through A12, retaining both
separate dtype modes and the full acceptance selection. Existing independent
A11 PH and A12 RMST suites are the acceptance evidence; do not edit them.
Run scripts/check.sh --fast --through A12 before review. If failures require
numerical changes, scope a bounded fix before implementation, stopping for
spec ambiguity or changes over about 300 lines. Commit, independent review,
then stop. Roadmap is outside this repo; report its stale status for maintainer
update rather than editing it here.

Status: complete and maintainer-approved (7 November 2026). Implementation
commit 120c750; independent review round 1/3: VERDICT merge, no findings,
../review/slices/A12.GATE.md. Reviewer independently reproduced the fast
through-A12 gate in both dtypes. No acceptance failures required code changes.
Remaining verification: full merge gate and hosted workflow execution.
Docs/demo/release closure is separate; do not reimplement PH or staged RMST.

## A12.GATE.RECORD — Close bookkeeping after approval

Start / previous step: 120c750; original worktree base: cc0e30f.
Files: PLAN.md, REVIEW.md, plans/release-agents/README.md and workspace
../ROADMAP.md (maintainer explicitly requests current bookkeeping).
Record approval and review outcome; mark prospective PH/RMST handoffs as
historical. Update the roadmap checkpoint and PH/RMST acceptance status,
without asserting full-gate, hosted CI or demo closure. No docs/, API,
dependency, spec or numerical changes. Validate whitespace and status strings;
commit repository records and run independent review, then stop.

## Rebase integration record

Rebased onto e461c1f (incoming demo evidence audit). Preserve that audit plan
verbatim in plans/demo-e1.md and its review below the current A12 review.
Original release plans were already archived by incoming commit 86d4a56;
the replayed archive must not label the demo plan as alpha-2 release history.
Original commit IDs above identify pre-rebase review evidence. A12 gate
implementation replayed as 3fbaa96; no numerical/test changes in resolution.
