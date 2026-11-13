# B8.DOCS — Sync documentation after the Slice B merge

Step starting commit: c908176edefe18797e1d1b81eeca55801be7b6b0
Previous step: [B7.AUDIT](../B7.AUDIT/PLAN.md), same commit.
Original worktree base: cc0e30f. Branch: li-censoring.
Cumulative gate: A12 and the existing five completed feature gates.

Dependencies: B6.1–B6.4 documentation and B7.AUDIT completion evidence.
One topic: current-checkout capability and acceptance guidance after integration.
Update README.md for all censoring/truncation modes, four families and AFT;
add an Unreleased CHANGELOG.md entry without rewriting historical releases;
expose the two interval probability methods in docs/api.md; reconcile
CONTRIBUTING.md and docs/releasing.md with the committed A12 CI/release gate.
Keep published-package history distinct from capabilities on this branch.

Spec coverage: sections 4.2, 6–7, 10.5 and 12.1–12.3 (revision 62).
No public API, runtime, dependency, protected-path or test changes. No new tests
are needed for this documentation-only sync. Target under 150 changed lines.
Inspect current docs, exports, workflows and previous evidence; run the strict
docs build in both dtypes and scripts/check.sh --fast --through A12. Inspect
rendered API entries and links. Commit records and changes, obtain independent
review and stop. Do not publish, merge or change release versions.

## Docs to-do

This step covers the remaining current-scope entry points and gate guidance.
Existing likelihood/data/diagnostic/verification guides already cover Slice B.
Historical release records and benchmark instructions stay explicitly historical.
Protected AGENTS.md and specification status text still mention Slice A;
report this separately, without editing either.
