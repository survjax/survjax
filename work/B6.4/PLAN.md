# B6.4 — Organize the verification page

Step starting commit: 1c47d6a0acce393cf57ff4727f78bf80baa6f398
Previous step: [B6.3](../B6.3/PLAN.md), same commit.
Original worktree base: cc0e30f. Cumulative gate: A12, existing features.

Maintainer-requested review and correction of docs/technical-notes/verification.md
organization, repeated R comparisons and heading nesting. The page currently
has R material under the wheel heading, a later separate R section, and a
top-level concordance reference section. Consolidate these into one H2
Comparisons with R: H3 exact/right references, H3 censoring/truncation with
H4 exclusions, H3 concordance, then H3 integrity/regeneration. Keep installation,
CPU performance and documentation build instructions as sibling H2 topics.

Remove duplicate provenance prose, scope old exact/right claims explicitly,
and make shared regeneration guidance account for the Slice B generator's
required destination. Preserve existing heading text/anchors where linked,
all numerical evidence and all executable Python cells in their current order.
One page/topic; target below 300 changed documentation lines including moves.
No API, source, tests, fixture, dependency or protected-path edits.

Spec 10.4–10.5 and 12.1–12.3; evidence inherited from B6.1–B6.3 and fixture
READMEs. Check source heading order and unchanged executable cells, run
scripts/build_docs.sh directly and scripts/check.sh --fast --through A12.
Inspect rendered heading/section nesting, unique IDs, existing inbound anchors,
downloads and local links in both dtypes. No new tests or reference rerun
needed for this structural edit. Commit handoff, use independent-review skill,
open diff, then stop.

## Remaining work

This is a documentation organization correction, not the final Slice B audit.
That audit and maintainer integration remain. Benchmark work stays in experiments.
