# AFT.MERGE.1 — Integrate approved AFT work

Starting/last AFT commit: 3e4cd40. Previous implementation step: 607a11b.
Original worktree base: cc0e30f. Local main at start: d8c85c7.
Maintainer approved the complete AFT scope and requested integration with main,
a spec-diff check and passing tests. No fetching, publishing or remote edits.

Preserve current AFT legacy records here before integrating main's newer
per-step bookkeeping. The command policy rejected git rebase main. Instead merge local main into
aft, preserving approved history, then fast-forward main after validation;
resolve exports/inventories by retaining both new families and AFT. Keep
main's root records and protected files; import spec tests unchanged. Preserve
both sides of doc navigation and API documentation. Numerical AFT code and
all its test assertions must remain intact. If conflicts expose a specification
question, stop and ask; do not change the normative spec or acceptance tests.

Review spec rev 47 through current rev 58 for AFT impact. Current diff changes
summary contracts, feature staging and family scope; AFT direct methods and
likelihood identities remain unchanged. Generic RMST/summaries and observation
extensions remain separate worktrees; do not enable their acceptance features.
Run full scripts/check.sh --through A12, plus direct both-dtype docs builds
because exports/docs integrate with new families. Record exact results.
Merge the validated integrated branch into local main; no remote publishing.

Conflict resolution: retain main's legacy root PLAN/REVIEW; AFT versions were
preserved in this directory before merging. Keep both plan-index links and
combine installed-wheel export inventories. Regenerate timing metadata for the
combined tree rather than choosing a stale side. No acceptance test edits.
Spec checkpoint imported from main is rev 57 (83a8e82); latest rev 58 changes
only generic RMST tail-cut machinery, not the enabled direct AFT contract.
No normative spec edit or unsupported feature activation is warranted.
