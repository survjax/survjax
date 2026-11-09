# B1.d2 — Explain applications and truncation terminology

Start / previous step: e0cdde2; original worktree base: cc0e30f.
Maintainer requested short practical applications for interval probabilities
and conditioning, using left truncation/delayed entry terminology. Edit only
docs/working-with-distributions.md plus PLAN.md/REVIEW.md. No code/API changes.
Use periodic assessment as interval censoring; event-free delayed recruitment
as left truncation; an events-only registry cutoff as right truncation.
Clarify time origin and distinguish truncation from retained censored follow-up.
Spec 2, 4.2, 12.2. Keep likelihood integration pending (B2.1 then B2.2).
Run scripts/build_docs.sh directly, fast A10 gate and whitespace checks; inspect
generated HTML; commit and independent review, then stop. No new numerical tests.

# B1.d1 — Document interval probabilities

Previous implementation step: 535b854; original worktree base: cc0e30f.
Maintainer requested useful work while the independent spec-test author handles
D-049. One docs topic in docs/working-with-distributions.md: interval mass,
conditioning on a truncation window, readable S/F formulas, executable examples,
boundary policy and the current likelihood-integration limitation.
No code/API/dependency/protected-path changes. Spec 4.2, 7.1–7.3 and 12.2.
Use already-implemented Exponential/Weibull hooks; do not imply likelihood
support is enabled. Run scripts/build_docs.sh directly in both dtype modes,
check rendered topic, write REVIEW.md, commit, independent review, then stop.
B2 remains pending below; docs do not resolve D-049.

# B2.1 — Untruncated left/interval likelihood integration (blocked on D-049)

Start / previous step: 535b854; original worktree base: cc0e30f.
Maintainer authorized likelihood integration. First bounded step enables left
and interval censoring without truncation; B2.2 adds truncation combinations.
Files planned: src/survjax/likelihood.py, distributions/base.py (correct existing
likelihood docstring), tests/unit/test_likelihood_core.py,
tests/unit/test_likelihood_intervals.py, PLAN.md, REVIEW.md; duration list if needed.
Spec 2, 4.2, 7.1–7.3: dispatch censored intervals through family hooks, preserve
exact/right contributions and gradients, current/stored diagnostics, dtype and
batching. Safe inactive set inputs use (0,inf]; density uses reference_time.
Tests: mixed kinds with hand-derived expected values, hooks dispatch, invalid
rows, transformed construction, jit/vmap, first/second derivatives and debug_nans.
No public API, module or dependency changes. Docs to-do: expanded likelihood
modes and diagnostics after integration. Truncation remains explicitly unsupported
in B2.1 and is implemented in B2.2.

STOP: tests/spec/test_a6_unsupported_modes.py unconditionally requires the new
modes to produce NaN. No SPEC_FEATURES switch exists in the current gate or
acceptance suite. Independent-author update/sync is needed; do not change, skip
or xfail protected tests. See D-049. No implementation changes yet.
After resolving the acceptance transition: touched tests both dtypes, fast A10
(or maintainer-designated Slice B) gate, commit, independent review, then stop.
Remaining: B2.2 truncation, B3 full numerical/reference matrix, B4 pruning/docs.


---

# Incoming main history (4239d61)

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
