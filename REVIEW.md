# B1.d2 — Applications and truncation terminology

Independent review: round 1/3

Read PLAN.md B1.d2, then the added application paragraphs in
 docs/working-with-distributions.md, under "Probabilities over intervals" and
"Conditioning on an inclusion window". Spec 2, 4.2 and 12.2.
Start / previous step: e0cdde2; original worktree base: cc0e30f.

One topic on one docs page, 25 changed lines; no code, API, examples or
protected paths changed. Periodic assessment/inspection motivates interval
censoring; delayed recruitment while event-free motivates left truncation
(delayed entry) without resetting the time origin. An events-only retrospective
registry motivates right truncation and contrasts with retained right-censored
follow-up. Two-sided truncation combines the inclusion restrictions.
The existing scope note continues to state that likelihood integration is
pending. B2.1 adds left/interval censoring; B2.2 adds truncation. D-049 remains
open while the independent spec-test author prepares acceptance updates.

Checks:
- scripts/build_docs.sh run directly with the previously required escalation:
  exit 0; both x32 and x64 Sphinx builds succeeded and notebooks executed.
- Inspected generated HTML text for both modes: the new applications, names,
  time-origin explanation and formulas are present. Executable outputs unchanged.
  No browser visual-layout inspection claimed (browser host unavailable earlier).
- scripts/check.sh --fast --through A10: exit 0, all gates passed (includes
  Ruff, mypy, implementer tests, doctests and spec tests in both dtypes).
- git diff --check: passed. No new numerical tests for this prose-only change;
  full non-fast test gate not rerun.

Remaining: B2 acceptance transition and likelihood integration; B3 full
numerical/reference matrix; B4 pruning and remaining docs. This docs step
neither resolves the acceptance blocker nor enables new likelihood modes.
