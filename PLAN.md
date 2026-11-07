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

Remaining substeps: resolve any acceptance failures in separately reviewed
steps; full merge gate belongs to the maintainer. Demo and final release
completion are separate from passing this PH/RMST gate.
