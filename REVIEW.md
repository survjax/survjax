# ALIGN.2a — Approved D-042 spec example cleanup

Independent review remains waived by maintainer while reviewer tokens are
unavailable. Starting/previous step: a4c8b42; worktree starting point: 3cb4cd7.
Read PLAN.md ALIGN.2a, then the design-repo diff for spec/SPECIFICATION.md
and the four affected use-case design notes. Only PH example spelling changes;
revision 46 and all requirements remain fixed. Constructor behavior and
executable guides were already migrated in ALIGN.1/ALIGN.d1.

Validation complete below. No tests/spec edits. D-043/D-044 still await separate
maintainer decisions. Preserve deviation statuses; record the D-042 approval
without independently marking entries resolved.

## Validation and outcome

- Design commit: 54bbc62e3880dbbe5544cf913fb95291e78904f9; five spec files,
  14 insertions and 14 deletions. Example spelling only; revision remains 46.
- Inspected the complete design diff. `git -C spec diff --check` passed.
- Scanned PH call snippets across SPECIFICATION.md and all use-case notes:
  all 12 remaining call examples use baseline=. Replaced a type-only shorthand
  with prose (PH with a Weibull baseline) to avoid an apparent positional call.
- Both Python sketches in 02-bayesian-survival-details retain identical ASTs
  after keyword normalization. Model/prior/route mathematics unchanged.
- `../scripts/spec-since.sh rmsttest --mark`: reports
  rmsttest/.spec-seen -> 54bbc62. Used the current worktree directory argument;
  no mutation of the separate dev checkout.
- D-042 approval and implementation recorded; statuses remain untouched.
- Implementation, tests, docs pages, dependency declarations and lockfile
  unchanged. No numerical tests/docs build repeated for spec example spelling.
- `git diff --check`: passes. Design working tree clean after its commit.

Independent review remains waived. D-043 and D-044 await separate decisions;
ALIGN.3 verification remains pending. Present D-043 now and wait.

## ALIGN.2b author handoff

Maintainer approved the D-043 expected mask [False,True,True]. Saved the
requested short instruction in SPEC-TEST-D043.txt. Checked it against the
read-only failing test, CappedRate predicate and spec 4.7. This is a planning/
handoff record, not an acceptance-test edit; no numerical tests rerun.
Starting/previous step: 3580ff2; worktree starting point: 3cb4cd7. D-044 still
requires its own decision. `git diff --check` passes before commit.
