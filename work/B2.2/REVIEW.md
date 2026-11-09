# B2.2 — Truncation likelihood integration

Independent review: round 1/3

Read this step's PLAN.md, spec 2/4.2/7.1–7.3, src/survjax/likelihood.py,
tests/unit/test_likelihood_truncation.py, then scripts/spec-features.txt.
Implementation base 5c271b4; previous B2.1 last checkpoint 47aa112;
original worktree base cc0e30f. B2.1 review is still pending its reference fix;
maintainer expressly authorized proceeding with B2.2 in the meantime.

Invariants: all current/stored data issues and parameter isolation propagate.
Exact rows use log f - log Z with nonfinite log Z rejected; censored rows use
the conditional hook, preserving built-in cancellation-resistant formulas.
Real censoring sets intersect the window before inactive operands are selected.
Inactive exact sets use their own window; inactive density uses the stopped
family reference. Invalid rows receive safe windows. No family-type dispatch,
probability floor, new API or dependency. Both observation features are enabled.

Evidence:
- Touched core, censoring and truncation unit files: 32 passed in each dtype.
- Full truncation acceptance files: 139 passed (x64), 132 passed/7 skipped
  (x32). Values cover all observation kinds/windows; diagnostics include
  zero-mass windows and partial/empty intersections; transformations,
  derivative/Hessian, deep-tail and conditional-hook contracts are covered.
- New unit tests independently derive mixed Exponential/Weibull values (spec 2),
  delayed-entry gradients/Hessians (7.3), invalid intersections (7.1), and
  conditional-hook dispatch with safe inactive operands (4.2/7.2).
- scripts/check.sh --fast --through A12: formatting/lint clean; mypy 23 source
  files clean; implementation x64 1086 passed/8 skipped, x32 1091 passed/3
  skipped; doctests 4 passed each. Acceptance x64 3395 passed/32 skipped,
  one failed, two existing PH reference warnings. Gate exits 1 before x32
  acceptance, solely on the known D-B2.1-001 tiny-tail reference error.
- git diff --check clean. Docs build not run for this code-only step.

Known blocker: ../B2.1/DEVIATIONS.md D-B2.1-001; independent spec-test author
is fixing it. No protected test or tolerance was changed to pass the gate.
No completed acceptance claim until the fix is integrated and checks rerun.
Remaining: B2.1 review, both steps' final acceptance, B3 references and B4
pruning/docs. No further step started here.

Full timing refresh is still running at this checkpoint: x64 implementation
1348 passed/8 skipped; x64 acceptance 3817 passed/32 skipped, one known
D-B2.1-001 failure and two PH reference warnings. Float32 is in progress.
The generated timing list will be checkpointed separately after it finishes;
its script tolerates failures and does not constitute a passing gate.

Timing refresh completed: 767 slow tests recorded. Full x32 implementation
1353 passed/3 skipped; full x32 acceptance 3775 passed/70 skipped, one known
D-B2.1-001 failure and two PH reference warnings. Both full acceptance runs
have only that reference failure. Generated timings are in a separate follow-up
commit; no code or test assertion changed after the review checkpoint.
