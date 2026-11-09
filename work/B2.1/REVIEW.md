# B2.1 — Untruncated censoring integration (blocked before review)

Independent review: not started; acceptance reference dispute needs resolution.
Read PLAN.md here, DEVIATIONS.md D-B2.1-001, scripts/spec-features.txt,
src/survjax/likelihood.py, distributions/_log_math.py, then unit tests.
Base/merge: c9fee45; previous step: 5fd6320; original worktree base: cc0e30f.

All untruncated censored rows dispatch through log_interval_prob; inactive sets
use (0,inf], inactive densities use stopped reference_time. Existing parameter
isolation, stored/current data checks, broadcasting and dtype promotion remain.
Truncation stays unsupported, with its A6 checks active and feature disabled.
Generic interval failure selection now follows endpoint handling, avoiding an
inactive NaN on valid inputs under eager debug_nans. Failure semantics remain.

Unit evidence:
- Touched core/new integration files: 27 passed in each dtype.
- After generic regression fix: core, integration, interval defaults,
  likelihood debug_nans and PH files: 64 passed (x64), 63 passed/1 skipped (x32).
- Mixed analytical values (spec 2), hook dispatch/safe inputs (4.2/7.2),
  analytical first/second derivatives plus check_grads (7.3), jit/vmap covered.
- Initial float64 finite-difference check failed with step 1e-4; reducing the
  step to 1e-5 resolves approximation error, with no tolerance change.
- Full timing run initially found three generic debug_nans regressions; the
  targeted rerun above verifies the fix. Timing-run output is not a clean gate.
- Ruff formatting/lint passed; mypy passed for 23 source files; diff check clean.
- Fast A12 gate: implementation 1084 passed/8 skipped (x64), 1089 passed/3
  skipped (x32); doctests 4 passed per dtype. Acceptance x64: 3520 passed, 32 skipped, 1 failed (D-B2.1-001),
  two existing PH reference warnings; gate exits 1 before x32 acceptance.

Uncertainty: independent tiny-tail reference must be corrected by its owner;
no independent review or completed B2.1 claim yet. No docs build for this code
step. Remaining B2.2 truncation, B3 numerical/reference matrix, B4 pruning/docs.

Timing refresh completed (693 slow tests). Its later x32 implementation run
passed: 1348 passed, 3 skipped. The x32 acceptance summary was not captured
before the script deleted its temporary logs; no pass is claimed. The timing
script tolerates test failures, so its exit 0 is not acceptance evidence.
Maintainer requested deferred instructions: SPECTEST-HANDOFF.md here.

Maintainer authorized continuing to B2.2 while the reference is fixed.
Checkpoint B2.1 for a separate reviewable diff; its review remains pending.
