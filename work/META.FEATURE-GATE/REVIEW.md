# META.FEATURE-GATE — Default feature gate and merge bookkeeping

Independent review: round 1/3

Read PLAN.md, scripts/check.sh and tests/unit/test_check_script.py, then the
wheel/inventory helpers, CI/release callers and current role/docs guidance.
The maintainer explicitly requested the protected check.sh/role changes,
formal deviation closure, Q-066 disposition, status edits and bookkeeping
commits in this task. This is a meta branch, not a numerical implementation.

The default is all baseline tests plus the committed feature list. Empty
SPEC_THROUGH is passed explicitly, overriding an inherited narrowing filter.
The existing collector's feature/until rules are unchanged. Optional --through
is retained for historical diagnostics and validates A<N> before running tools.
The slow inventory now reads the same committed list; check_wheel no longer
silently limits acceptance to A9. No features are added or removed.

Subprocess tests exercise the actual shell with stub external commands, both
dtype invocations, inherited filter override, comment/blank feature parsing,
explicit legacy limit and malformed arguments. The harness uses the existing
Git Bash fixture and LF shell files; Windows execution is not claimed locally.
No acceptance assertions, numerical behavior or dependencies changed.

Bookkeeping: B2.1/B4.3 defects are formally resolved using independent fixes
and B7 evidence. B8's formal maintainer verdict is preserved and its acceptance
recorded. Design status-only edits are separate commit e95cfa1 (no revision
bump); .spec-seen acknowledges them. Q-066 is separately confirmed in spectest
9ea1ed9 and will be synced on a spectests branch, without changing assertions.
Root workflow/reviewer instructions and roadmap are separate workspace records.

## Validation

- Full scripts/check.sh with no flags: exit 0. Format/lint/mypy passed.
  Implementation float64 1527 passed/8 skipped; float32 1526 passed/9 skipped.
  Source doctests 4 passed per dtype. Acceptance float64 4864 passed/46 skipped;
  float32 4776 passed/124 skipped, two existing PH boundary-reference warnings
  per mode. All five committed features selected, baseline=all.
- Portable shell harness improvements landed during the full run; the final
  focused regression run passed all 7 tests (2.60s), format/lint clean.
  Production runner and acceptance sources were unchanged throughout the run.
- scripts/build_docs.sh: exit 0, both dtype builds, completed
  2026-11-13T10:50:22Z. Parsed both rendered releasing/contributing pages:
  check.sh present, --through A12 absent. No browser layout check claimed.
- bash -n on all three changed scripts and git diff --check passed.

Known pre-existing limitation: check.sh's protected-path guard uses branch
names, so a detached reviewer may fail the guard for authorized meta edits.
Report that honestly; no guard bypass or permission expansion is introduced.
The slow-inventory script's existing treatment of pytest failures is unchanged.
No timed inventory regeneration is needed for these sub-second unit cases.

Starting/previous-step/original base: 65340045dce1e894c25b1fd7a78086775572138f.
Remaining: independent review, documentation-only Q-066 sync, root tracker
commit. No publication or main integration is claimed by this step.
