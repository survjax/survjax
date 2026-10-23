# Review: default spec-test sync path

Read `scripts/sync_spec_tests.sh` first, then the 2026-10-23 entry in `PLAN.md`.

The source defaults to `../spectest`; callers can still supply a path.
Usage text documents the optional argument. Existing branch, clean-tree and
spec-version guards are unchanged. No public API or spec requirements change.

Checks actually run:

- `bash -n scripts/sync_spec_tests.sh`: exit 0.
- Temporary Git repo smoke checks: default and explicit paths both copied the
  fixture, removed an obsolete file and recorded the tester commit. A missing
  explicit path returned exit 1 with the updated usage text. All passed.
- `git diff --check`: exit 0 after removing an extra blank line in the plan.

No permanent tests were added for this small shell default change. No library
suite was run. The real tester repo was not read or synced. No open concerns.
The script change was already captured by concurrent commit `1db22c4` during
verification; the follow-up commit records this review and plan whitespace fix.
