# TEST.STAGING handoff

Independent review: round 1/3.
Final verdict: maintainer; accepted by the maintainer in the conversation.
Reviewed implementation commit: 9a274211745e063a4c040028a668b503caa7fd72.
Review file: ../../../review/slices/TEST.STAGING.md.

S-1 requested confirmation of the protected gate edit and branch exception;
the maintainer agreed after receiving the review outcome. N-1 notes that the
tester conftest does not yet consume SPEC_FEATURES; the task explicitly permits
this while the feature list is empty. Verify the consumer when tests are synced.
The reviewer could not reproduce the gate: the detached checkout failed the
branch guard, and a direct pytest attempt was blocked by uv cache permissions.
The implementer's passing results below remain separate evidence.

Read PLAN.md, scripts/spec-features.txt, then the guard and spec-test sections
of scripts/check.sh. Workspace docs/workflow.md, "Staging after the release",
defines this workflow. The maintainer's task explicitly authorizes the gate edit.

Invariants: --through still selects the release baseline; feature selection
comes from the committed list rather than an inherited environment variable;
comments and blank lines contribute no names; both dtype processes inherit the
same comma-separated list, including the empty string. No feature is enabled.
The branch guard exception permits only scripts/check.sh on test-staging.
CI and release already invoke scripts/check.sh --through A12.

Validation:
- bash -n scripts/check.sh and git diff --check: exit 0.
- Parser extracted from the actual script: 3 cases passed (empty input,
  comments/blank lines, whitespace/inline comments/multiple names).
- Disposable-repository orchestration with mocked uv: empty and two-feature
  lists reach both spec-test processes with A12 unchanged; inherited
  SPEC_FEATURES is overwritten. Passed. Initial harness attempt lacked a Git
  commit and exited 128; adding a temporary baseline commit fixed the harness.
- scripts/check.sh --fast --through A12: exit 0, all gates passed.
  Ruff: 198 files formatted, lint clean; mypy: 22 source files clean.
  Implementation x64: 1072 passed/8 skipped; x32: 1077 passed/3 skipped.
  Doctests: 4 passed per mode. Spec x64: 3397 passed/32 skipped;
  x32: 3391 passed/38 skipped. Each spec run reports two NumPy warnings from
  the expected-value calculation at test_a11_ph_direct_methods.py:78.

No library behavior changed; no numerical tests or mutations were added.
Full slow gate and docs build have not run. Independent review ran once.

The maintainer authorized creation of the missing reviewer checkout in this
conversation. The pre-existing .spec-seen edit was stashed for the clean-tree
precondition and restored after review. On acceptance, the maintainer requested
updating and committing it: now aligned to design commit e991e38, revision 51.

Starting base: 3abefc5; previous step tip: ff07cf9 (A12 acceptance).
Remaining: maintainer integration into main and a
merge of main into rmst. The separate rmst D-049/plan update is not this step.
Roadmap handoff: staging mechanism implemented, no post-release feature enabled;
review accepted; integration pending. No shared progress files changed.
