# CI.WINDOWS.1 — repair Windows test tooling

Feature: repair the scheduled Windows CPU CI failures in run 42611200442.
Base commit and original worktree base: 0c494ad05376aa3407e98427cd2d11e995f95eb7.
Preceding step: D.LOGLOGISTIC.2, final commit 54b242c
([handoff](../D.LOGLOGISTIC.2/REVIEW.md)). No feature dependencies.
Cumulative gate target: A12. Step ID selected for this unnumbered repair task.

## Scope

- tests/install/test_wheel.py: decode the UTF-8 README explicitly.
- tests/conftest.py and both docs shell test modules: invoke Git Bash explicitly
  on Windows, use shell launchers for fake uv, and compare native paths correctly.
- scripts/build_docs.sh: handle Git Bash/native Windows path conversion and
  the native Python path separator when preparing Python environments.
- Keep every existing shell test running; no numerical, API, dependency,
  documentation-page or protected-path changes. Target under 300 changed lines.

Contract: spec 10.2–10.3 packaging and verification; no new supported numerical
platform claim. These changes repair existing CI tooling only.

## Validation

Run the three affected test modules in full in both dtype modes, six workers;
run scripts/check.sh --fast --through A12. Record Windows verification as pending
unless an actual Windows runner is available. Commit records with the repair.
Independent Claude review explicitly waived by the maintainer for this step.

## Docs to-do

None. No public behavior or API is added.
