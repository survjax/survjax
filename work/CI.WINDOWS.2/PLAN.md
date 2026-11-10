# CI.WINDOWS.2 — locate Git Bash across Git layouts

Feature: repair the Windows shell-test setup in CI run 42701729597.
Base and original worktree base: 420b56c105da8e8a275337a1afcf1e6acebbe005.
Preceding step: [CI.WINDOWS.1](../CI.WINDOWS.1/REVIEW.md), commit 420b56c.
Dependencies: CI.WINDOWS.1. Cumulative gate target: A12.
Step ID selected for this unnumbered repair task.

## Scope

- tests/conftest.py: search Git executable ancestors for bin/bash.exe,
  supporting cmd, bin and ucrt64/bin Git executable locations.
- tests/tooling/test_bash_executable.py: reproduce those layouts using a
  temporary installation and check explicit failure when Bash is absent.
- No public API, new module in src, dependencies or protected paths touched.
  Spec 10.2–10.3 verification tooling; target below 300 changed lines.

## Validation

Run the new test and both docs shell-test modules in both dtype modes with six
workers; then scripts/check.sh --fast --through A12. Commit and run independent
review. Actual Windows execution requires maintainer integration and CI rerun.

## Docs to-do

None.
