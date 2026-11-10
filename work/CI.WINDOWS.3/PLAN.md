# CI.WINDOWS.3 — normalize paths with current Git Bash

Feature: repair seven docs shell-test failures per Windows dtype in full-matrix
run 42781567128. Step ID selected for this unnumbered repair task.
Base and original worktree base: b8454db475ae20465ac942346bd9d4922b097f76.
Dependency and preceding step: [CI.WINDOWS.2](../CI.WINDOWS.2/REVIEW.md), b8454db.
Cumulative gate target: A12.

## Scope

- scripts/build_docs.sh: recognize both msys and cygwin OSTYPE values for native
  Windows path conversion and Python search-path separator.
- tests/tooling/test_build_docs.py: apply the same platform detection in fake uv.
- tests/unit/test_docs_build_script.py: exercise both OSTYPE values even on POSIX,
  using a fake cygpath there; retain native path and isolation assertions.
- No public API, dependency, docs page, spec or protected-path changes.
  Spec 10.3 verification tooling only. Target below 300 changed lines.

## Validation

Run touched test modules in both dtype modes with six workers, and the A12 fast
gate. Commit and run independent review, then push to existing PR #6 and dispatch
the full Windows matrix. No merge as part of this repair.

## Docs to-do

None.
