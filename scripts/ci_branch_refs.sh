#!/usr/bin/env bash
# Prepare branch references for the protected-path guard in scripts/check.sh,
# which diffs main...HEAD and grants access by the current branch name.
#
# - Creates a local main from origin/main when missing (needs a checkout with
#   fetch-depth: 0).
# - On pull_request, names the detached merge commit after the PR source branch
#   (CI_SOURCE_BRANCH, from github.head_ref), keeping its tree and parents.
set -euo pipefail

if ! git show-ref --verify --quiet refs/heads/main; then
  git branch main refs/remotes/origin/main
fi
if [ "${GITHUB_EVENT_NAME:-}" = pull_request ]; then
  guard_branch="$CI_SOURCE_BRANCH"
  # Preserve the baseline for PRs from a branch also named main.
  if [ "$guard_branch" = main ]; then
    guard_branch=ci-pr-main
  fi
  git switch -c "$guard_branch"
fi
