#!/usr/bin/env bash
# CI plan job: decide whether the CPU matrix runs. Writes run=true|false to
# $GITHUB_OUTPUT. The weekly (schedule) run is skipped when main has not moved
# since the last successful weekly run; a skipped run also counts, for the same
# commit. Every other event runs.
#
# Needs GITHUB_EVENT_NAME, GITHUB_OUTPUT; on schedule also GH_TOKEN,
# GITHUB_REPOSITORY, GITHUB_SHA and GITHUB_STEP_SUMMARY (set by Actions).
set -euo pipefail

run=true
if [ "$GITHUB_EVENT_NAME" = schedule ]; then
  last="$(gh api "repos/$GITHUB_REPOSITORY/actions/workflows/ci.yml/runs?event=schedule&status=success&per_page=1" \
    --jq '.workflow_runs[0].head_sha // ""')"
  if [ "$last" = "$GITHUB_SHA" ]; then
    run=false
    echo "Weekly run skipped: $GITHUB_SHA already passed the weekly matrix." >> "$GITHUB_STEP_SUMMARY"
  fi
fi
echo "run=$run" >> "$GITHUB_OUTPUT"
