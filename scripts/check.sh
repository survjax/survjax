#!/usr/bin/env bash
# The maintainer's gate. Usage: scripts/check.sh --through A<N>
#
# 1. protected-path guard (branch name decides what a branch may touch)
# 2. ruff format + lint, mypy
# 3. the implementer's own tests, both dtype modes (scripts/run_tests.sh)
# 4. source doctests, both dtype modes
# 5. independent spec tests up to step <N>, both dtype modes
#
# PYTEST_WORKERS sets the pytest-xdist worker count for steps 3 and 5
# (default 6; 0 runs serially).
set -euo pipefail

usage() { echo "usage: scripts/check.sh --through A<N>" >&2; exit 2; }
through=""
while [ $# -gt 0 ]; do
  case "$1" in
    --through) through="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$through" ] || usage
workers="${PYTEST_WORKERS:-6}"
export PYTEST_WORKERS="$workers"

cd "$(git rev-parse --show-toplevel)"

# --- 1. guard -------------------------------------------------------------
# Implementation branches must not touch these. Branch prefixes grant access:
#   spectests-*  may change tests/spec/
#   meta-*       may change CLAUDE.md, AGENTS.md, scripts/check.sh,
#                scripts/sync_spec_tests.sh
# The spec itself lives in ../design (spec is a symlink); no branch here may
# change the symlink.
branch="$(git rev-parse --abbrev-ref HEAD)"
allowed=""
case "$branch" in
  spectests-*) allowed='^tests/spec/' ;;
  meta-*)      allowed='^(CLAUDE\.md$|AGENTS\.md$|scripts/check\.sh$|scripts/sync_spec_tests\.sh$)' ;;
esac
protected='^(tests/spec/|spec$|spec/|CLAUDE\.md$|AGENTS\.md$|scripts/check\.sh$|scripts/sync_spec_tests\.sh$)'
if git rev-parse --verify -q main >/dev/null; then
  touched="$(git diff --name-only main...HEAD | grep -E "$protected" || true)"
  if [ -n "$allowed" ]; then
    touched="$(printf '%s\n' "$touched" | grep -Ev "$allowed" || true)"
  fi
  if [ -n "$touched" ]; then
    echo "GUARD FAILED: branch '$branch' changes protected paths:" >&2
    printf '%s\n' "$touched" >&2
    exit 1
  fi
fi
echo "guard: ok ($branch)"

# --- 2. static checks -----------------------------------------------------
[ -x scripts/run_tests.sh ] || { echo "scripts/run_tests.sh missing: step A0 first" >&2; exit 1; }
uv run ruff format --check .
uv run ruff check .
uv run mypy src

# --- 3. implementer tests (both dtype modes handled inside) ---------------
scripts/run_tests.sh

# --- 4. executable source documentation, one process per dtype mode ------
for x64 in 1 0; do
  echo "source doctests, JAX_ENABLE_X64=$x64"
  JAX_ENABLE_X64=$x64 uv run --group test pytest src --doctest-modules -q
done

# --- 5. spec tests, one process per dtype mode ----------------------------
# pytest exit code 5 means "no tests collected": fine while tests/spec is empty.
for x64 in 1 0; do
  echo "spec tests through $through, JAX_ENABLE_X64=$x64"
  set +e
  JAX_ENABLE_X64=$x64 SPEC_THROUGH="$through" uv run --group test pytest tests/spec -q -n "$workers"
  rc=$?
  set -e
  if [ $rc -ne 0 ] && [ $rc -ne 5 ]; then
    echo "spec tests failed (exit $rc) with JAX_ENABLE_X64=$x64" >&2
    exit $rc
  fi
done

echo "check: all gates passed through $through"
