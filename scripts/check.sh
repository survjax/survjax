#!/usr/bin/env bash
# The maintainer's gate. Usage: scripts/check.sh [--fast] --through A<N>
#
# 1. protected-path guard (branch name decides what a branch may touch)
# 2. ruff format + lint, mypy
# 3. the implementer's own tests, both dtype modes (scripts/run_tests.sh)
# 4. source doctests, both dtype modes
# 5. independent spec tests up to step <N>, both dtype modes
#
# --fast deselects the tests listed in tests/slow_tests.txt (marker `slow`) in
# steps 3 and 5: the step gate for the implementer-reviewer loop. Without it
# every test runs: the merge gate, also run by CI.
#
# PYTEST_WORKERS sets the pytest-xdist worker count for steps 3 and 5
# (default 6; 0 runs serially). SURVJAX_X64_MODES picks the dtype modes
# (default "1 0"). SURVJAX_JAX_CACHE sets the persistent JAX compilation cache
# directory (default <git common dir>/survjax-jax-cache, shared by worktrees;
# "off" disables it). SURVJAX_WHEEL names a built wheel for the spec wheel
# tests; when unset, one is built into a temporary directory.
set -euo pipefail

usage() { echo "usage: scripts/check.sh [--fast] --through A<N>" >&2; exit 2; }
through=""
fast=""
while [ $# -gt 0 ]; do
  case "$1" in
    --through) through="${2:-}"; shift 2 ;;
    --fast) fast="--fast"; shift ;;
    *) usage ;;
  esac
done
[ -n "$through" ] || usage
workers="${PYTEST_WORKERS:-6}"
export PYTEST_WORKERS="$workers"
modes="${SURVJAX_X64_MODES:-1 0}"
export SURVJAX_X64_MODES="$modes"
marker=""
[ -z "$fast" ] || marker="not slow"

cd "$(git rev-parse --show-toplevel)"

# --- 1. guard -------------------------------------------------------------
# Implementation branches must not touch these. Branch prefixes grant access:
#   spectests-*  may change tests/spec/
#   meta-*       may change CLAUDE.md, AGENTS.md, scripts/check.sh,
#                scripts/sync_spec_tests.sh, .agents/ (skills), .workmux.yaml
# The spec itself lives in ../design (spec is a symlink); no branch here may
# change the symlink.
branch="$(git rev-parse --abbrev-ref HEAD)"
allowed=""
case "$branch" in
  spectests-*) allowed='^tests/spec/' ;;
  meta-*)      allowed='^(CLAUDE\.md$|AGENTS\.md$|scripts/check\.sh$|scripts/sync_spec_tests\.sh$|\.agents/|\.workmux\.yaml$)' ;;
esac
protected='^(tests/spec/|spec$|spec/|CLAUDE\.md$|AGENTS\.md$|scripts/check\.sh$|scripts/sync_spec_tests\.sh$|\.agents/|\.workmux\.yaml$)'
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

# --- 3. implementer tests (dtype modes handled inside) --------------------
# One wheel for the install tests here and the spec wheel tests in step 5.
if [ -z "${SURVJAX_WHEEL:-}" ]; then
  wheel_dir="$(mktemp -d)"
  trap 'rm -rf "$wheel_dir"' EXIT
  uv build --wheel -q -o "$wheel_dir"
  SURVJAX_WHEEL="$(ls "$wheel_dir"/*.whl)"
  export SURVJAX_WHEEL
fi
# run_tests.sh also sets up the shared JAX compilation cache; repeat it here
# so step 5 uses the same cache.
cache="${SURVJAX_JAX_CACHE:-$(git rev-parse --path-format=absolute --git-common-dir)/survjax-jax-cache}"
export SURVJAX_JAX_CACHE="$cache"
if [ "$cache" != off ]; then
  export JAX_COMPILATION_CACHE_DIR="$cache"
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
  # No JAX_COMPILATION_CACHE_MAX_SIZE: its LRU locking stalls xdist workers.
  # Bound the size here instead: drop the whole cache past 4 GB.
  case "$cache" in *jax-cache*) ;; *) echo "SURVJAX_JAX_CACHE must name a *jax-cache* directory" >&2; exit 2 ;; esac
  if [ -d "$cache" ] && [ "$(du -sk "$cache" | cut -f1)" -gt 4000000 ]; then
    rm -rf "$cache"
  fi
fi
scripts/run_tests.sh $fast

# --- 4. executable source documentation, one process per dtype mode ------
for x64 in $modes; do
  echo "source doctests, JAX_ENABLE_X64=$x64"
  JAX_ENABLE_X64=$x64 uv run --group test pytest src --doctest-modules -q
done

# --- 5. spec tests, one process per dtype mode ----------------------------
# pytest exit code 5 means "no tests collected": fine while tests/spec is empty.
for x64 in $modes; do
  echo "spec tests through $through, JAX_ENABLE_X64=$x64${fast:+, fast}"
  set +e
  JAX_ENABLE_X64=$x64 SPEC_THROUGH="$through" uv run --group test pytest tests/spec -q -n "$workers" -m "$marker"
  rc=$?
  set -e
  if [ $rc -ne 0 ] && [ $rc -ne 5 ]; then
    echo "spec tests failed (exit $rc) with JAX_ENABLE_X64=$x64" >&2
    exit $rc
  fi
done

echo "check: all gates passed through $through${fast:+ (fast: slow tests skipped)}"
