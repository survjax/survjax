#!/bin/sh
# The implementer's own tests, one process per dtype mode.
# Usage: scripts/run_tests.sh [--fast]
#
#   --fast               deselect tests marked slow (tests/slow_tests.txt)
#   PYTEST_WORKERS       pytest-xdist worker count (default 6; 0 runs serially)
#   SURVJAX_X64_MODES    dtype modes to run, in order (default "1 0")
#   SURVJAX_JAX_CACHE    persistent JAX compilation cache directory; default
#                        <git common dir>/survjax-jax-cache, shared by all
#                        worktrees, cleared past 4 GB; "off" disables it
set -eu

marker=""
case "${1:-}" in
  "") ;;
  --fast) marker="not slow" ;;
  *) echo "usage: scripts/run_tests.sh [--fast]" >&2; exit 2 ;;
esac

workers="${PYTEST_WORKERS:-6}"
modes="${SURVJAX_X64_MODES:-1 0}"

cache="${SURVJAX_JAX_CACHE:-$(git rev-parse --path-format=absolute --git-common-dir)/survjax-jax-cache}"
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

for x64 in $modes; do
  JAX_ENABLE_X64=$x64 uv run --group test pytest -q -n "$workers" --ignore=tests/spec -m "$marker"
done
