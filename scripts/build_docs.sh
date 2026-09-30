#!/usr/bin/env bash
# Build and execute documentation in separate JAX dtype processes (spec 10.5).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

for mode in 0 1; do
  if [ "$mode" = 1 ]; then
    output=docs/_build/html-x64
  else
    output=docs/_build/html-x32
  fi
  echo "Building documentation with JAX_ENABLE_X64=$mode"
  JAX_ENABLE_X64="$mode" uv run --locked --group docs sphinx-build \
    -b html -E -n -W --keep-going docs "$output"
done
