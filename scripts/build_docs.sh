#!/usr/bin/env bash
# Build and execute documentation in separate JAX dtype processes (spec 10.5).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

root="$PWD"
build="$root/docs/_build"
mkdir -p "$build"
identity="$build/build-identity.txt"
# Invalidate old success before any setup or execution can fail.
printf 'state: building\ncheckout: %s\n' "$root" > "$identity"
finish() {
  result=$?
  if [ "$result" -ne 0 ]; then
    printf 'state: failed\nexit: %s\n' "$result" >> "$identity"
  fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

commit=$(git rev-parse HEAD)
status=$(git status --porcelain --untracked-files=normal)
# Track source edits during execution; dirty builds are allowed but labelled.
fingerprint=$(git diff HEAD --binary | git hash-object --stdin)
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf 'commit: %s\nstarted_utc: %s\ndtypes: x32 x64\n' "$commit" "$started" >> "$identity"
if [ -n "$status" ]; then
  printf 'tree: dirty\n%s\n' "$status" >> "$identity"
else
  printf 'tree: clean\n' >> "$identity"
fi

export JUPYTER_RUNTIME_DIR="$build/runtime/jupyter"
export IPYTHONDIR="$build/runtime/ipython"
export MPLCONFIGDIR="$build/runtime/matplotlib"
mkdir -p "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR" "$MPLCONFIGDIR"
# Retain the shared hash-checked MNIST cache and any explicit override.
# A cold cache needs network and a writable SURVJAX_EXAMPLES_DATA location.
echo "MNIST cache: ${SURVJAX_EXAMPLES_DATA-~/.cache/survjax-examples/mnist}"
echo "Build identity: $identity"

for mode in 0 1; do
  if [ "$mode" = 1 ]; then
    output="$build/html-x64"
  else
    output="$build/html-x32"
  fi
  echo "Building documentation with JAX_ENABLE_X64=$mode"
  JAX_ENABLE_X64="$mode" uv run --locked --group docs --group examples sphinx-build \
    -b html -E -n -W --keep-going docs "$output"
done

if [ "$commit" != "$(git rev-parse HEAD)" ] ||
   [ "$status" != "$(git status --porcelain --untracked-files=normal)" ] ||
   [ "$fingerprint" != "$(git diff HEAD --binary | git hash-object --stdin)" ]; then
  echo "Checkout changed during build; rerun before inspecting." >&2
  exit 1
fi
printf 'completed_utc: %s\nstate: success\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$identity"
echo "Verified both dtypes. Identity: $identity"
printf 'HTML: %s/index.html\n' "$build/html-x32" "$build/html-x64"
