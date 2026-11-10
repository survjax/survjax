#!/usr/bin/env bash
# Build and execute documentation in separate JAX dtype processes.
set -euo pipefail

usage() {
  echo "usage: scripts/build_docs.sh [--output-dir DIR]"
  echo "Build both dtype modes; DIR defaults to docs/_build in this worktree."
  echo "Relative DIR paths are resolved from the caller's current directory."
}

caller_dir="$PWD"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# Native Python needs Windows paths and its own search-path separator.
python_path_separator=:
if [[ "$OSTYPE" == msys* ]]; then
  caller_dir="$(cygpath -m "$caller_dir")"
  repo_dir="$(cygpath -m "$repo_dir")"
  python_path_separator=';'
fi
output_root="$repo_dir/docs/_build"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --output-dir)
      if [ "$#" -lt 2 ] || [ -z "$2" ] || [[ "$2" == --* ]]; then
        usage >&2
        exit 2
      fi
      output_root="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
case "$output_root" in
  /*|[A-Za-z]:/*) ;;
  *) output_root="$caller_dir/$output_root" ;;
esac

cd "$repo_dir"
# Use this worktree even when invoked by a portal in another environment.
export UV_PROJECT_ENVIRONMENT="$repo_dir/.venv"
export PYTHONPATH="$repo_dir/src$python_path_separator$repo_dir"
# Invalidate an existing certificate before dependency/path setup can fail.
if [ -f "$output_root/build-identity.txt" ]; then
  printf 'state: building\n' > "$output_root/build-identity.txt"
fi
output_root="$(uv run --project "$repo_dir" --locked --group docs --group examples python -c \
  'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve().as_posix())' "$output_root")"
# Windows Python writes CRLF even when its stdout is captured by Git Bash.
output_root="${output_root%$'\r'}"
case "$output_root" in
  "$repo_dir"|"$repo_dir/docs")
    echo "Output must be a dedicated build directory, not the repository or docs root." >&2
    exit 2 ;;
  "$repo_dir/docs/_build"|"$repo_dir/docs/_build/"*) ;;
  "$repo_dir/docs/"*)
    echo "Output inside docs must be under docs/_build to stay outside source discovery." >&2
    exit 2 ;;
esac
mkdir -p "$output_root"
# A caller-selected build root must stay untracked even inside this repository.
printf '*\n' > "$output_root/.gitignore"
identity="$output_root/build-identity.txt"
# Invalidate old success before any setup or execution can fail.
printf 'state: building\ncheckout: %s\n' "$repo_dir" > "$identity"
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

export JAX_COMPILATION_CACHE_DIR="$output_root/.cache/jax"
export MPLCONFIGDIR="$output_root/.cache/matplotlib"
export IPYTHONDIR="$output_root/.cache/ipython"
export JUPYTER_RUNTIME_DIR="$output_root/.cache/jupyter-runtime"
mkdir -p "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR" "$MPLCONFIGDIR"
export JUPYTER_PATH="$repo_dir/.venv/share/jupyter"
export JUPYTER_PREFER_ENV_PATH=1

# Retain the shared hash-checked MNIST cache and any explicit override.
# A cold cache needs network and a writable SURVJAX_EXAMPLES_DATA location.
echo "MNIST cache: ${SURVJAX_EXAMPLES_DATA-~/.cache/survjax-examples/mnist}"
echo "Build identity: $identity"

for mode in 0 1; do
  if [ "$mode" = 1 ]; then
    output="$output_root/html-x64"
  else
    output="$output_root/html-x32"
  fi
  echo "Building documentation with JAX_ENABLE_X64=$mode into $output"
  JAX_ENABLE_X64="$mode" uv run --project "$repo_dir" --locked --group docs --group examples python -m sphinx \
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
printf 'HTML: %s/index.html\n' "$output_root/html-x32" "$output_root/html-x64"
