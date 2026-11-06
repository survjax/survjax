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
  /*) ;;
  *) output_root="$caller_dir/$output_root" ;;
esac

cd "$repo_dir"
# Use this worktree even when invoked by a portal in another environment.
export UV_PROJECT_ENVIRONMENT="$repo_dir/.venv"
export PYTHONPATH="$repo_dir/src:$repo_dir"
output_root="$(uv run --project "$repo_dir" --locked --group docs --group examples python -c \
  'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$output_root")"
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
export JAX_COMPILATION_CACHE_DIR="$output_root/.cache/jax"
export MPLCONFIGDIR="$output_root/.cache/matplotlib"
export IPYTHONDIR="$output_root/.cache/ipython"
export JUPYTER_PATH="$repo_dir/.venv/share/jupyter"
export JUPYTER_PREFER_ENV_PATH=1

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
