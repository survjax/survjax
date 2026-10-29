#!/usr/bin/env bash
# Build once, retain the artifact for review, and run the unchanged gate.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ "$#" -eq 0 ]; then
  set -- --through A9
fi
export UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never UV_LOCKED=true
wheel_dir="$(mktemp -d "${TMPDIR:-/tmp}/survjax-wheel.XXXXXX")"
uv build --wheel --out-dir "$wheel_dir"
wheels=("$wheel_dir"/*.whl)
if [ "${#wheels[@]}" -ne 1 ] || [ ! -f "${wheels[0]}" ]; then
  echo "expected one built wheel in $wheel_dir" >&2
  exit 1
fi
export SURVJAX_WHEEL="${wheels[0]}"
printf 'Wheel: %s\n' "$SURVJAX_WHEEL"
scripts/check.sh "$@"
