#!/usr/bin/env bash
# Copy the independent spec tests from the tester repo into this repo.
# Usage (on a branch named spectests-*): scripts/sync_spec_tests.sh ../spectest
# Afterwards review `git diff`, run scripts/check.sh, commit, merge to main.
set -euo pipefail

die() { echo "sync: $*" >&2; exit 1; }
src="${1:-}"
[ -n "$src" ] && [ -d "$src/tests/spec" ] || die "usage: $0 <path to spectest>"

cd "$(git rev-parse --show-toplevel)"
branch="$(git rev-parse --abbrev-ref HEAD)"
case "$branch" in spectests-*) ;; *) die "must be on a spectests-* branch (on '$branch')" ;; esac
[ -z "$(git status --porcelain)" ] || die "working tree not clean"
[ -z "$(git -C "$src" status --porcelain)" ] || die "tester repo has uncommitted changes"

# The tester must have worked from the same spec version as this repo.
[ "$(cat .spec-seen)" = "$(cat "$src/.spec-seen")" ] \
  || die ".spec-seen differs from the tester's: align both with the spec first (../scripts/spec-since.sh)"

# Replace tests/spec wholesale so files deleted by the tester disappear here too.
rm -rf tests/spec
mkdir -p tests/spec
(cd "$src/tests/spec" && tar --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='SOURCE_COMMIT' -cf - .) | tar -xf - -C tests/spec
git -C "$src" rev-parse HEAD > tests/spec/SOURCE_COMMIT

echo "synced tests/spec from $(cat tests/spec/SOURCE_COMMIT)"
git status --short tests/spec
