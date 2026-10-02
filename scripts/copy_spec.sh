#!/usr/bin/env bash
# Copy the spec between dev and the tester repo (spectest).
#
#   scripts/copy_spec.sh to-tester   [tester-path]   dev main -> tester
#       copies spec/SPECIFICATION.md, spec/GOALS.md and roles/TESTER.md (-> CLAUDE.md)
#   scripts/copy_spec.sh from-tester [tester-path]   tester -> dev, on a spec-* branch
#       copies spec/SPECIFICATION.md and spec/GOALS.md
#
# tester-path defaults to ../spectest. Run from dev.
# Only copies: review the diff and commit yourself.
set -euo pipefail

die() { echo "copy_spec: $*" >&2; exit 1; }
usage="usage: $0 to-tester|from-tester [path to spectest]"

direction="${1:-}"
top="$(git rev-parse --show-toplevel)"
tester="${2:-$top/../spectest}"
[ -d "$tester/spec" ] || die "no spec/ in $tester"
tester="$(cd "$tester" && pwd)"
cd "$top"

spec_files=(spec/SPECIFICATION.md spec/GOALS.md)
branch="$(git rev-parse --abbrev-ref HEAD)"
# no uncommitted changes (staged or not) to the given paths in repo $1
clean() { git -C "$1" diff --quiet HEAD -- "${@:2}"; }

case "$direction" in
  to-tester)
    [ "$branch" = main ] || die "dev must be on main (on '$branch'): the spec lands there first"
    clean . spec roles/TESTER.md || die "dev: uncommitted changes in spec/ or roles/TESTER.md"
    clean "$tester" spec CLAUDE.md || die "tester repo: uncommitted changes in spec/ or CLAUDE.md"
    for f in "${spec_files[@]}"; do cp "$f" "$tester/$f"; done
    cp roles/TESTER.md "$tester/CLAUDE.md"
    from="dev $(git rev-parse --short HEAD)"
    target="$tester"; paths=(spec CLAUDE.md)
    ;;
  from-tester)
    case "$branch" in spec-*) ;; *) die "dev must be on a spec-* branch (on '$branch')" ;; esac
    clean "$tester" spec || die "tester repo: commit the spec change there first"
    clean . spec || die "dev: uncommitted changes in spec/"
    for f in "${spec_files[@]}"; do cp "$tester/$f" "$f"; done
    from="spectest $(git -C "$tester" rev-parse --short HEAD)"
    target="$PWD"; paths=(spec)
    ;;
  *) die "$usage" ;;
esac

if git -C "$target" diff --quiet -- "${paths[@]}"; then
  echo "copy_spec: already in sync, nothing copied"
  exit 0
fi
git -C "$target" --no-pager diff --stat -- "${paths[@]}"
echo
echo "Review with:  git -C $target diff -- ${paths[*]}"
echo "Then commit:  git -C $target commit -m \"spec: <what> (from $from)\" -- ${paths[*]}"
