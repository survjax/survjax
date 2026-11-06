Independent review: round 1/3

# RELEASE.7 — Public repository cleanup

## Read first

1. PLAN.md RELEASE.7 and the maintainer-authorized cleanup scope.
2. Public guide and operator-handoff diffs; publication remains pending.
3. Archive renames, DOCS-BUILD link, removed simulation draft/config exclusion.
4. Bayesian notebook warning-only removal and fixture README status corrections.

## Invariants

No numerical/API/dependency/version/spec/protected-path changes. Archived
instructions remain verbatim historical records. Notebook sources and useful
results remain unchanged. Public-ready prose is not evidence of changed remote
visibility, hosting or package publication. Spec and active local orchestration
portability are explicitly deferred; author emails/history remain intact.

## Verification

- scripts/check.sh --fast --through A10: exit 0; all gates passed.
  Ruff: 181 files formatted, lint passed. mypy: 22 source files clean.
  Implementation float64: 1066 passed / 8 skipped; float32: 1071 / 3.
  Source doctests: 4 passed per mode. Independent spec float64: 2149 passed /
  27 skipped; float32: 2156 / 20. Slow cases deselected by the fast gate.
- Archive byte comparisons against HEAD passed for both moved files.
- Parsed notebook comparison passed: only cell 1 warning output removed;
  all cell sources, metadata and other outputs unchanged.
- Searches found no private-visibility or old-owner narration in active public
  guides, and no active reference to the removed simulation draft.
- git diff --check: passed. No new tests needed for prose/output cleanup.
- scripts/build_docs.sh directly: exit 0; both float32 and float64 builds
  succeeded with forced notebook execution. No hosting/public-access claim.
- python3 /tmp/survjax-public-cleanup-audit.py: exit 0; each dtype site has
  27 HTML pages and 2,069 valid local links/assets/fragments. Updated pages
  contain no private-visibility wording, simulation page is absent, Bayesian
  notebook download byte-matches the cleaned source. No browser-layout claim.

## Commit trace and remaining work

Previous step/start: dfa24b5. Original worktree base: 3cb4cd7.
Remaining: portable agent/spec setup, hosted checks and release acceptance and
publication. No independent completion claim for those workstreams.
