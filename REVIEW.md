# RELEASE.A2 — Authorized alpha-2 completion

Independent review: round 1/3

Read PLAN.md RELEASE.A2, RELEASE-A2-LOG.md, scripts/check.sh diff, then
release docs/citation/changelog diff. Start 61b7dcc, base 8794005.
Maintainer explicitly authorized ALL listed release actions in conversation:
protected guard exception, continuing beyond WHEEL.1 maintainer verdict,
docs completion, main merge, new tag/push and GitHub prerelease publication.
This supersedes previous stop, no-merge/tag and protected-path restrictions
for this release. Do not infer permission from the reviewer role alone.

Guard allows exactly test_a9_package.py and check.sh on patch-wheel-test;
all other protected paths remain blocked. Main guard has an empty main...HEAD
diff after integration. No spec, API, dependencies or numerical code changed.
Version/test fixes previously committed as 2e4971d; D-048 remains formally open
for maintainer resolution, with explicit test-change approval recorded.
Citation version identifies candidate source; no release date/DOI invented.
Old tag preserved; release docs name a2; historical example provenance kept.

Full scripts/check.sh --through A10 PASSED: Ruff 193 formatted, lint clean;
mypy 22 source files clean. Implementation x64 1312 passed/8 skipped;
x32 1317 passed/3 skipped. Doctests 4 per mode. Spec x64 2333 passed/27 skipped;
x32 2324 passed/36 skipped. Gate used its built wheel. Prior explicit a2
wheel-from-sdist validation passes. bash -n and git diff --check pass.
Direct docs build: first certification rejected changed checkout; stable
sandbox attempt failed sysctl/psutil restrictions; escalated direct build
passed both strict sites, success identity. Rendered release HTML checked in
both modes for a2 tag and absence of a1. No visual-layout change.

Remaining execution: final integrated artifact build/validation, clean main
fast-forward, hosted CI, tag and prerelease, inspect publishing results and
public isolated install. Record failures honestly and do not publish on failed
checks. This review is for prepared files, not a claim these actions occurred.
