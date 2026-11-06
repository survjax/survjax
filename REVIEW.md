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

Round 1: VERDICT maintainer (../review/slices/RELEASE.A2.md). Reviewer repeats
B-1/S-1 protected-policy concerns and cannot see the conversation. Maintainer
explicitly answered "Authorize all these release actions" to the question
listing protected guard repair, continuing past review stop, merge, tag/push
and prerelease publication. Proceed under that authorization; no new technical
blocker identified. Reviewer could not run tests due to sandbox uv access.
N-1 double blank, N-2 candidate wording, N-3 root execution log are editorial;
candidate wording is accurate until publication, and maintainer requested the
execution log. D-048 remains formally open; no spec edit made.

Independent review: round 2/3 (publication record follow-up)

Read PLAN.md publication follow-up and RELEASE-A2-LOG.md items 17 onward,
then citation/changelog/release guide status diffs. Maintainer explicitly
approved v* environment tag rule and retry; publish job now succeeds.
Public JSON artifact hashes match tested artifacts. Public isolated smoke
passes both dtype modes with Python 3.12.9/JAX 0.12.0.1/NumPy 2.5.4; no broad
compatibility claim. Only release record/prose/date changes since main.
No protected/code/API/dependency edits in this follow-up, no tag movement.
Direct both-dtype docs build is the validation for these status edits.

Follow-up validation: direct escalated scripts/build_docs.sh passed both strict
sites with success identity. Both rendered release pages contain publication
status. git diff --check passed. No repeat numerical tests for prose/date/log.
