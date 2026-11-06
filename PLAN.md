# WHEEL.1 — Compare installed version with project metadata

Start / previous step / original worktree base: 8794005 (release commit).
Plan archive: 69a66df. Files: tests/install/test_wheel.py, PLAN.md, REVIEW.md.
No public API, dependencies, docs or protected paths touched.
Spec 10.2–10.3: wheel metadata and installation verification.
Replace stale development-version literal with the project version read using
stdlib tomllib in the host test, passed to the isolated installed interpreter.
Retain an exact equality check with actual/expected diagnostics. Existing
installed-wheel test covers the regression; run the complete touched file in
both dtype modes, then scripts/check.sh --fast --through A10.
Write REVIEW.md, commit, independent review (at most three rounds), then stop.
Remaining substeps: none after checks and review.

Gate stopped on D-048: protected A9 test still requires the initial dev version.
Wheel tests pass in both modes; implementation uncommitted and independent
review pending maintainer decision under the stop-on-deviation rule.

## WHEEL.1 scope update — Prepare alpha 2

Maintainer requests coordinated release updates and proposes alpha 2.
GitHub v0.0.1a1 is already published as a prerelease; publish job skipped
because the build gate failed. Prepare 0.0.1a2 without moving the old tag.
Additional code-step files: pyproject.toml, uv.lock. No dependency changes.
Protected tests/spec/test_a9_package.py edit requires explicit pending approval;
propose metadata equality with pyproject.toml, matching the installed smoke.
This resolves the initial-version regression for future releases too.
Validate touched suites both dtypes, locked metadata, wheel/sdist validation
against v0.0.1a2, then fast A10 gate and independent review.
Subsequent docs step: release guide commands, changelog and citation status;
no invented publication date or completed hosting/acceptance claims.
Tag creation must follow reviewed commit integration by maintainer; this
checkout's agreement forbids agent tag mutations or publishing.

Maintainer explicitly approved the protected version check update in this
conversation. No other protected file changes. The branch guard cannot pass
on patch-wheel-test because it allows protected tests only on spectests-*;
run it honestly, then run the remaining gate components directly if blocked.

# RELEASE.A2 — Authorized release completion

Start/previous step: 61b7dcc; original worktree base: 8794005.
Maintainer explicitly authorizes guard exception, continuing past review stop,
merging to main, tagging/pushing v0.0.1a2 and GitHub prerelease publication.
Files: scripts/check.sh (narrow branch exception for itself and approved A9
version test), docs/releasing.md, CITING.md, CITATION.cff, CHANGELOG.md,
RELEASE-A2-LOG.md, PLAN.md, REVIEW.md. No dependency/API/spec changes.
Combine bounded release metadata/docs under this explicit release override.
Run full A10 gate on exact built artifact, direct docs build, inspect changed
rendered topic, commit, independent review. Stop on check failures. Verify
main worktree clean before fast-forward integration; never replace old tags.
Record commands, results, commits, hosted CI and publishing/public-install
outcomes. Use actual release date only at publication, no invented DOI.

## RELEASE.A2 publication record follow-up

Maintainer approved v* tag environment rule and failed-publish retry. Publish
succeeded; public smoke passes. Update root execution log, changelog release
date, citation date, citation prose and release-guide status; no code or tag
changes. Direct docs build, whitespace and metadata checks, round-2 review,
then integrate/push this record (published tag remains fixed at 56311e3).
