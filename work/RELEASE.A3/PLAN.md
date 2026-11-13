# RELEASE.A3 — Prepare alpha 3

Step starting commit: cd383be231739ee8851d2dac540b00289128f622
Previous-step final / branch and worktree base: cd383be231739ee8851d2dac540b00289128f622
Branch: dev-release-a3. Feature: release preparation for 0.0.1a3.

Dependencies: merged families, AFT, Slice B and default feature-gate work.
Read [B7 audit](../B7.AUDIT/REVIEW.md),
[B8 docs](../B8.DOCS/REVIEW.md),
[feature gate](../META.FEATURE-GATE/REVIEW.md) and
[AFT integration](../AFT.MERGE.1/REVIEW.md).

## Scope

One release-metadata/documentation topic, under 300 changed lines. Change
pyproject.toml and uv.lock to 0.0.1a3; align CITATION.cff and CITING.md;
prepare an undated alpha-3 changelog entry; update docs/releasing.md candidate
commands and canonical publisher owner. Keep published alpha-2 history.
No invented release date: omit candidate citation date until publication is
assigned. No public API, numerical code, dependency or protected-path changes.
No tags, push, merge, hosted configuration or publication.

Spec coverage: 10.1–10.5 packaging/documentation and 12.1–12.2 verification;
this incremental alpha does not claim completed v0 scope.
Completed feature selection remains scripts/spec-features.txt: observation
censoring, truncation, traced metadata, LogNormal and LogLogistic, plus all
baseline tests including AFT. No feature-list changes.

## Validation

Use existing release-validator, wheel/install and full acceptance tests;
no tests added for metadata/prose changes. Run uv lock and locked sync;
build wheel from sdist with uv_build 0.11.19 into a temporary artifact directory,
validate against v0.0.1a3, hash artifacts, run full scripts/check.sh on that
exact wheel (both dtypes), then scripts/build_docs.sh directly. Inspect built
release/changelog/citation pages and package metadata/contents. Record exact
results and limitations in REVIEW.md; commit then independent-review skill.

## Docs to-do

Covered here: release notes, candidate commands, citation and publication
handoff. Operator retains final-commit CPU CI, hosted/tag documentation checks,
publication-date assignment, tagging, publishing and public install checks.
GPU validation is needed only for a new accelerator-support release claim.

## Authorized release completion — 2026-11-13

After approving the reviewed preparation, the maintainer instructed: "approved;
go as far as you can, stop only when you need my interaction". Continue this
release through final date assignment, reviewed integration, remote push,
final-commit CI, tagging, hosted docs and publication where access allows.
This explicitly supersedes the preparation-only scope above for these operator
actions; do not rewrite or move published history/tags or unrelated work.
Assign 2026-11-13 for alpha 3 in CITATION.cff and CHANGELOG.md, retain honest
pending-publication wording, clarify the release guide and citation prose.
Commit the dated metadata with this authorization record and request review
round 3 before integration. Runtime/dependencies/tests/features stay unchanged.

Validate fresh wheel/sdist and full exact-wheel gate plus both docs builds on
the final dated snapshot; reviewer runs the fast gate. Integrate by fast-forward
only after checking main has not advanced; retain both implementation and
reviewer worktrees. Push main, wait for its full CPU CI, then tag the exact
validated commit, verify tagged hosted documentation, publish the GitHub
prerelease, inspect workflow artifacts/hashes and public PyPI install. Preserve
existing account protections. Stop for authentication, approval gates or
unavailable permissions rather than changing credentials or weakening controls.
Keep operator execution evidence in work/RELEASE.A3/RELEASE-LOG.md; no shared
roadmap edit. Service/account and CI results must be observed, not assumed.
