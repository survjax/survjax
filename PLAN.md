# DOCS-WORKFLOW.1 — Reliable local build and inspection (2026-11-06)

Start/previous step/worktree base: e6b3f9e on maintainer-created docs-workflow.
Prior plan records are preserved verbatim in plans/pre-docs-workflow.md.

One tooling/docs topic, spec 10.1/10.5; no public API, dependencies, protected
paths or spec changes. About 300 implementation/test/docs lines maximum.

- Change scripts/build_docs.sh: ignored absolute build-local Jupyter runtime,
  IPython and Matplotlib directories; preserve MNIST cache/explicit override.
  Invalidate one whole-run identity before either build, publish success only
  after both strict fresh builds, record checkout/commit/dirty status/times.
  Detect tracked edits during execution and fail rather than certify them.
- Extend docs/index.md's existing local-workflow topic with identity checks,
  actual local HTML/screenshot inspection, native Chrome fallback and narrowly
  scoped escalation for reproduced process-inspection failures.
- Add tests/tooling/test_build_docs.py using a temporary checkout and stub uv:
  verify successful identity/environment and stale-marker invalidation on
  failure/interruption, without running numerical suites.
- Validate direct scripts/build_docs.sh, bash syntax, focused tooling tests,
  Ruff, mypy, whitespace and native rendered inspection if available. No
  unrelated numerical gate rerun for this tooling step (maintainer instruction).
- Write REVIEW.md with exact results, commit, invoke independent-review for
  DOCS-WORKFLOW.1, fix required findings within three rounds, then stop.

Remaining candidates, each needing its own decision/step: single-dtype authoring
preview, offline MathJax, docs CI/artifacts, browser-host repair. No caching,
global config/permissions or main-checkout changes are authorized here.

Implementation committed as df167c6. Both docs builds and focused checks pass;
REVIEW.md records exact evidence. Independent review invocation and its one
permitted no-round retry both hit the reviewer five-hour limit; no round file
or verdict exists. Stop with independent review pending; do not start another
implementation step. Metadata-only handoff records this external limitation.
