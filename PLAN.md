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

## Round-1 follow-up — F-1, F-2, N-1

Maintainer approved the proposed fixes and will merge main into this branch
later. B-1 remains deferred to that maintainer action; no merge/rebase here.
Keep this within DOCS-WORKFLOW.1, no new API, dependency or protected edit.

- F-1: move agent-specific browser, screenshot/review and sandbox escalation
  guidance from docs/index.md to root CONTRIBUTING_DOCS.md. Keep public build,
  identity, cache and ordinary viewing instructions; retain the public offline
  MathJax limitation. No numerical or runner changes.
- F-2: extend tests/tooling/test_build_docs.py with a pre-dirtied tracked file
  that changes again during x64, keeping porcelain status unchanged. Verify
  failed certification and detect a mutant with fingerprint checking removed.
- N-1: log the stub's dtype calls under ignored bin/; assert exactly 0 then 1
  on success and detect an x64-only-loop mutant.
- Validate focused tooling tests in both dtype processes, the two mutants in
  temporary fixtures (not the source runner), Ruff/mypy/bash/whitespace, direct
  two-dtype scripts/build_docs.sh and changed-topic rendered inspection.
  Record exact results in REVIEW.md, commit separately, request independent
  review round 2/3, then stop under the skill verdict. No unrelated numerical
  rerun or new feature step; B-1 may still require a maintainer verdict.
