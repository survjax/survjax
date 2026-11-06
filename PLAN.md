# Merge resolution - Main wheel-test patch

Resolve only PLAN.md and REVIEW.md; leave other staged merge entries intact.
Preserve the incoming patch plan below and the docs-workflow plan unchanged.
Incoming historical plans remain verbatim in plans/incoming-main-docs-workflow.md.
No code, public API, dependency or spec change is authored in this resolution.
Stage resolutions for the maintainer; do not create the merge commit.

# KWPATCH.1 — Align installed-wheel PH calls with keyword-only constructors

Start / previous step / worktree base: d9832baaa19559b7d49eb0e52279695c27efde34.
Files: tests/install/test_wheel.py, PLAN.md, REVIEW.md.
Public API: unchanged. Spec 4.4 requires baseline= for ProportionalHazards;
spec 10.3 requires the installed-wheel smoke test to exercise usable exports.
Change the two stale positional baseline calls in the existing smoke test.
No new tests: run the entire touched install file in both dtype modes,
then scripts/check.sh --fast --through A12 and independent review.
No dependency, docs, protected-path or spec edits. This is one bounded
maintenance patch; inherited merged plan records remain below unchanged.
Remaining substeps: none after validation and review.

---

---

# DOCS-WORKFLOW.1 — Main merge resolution

Maintainer started a merge and requests conflict resolution. HEAD d89f2ba;
incoming main d9832ba; original worktree/previous-step base e6b3f9e.
Resolve/stage only; the maintainer owns the merge commit. Review round 2
confirmed F-1/F-2/N-1 fixed, B-1 required this integration. Round 3 must review
the committed merge; independent review cannot check uncommitted resolutions.

Files and intent, before implementation (spec 10.5; no public API/dependencies):
- scripts/build_docs.sh: preserve incoming --output-dir, worktree environment,
  cache and kernel isolation; put identity and Jupyter runtime in output_root.
  Keep this branch's two-dtype certification/traps/source-change detection.
- docs/index.md: incoming reorganized page. Move retained public cache/identity/
  viewing guidance into docs/technical-notes/verification.md's build section.
  DOCS-BUILD.md and CONTRIBUTING_DOCS.md: align root/runtime descriptions and
  contributor links; CONTRIBUTING.md links contributor inspection instructions.
  docs/contributing.md excludes that agent-only section from its public include.
- tests/tooling/test_build_docs.py: teach stub about incoming path resolver and
  .cache layout. tests/unit/test_docs_build_script.py: temporary real Git repo
  for identity checks; assert selected-root identity/runtime. Existing dtype/
  output assertions cover round-2 N-2, no new runner mode or weakened test.
- PLAN.md: keep current topic below, preserve incoming plan verbatim in
  plans/incoming-main-docs-workflow.md; combine plans/README.md links.
  REVIEW.md: current merge evidence then both parent reviews verbatim.

Validate both focused suites in both dtypes, Ruff/mypy/bash/whitespace and
strict direct scripts/build_docs.sh. Check no unresolved entries/markers and
all other stage-0 entries remain byte-identical to the initial merge index.
No unrelated numerical reruns, protected paths or main-checkout changes.
Keep integration changes near 300 authored implementation lines. No commit,
merge, rebase or review invocation until the maintainer completes the merge.

Resolution complete and staged, no merge commit by agent. Thirteen focused tests
pass per dtype; static checks pass. Direct both-dtype docs build passes using
the approved escalation after a reproduced psutil restriction and corrected
include-boundary/link errors. REVIEW.md records exact evidence and incoming
whitespace limitations. Round 3 awaits the maintainer's merge commit; visual
inspection awaits working native/browser tools. No next implementation step.

---

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
