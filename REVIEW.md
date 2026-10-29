Independent review: round 1/3

# A10.d1 — Executable concordance guide

Authorized by “ok move on to next part of A10”. Branch A10.4, clean base
7c4f57b. One docs topic: docs/concordance.md and index/API links. No source,
API, dependencies, fixtures, protected paths or spec changes.

## Reading order

1. docs/concordance.md: synthetic five-pair example, score draws and failure
   diagnostics; score direction, support and reference evidence.
2. docs/index.md and docs/api.md: topic links and current capability wording.
3. PLAN.md A10.d1: scope, prerequisite through-A10 result, later work.

## Invariants and spec mapping

Spec 6.1: scores first, prepared data or raw time/event keywords. Spec 9.1–9.2:
observation axis last, no broadcasting it, leading cohorts never pooled;
higher scores predict earlier events; event/censor time ties comparable,
other time ties excluded, tied scores get half credit; failed cohorts have
zero counts, no-pair masks only for evaluated cohorts. Exact counts use
uint32 hi/lo words; returned ratios follow the observation float dtype.
No gradients or calibration promised. O(N²) work and O(N) pair workspace.
Spec 10.4/12.1: reference claims limited to committed synthetic pinned-R
checks, no R required for examples. Spec 10.5: examples execute freshly in
both dtype processes; no outputs committed. Spec 12.2: this is the guide,
not the playground, held-out example or final completion audit.

## Validation ownership

The maintainer explicitly requested: “only you or the reviewer run them not
both as it would be exactly the same no?” The implementer owns this step's
build and routine checks. Please review source and recorded evidence without
repeating those same checks. This preserves the maintainer's preference;
no changes to the required reviewer prompt or protected settings.

Prerequisite full fresh-wheel gate on 7c4f57b already passed through A10:
own suites 1048 passed, 7 skipped / 1054 passed, 1 skipped; doctests 4 each;
independent suites 2332 passed, 28 skipped / 2322 passed, 38 skipped.
No production/test changes for this step, so that gate is not repeated.

## Checks and rendered inspection

`scripts/build_docs.sh > /private/tmp/survjax-a10d1-docs.log 2>&1` exited 0,
both Sphinx builds succeeded, but notebook shutdowns logged macOS sysctl
PermissionError tracebacks inside the sandbox. Repeated unchanged using the
approved escalated build command, logging to
`/private/tmp/survjax-a10d1-docs-unsandboxed.log`: exit 0, both builds
succeeded; no PermissionError, execution error or Sphinx warning. Existing
IPKernelApp TCP-transport warnings still appear on stderr; they do not
originate from the new page or from Sphinx warning checks. Optional docs
dependencies were synced from the unchanged lockfile on the first run.

The guide's eight code cells execute freshly in each process (3.30s float32,
3.19s float64 on the final build). Outputs in both renders:
- Hand table: C = 0.700, counts (3,1,1,5), raw C = 0.7.
- Score draws: [0.7,0.3,0.5]; jit and vmap give the same array.
- Invalid draw: [0.7,NaN,0.5], invalid [False,True,False], zero failed counts.
- Valid delayed-entry cohort: NaN with unsupported True and invalid False.
- All censored: NaN with no_comparable_pairs True, no failure masks.
Assertions for counts, shapes, transformation agreement and masks all pass.

`uv run ruff format --check .`: final run 112 files already formatted
(initial run before docs artifacts: 105).
`uv run ruff check .`: All checks passed! `git diff --check`: clean.
No new pytest/mypy/full-wheel invocation: this docs-only step does not
change source or tests; the prerequisite gate is listed above.

Rendered evaluation: served docs/_build over loopback port 8764 using
Python's http.server. CUA browser host was unavailable, so used native
Brave controls in a new tab. Inspected html-x32/concordance.html and
html-x64/concordance.html, with screenshots and accessibility text.
Page title, sidebar and section navigation are present. The complete pair
table and rendered fraction are readable; count/mask/shape tables fit the
content column. Score-draw and failure outputs match the values above;
code blocks stay within their containers. Verified the API's worked-guide
backlink navigates to the guide and inspected the updated index wording,
prose link and toctree entry. No layout issue found at the inspected desktop
viewport; no mobile-layout claim. Generated HTML and execution outputs are
ignored and uncommitted.

Known gaps: the existing playground still says likelihoods are unimplemented
and has no concordance cells; its update is the next separate docs topic.
The held-out synthetic workflow and final evidence/install audit remain
unfinished. This guide describes current support, undefined/failed results,
exact-pair complexity and the scope of committed R evidence without claiming
these remaining deliverables or Slice A completion. No new deviation.

Final planned-file/whitespace audit, docs commit and independent review
follow. The maintainer's single-runner preference above applies to review.
