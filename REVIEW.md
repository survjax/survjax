# PH.appendix — Clear scaling names, evaluated shapes and permanent mathematics

Independent review: round 2/3

The maintainer approves the PH.1 code except naming/shape-docstring nits,
requests a permanent appendix outside standard docs, and explicitly requests
the next PH implementation step after this one. Work stays on release-ph.

## Read first

1. PLAN.md's new PH.appendix section; the maintainer's accepted PH.1 outcome
   remains in Git at 68bd461 and merge 245219d's preserved records.
2. src/survjax/distributions/wrappers.py::_scale_log_survival: private naming,
   Args/Returns docstring and three stable appendix references only.
3. appendices/ph-log-survival.md and adjacent PDF; appendices/README.md and
   the small pagination filter. This follows release-rmst's appendix layout.
4. Existing PH test modules and spec 3.3, 4.4–4.5, 5, 7.2–7.3. Revision 38's
   staged RMST amendments do not change these PH requirements.

## Invariants and mathematical scope

- No public API/export change or numerical operation changes. Full module AST
  comparison is identical after reversing four private name changes and
  removing docstrings; comments do not enter that comparison.
- Log cumulative hazard is named explicitly as baseline versus scaled.
  baseline_log_cdf is a log CDF, not a probability or a batch-only array.
- The docstring describes evaluated shape E, batch shape B, scalar/infinity
  shape B, grid shape (K,*B), ordinary trailing broadcasting and no component
  axes. Internal callers supply already-broadcast arrays of the same E.
- The appendix contains mathematical definitions, real-arithmetic lower-tail
  value/first-derivative bounds, guard rationale and information-loss limits.
  It makes no uniform floating-point or tail-Hessian claim. Step/review/check
  history is kept here and in PLAN.md, not in the permanent appendix.
- Stable section references link code to lower-tail recovery, overflow and
  zero-limit tangents. User docs/navigation, spec, dependencies, RMST-owned
  files, exports, tolerances and acceptance tests are not edited by this step.
- README follows the RMST worktree's shared table convention but lists only
  the PH appendix available in this branch. Maintainer integration retains
  both rows. The adjacent PDF is rebuilt from source. The older temporary output artifacts
  have been moved to a private temporary backup after migration; only the
  canonical appendix source/PDF will be committed.

## Actual verification

- `uv run ruff format src/survjax/distributions/wrappers.py`: 1 reformatted.
  Ruff lint: All checks passed! Mypy: no issues found in 21 source files.
- Existing PH tests, both modules: JAX_ENABLE_X64=1 → 21 passed, 1 skipped
  in 9.17s; JAX_ENABLE_X64=0 → 20 passed, 2 skipped in 9.17s. Existing tests
  cover the unchanged scalar/grid/infinity shapes and gradient guards;
  no implementation-mirroring tests were added for renames/docstrings.
- Standard-library AST comparison of the full old/new wrapper module passed
  after reversing baseline_log_cdf, log_baseline_cumulative_hazard,
  log_scaled_cumulative_hazard and cumulative_hazard_magnitude and stripping
  docstrings. This verifies the stated non-numerical scope directly.
- Pandoc/XeLaTeX with the pagination filter and plain default Pandoc both
  exited 0 with --fail-if-warnings. Five-page A4 PDF rendered with Poppler;
  every page inspected for equations, table/code widths, clipping and page
  transitions. Final formatting check passes: 121 files already formatted; lint and mypy
  pass. Both final PDF builds also exited 0 after the code-fence correction;
  the updated page 3 was re-rendered for inspection.
- An initial whole-tree Ruff format check found the illustrative code fence
  unformatted. Shorten it to the exact exponential subexpression, then rerun
  the formatter and rebuild both PDF paths; no numerical source change.
- Appendix anchors and its relative code link were checked. No standard-docs
  build was run: this appendix is intentionally outside the docs site, and
  this clarity step does not change those pages or their executed cells.
- Fresh-wheel `scripts/check_wheel.sh --through A10` exited 0 and ended
  `check: all gates passed through A10`. Guard ok; Ruff 120 files already
  formatted / All checks passed!; mypy no issues in 21 source files.
  Own suites: 1069 passed, 8 skipped (66.20s) float64; 1074 passed, 3 skipped
  (59.46s) float32. Source doctests: 4 passed per mode (0.99s/0.95s).
  Independent through-A10: 2332 passed, 28 skipped (67.35s) float64;
  2322 passed, 38 skipped (61.96s) float32. Existing skips only.
  Wheel: /var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.m0Pb7W/survjax-0.0.1.dev0-py3-none-any.whl
  SHA-256: 865ca87ab7d7d634cc843da45f86f4a76d89dfc42bcdc2fb3626b9f3d0a2d41a.
  Appendix PDF SHA-256:
  927d28c95efca63cb13ac5af247c30c7916fa8509e0ee16edaa4a70d44577274.
- Final whitespace and seven-path scope audit clean. Source, appendix and PDF
  are unchanged after these checks; only plan/review result records follow.

## Existing main merge and commit/review precondition

A maintainer-started merge of 14155edfccd26a6ebfb984bad67bf6e3523dfe55 into
68bd461 was already in progress before this request, with conflicts only in
PLAN.md and REVIEW.md. Their merge-only resolutions preserve each parent's
full contents verbatim; every other incoming stage-0 entry was checked equal
to the initial index. Originals and the original index are saved at
/private/tmp/ph-appendix-merge. The maintainer subsequently answered “jok agreed”, explicitly authorizing
completion of that prepared merge. Commit 245219d19f8575a115936d8976c58a1036b7a94d
contains only the staged incoming integration and preserved records; the PH
changes remained unstaged for this separate clarity commit.

The merge is now committed and the code/appendix step can be committed and
reviewed separately. No new merge, rebase or publication is authorized by
that one integration approval. The independent-review skill will be invoked
from a clean committed state using the exact request command.

Next: commit/review PH.appendix as a new clarity
step, then proceed to the explicitly authorized next PH.2 identity/evidence
portion with its concrete <=300-line plan. Do not start RMST closure work or
export integration, and do not count this as a fourth PH.1 numerical review.

Temporary output backup (not a repository artifact): /private/tmp/ph-temporary-note-d5iw1e_1/output.


## Independent round 1 and focused follow-up

Round 1 on 5ff2768 returns fix-first. It confirms AST equivalence, the
appendix's real-arithmetic bounds and examples, input/output shapes and all
three code-to-appendix anchors. No mathematical or numerical-source finding.

- F-1: add finite-time PH cases just below and above log(max), rather than
  relying on base-supplied t=0/infinity values. Mathematical origin: at rate=1,
  t=1, log survival=-exp(eta). Below the threshold it is finite; above it IEEE
  overflow gives -inf, and survival gives zero. This catches deleting the
  restore and an over-eager overflow threshold in the lower case.
- N-1: explicitly state that only CDF entries at zero baseline log survival
  are used; other entries can be evaluated at the inactive reference time.

Reviewer checks: own suites 1069/8 and 1074/3 (passed/skipped); doctests 4
per mode; Ruff/mypy clean; float64 acceptance 2327/28 with 5 unsupplied-wheel
fixture errors, then no float32 acceptance run. It could not run the approved
PDF build or check_wheel.sh under its permissions. PDF text was extracted;
mathematics checked by hand. Its A2 mutant (delete overflow restore) survived,
while A1/A3 were caught. It reports refused temporary-worktree cleanup; I do
not edit reviewer-owned paths. No re-run of PH.1 or new spec decision is needed.

Focused fix-up validation: both PH modules, JAX_ENABLE_X64=1 → 23 passed,
1 skipped in 11.37s; JAX_ENABLE_X64=0 → 22 passed, 2 skipped in 11.37s.
Ruff format: 120 files already formatted; lint: All checks passed!; mypy:
no issues in 21 source files. Whitespace/scope clean. The initial full
fresh-wheel gate above is pre-fix-up evidence; not rerun for this small
assertion/docstring-only change. No PDF change is needed: the appendix already
describes the cases and relevant masks. Commit and request round 2/3.
