# B2.1 — Untruncated censoring integration

Independent review: round 1/3

Read work/B2.1/PLAN.md, spec 2/3.3/4.2/7.1–7.3, the B2.1 implementation
commit 5c271b4 (parent edba58a), then current src/survjax/likelihood.py,
distributions/_log_math.py, and the corresponding unit tests.
Original step base c9fee45; preceding step 5fd6320; original worktree base
cc0e30f. Last B2.1 checkpoint 47aa112. Review resumption starts at 157a240.

B2.1 enabled untruncated left/interval likelihoods through the family interval
hook, retaining exact/right behavior, common dtype, trailing-axis broadcasting,
stored/current diagnostics and parameter isolation. Inactive sets used (0,inf]
and inactive densities used stopped reference_time. A generic helper fix delays
failure selection until endpoint cases resolve, avoiding inactive NaNs under
jax_debug_nans without relabelling precision loss as zero mass.

The maintainer authorized B2.2 while B2.1 awaited a spec-reference correction.
B2.2 now routes censored rows through the conditional hook and exact rows through
log f - log Z; its separate round-1 review returned merge at 9f1b264. Both
observation features are enabled on the present tree. The B2.1 integration tests
still exercise untruncated cases. This is the deferred first B2.1 review, not a
request to repeat B2.2's verdict. No numerical change accompanies this handoff.

Evidence already obtained on 157a240 (no source changes since):
- The formerly failing tiny-tail spec test passes separately in x64 and x32:
  one passed each. Main's independent reference fix uses stable arithmetic.
- scripts/check.sh --fast --through A12 exited 0 with both observation features.
  Formatting: 233 files; lint clean; mypy 25 source files clean.
  Implementation: x64 1308 passed/8 skipped; x32 1313 passed/3 skipped.
  Doctests: 4 passed in each dtype.
  Acceptance: x64 3800 passed/32 skipped; x32 3765 passed/63 skipped.
  Both acceptance runs emit the two existing PH NumPy-reference warnings.
- Earlier full touched-file run after the generic fix: 64 passed x64;
  63 passed/1 skipped x32 (core, censoring, defaults, debug_nans, PH).
  Full truncation acceptance later passed 139 x64; 132 passed/7 skipped x32.
- git diff --check is clean. The full non-fast gate was not rerun on the merged
  tree; no docs build for this code/review step. Timing metadata was resolved
  to main's recalibrated list, broadening the fast gate's selection.

Coverage: test_likelihood_intervals.py covers mixed analytical values (spec 2),
hook dispatch/safe inputs (4.2/7.2), jit/vmap and analytical parameter gradients
and Hessians plus check_grads (7.3). test_interval_defaults.py covers the generic
debug_nans regression; existing core/debug_nans tests cover diagnostics, dtype,
broadcasting, invalid-row isolation and exact/right regressions.
The x64 finite-difference step was reduced from 1e-4 to 1e-5 to reduce
approximation error; test tolerances were not relaxed.

Uncertainty: D-B2.1-001 documents the historical reference defect. Its fix is
integrated and tested, but its formal status remains for the maintainer.
No current acceptance failure is known. Remaining implementation work:
B3 numerical/reference matrix and B4 pruning/docs, after this review.
