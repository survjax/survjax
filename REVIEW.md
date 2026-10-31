Independent review: round 2/3

# MNIST.6 — Evaluate CNN and references

Read first: PLAN.md MNIST.6; use-case 03 References and metrics/Precision;
examples/survival_mnist/model.py evaluate; tests/unit/check_mnist_metrics.py.

Invariants:
- Caller supplies already-selected CNN predictions or the fitted intercept/
  stored oracle parameters. No fitting, checkpoint selection or labels enter
  evaluate. All observations enter both metrics; no partial-batch dropping.
- Explicit float32 parameter casts, float32 data requirement. Scalar shared
  shape, scalar intercept or aligned per-row scales. Never change global JAX
  precision. Reporting is eager host code, not a transformed numerical API.
- Public survjax Weibull log_likelihood mean and concordance_index; higher
  risk is -log scale. Scalar intercept broadcasts to all rows and has C=0.5
  when pairs exist. Undefined/nonfinite metrics fail loudly. Invalid observation
  diagnostics raise before evaluation; unsupported modes still fail via library.
- C wall timing includes possible compilation and explicitly waits by converting
  the scalar result to a Python float. It measures the C call, not likelihood.
- No dependency, core API, protected file or spec edits. D-030 initialization
  decision remains in effect; this step does not reopen or alter it.

Tests and requirements:
- Exponential hand algebra pins mean LL and C direction (C=1 versus C=0).
- Mixed exact/right intercept algebra pins scalar broadcast and C=0.5.
- Independent generating rule pins oracle k=1.38 and per-digit scales.
- Real CNN forward output reaches metrics without digit labels.
- 129-row example pins full-tail observation weighting, not minibatch mean.
- Near-equal float64 predictions pin float32 casting before score comparison.
- Invalid shapes/observations, empty cohort, nonfinite predictions, no comparable
  pairs and float64 observations pin explicit failures in either dtype process.

Checks: results appended below. A first gate attempt stopped at Ruff C408
(dict constructor); replaced it with a literal and reran the complete gate.
No tolerances changed and no additional testing requirement inferred.

Synthetic timing (CPU, x64-enabled process but float32 evaluation): permutation
of 10,000 distinct exact event times, log scales=log times, hence C=1 by strict
ordering. C=1 observed both times; first-call 0.298767 seconds, warm 0.184139
seconds. This is a synthetic public-API timing probe, not held-out MNIST or
full-run performance evidence. Actual 10k MNIST timing remains for the runner.

No full MNIST training, test-set metrics, docs build or R generation run.
No new uncertainty/deviation. Runner, result artifacts and docs remain later.
Previous step final commit: 4a4fb99d9c39e64430c6fd29ef4ad35da5b8a419.
Worktree original base: ee72c837549d14cb26e9983bf6d63ac54563f189.

Completed checks:
- Optional six-file MNIST suites: x64 72 passed/1 skipped (44.65s), float32
  51 passed/22 skipped (25.60s). New metrics file alone: x64 14 passed.
- scripts/check_wheel.sh --through A10: all gates passed. Ruff format 134
  files already formatted; Ruff all checks passed; mypy no issues in 21 files.
  Own suites: x64 1119 passed/7 skipped (110.03s), float32 1125 passed/1
  skipped (78.82s). Doctests: 4 passed each mode. Acceptance: x64 2333
  passed/27 skipped (67.40s), float32 2324 passed/36 skipped (60.88s).
- git diff --check clean.


Round 1 findings and fix-up:
- F-1: add an x64-only test with a real float64 shape. Observe the dtype of
  the actual public log_likelihood result during evaluate; require float32.
  The wrapper forwards unchanged arguments/results and does not fake values.
  This directly detects accidental likelihood promotion from a missing cast.
- N-1: rename the constant CNN vector-output test to state what it demonstrates.
- N-2: module remains about 300 lines; runner/figure steps will use the named
  run.py rather than adding responsibilities to model.py.
- Reviewer could not run the cumulative gate due sandbox git/cache restrictions;
  their exact archive checks reproduced both optional suite counts and timing.
  Our complete gate result above remains the cumulative evidence.
- Fix-up checks: Ruff format 134 files already formatted; Ruff all checks
  passed; metrics x64 15 passed (5.84s), float32 13 passed/2 skipped (5.49s).
  No production code changed after the complete gate; no redundant gate rerun.
