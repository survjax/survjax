Independent review: round 1/3

# MNIST.5s — Replace the custom optimizer with stock BFGS

Maintainer explicitly rejected the custom RoundoffArmijo optimizer and directed
plain optx.BFGS for the intercept estimate. This corrective step removes both
optimizer subclasses; MNIST.5's three review rounds remain recorded separately.

Read first:
1. PLAN.md, MNIST.5s, and the maintainer decision in D-029.
2. examples/survival_mnist/model.py, fit_intercept.
3. tests/unit/check_mnist_reference.py, independent profile optimum check.

Invariants: stock optx.BFGS(rtol=1e-5, atol=1e-5), default line search and
AD, maximum 256 steps. Public likelihood, float64 process boundary, input
validation and explicit unsuccessful/nonfinite result errors remain. No
custom convergence logic, likelihood adjustment, new dependency or core API.
Maintainer additionally confirmed "Initialize the CNN too". train now requires
initial_intercept=(log shape, log scale) from the separate float64 training-only
fit. It casts to float32, validates finite shape (2,), initializes the shared
log shape and CNN final bias. CNN takes log scale directly. D-030 records this
authorized change from the documented zero/observed-mean initialization;
no spec edit is requested or made.

The previous coefficient-precision target was unnecessary for this estimate.
The replacement test independently derives the optimum mean NLL from the
50-digit mpmath profile-score root and compares fitted likelihood using the
unchanged float64 tolerance constants. Eight exact/mixed cohorts include
censoring at zero and both fixed random seeds. Existing transformation,
second derivative, precision, invalid-observation and solver-failure checks
remain. Tests specific to the deleted custom line search are removed.

Evidence:
- Before selecting 1e-5, stock BFGS at 1e-6 passed the small cohorts but hit
  max_steps on the actual smoke cohort; this failure was reported, not accepted.
- Stock BFGS at 1e-5 succeeds in seven iterations for both cached training
  cohorts: smoke log parameters [0.16221817, 1.42582022], full
  [0.17199414, 1.43252349]. No held-out metric or seed selection.
- Optional five-file checks at 1e-6: x64 55 passed, 1 skipped; float32
  35 passed, 21 skipped. Final reference checks at 1e-5 recorded below.
- Cumulative fresh-wheel gate and final reference results recorded below.

The training schedule test captures the supplied log scale, verifies actual
initial bias and nonzero log shape before the first Adam update, and checks
float32 dtype even with a float64 input. Invalid initialization checks cover
wrong shape, NaN and infinity. The CNN geometry/algebra tests now supply log
scale directly; their independent expected values remain unchanged.

Open records: D-029 and D-030 retain maintainer-owned open statuses, with the
explicit decisions above recorded. No unresolved implementation choice.

Final checks:
- Optional five-file suite after both corrections: x64 58 passed, 1 skipped;
  float32 38 passed, 21 skipped.
- Ruff format: 126 files already formatted; Ruff: All checks passed;
  mypy: Success, no issues found in 20 source files; git diff --check clean.
- scripts/check_wheel.sh --through A10: all gates passed. Implementation
  suites: x64 1087 passed/7 skipped, float32 1093 passed/1 skipped; source
  doctests 4 passed in each mode; acceptance suites: x64 2333 passed/27 skipped,
  float32 2324 passed/36 skipped. Gate started before the CNN initialization
  edit; the final optional suites above cover that edit (core was unchanged).
