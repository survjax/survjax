# B3.5 — PH/AFT mixed-observation derivative evidence

Independent review: round 1/3

Base / previous step: 40c022f78160f7dc1224e966bdf8e0c4fe5a2b95 (B3.4).
Original worktree base: cc0e30f. Cumulative gate: A12 with the existing two
observation features. This addresses gap 2 of [B3.3](../B3.3/REVIEW.md).

## Read first

1. [Plan](PLAN.md), spec 2, 4.4, 7.1–7.3 and 12.1.
2. tests/unit/test_wrapper_censoring_derivatives.py: parameter points and rows,
   independent survival/density formulas, reference derivatives, then tests.
3. tests/impl_tolerances.py: three explanatory lines, no changed values.
4. tests/slow_tests.txt: mechanically refreshed timing classification.

## Invariants and evidence

No library, public API, dependency, protected path, documentation or feature
flag changes. Both wrappers are tested over Exponential, Weibull, LogNormal and
LogLogistic. Each has two distinct joint baseline/eta parameter vectors;
eta is 0.25 or -0.375. Inputs are dyadic and exactly representable in each dtype.
All parameters are in their smooth interiors. This is ordinary-range evidence,
not a claim about PH extreme scaling, endpoint derivatives, invalid-parameter
isolation or arbitrary custom baselines.

The 21-row batch crosses all four observation kinds with absent, left, right
and two-sided truncation, adds partial overlaps at L/U, and includes three
whole-window sets. The latter cover left censoring above U, an interval around
a two-sided window, and right censoring at entry in a left-truncated window.

The independent oracle uses 80-digit mpmath survival/density definitions.
PH uses S0(t)**exp(eta) and exp(eta)*f0(t)*S0(t)**(exp(eta)-1).
AFT uses S0(t/exp(eta)) and f0(t/exp(eta))/exp(eta).
It then conditions the exact density or intersected censoring mass by
Z=S(L)-S(U). mp.diff differentiates these definitions with respect to every
baseline parameter and eta, including all cross terms. It never calls survjax,
JAX differentiation, or independent acceptance-test helpers for expectations.

- test_mixed_wrapper_likelihood_and_joint_derivatives covers section-2 values,
  section-4.4 wrapper laws, and section-7.3 Jacobians/Hessians for each row.
  Eager checks use the first vector; jit(vmap) checks both. Both Jacobian modes
  and all four forward/reverse Hessian compositions use ordinary ATOL/RTOL.
  Values run eagerly with debug_nans. Whole-window values and all derivatives
  must be exactly zero; data diagnostics must be clean and uninformative there.
- test_wrapper_censoring_order_two_finite_differences independently checks the
  compiled mean objective with check_grads(order=2, modes=(fwd,rev)), on the
  first vector for each wrapper/baseline. Existing finite-difference tolerance
  values and step sizes are unchanged; the second vector has mpmath evidence.

## Validation

- Focused Ruff format check: 2 files already formatted. Ruff lint: passed.
- JAX_ENABLE_X64=1 uv run --group test pytest -n 6
  tests/unit/test_wrapper_censoring_derivatives.py --durations=20:
  16 passed in 97.23s.
- Same command with JAX_ENABLE_X64=0: 16 passed in 111.62s.
- PYTEST_WORKERS=6
  SPEC_FEATURES=observation.censoring,observation.truncation
  scripts/update_slow_tests.sh --through A12: exit 0; 724 slow tests written.
  All 16 new tests are classified as slow. The mechanical refresh is kept in
  a separate commit from the 191-line test file and tolerance explanation.
  All four underlying pytest summaries were retained:
  float64 implementation 1494 passed/8 skipped (349.44s);
  float64 acceptance 3867 passed/32 skipped/2 warnings (350.63s);
  float32 implementation 1493 passed/9 skipped (390.22s);
  float32 acceptance 3813 passed/82 skipped/2 warnings (353.61s).
- scripts/check.sh --fast --through A12: exit 0, all gates passed. Guard clean;
  Ruff formatting (261 files), lint and mypy (25 source files) clean.
  Implementation: float64 1200 passed/8 skipped; float32 1201 passed/7 skipped.
  Source doctests: 4 passed per mode. Acceptance: float64 3442 passed/32 skipped;
  float32 3425 passed/45 skipped. Each acceptance mode reports the two existing
  PH reference-arithmetic warnings in test_a11_ph_direct_methods.py.
  All 16 new tests were run in full above, independently of this fast gate.
- git diff --check: clean. Timing refresh commit: d14c4fa (47 additions,
  142 deletions); changed fast-suite counts reflect measured classification.

No R regeneration, docs build or full nonfast check.sh invocation in this
test-only step. No spec ambiguity or numerical failure found. No tolerance
value was changed. The timing script tolerates pytest failures; its four
retained passing summaries above, rather than its exit status alone, supply
the full-suite test evidence.

## Remaining

B4 static-metadata pruning and measurements, family-feature activation decisions,
and likelihood documentation remain. No Slice B completion claim or shared
roadmap edit. Reviewer-workflow investigation remains deferred.
