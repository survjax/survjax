# B3.4 — Truncated-family derivative evidence

Independent review: round 1/3

Base / previous step: a28318fe8094c22fe7cdb68e547b7d969c5aa3bc (B3.3).
Original worktree base: cc0e30f. Cumulative gate: A12 with the existing two
observation features. This closes the specific brought-forward-family derivative
gap identified by [B3.3](../B3.3/REVIEW.md), not the whole Slice B gate.

## Read first

1. [Plan](PLAN.md), spec 2, 7.1–7.3 and 12.1.
2. tests/unit/test_likelihood_truncated_families.py: ROWS, WINDOWS, POINTS,
   then the mpmath reference, then the four test behaviors.
3. tests/impl_tolerances.py: rationale only, no tolerance values changed.
4. tests/slow_tests.txt: generated timing classification, separate from the
   functional test addition if a separate commit is needed.

## Invariants and evidence

The library, public API, dependencies, feature flags and protected paths are
unchanged. All twenty tests run in both dtype modes. Each family has two
distinct natural parameter vectors; LogLogistic includes shape below and above
one, and LogNormal includes positive and negative meanlog. Inputs are dyadic
and exactly representable in both modes. Only parameter derivatives are tested;
no guarantee is inferred for endpoint derivatives, nonsmooth or invalid cases.

The reference computes S(t) from the normal erfc definition or the log-logistic
survival power, and f(t) from the corresponding density formula. It computes
log(f/Z) for exact rows and log((S(max(a,L))-S(min(b,U)))/Z) for censored rows,
with Z=S(L)-S(U), using 80-digit mpmath throughout. mp.diff differentiates that
independent mathematics, never survjax values, JAX formulas or spec-test helpers.
The selected masses are moderate; this is not additional extreme-tail evidence.

Six rows per window exercise exact, right, left, interior interval and partial
overlap at each finite truncation boundary. Left-only, right-only and two-sided
windows all occur. The overlapping interval is naturally interior where that
window has no corresponding truncation boundary.

- test_truncated_mixed_values_match_independent_ratios: per-row section-2
  values, eager debug_nans and jit(vmap), two parameter vectors per window.
- test_truncated_row_jacobians_and_hessians_match_mpmath: each row separately
  (no sum can cancel row errors); both Jacobian modes and all four Hessian
  compositions. Eager checks use the first vector; jit(vmap) checks both.
  Ordinary ATOL/RTOL apply to values and these high-precision derivatives.
- test_truncated_interior_order_two_finite_differences: check_grads(order=2,
  modes=(fwd,rev)) of the compiled mean objective, on the first vector in each
  family/window. Existing finite-difference tolerances and eps are unchanged.
- test_whole_windows_have_exact_zero_parameter_derivatives: a left set above
  U, an interval containing a two-sided window, and right censoring at entry
  in a left-only window. Exact zero for values, Jacobians and all four Hessian
  compositions under jit(vmap), plus clean/uninformative data diagnostics.

## Validation

- uv run ruff format on the new test file and impl_tolerances.py: formatted.
  Initial Ruff check found a loop-closure binding warning in the reference;
  binding row inputs as defaults fixed it. Final focused Ruff check passed.
- JAX_ENABLE_X64=1 uv run --group test pytest -n 6
  tests/unit/test_likelihood_truncated_families.py --durations=20:
  20 passed in 40.51s. This run began before the closure-binding lint fix;
  the fix does not change the immediate per-row reference evaluation.
- Same command with JAX_ENABLE_X64=0 after the lint fix:
  20 passed in 40.77s. Longest tests were the per-row LogNormal derivatives,
  around 29 seconds under six-worker contention in each mode.
- Required scripts/update_slow_tests.sh --through A12 with PYTEST_WORKERS=6
  and SPEC_FEATURES=observation.censoring,observation.truncation: exit 0;
  819 slow tests written, including 15 of this step's 20 tests. The generated
  refresh is 168 additions/23 deletions (191 lines), kept in a separate commit
  from the 185-line test file and four-line tolerance explanation.
  Its completed float64 implementation subrun passed 1478 tests, with 8 skips
  (350.16s), confirming the final test file after the closure-binding lint fix.
  Float64 spec: 3867 passed, 32 skipped, 2 warnings (349.77s).
  Float32 implementation: 1477 passed, 9 skipped (434.96s).
  The last float32 spec summary was not retained before the script deleted its
  temporary logs; no complete test-pass claim is made from the script status.
- scripts/check.sh --fast --through A12: exit 0, all gates passed. Guard,
  Ruff formatting (258 files), lint and mypy (25 source files) clean.
  Implementation: float64 1173 passed/8 skipped; float32 1174 passed/7 skipped.
  Doctests: 4 passed per mode. Spec: float64 3360 passed/32 skipped;
  float32 3344 passed/44 skipped. The two existing PH reference-arithmetic
  warnings appear in each spec mode. The new timing list explains changed
  fast-suite counts relative to B3.3; no test was removed or assertion changed.
- git diff --check: clean.

No R regeneration, docs build or full nonfast check.sh gate in this test step.
The timing-refresh script tolerates pytest failures, so its exit code alone
will not be treated as passing-suite evidence.

## Remaining

Wrapper evidence (B3.3 gap 2), B4 pruning/measurements, family feature-activation
decisions and likelihood documentation remain. No shared roadmap edited.
Reviewer-workflow investigation remains deferred. No new spec ambiguity found.
