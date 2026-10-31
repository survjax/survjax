# PH.2c — High-precision tails and direct time-value policy

Independent review: round 1/3

Base: 796085ec9c000996d467609b94090235a2a4548f on release-ph; clean before
planning. That maintainer merge has parents c25c0d7 and 1953623 (agreement
updates), and is a prerequisite rather than a change made in this step. Authorized continuation: “ok on to next part”. Current spec revision
39 (RMST log-parameter tolerance clarification); no RMST core or ownership
handoff is in this checkout. This is test-only: 288 new lines, 34 cases.
PLAN/REVIEW are excluded from the updated implementation diff budget.

## Read first

1. PLAN.md PH.2c, plans/release-agents/ph.md item 2; spec 3.3, 4.4–4.5,
   5, 7.1–7.3 and 12.1–12.2.
2. tests/unit/test_proportional_hazards_tails.py: inputs, independent reference,
   value tests, eta AD, baseline AD, then time-policy tests.
3. Unchanged distributions/wrappers.py and appendices/ph-log-survival.md,
   especially recovery guards and the final information-loss limitations.
4. Existing built-in implementations and _survival_methods.py as background;
   no source code changes here and RMST-owned files remain untouched.

## Oracle, scales and invariants

Reference calculations use mpmath with 100 decimal digits and the actual
float32/float64 input values (host conversion of each already-rounded input).
No distribution/wrapper methods compute expected values. PH Exponential has
log(h)=log(r)+eta and H=exp(log(h))*t. PH Weibull has
log(h)=log(k/b)+(k-1)*log(t/b)+eta and H=exp(eta+k*log(t/b)).
From these: log S=-H, S=exp(-H), log f=log(h)-H. Reference log F uses
log(-expm1(-H)) below H=1 and log1p(-exp(-H)) otherwise; it never subtracts a
tiny survival probability from one. High precision alone would not fix the
latter cancellation at H=600. Host conversion to binary64 happens only after
the high-precision formula, and retains tiny normal expected tails.

The four input rows per family cover:

- Baseline H0 underflow with representable scaled H, using the accurate log CDF
  recovery path. The baseline's own log survival is asserted zero explicitly
  in the first-baseline-derivative test.
- exp(eta) overflow with finite H/log(h)/h/log(f), plus legitimately underflowed
  survival and log CDF. No cap, clipping, epsilon or altered time is introduced.
- exp(eta) underflow with large but finite H0 and representable scaled H.
- Late tails H approximately 600 (x64) / 60 (float32), with nonzero normal S
  and log F; zero absolute tolerance prevents accepting a rounded zero tail.

First and second eta derivatives of log S are exactly -H. The second-order
compositions include all four forward/reverse combinations, as unmapped eager
calls, vmap and jit(vmap). Eta is passed as a distinct constructor argument,
with the baseline parameters independent of it. This tests the eta direction
without traversing extreme natural baseline Hessian intermediates. Baseline
first derivatives on recovery rows are [-H/r, -H] for Exponential and
[-H*log(t/b), H*k/b, -H] for Weibull. These expected derivatives are evaluated
at 100 digits before host conversion. Log H is roughly in [-421,30]; existing
TAIL_ATOL=0 and TAIL_RTOL=1e-8/1e-4 cover log reconstruction/AD rounding, not
finite-difference errors. No tolerance is changed or added. Every allclose
sets equal_nan=False. Only the explicit NaN-time output is checked as NaN.

Time-policy expected values come from spec 4.5 and analytical boundaries:
negative times precede the origin; -0 and +0 share the zero limit; proper
survival at infinity is zero; Exponential hazard is constant; the selected
Weibull k=1.4 has hazard 0 at zero and infinity at +inf; density at +inf is
zero. A positive t=2 entry is checked against the independent reference.
Finite policy values are constant in all baseline parameters and eta, so
both forward/reverse parameter Jacobians must be exactly zero.

## Requirement-to-test map

- test_tail_values_match_high_precision_ph_formulas: 14 cases (7 methods x
  2 families), all four tail rows, scalar eager/jit and vmap/jit(vmap).
  Spec 4.4/7.1–7.2: log-space PH, accurate two-tail probabilities, finite
  log density/hazard even when survival underflows, no premature eta overflow.
- test_tail_eta_first_and_second_derivatives_equal_log_survival: 2 cases,
  6 AD orders, four tail rows, eager/vmap/jit(vmap). Spec 7.3 finite smooth
  eta direction to second order. The transform-order comment also follows
  PH.2b review N-2's suggested convention (that older file is unchanged).
- test_lower_tail_natural_parameter_gradients_match_analytic_chain_rule:
  2 cases, baseline and eta first derivatives, forward/reverse eager/jit,
  with a baseline rounded-zero precondition. Spec 4.4/7.2–7.3.
- test_direct_and_grid_methods_follow_complete_time_value_policy: 14 cases,
  all seven methods, scalar-batch paired vector/direct and time-first grid,
  eager/jit and vmap; exact NaN mask separately from value comparisons.
  Spec 4.5/5; earlier PH.2a covers the other Weibull shape boundary regimes.
- test_finite_policy_values_have_zero_parameter_gradients: 2 cases, first
  Jacobians eager/jit in both modes at finite constant policy outputs.
  Spec 4.5/7.2 safe inactive operands, including proper +inf survival/log F.

## Validation and draft changes

Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, mpmath 1.4.1.

- Initial formatted draft was 315 lines. Split off the extra 28-line mixed
  Weibull likelihood/debug_nans check before finalizing the step, leaving
  288 final implementation lines. No assertion was removed to address a
  failed result; the deferred check was not reached in the initial -x run.
- Initial x64 focused run: 14 passed, 1 failed in 7.44s. Its eta test rebuilt
  the full parameter vector using p.at[-1].set(eta); forward-over-forward
  on the lower-tail Exponential row returned NaN. All other orders returned
  the expected -1.403592e-183. A standalone reproduction compared that
  construction with independent baseline/eta arguments: all six AD orders
  pass with the independent arguments. The final test passes eta directly
  to the wrapper, without rebuilding/routing tangents through the baseline.
  No production expression or tolerance was changed to obtain the pass.
- `JAX_ENABLE_X64=1 uv run pytest -q tests/unit/test_proportional_hazards_tails.py -x`:
  final **34 passed in 16.32s**.
- `JAX_ENABLE_X64=0 uv run pytest -q tests/unit/test_proportional_hazards_tails.py -x`:
  final **34 passed in 17.13s**.
- `uv run ruff format --check .`: 126 files already formatted.
- `uv run ruff check .`: All checks passed.
- `uv run mypy src`: Success, no issues found in 21 source files.
- `git diff --check`: clean.
- `scripts/check_wheel.sh --through A10`: **exit 0, all gates passed**.
  Guard, Ruff and mypy passed. Own tests: **1210 passed, 8 skipped** x64
  (114.86s), **1215 passed, 3 skipped** float32 (109.95s). Source doctests:
  **4 passed** each mode (0.97s/0.91s). Independent acceptance: **2332 passed,
  28 skipped** x64 (67.88s), **2322 passed, 38 skipped** float32 (108.57s).
- Wheel: `/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.yg9x7K/survjax-0.0.1.dev0-py3-none-any.whl`.
  SHA256 `3af2391ef81dacf1dfadeb997fbdc9d22d53a39c3091657d0e910eeb0451ee3c`;
  byte-identical to PH.2b, consistent with no source changes.
  Full gate log: `/private/tmp/ph2c-wheel-gate.log`.

## Limits and uncertainties

The independent-eta second-order tests do not claim arbitrary extreme
natural-parameter or rebuilt-vector tail Hessians. The initial rebuilt-vector
failure above remains an observed limitation; the final tests exercise the
explicit eta argument. The existing appendix likewise makes no arbitrary
tail-Hessian claim. Baseline underflow/infinite log survival with no accurate
CDF cannot be repaired by the helper; no unavailable magnitude is invented.
No new spec interpretation, deviation resolution or public API is proposed.

PH.2b N-1 is clarified in the new plan section: its committed derivative loop
uses vmap/jit(vmap), with unmapped eager evidence supplied by its reviewer.
Its older plan/test history is preserved. No N-2 change to the old file.

RMST closure, ownership handoff, public exports/API-reference agreement,
executable PH documentation and final release evidence remain separate.
The deferred extra Weibull mixed likelihood/debug_nans coverage is named in
PLAN.md. Existing Exponential PH debug_nans and built-in mixed likelihood
checks remain in the regression gate. No docs/PDF build or R fixture
regeneration is needed by this test-only step. No protected-path, dependency,
tolerance, source, RMST-owned or export changes; no merge/push/publish.
