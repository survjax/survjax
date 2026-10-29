# PH.1 — Construction, diagnostics and direct PH composition

Independent review: round 2/3

Branch: `release-ph`. Actual full base SHA:
`8757c0fda78e34e6e818249adedd2a144a33c33d` (requested snapshot 8757c0f).
Unfinished A9/A10 PLAN history is preserved. Reviewer checkout already exists
at `../review/code-release-ph`; no branch creation or integration performed.

## Read first, in order

1. `plans/release-agents/README.md`, `plans/release-agents/ph.md`, PH.1 at the
   top of PLAN.md, and spec 4.1–4.2, 4.4–4.7, 5–7, 8.7, 10.2, 12.1–12.2.
2. `src/survjax/distributions/wrappers.py` (153 lines): the complete new code.
3. `tests/unit/test_proportional_hazards.py` (290 lines): analytical evidence.
4. Existing registration/policy in `distributions/base.py` and
   `_survival_methods.py`, nested diagnostics in `_parameter_issues.py`, and
   the shared `likelihood.py` evaluator (all unchanged).

## Scope and invariants

- This is a defining-module implementation, not public-export integration.
  Both __init__.py files and all exact export/install/reference assertions
  remain unchanged. A later export step needs the maintainer's docs exception.
- The wrapper is a registered frozen dataclass with a nested `baseline` and
  real-constrained `log_hazard_ratio`. The baseline may be positional or keyword, by the maintainer decision on
  D-028; eta remains keyword-only. Construction stays pure and value-unchecked.
  Registration owns dtype normalization, nested broadcasting,
  PyTree reconstruction and indexing. Baseline leaves retain their dtypes.
- H=exp(eta)*H0, log h=log h0+eta and log f=log h+log S. The wrapper never
  constructs or delegates to a PH-equivalent built-in distribution. The seven
  grid methods and the exact/right likelihood evaluator are inherited.
- Scaling uses sign and log magnitude to avoid premature exp(eta) overflow.
  Zero/infinite baseline log survival is substituted before arithmetic, and
  an overflowing final magnitude is restored after a safe exponential.
  This prevents inactive overflow tangents from poisoning boundary gradients.
  Where baseline log survival underflows to zero, use the baseline's existing
  log-CDF lower tail as log H0 (equal to rounding precision in this regime),
  then scale in log space. This recovers the value and its eta/baseline
  derivatives where the baseline retains that tail information. A genuinely
  zero log-limit keeps the exp(eta)-scaled baseline tangent through a safe
  linear branch when the factor is representable.
  No floors, clipping, epsilon or new derivative rules are used.
- The baseline infinity hook is scaled, so an improper baseline can retain
  its cure limit. Reference times retain the promoted baseline/predictor dtype;
  no narrowing cast or fallback remains.
- RMST ownership is respected: no edits to base.py, exponential.py, weibull.py
  or tolerances. No exports, docs, dependencies, protected paths, AFT, generic
  summaries or sampling. Initial source/test total was 292 added lines;
  round-1 fix-up changes 191 source/test lines (171 additions, 20 removals),
  below its 300-line budget. Current files total 443 source/test lines.
- The PH-only initializer adapter calls the saved registered initializer with
  keywords, preserving its checks/normalization. Registration's shared rule
  stays untouched while PH accepts exactly one optional positional argument.

## Requirement-to-test evidence

- `test_broadcast_pytree_indexing_and_dtypes`: spec 3.3/4.1–4.2/4.4,
  (2,1) baseline × (2,2) predictor, preserved float32 nested dtype with
  mode-default predictor dtype, frozen leaves, tree round trip and slicing.
- `test_traced_diagnostics_include_predictor_and_prefixed_baseline`: spec
  4.7, pure traced construction and eager/jit/vmap masks, real predictor
  versus positive baseline violations, dotted keys and indexed diagnostics.
- `test_static_misuse_raises`: spec 4.1–4.2 / D-028 decision, keyword-only eta, registered
  nested type, nonreal predictor and incompatible construction shapes.
- `test_direct_and_grid_exponential_formulas` (seven cases): spec 4.4–4.5/5,
  all direct methods via shared grids and vmap of pointwise methods, values
  independently derived from hazards [1,4], at t=0, 1 and +inf; eager/jit/vmap.
- `test_likelihood_and_second_order_interior_derivatives`: spec 6–7/12.1,
  independently derived exact/right contributions, prepared/raw forms,
  eager/jit, debug_nans on a valid mixed batch and forward/reverse order-2
  finite-difference checks in the baseline rate and eta. Existing central
  derivative scales apply to this O(1) averaged likelihood, unchanged.
- `test_extreme_predictors_do_not_form_overflow_times_zero`: spec 7.1–7.2,
  representable H despite overflowing exp(eta), compared to the analytical
  log H in host float64 with zero absolute tolerance; zero/infinity survival
  and exact zero reverse-mode boundary Jacobians for eta of both signs.

## Initial-commit checks (965cbeb; retained history)

Environment: own worktree .venv; `uv sync --locked --offline --dev --group test`
exited 0 (155 packages resolved, 34 checked; no metadata changes).
Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, pytest 9.1.1, CPU device 0.

- Initial focused command below: 11 passed / 1 failed in each dtype
  (18.48s each). The extreme predictor test exposed NaN boundary reverse-mode
  derivatives from an inactive overflowing exponential. Fixed by substituting
  its input and restoring the overflow value separately; test unchanged.
- Final focused commands:
  `JAX_ENABLE_X64=1 uv run --group test pytest tests/unit/test_proportional_hazards.py -q`
  → 12 passed in 10.05s;
  same command with `JAX_ENABLE_X64=0` → 12 passed in 9.99s. No skips.
- `uv run ruff format --check .` → 113 files already formatted;
  `uv run ruff check .` → All checks passed!;
  `uv run mypy src` → Success: no issues found in 21 source files.
- `git diff --check` → exit 0, no output.
- `scripts/check_wheel.sh --through A10` → exit 0,
  `check: all gates passed through A10`. Unchanged guard passes on release-ph.
  Ruff/mypy repeat the clean outputs above. Fresh-wheel install expectations
  and all own suites: 1060 passed, 7 skipped (126.40s) float64; 1066 passed,
  1 skipped (174.25s) float32. Source doctests: 4 passed per mode (1.75s/1.61s).
  Independent through-A10: 2332 passed, 28 skipped (205.76s) float64;
  2322 passed, 38 skipped (179.15s) float32. Existing skips only.
  Wheel: `/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.XcrWfu/survjax-0.0.1.dev0-py3-none-any.whl`.
  SHA-256: `7f28b3791c7c6130dee9d7f2cf4bf11dfe8347d655fba0e862575cea4aec928f`.
- Final staged whitespace and four-path scope audit are clean. The first
  sandboxed git add could not create the shared worktree index.lock; rerunning
  the same staging command with the approved escalation succeeded.

## Limits and follow-up

PH.1 is not the complete first-release PH gate. Full exponential/Weibull
identity values and parameter derivatives, draw/observation/time broadcasting,
additional custom and saturated baselines, invalid-row first/second derivative
isolation, nonzero boundary derivatives and the extended tail matrix belong
to PH.2. These are not claimed verified by this initial test selection.
RMST/closure waits for the reviewed RMST integration. The default log-CDF
inherits the generic lower-tail range (spec 4.2); no new log-H extension hook
or built-in dispatch is introduced here. Public export and executable topic
integration remain separate. No R generation, demos or docs build were run.
Round 1 identified a constructor question (S-1 / D-028). The maintainer has
now chosen a positional baseline. This follow-up implements the decision and
records it in D-028 without editing the read-only spec or marking it resolved.


## Independent round 1 outcome — historical record before the decision

Reviewed implementation commit: `965cbeb0b61efafc0b6a981fa5fedeec5044d3dd`.
Report: `../review/slices/PH.1.md`, round 1. Final `VERDICT: maintainer`.
One review invocation, no retry. No fix-up implementation or second step begun.

- S-1 / D-028 (open): spec 4.2 keyword-only constructors conflict with the
  positional baseline in 4.4/11.3 examples. The implementation followed 4.2;
  the reviewer requires a maintainer decision before the next fix-up.
- F-1 (open, should-fix): a baseline log survival rounded to zero at positive
  time is returned unscaled, losing the exp(eta) factor in its baseline
  tangent. Review probes: float32 r=1e-30, t=1e-20, eta=40 gives d/dr=-1e-20
  instead of about -2.35e-3; float64 r=t=1e-200, eta=300 also fails. This
  refutes a general derivative-correctness claim; focused tests missed it.
- F-2 (open, should-fix): improper-baseline infinity scaling and NaN eta hook
  behavior need tests in the same step. Deferring these implemented behaviors
  to PH.2 was insufficient. Mutants removing either behavior survived.
- N-1 / N-2 (open nits): promoted reference dtype versus predictor downcast;
  an overflow-threshold regression near log(max). Record for the fix-up scope.

Reviewer checks: own suites 1060 passed/7 skipped and 1066 passed/1 skipped;
4 source doctests per dtype; Ruff/mypy clean. Its plain check.sh gate exits 1
because SURVJAX_WHEEL was not supplied: float64 independent suite 2327 passed,
28 skipped, 5 wheel-fixture setup errors, then stops before float32 acceptance.
The reviewer could not invoke check_wheel.sh under its permissions. This does
not supersede the successful fresh-wheel gate above; it is an independent
verification limit. Mutation checks caught 2/5 mutants in both modes.

The reviewer reports a leftover prunable temporary-worktree registration in
shared Git metadata after refused cleanup. I have not touched reviewer-owned
paths or pruned other worktrees. No merge, push, publication or spec edit.
This record and D-028 are a follow-up bookkeeping commit only; the reviewed
implementation remains unchanged and is not approved for merge.


## Round-1 follow-up for independent round 2

Maintainer reply: “allow for positional baseline in PH wrapper”. Resume PH.1
from b0b0f8b, retain release-ph ownership boundaries and public-export deferral.
I am using the independent-review skill again after the fix-up commit; this
is round 2/3 for PH.1, not a reset of the review count.

- S-1 / D-028: implement the maintainer's constructor decision. Positional and
  keyword baseline forms share registration; eta stays keyword-only. New
  `test_positional_baseline_matches_keyword_under_tracing` compares both forms
  eagerly/jit/vmap; `test_static_misuse_raises` checks extra positional eta,
  duplicate baseline, missing eta and existing malformed types/shapes. The
  obsolete positional-baseline rejection was changed for this explicit decision.
- F-1: recover a zero baseline log survival from an accurate baseline log-CDF
  lower tail, rather than returning an underflowed zero. The new test
  `test_lower_tail_recovers_value_and_first_derivatives` checks the reviewer's
  r/t/eta cases in both dtypes against hand-derived -exp(eta)*r*t and
  (-exp(eta)*t, -exp(eta)*r*t), with zero absolute tolerance; eager/jit values,
  jit forward/reverse first derivatives. No expectation uses production code.
- F-2: `CureBaseline` supplies log S0(t)=log(p)*(1-exp(-t)). The new infinity
  test checks S(inf)=p**exp(eta) and its p/eta derivatives, for p=.3 and p=1,
  including the true zero log-limit tangent. A separate test checks NaN eta
  in the direct hook and all seven infinity direct methods. Both reported
  infinity-hook mutants are now directly targeted.
- N-1: retain reference dtype by ordinary promotion; the new x64-only test
  keeps the baseline reference 1e60 with a float32 eta. Its float32-mode skip
  states why mixed float64 inputs cannot be tested there.
- N-2: the extreme-predictor test now includes log(max)+.25 in each dtype;
  exact zero boundary Jacobians pin the overflow guard's near-threshold band.

Checks of the follow-up (source/tests unchanged after these runs):

- `uv run ruff format --check .` → 113 files already formatted;
  `uv run ruff check .` → All checks passed!;
  `uv run mypy src` → Success: no issues found in 21 source files.
  Initial mypy check rejected the initializer's dataclass-derived Array-only
  annotation/signature. The saved initializer now has a Callable[..., None]
  annotation; the method assignment explicitly ignores the generated-signature
  mismatch, as registration already does for its own initializer replacement.
- `JAX_ENABLE_X64=1 uv run --group test pytest tests/unit/test_proportional_hazards.py -q`
  → 18 passed in 9.98s.
  Same command with `JAX_ENABLE_X64=0` → 17 passed, 1 skipped in 10.12s.
- Fresh-wheel `scripts/check_wheel.sh --through A10` → exit 0,
  `check: all gates passed through A10`; guard, Ruff and mypy pass.
  Own suites: 1066 passed, 7 skipped (70.01s) float64; 1071 passed, 2 skipped
  (73.11s) float32. Source doctests: 4 passed in each mode (1.35s/1.19s).
  Independent through-A10: 2332 passed, 28 skipped (95.92s) float64;
  2322 passed, 38 skipped (84.82s) float32. The one additional float32 skip
  versus round 1 is the new x64-only mixed-reference-dtype test.
  Wheel: `/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.YhuvWQ/survjax-0.0.1.dev0-py3-none-any.whl`.
  SHA-256: `b9735cf8f54a912fb3720ce0ca5611eed64f3e7f4a2e3be75547c9c2faf2709a`.
- `git diff --check` → exit 0, no output. Five-path scope audit only:
  wrappers.py, PH tests, PLAN.md, REVIEW.md and DEVIATIONS.md. RMST-owned
  files, exports, tolerances and all protected paths are unchanged.

No new spec question, dependency, protected-path edit or shared ownership
change is needed for this follow-up. Full PH.2 evidence remains separate.
