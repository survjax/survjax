Independent review: round 2/3

# SIM.4 — Single-replicate PH fitting

Step start: 95f453279ab9bffad1c67dc234a56d46b534a677 (clean release-simulation,
maintainer-integrated PH/RMST). Previous simulation step final:
afc266a684be990406ec6152b7eb035b6bb1aa26. Original worktree base:
8757c0fda78e34e6e818249adedd2a144a33c33d. No integration performed here.
Preserve all merged PLAN history; prepend this use-case continuation only.


## Round 1 response (2026-11-02)

Written round 1 on 1005ac7 returned fix-first; no spec questions.
- F-1 fixed: extract the existing post-solve policy into private _assess_fit,
  called only for estimable optimizer results. Preserve all acceptance tests,
  Cholesky substitutions, raw diagnostics and inference arithmetic. BFGS,
  starts and the public demo interface are unchanged.
- Nine hand-input acceptance cases run eagerly and jitted: positive control,
  unsuccessful solver, score 2e-5, score exactly 1e-5, NaN score, NaN theta,
  NaN objective, NaN Hessian, indefinite Hessian [[1,2],[2,1]]. Each failure
  independently exercises a gate; all raw diagnostic fields are retained;
  rejected SE/interval are NaN. Positive H=[[4,2],[2,2]] has inverse-gamma
  variance 1 and verifies the accepted inference path.
- N-2 fixed: check-file command now uses --locked like PLAN/REVIEW.
- N-1 remains a maintainer-owned decision about the protected ordinary gate;
  no protected scripts or dependency-discovery changes are made.

This is a separate review-fix sub-step/commit, about 120 changed source/check
lines (mostly moving the existing block), not a new simulation feature.
The original implementation had 308 changed implementation lines. The
cumulative implementation diff now has 374 lines after required guard
coverage; study.py is 306 lines, close to its ~300-line guideline. No new
module, dependency, public API or tolerance. This records the review-driven
scope increase explicitly; the next scenario work remains separate.

Fix-up checks:
- Optional fit command below: 15 passed in 18.63s (six original + nine cases).
- Generator checks below: 17 passed in 2.72s (x64) / 17 in 2.69s (x32).
- Ruff: 140 files already formatted / All checks passed!
- mypy: Success: no issues found in 22 source files.
- uv lock --check: Resolved 171 packages, exit 0. git diff --check: clean.
No full-gate repeat for this example-only refactor; the initial fresh-wheel
through-A10 evidence below remains the core regression evidence. All changed
example code is covered by the final 15 optional checks and generator checks.

Reviewer round 1 independently ran the six original fit checks (6 passed in
19.83s). Its first `--through SIM.4` gate was an invocation error, not a code
result; the corrected `--through A10` gate passed Ruff/mypy, both own suites
(1282/1287 passed, 8/3 skipped), doctests 4 per mode, then x64 spec tests gave
2328 passed / 27 skipped / 5 wheel setup errors (SURVJAX_WHEEL unset). It
stopped before x32 spec tests. Our fresh-wheel wrapper passed both modes as
recorded below. Reviewer-owned temporary-worktree cleanup remains untouched.

## Reading order

1. PLAN.md SIM.4 and use-case 01 details Models and fitting.
2. examples/weibull_simulation/study.py, from FitResult onward: real PH
   objective, exact prescribed BFGS setup, raw diagnostics and Wald inference.
3. examples/weibull_simulation/check_fit.py: explicit optional-environment
   verification, analytical identities and acceptance cases.
4. tests/impl_tolerances.py: new solver-only bound, derived before testing.
5. DEPENDENCIES.md: trivial SIM.3 N-2 fix, cite afc266a so the probe source
   remains findable when this review file is overwritten. SIM.3 N-1 is still
   a maintainer-owned protected-gate decision; no protected edit here.

About 300 changed implementation/test lines (308 including the standalone
check file and central solver bound); study.py is 292 lines. Core, protected
paths, dependencies and other agents' files unchanged. No docs step here.

## Invariants and specification mapping

- Use-case 01 / spec 4.4: both objectives use the integrated public
  ProportionalHazards(baseline, log_hazard_ratio=lp), with lp=gamma*arm on its
  own line. Weibull theta=(log k,log b0,gamma), Exponential=(log rate0,gamma).
  Return negative sum of log_likelihood(data). No substitute PH implementation.
- Validated nonempty (N,) exact/right untruncated data and matching binary arm
  are caller preconditions. Host data.issues.raise_if_invalid() precedes JIT
  in checks. No events in either arm is explicitly unestimable, not invalid
  data. lax.cond returns NaN diagnostics/intervals and false flags for that
  case; jit(vmap(fit)) preserves supported fits next to it. Under vmap both
  branch computations can execute, so skipping is semantic, not a claim of
  avoiding all optimizer work in a mixed compiled batch. No derivative-through-
  fit guarantee. Ordinary likelihood parameter derivatives remain supported.
- Exact BFGS rtol=atol=1e-8, max_steps=1000, throw=False. Start log k=0,
  log scale=log(mean observed time), gamma=0; Exponential starts negative log
  mean time. Only float64 fitting is supported, per this use case; other
  precision fails explicitly. No global JAX configuration changes.
- Acceptance is exactly solver success, finite theta/objective, max absolute
  gradient <1e-5, and successful Cholesky of the exact objective Hessian.
  Nonfinite Hessians are substituted before Cholesky; failed factors are
  substituted before the SE solve. Inactive interval operands are finite.
  No probability clipping, floors, epsilon or added acceptance criterion.
- H=L L^T implies [H^-1]_gamma,gamma=||L^-1 e_gamma||^2. SE is its square root;
  HR interval=exp(gamma+/-1.96 SE). Undefined inference is NaN; rejected fits
  keep raw diagnostics for later failure accounting. Result has solver,
  Hessian, estimability and acceptance flags; no arbitrary failure-priority
  rule is introduced. Scenario categories/counts remain the next step.
- Optimistix imports only when fitting is called. Shared generator imports
  still require only JAX/NumPy/core survjax. Existing generator signature,
  keys and calibration are unchanged.

## Test mapping and independent expectations

`check_fit.py` is deliberately outside default pytest filename discovery.
Run it explicitly with --group examples and JAX_ENABLE_X64=1; it raises a
clear error if run in the wrong dtype mode, without dependency skips. Ordinary
offline core tests do not acquire optional fitting dependencies.

- Both model objectives: independent exact/right PH formulas sum H-event*log h,
  checked eagerly under debug_nans, jit, vmap, forward/reverse gradients and
  Hessians. check_grads(order=2,modes=fwd/rev) on a mean objective uses unchanged
  central derivative tolerances/steps. Ordinary-range cases only, respecting
  spec 7.3's integrated extreme-range PH derivative limitation.
- Exponential fit: hand cohort has d0=d1=2, Y0=6,Y1=12. Unique MLE rates are
  1/3 and 1/6, gamma=log(.5). At any theta, mu_g=lambda_g*Y_g and
  H=[[mu0+mu1,mu1],[mu1,mu1]], giving SE_gamma=sqrt(1/mu0+1/mu1).
  Check fitted theta against the exact MLE with a solver-accuracy bound; check
  returned H, SE and Wald limits against algebra at the returned theta with
  ordinary float64 tolerances. Check the specified score threshold explicitly.
- Solver tolerance is separate from numerical formula accuracy: at the MLE
  H^-1=[[.5,-.5],[-.5,1]], so score infinity norm <1e-5 implies local theta error
  <=1.5e-5 to first order. Fixed 2e-5 allows the O(score**2) remainder. Added
  before running checks, not loosened after a failure. Existing value/derivative
  tolerances are unchanged.
- Both model fits on one replicate: n=100 fixed equal arms,
  k=1.38,b0=11.42,gamma=log(.7), calibrated
  .55 censoring; root key 20261025 -> scenario 1 -> split R=1 -> U/C split.
  Both jitted fits meet every acceptance condition and have positive SE. This is a
  one-replicate check, not full-run coverage or parameter-recovery evidence.
- A compiled mixed two-replicate Exponential batch retains the analytical MLE
  in its estimable member and reports NaN diagnostics/unestimable in the other.
  The entire batch passes structural data validation.

## Commands actually run

`JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q examples/weibull_simulation/check_fit.py`:
6 passed in 35.40s. Earlier five-test executions: 5 passed in 36.03s and
5 passed in 25.56s; the sixth case fits Exponential to the same seeded
Weibull cohort rather than only to the analytical Exponential cohort.
`JAX_ENABLE_X64=1 uv run pytest -q tests/unit/test_simulation_generator.py`:
17 passed in 5.42s.
`JAX_ENABLE_X64=0 uv run pytest -q tests/unit/test_simulation_generator.py`:
17 passed in 3.79s. No skips.
`uv run ruff format --check .`: 140 files already formatted.
`uv run ruff check .`: All checks passed!
`uv run mypy src`: Success: no issues found in 22 source files.
`uv lock --check`: Resolved 171 packages, exit 0.

Initial source lookup used an obsolete guessed Optimistix bfgs.py filename
and exited 2; inspected the installed minimise implementation instead. Initial
Ruff caught one assigned lambda in the check file; replaced it with a plain
function. No numerical check failed and no tolerance was loosened.

`scripts/check_wheel.sh --through A10`: exit 0,
final `check: all gates passed through A10`.
Log: /private/tmp/survjax-sim4-gate.log.
Own suites: 1282 passed / 8 skipped in 160.22s (x64); 1287 passed / 3 skipped
in 215.27s (x32). These include the integrated PH/RMST own suites; the six
optional demo checks are separate as above. Doctests: 4 passed per mode
(1.93s/1.67s). Independent through-A10: 2333 passed / 27 skipped in 157.31s
(x64), 2324 passed / 36 skipped in 125.86s (x32).
Wheel SHA-256: `e9aa003d19c07d4290041a300a45288b7b8a942982dae204ceab57b4f19ec46d`.
`git diff --check`: exit 0, no output.

## Limits and next sub-steps

Single-replicate fitting only. No scenario runner, mutually exclusive failure
classification, accepted-fit denominator summaries, smoke/full CLI, full study,
performance timing or executable simulation docs yet. No full study acceptance
claim. No fit differentiation or extreme-PH derivative claim. No docs/shared
navigation/config/protected edits. Prior reviewer-owned temporary worktrees
are untouched. Report final/previous/base commits and stop after review.
