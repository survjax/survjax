Independent review: round 1/3

# SIM.5 — Compiled scenarios and independent failure columns

Start/previous simulation final: d3ceb0fadb5052564079a46ac8e799bf0a7fbbc3.
Integration base: 95f453279ab9bffad1c67dc234a56d46b534a677.
Original worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Branch release-simulation; initial changes were the preceding planning-only
PLAN.md/D-033 records. Current user settles D-033 and explicitly authorizes
this step AND the next performance/MCSE step. Review SIM.5 before proceeding;
no new CLI, full run or docs in this step. Preserve merged plan history.

## Reading order

1. DEVIATIONS.md D-033 decision and PLAN.md SIM.5.
2. examples/weibull_simulation/run.py: host preparation/validation, compiled
   batch fitter, independent reason columns and unique-total host counts.
3. examples/weibull_simulation/check_scenario.py: seed law, fitting equivalence,
   overlap totals and a real optimizer-failure status regression (SIM.4 N-3).
4. Existing reviewed study.py is untouched (306 lines); do not grow it here.

## Invariants and requirements

- Use-case 01 DGP/seeds: fixed equal allocation, k=1.38,b0=11.42, HR=.7;
  root key 20261025 -> fold_in(scenario_index) -> split(R) -> U/C split inside
  the shared generator. Caller supplies c calibrated once per target; no
  recalibration per replicate or sample size. Scenario order remains caller's
  responsibility (full n ascending then target, smoke's own order).
- Float64 required, positive even n and R, nonnegative index, positive finite c.
  Preparation is host orchestration with vmap generation. The resulting (R,N)
  SurvivalData is explicitly host-validated before any fit.
- Fitting is one module-level jit(vmap(fit_replicate)) with static model and
  shared (N,) arm, returning the unchanged reviewed FitResult batches. No
  optimizer setting, start, acceptance condition or numerical formula changes.
- D-033: every applicable failed gate is recorded, with overlapping reason
  counts. Overall failed=~accepted is authoritative: accepted+failed=R.
  Solver/nonfinite/gradient/hessian are applicable only to estimable attempts;
  unestimable placeholder NaNs are not evidence of attempted optimizer failure.
  Attempted counts requested replicate assessments, including unestimable ones,
  not physical optimizer execution (which can occur in both vmap branches).
- nonfinite checks theta/objective as in the reviewed acceptance policy;
  gradient includes nonfinite score and failure of strict max|grad|<1e-5;
  hessian uses the reviewed Cholesky flag. No reason priority/single label.
- failure_counts performs explicit host scalar extraction, not a JIT API.
  failure_columns is pure JAX. No tracer-to-NumPy conversion in compiled code.
- No core/API, generator-contract, dependency, protected scripts, docs or other
  workstream edits. run.py is named by the use-case layout; currently it is an
  importable engine, not the later --config/--out CLI. Timings and high-level
  orchestration move to SIM.6 with performance summaries to keep steps small.

## Independent checks and investigations

Optional check command:
`JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q examples/weibull_simulation/check_scenario.py`
Final scenario suite: 10 passed in 11.26s (adds exact-threshold and NaN-score
cases to the earlier 8 passed in 9.58s). Source uses the same explicit dtype/environment convention
as check_fit.py; no optional dependency skips or protected gate change.

- Primitive generation expectation is independent NumPy inverse-CDF algebra
  with JAX primitive uniform draws on the exact prescribed subkeys. Checks
  both observed times and event flags, arm allocation and output shape/dtype.
- Both compiled model batches compare theta, objective, H, SE and intervals
  against reviewed single-cohort fits at unchanged ordinary tolerances.
  Scores are checked against exact PH score algebra at their OWN fitted points,
  and both must meet the prescribed score threshold.
- Initial attempted comparison of gradients at two different converged points
  failed for Weibull: residuals 5.32424e-8 vs 7.10061e-8 (difference 1.77637e-8).
  Both are well below 1e-5. Theta/objective and other fields agree. No tolerance
  was loosened: instead check the score against its independent analytical
  formula at each actual theta, with ordinary atol=1e-10/rtol=1e-8. Fixed-input
  forward/reverse/Hessian transformations remain covered by check_fit.py.
- Hand failure record: one accepted, one rejected with solver/nonfinite/
  gradient/hessian failures, and one unestimable. Reason counts are 1 each
  (sum 5), unique failed=2, accepted=1, attempted=3. The unestimable placeholders
  do not inflate the other four reason counts. Jitted masks agree exactly.
- SIM.4 N-3: four exact tied times 1e-50, two per arm, have an unbounded Weibull
  shape likelihood. In the locked environment a real jitted single fit reports
  solver failure; stack that real result and require solver and failed counts
  of 1. No BFGS setting is changed or mocked. This is failure/status plumbing,
  not PH extreme-range derivative accuracy or a claim of a finite MLE.
  Initial use of the same cohort with vmap yielded solver_success=True instead;
  both runs reject the fit. Optimizer status at extreme iterates need not be
  invariant across transformations. The regression uses the observed single
  fit failure (real output), while ordinary batch equivalence uses regular data.

## Actual checks

`JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q examples/weibull_simulation/check_fit.py examples/weibull_simulation/check_scenario.py`:
23 passed in 23.87s before the two additional boundary-score cases. The
unchanged 15 fit checks plus the final 10 scenario checks above all pass.
Generator commands (`JAX_ENABLE_X64=1/0 uv run pytest -q tests/unit/test_simulation_generator.py`):
17 passed in 3.11s / 17 passed in 3.03s, no skips.
`uv run ruff format --check .`: 142 files already formatted.
`uv run ruff check .`: All checks passed!
`uv run mypy src`: Success: no issues found in 22 source files.
`uv lock --check`: Resolved 171 packages, exit 0.
`git diff --check`: clean.
`scripts/check_wheel.sh --through A10`: exit 0;
final `check: all gates passed through A10`.
Own suites: 1282 passed / 8 skipped in 124.78s (x64), 1287 passed / 3 skipped
in 125.45s (x32). Doctests: 4 per mode in 1.03s / .87s. Independent:
2333 passed / 27 skipped in 70.97s (x64), 2324 passed / 36 skipped in 72.86s
(x32). Log: /private/tmp/survjax-sim5-gate.log.
Wheel SHA-256: `e9aa003d19c07d4290041a300a45288b7b8a942982dae204ceab57b4f19ec46d`.
No core/packaging changes. Dedicated examples checks are separate from the
ordinary gate by design; no scripts changed.

## Remaining scope and continuation authorization

After a merge verdict, report SIM.5 in commentary and continue to SIM.6
performance/MCSE summaries and synchronized timings, as the user explicitly
requested both steps. Still stop for any maintainer verdict/spec question.
Remaining after SIM.6: CLI smoke, six-scenario full study/results/figures and
executable simulation docs. No remote, merge, rebase or publication actions.
