Independent review: round 1/3

# SIM.7a.batch — Batch same-size censoring scenarios in JAX

Start/previous final: 6a9c96b7c2935f18d8467d3266625bd12aaaba6e.
Maintainer merge base: d2469e0a7e719dbcb276277e6fd80a447626ad4e.
Original worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
User explicitly requests JAX batching across same-size scenarios and comments
justifying remaining Python loops. No export helper approval is inferred.
Read PLAN.md's correction, run.py and the focused CLI/scenario/summary checks.
Numerical fitter/core/generator contract/dependencies remain unchanged.

Counts and summaries must retain censoring-level axes and reduce replicates
only. Timings are for the whole same-size batch; CSV states timed_scenarios
and throughput uses C*R. No made-up per-scenario time allocation. Existing
scalar demo helpers remain usable. Exact scenario/replicate/U-C keys and host
observation validation must be preserved. No numerical Python scenario loop.

## Reading order / implementation

1. PLAN.md SIM.7a.batch and use-case 01 seed/fitting/performance semantics.
2. run.py prepare_scenario, _fit_batch, last-axis counts/performance, _summarize,
   run_scenario and the corrected sample-size loop in run_config.
3. check_scenario.py grouped numerical equivalence and trace-count reuse;
   check_performance.py independent grouped hand statistics;
   check_cli.py batched dispatch and independent real smoke key checks.
4. study.py: three explanatory comments only; generator/calibration/fitter
   numerical code and the frozen shared import contract are unchanged.

Changed implementation/check lines: about 295, excluding PLAN/REVIEW.
run.py is 321 lines, near the approximate module-size target; the proposed
export helper remains unapproved/uncreated, and no extra helper layer or
library API has been added. The current correction stays self-contained.

## Invariants / tests

- Scalar preparation still yields (R,N). Matching nonempty (C,) index/limit
  vectors yield (C,R,N); positive even n/R, integer nonnegative indices,
  matching shapes, finite positive limits and x64 are checked on the host.
  Four new invalid vector/type cases cover these guards.
- Outer generation vmap owns fold_in(scenario_index), inner vmap owns split(R)
  and the unchanged generator splits U/C. The exact scalar keys are preserved;
  no flattened split(C*R), common random keys across levels or padding.
  Host observation validation still runs on the complete generated batch.
- One module-level fit_batch jit contains the two vmaps. Scalar R-only calls
  retain one vmap. No Python runtime iteration over censoring levels or fits.
  Two model graphs have different theta dimensions; their static dispatch
  loop is explained, rather than padding a heterogeneous parameter vector.
- failure_counts is now pure JAX with scalar/(C,) outputs, not host Python int
  extraction. This is a demo helper change, not the frozen generator contract.
  Counts and every performance reduction use only the last replicate axis.
  The means broadcast back via mean[...,None]; count casts precede division.
  Rejected NaNs are substituted before arithmetic. Strict acceptance, SD,
  MCSE and overlap/unique-failure formulas are unchanged.
- A stable module-level _summarize jit jointly computes counts/performance
  over all levels. The existing scalar hand/timing tests still pass.
- Independent two-level summary fixture: accepted gamma=[0,3] with SE=[1,.5]
  and true gamma=.5 yields bias=1, SD=3/sqrt(2), RMS SE=sqrt(.625), coverage/
  power=.5. The second level has only gamma=1,SE=1 accepted: bias=.5,
  RMS SE=1, coverage=1,power=0, zero Bernoulli MCSEs and undefined SD/SD+bias
  MCSE. Accepted counts [2,1], failed [2,3], attempts [4,4] must stay separate.
  Eager/jit agree at unchanged ordinary x64 tolerances. No pooled denominator.
- For both actual models, grouped (C=2,R=2,N=20) generation/fit outputs match
  the scalar path for each level at the existing value tolerances (theta,
  objective,H,SE,CI and acceptance). Independent real CLI inverse-CDF/key
  reconstruction also remains green. Scores at different fitted points are
  not compared as identical residues; fixed-input derivative checks remain.
- Actual trace counter: run the grouped fitter, change observed values with
  the same shape/static observation metadata, run again, require one trace.
  No private cache-size API or clearing caches. Eager/Python callback counting
  is test instrumentation at trace time only.
- run_config dispatches once per n, with indices [0,1], [2,3], [4,5] and both
  calibrated limits together. Fake full-grid checks assert these exact calls.
  Calibration still runs once per target on the host before numerical batches.
- Remaining Python loops/comprehensions are explained: static result-schema
  tracing, distinct model graphs, host root calibration, synchronized host
  timing samples, n-dependent compilation and post-device_get CSV writing.
  study.py comments explain host quadrature/bracketing/bisection too.
- Timings cover the C*R-fit batch. CSV repeats the shared first/warm timings
  with timed_scenarios=C and computes rate=C*R/warm, never R/warm or invented
  per-scenario times. Each row's statistics/counts retain its own R. This is
  the direct measurement scope of the user's explicit batching correction;
  D-034's later full-run compile label/cache question is not silently closed.

## Actual checks / investigations

Initial updated CLI run: 3 failed,23 passed (27.86s); all failures were model
CSV order. jax.device_get rebuilds dict pytrees in sorted-key order, so model
order became exponential first. Fixed serialization to use the explicit
weibull/exponential order after the one host transfer. No numerical failure,
expected-value/tolerance relaxation or change to the generator.
Grouped performance/scenario checks: 24 passed in 27.83s.
`JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q -s examples/weibull_simulation/check_cli.py examples/weibull_simulation/check_performance.py examples/weibull_simulation/check_scenario.py examples/weibull_simulation/check_fit.py`:
44 passed in 43.28s; real direct-script smoke 5.20s. After four new invalid
configuration cases, scenario file alone: 20 passed in 18.57s; unchanged
CLI/performance/fit files contribute 5/8/15 cases. No full-grid execution.
`JAX_ENABLE_X64=1/0 uv run pytest -q tests/unit/test_simulation_generator.py`:
17 passed in 3.62s / 17 passed in 3.66s, no skips.
Ruff format: 144 files already formatted; Ruff lint: All checks passed!
`uv run mypy src`: no issues found in 22 source files.
`uv lock --check`: Resolved 171 packages, exit 0. Whitespace check clean.
`scripts/check_wheel.sh --through A10`: exit 0, final
`check: all gates passed through A10`. Own x64: 1282 passed/8 skipped in
127.59s; x32: 1287 passed/3 skipped in 119.02s. Doctests: 4 per mode in
1.03s/1.00s. Independent x64: 2333 passed/27 skipped in 67.08s; x32:
2324 passed/36 skipped in 61.06s. Log:
/private/tmp/survjax-sim7a-batch-gate.log. Fresh wheel/install checks included;
optional demo checks remain separate, with no protected gate changes.

## Remaining scope

Complete replicate/metadata exports (new helper proposal still awaits explicit
approval), full study/figures and executable docs. No new dependency, core,
protected-path, docs or spec edit. Preserve unfinished plan history. Stop
when this correction's independent review finishes; never merge/rebase/push.
