Independent review: round 1/3

# SIM.7b.1 — Complete per-replicate CSV export

Start/previous final: 49f57462583259606abca2ecceed2ef846392c03.
Reviewed numerical batching: 877daf120aa43a4d11b49f04fd04b7ccb39af2d6.
Merge base: d2469e0a7e719dbcb276277e6fd80a447626ad4e.
Original worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
The maintainer accepts shared-batch timing and authorizes the next export
step after the _results.py proposal. Record D-034's decision without changing
its formal open status or editing read-only spec. The previous maintainer
verdict's timing choice is settled; no need to reopen that approval.

Read PLAN.md SIM.7b.1, _results.py, run.py integration and check_results/CLI.
All fits and masks are computed together in JAX. The helper serializes host
records, including every rejected/unestimable fit; no Python numerical loop,
new core API/dependency, protected path or shared study.py change.
Metadata/provenance is the next separate step before full study outputs.
## Reading order and scope

1. DEVIATIONS.md D-034's explicit maintainer decision; PLAN.md SIM.7b.1 split.
2. _results.py host-only writer and fixed two-model numeric CSV schema.
3. run.py: _summarize returns masks with counts/performance; ModelRun carries
   them; create output directory before calibration; collect already-host
   batches and call write_results. Existing numerical reductions/fit unchanged.
4. check_results.py round-trip fixture, check_cli.py fake grid and real smoke.

The helper was proposed with its exact path and purpose before this step;
"on to next bit" follows that proposal and accepts proceeding with exports.
No other new module/API/dependency is introduced. The full CSV+metadata step
was split before implementation: metadata is SIM.7b.2. No complete full-run
artifact/evidence claim before metadata and figures exist.

## Invariants / requirement mapping

- Computation stays in the C/R JAX batch; failure_columns joins counts and
  performance in the existing compiled _summarize call. Every returned mask
  has shape (C,R). ModelRun carries the masks; no host reimplementation of
  acceptance conditions, per-scenario JAX calls or estimator work in writer.
  Device transfer remains once per n, before the serialization helper.
- write_results receives only completed host batches. All its loops are
  explicitly host serialization: static schema, levels, heterogeneous model
  record shapes, CSV rows and vector/matrix cells. No filtering by accepted;
  every (scenario,model,replicate) is retained, including unestimable cases.
- performance.csv has fixed count/measure order, avoiding device_get dict
  sorting (batch review N-1). Existing values, scenario order, unique failed
  totals and independent overlap counts remain unchanged. Shared group times,
  timed_scenarios=C and C*R/warm are now explicitly approved in D-034. No
  timing-code change, separate timing runs or allocated per-scenario times.
- replicates.csv includes theta_i, gradient_i and hessian_i_j raw numeric
  cells, objective, log_hr and SE/HR interval, raw solver_success/hessian_ok/
  estimable/accepted flags, plus failed and every independent reason. Parameter
  count disambiguates the Weibull's 3 vs Exponential's 2 parameters. Missing
  Exponential third-parameter cells are blank; real diagnostic NaN/inf values
  remain numeric CSV spellings. No zero fill, clipping or hidden row dropping.
- Fixed schema is for the two specified models only, not a new extension API.
  Parameter order remains Weibull(log k,log b0,gamma) and Exponential(log
  lambda0,gamma), exactly the reviewed fitter. Formal metadata map is next.
- out.mkdir runs after config/x64 guards but before calibration/fitting,
  fixing the carried SIM.7a nit for an existing-file --out. CSV writes still
  occur only after all n batches have completed. This is not a claim to
  detect every possible later filesystem failure before calculation.
- Shared study.py/core/dependencies/protected scripts unchanged. Runner is
  back to 297 lines. _results.py is 130 lines. The implementation/check diff
  is 337 lines including 39 deleted lines of moved serialization; 298 added.
  This is near the 300-line target after splitting metadata into the next step.

## Meaningful checks

- For P=3 and P=2, construct three hand diagnostic rows using reviewed
  _assess_fit: accepted; rejected with nonfinite theta/objective/score/Hessian
  plus solver failure; unestimable with placeholder diagnostics. Stack two
  levels with opposite replicate order so a wrong-level export is visible.
  Positive-definite H=L L^T has distinct diagonal values, good gradients are
  distinct nonzero values below 1e-5, and raw nonfinite values include NaN,
  +inf and -inf. Expected output is the supplied input records, not a second
  run of the serializer. Compare every raw numeric/vector/matrix/flag/reason
  exactly, including nonfinite values; no numerical tolerance was added.
- Twelve raw rows (2 levels * 2 models * 3 replicates), all unique IDs. Each
  level/model has attempted=3, accepted=1, unique failed=2 and reason sums=5.
  Raw flag/reason sums reproduce each summary column, including the gated
  unestimable reason. Assert intended count-column order and unused blanks.
- Existing mocked smoke/full CLI grid still checks n/order/R/calibration once
  and every summary field. Its artificial raw fields now satisfy the export
  interface; these fakes are for dispatch/serialization, not likelihood claims.
- Real direct-script smoke must additionally emit all 80 unique raw rows and
  correct parameter counts for both models. Existing primitive-key/NumPy
  inverse-CDF censoring checks, timing scope and count conservation stay green.
- Invalid --out is a regular file with known contents. Under a calibration
  stub that fails if called, valid smoke request raises FileExistsError and
  leaves those contents intact. This proves rejection precedes numerical work.

## Actual commands/results

`JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q -s examples/weibull_simulation/check_results.py examples/weibull_simulation/check_cli.py examples/weibull_simulation/check_performance.py examples/weibull_simulation/check_scenario.py examples/weibull_simulation/check_fit.py`:
49 passed in 40.22s; direct-script smoke 4.90s. After strengthening the
round-trip fixture with distinct gradients/SPD Hessian, check_results alone:
1 passed in 1.45s. No failure/tolerance adjustment.
`JAX_ENABLE_X64=1/0 uv run pytest -q tests/unit/test_simulation_generator.py`:
17 passed in 3.40s in each mode, no skips.
Ruff format: 146 files already formatted; lint: All checks passed!
`uv run mypy src`: Success: no issues found in 22 source files.
`uv lock --check`: Resolved 171 packages, exit 0. Whitespace check clean.
`scripts/check_wheel.sh --through A10`: exit 0, final
`check: all gates passed through A10`. Own x64: 1282 passed/8 skipped in
127.03s; x32: 1287 passed/3 skipped in 115.49s. Doctests: 4 per mode in
.95s/.86s. Independent x64: 2333 passed/27 skipped in 65.82s; x32:
2324 passed/36 skipped in 61.01s. Log /private/tmp/survjax-sim7b1-gate.log.
Fresh wheel built; installed wheel/extension checks included. Dedicated demo
checks remain explicit separate commands; no protected runner changes.

## Remaining / limits

Metadata/provenance, then full six-scenario results and figures, then
executable docs. No full R=1000 fit or release acceptance claim here.
D-034 approval is recorded but its formal status stays maintainer-owned.
Do not reopen the settled timing choice or the simulation D-033 policy.
The separate CI D-033 collision from the merge remains for maintainer
bookkeeping. No merge/rebase/push/publication. Stop after independent review.
