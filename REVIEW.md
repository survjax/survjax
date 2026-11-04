Independent review: round 1/3

# SIM.7a — CLI configurations and smoke performance CSV

Start / integration base: d2469e0a7e719dbcb276277e6fd80a447626ad4e.
Previous simulation final: 8d0fb5edb248038dc7bcc9a9af1e6326f120da15.
Original worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Read PLAN.md SIM.7a, use-case 01 main/details, then run.py and check_cli.py.
The maintainer completed the merge and explicitly authorized the next bit.
Preserve plan history; previous review records remain in Git at the start.

One small CLI/configuration step; performance.csv only. Complete per-replicate
CSV/metadata is SIM.7b before full-run evidence; figures/full results/docs are
later steps. No new dependency, library API, unnamed module, core or protected
path change. Shared study.py and generator imports stay unchanged.

D-034 remains for the full-run label/cache decision. This step retains the
reviewed honest first_call_seconds field and changes no timing/cache policy.
The two merged D-033 entries are distinct: simulation failure policy is settled
by the recorded user decision, CI guard is another workstream's record. No
formal status/ID changes in this step.

## Reading order and scope

1. PLAN.md SIM.7a and use-case 01 main/details: grids, calibration, seed order,
   performance measures, smoke budget and documented command.
2. run.py import branch, run_config and main. Existing numerical code unchanged.
3. check_cli.py: mocked grid/export tests and the actual documented script run.
4. Existing check_fit/check_scenario/check_performance and generator tests.

This step is about 297 changed source/check lines: run.py +73/-2 and
check_cli.py 222 lines; run.py totals 297. Complete replicate/metadata exports
remain SIM.7b, followed by full results/figures and executable docs. No full
study or complete-artifact claim. Do not grow study.py or add a new module.

## Invariants and independent checks

- Exactly smoke n=100,R=20, or full n=100,300,1000,R=1000; both targets .25/.55.
  Config validation and the precision guard run before calibration/output.
- Calibrate exactly once per target, with k=1.38,b0=11.42 and equally weighted
  arm LPs [0,log(.7)]; no recalibration across sample sizes. Enumerate n first,
  target second, passing indices 0..1/0..5 to the unchanged reviewed key scheme.
  Mocked checks capture every call and independently assert these exact inputs,
  grid/order/R values and one calibration per target for both configurations.
- Reuse run_scenario without changing BFGS, acceptance, performance formulas,
  thresholds or timing. Cast summary scalars on the host only; no tracer
  conversion/new JAX numerical path. Every model is a row, retaining all
  reason counts plus unique failed, attempted and accepted totals.
- CSV contains scenario/n/target/calibrated c/realized censoring/model, every
  reviewed summary and failure field, first_call_seconds, warm_seconds and
  fits_per_second. Fake records have different metric/reason values so the
  round-trip assertions distinguish their names. Reason sums exceed failed
  in that fixture; nothing sums overlapping reasons into the overall failed.
  All expected CSV values come from explicit fake input records, not by
  invoking the export under test. No pandas/optional dependency is added.
- Direct-script import uses local study.py; package import stays relative.
  Both refer to the same unchanged shared generator. No broad ImportError
  fallback that could hide a missing dependency. Only CLI main selects x64;
  imported run_config raises on disabled x64. The float32 subprocess verifies
  import preserves mode and run_config refuses before filesystem work.
- Actual smoke runs the documented script entry in a fresh subprocess whose
  initial JAX_ENABLE_X64=0. Main enables float64. It must complete within the
  use case's 120s docs timeout (not a new speed target). Four rows must be in
  the exact specified scenario/model order, with R=20 and count conservation.
- Reconstruct smoke censoring independently from primitive JAX uniform keys
  and NumPy inverse-CDF algebra on n=100,R=20, using the emitted c. Compare
  that integer censored count / 2000 against the emitted pooled censoring.
  Existing analytical/empirical calibration tests cover the root itself.
  Require positive finite timing/rates and rate=20/warm. Do not assert
  nominal coverage or all accepted at R=20; those are not smoke requirements.
- Unknown config rejects before calibration/creating output; argparse also
  rejects an invalid --config. No new numerical tolerances or silent skips.

## Actual commands/results

`JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q -s examples/weibull_simulation/check_cli.py`:
5 passed in 7.28s; direct-script smoke 4.41s. After adding the argparse
invalid-choice assertion, the combined CLI/performance/scenario/fit command
(same prefix, all four explicit check_*.py files): 41 passed in 30.99s;
direct-script smoke 4.54s. The argparse error text is an expected rejection,
not a failed test. Full grid was mocked, not executed.
`JAX_ENABLE_X64=1/0 uv run pytest -q tests/unit/test_simulation_generator.py`:
17 passed in 3.42s / 17 passed in 3.44s, no skips.
`uv run ruff format --check .`: 144 files already formatted.
`uv run ruff check .`: All checks passed!
`uv run mypy src`: Success: no issues found in 22 source files.
`uv lock --check`: Resolved 171 packages, exit 0. `git diff --check`: clean.
Initial Ruff/full-gate attempt stopped at C408 for three dictionary-literal
style issues in the new tests. Replaced dict(...) with literals; no test
assertion/numerical failure and no tolerance change. Initial gate log:
/private/tmp/survjax-sim7a-gate-initial.log.
`scripts/check_wheel.sh --through A10`: final exit 0,
`check: all gates passed through A10`.
Own x64: 1282 passed/8 skipped in 134.91s; x32: 1287 passed/3 skipped in
119.10s. Doctests: 4 passed per mode in 1.02s/.86s. Independent x64:
2333 passed/27 skipped in 67.60s; x32: 2324 passed/36 skipped in 62.67s.
Final log: /private/tmp/survjax-sim7a-gate.log. Fresh wheel built; wheel and
installed extension checks included. Optional demo checks are separate from
this ordinary gate. No protected/standard runner changes.

## Remaining scope / questions

SIM.7b must export every per-replicate diagnostic/failure flag and metadata
with CPU model, JAX/Python/dependency versions, thread settings, seeds/key
scheme, solver settings, calibrated/realized censoring and exact command.
Do not record full-run evidence until those exports and figures are ready.
D-034's wider compile-report/cache question remains for that full-run step;
CSV currently uses the reviewed first_call_seconds name and never implies
that a cached first call incurred compilation. The maintainer explicitly
resumed after the prior maintainer verdict; no new policy is implemented.
The ledger's two D-033 headings must be distinguished by topic; this step
does not resolve/relabel either entry or another agent's history.
Stop after this step's independent review. No merge/rebase/push/publication.
