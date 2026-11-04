Independent review: round 2/3

# SIM.6 — Accepted-fit performance and synchronized timings

Start/previous final: 458ebbd559fb5540b4eb7a32784d93143fe4d79b.
Integration base: 95f453279ab9bffad1c67dc234a56d46b534a677.
Original worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Branch release-simulation. SIM.5 reviewed merge round 2; user explicitly
requested "implement this and then the next bit as well", authorizing this
step after SIM.5's review. Stop after this step's review. No CLI/full study yet.
Preserve merged/unfinished plan history. No public library API, dependency,
protected path, config, docs or other workstream changes.

## Reading order

1. PLAN.md SIM.6 and use-case 01 details Performance measures/Timing.
2. examples/weibull_simulation/run.py performance, ModelRun/ScenarioRun,
   _time_model and run_scenario. Earlier SIM.5 functions are unchanged.
3. examples/weibull_simulation/check_performance.py.
4. Existing study.py (306 lines) and shared generator import contract stay
   untouched. The importable engine intentionally lives in run.py; direct
   script imports and docs imports/layout are tracked for the CLI/docs steps
   (SIM.5 N-1). No attempt to deliver the CLI in this commit.

## Invariants and requirement mapping

- Only accepted fits contribute to performance. Retain every raw fit and its
  independent reasons/unique failed count. No extra acceptance gates, retries,
  or silently missing rows. Summary counts remain in failure_counts.
- Gamma is the final theta entry for either model. Bias is mean gamma - truth;
  empirical SE is sample SD (ddof=1); model SE is sqrt(mean(se**2)), not mean
  SE. Bias MCSE=SD/sqrt(R_acc), SD MCSE=SD/sqrt(2*(R_acc-1)). No model-SE MCSE.
- Coverage uses inclusive log-Wald endpoints gamma +/-1.96*SE. Equivalent to
  the HR interval containment for finite values, avoiding unnecessary exp.
  Power rejects strictly |gamma/SE|>1.96. Bernoulli MCSE=sqrt(p*(1-p)/R_acc).
- Cast integer counts to the floating parameter dtype before division (spec
  3.3). Replace rejected gamma/SE before subtracting, squaring or dividing
  (7.2), so rejected huge/NaN estimates and NaN inference cannot poison sums.
  No floors, epsilons or clipping; substitute denominators only where no
  statistical estimate is defined. Zero accepted: all statistics undefined
  NaN; one accepted: sample SD and SD/bias MCSE undefined, other measures
  remain defined. This is ordinary ddof=1 arithmetic, not a new failure policy.
- performance is pure JAX, eager/jittable; host timing/count extraction stays
  outside jit. No differentiation guarantee through optimization or discrete
  acceptance/statistical decisions. Core derivatives are unchanged.
- run_scenario prepares/host-validates once, uses the same generated cohort
  for both models, and records realized pooled censoring as a float mean.
  Calibration stays caller-owned, once per target, unchanged from SIM.5.
- _time_model synchronizes data/arm before the clock, then one first call and
  exactly three subsequent warm calls, each blocking the entire FitResult.
  Warm time is median(3), rate=requested R / warm seconds, never R_acc/time.
  First-call seconds include compilation if not already cached. No cache
  clearing or speed threshold. Later metadata will report these honestly.
- run.py stays under 300 lines (226); the unchanged study.py does not grow.
  Source/check diff is about 290 lines, including a focused SIM.5 N-3 check.

## Analytical and orchestration checks

- Two accepted estimates gamma=[0,3], SE=[1,.5], true gamma=.5: bias=1,
  empirical SD=3/sqrt(2), bias and SD MCSE=1.5, RMS model SE=sqrt(.625),
  coverage=power=.5, each Bernoulli MCSE=1/sqrt(8). These are hand algebra,
  not expectations obtained from code under test. Check eager and jit at the
  unchanged ordinary x64 tolerances from tests/impl_tolerances.py.
- Rejected huge gamma and rejected NaN gamma/SE do not contribute. The
  theta-only NaN row has finite objective and the actual reviewed acceptance
  flag; require its nonfinite reason (closes nonblocking SIM.5 N-3).
- Zero accepted: every statistic is NaN and failed=4. Singleton z=1.96 has
  inclusive coverage, strict nonrejection, model SE=1, bias=1.96 and zero
  Bernoulli MCSEs; ddof-dependent quantities are NaN.
- Timing spy delegates real synchronization and asserts the whole result
  object, exact block/clock/fit order, four calls, first 10 and warm [1,7,2]
  -> median 2 (different from mean). A controlled scenario with 4 requested,
  2 accepted and warm=2 verifies rate=2, not 1. No wall-clock thresholds.
- One real n=100,R=2 scenario checks both models, all retained rows/counts,
  realized censoring against independent NumPy event counts and positive
  finite timings/rates. Not a smoke acceptance/full study or coverage claim.

## Actual commands/results

`JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q examples/weibull_simulation/check_performance.py examples/weibull_simulation/check_scenario.py examples/weibull_simulation/check_fit.py`:
35 passed in 68.80s (6 performance, 14 scenario, 15 fit).
After stronger timing-spy/throughput assertions, performance file alone:
6 passed in 8.60s. Earlier performance runs: 5 passed in 13.75s, then 6 passed
in 10.26s. No numerical failure or tolerance adjustment.
Generator commands (`JAX_ENABLE_X64=1/0 uv run pytest -q tests/unit/test_simulation_generator.py`):
17 passed in 8.65s / 17 passed in 8.15s, no skips.
`uv run ruff format --check .`: 143 files already formatted.
`uv run ruff check .`: All checks passed!
`uv run mypy src`: Success: no issues found in 22 source files.
`uv lock --check`: Resolved 171 packages, exit 0. `git diff --check`: clean.
`scripts/check_wheel.sh --through A10`: exit 0, final
`check: all gates passed through A10`. Own x64: 1282 passed/8 skipped in
180.18s; x32: 1287 passed/3 skipped in 172.67s. Doctests: 4 per mode in
1.29s/1.22s. Independent x64: 2333 passed/27 skipped in 74.84s; x32:
2324 passed/36 skipped in 61.12s. Log: /private/tmp/survjax-sim6-gate.log.
Wheel built fresh with unchanged core/packaging, install/spec smoke included.
Optional example checks are explicit separate commands, not ordinary-gate
coverage; no standard/protected script modifications.
Tested versions (metadata query, no dependencies changed): Python3.12.9,
JAX/jaxlib0.11.2, NumPy2.5.3, Optimistix0.1.0, pytest9.1.1. CPU reference.

## Prior review and remaining work

SIM.5 round 2 written verdict merge; F-1 fixed, N-3 now tested here.
Reviewer round-2 raw gate exited 1: own suites 1282/1287 pass, doctests pass,
x64 spec 2328 pass/27 skips/5 wheel-fixture setup errors (SURVJAX_WHEEL unset),
x32 spec not run after that exit. Focused fit/scenario: 29 passed. The
implementer's SIM.5 fresh-wheel wrapper passed both spec modes; these are
separate outcomes, not a claim that the reviewer's raw gate passed.
Independent review file: /Users/wamster3/dev/survjax/review/slices/SIM.5.md.
N-1 direct-script imports is deferred to CLI/docs; N-2 real solver-failure
fixture is pinned-environment/status plumbing, not transformation invariance
or extreme PH derivative accuracy. D-033 status remains maintainer-owned.
Remaining after this step: CLI smoke, six-scenario full study/results/figures,
executable docs. Never merge, rebase, push or publish.


## Round 1 response

F-1 fixed: new three-accepted-row test uses gamma=[0,1,2], truth=0.
Hand mean=1, sum squared deviations=2 and ddof=1 SD=1, so bias MCSE is
1/sqrt(3), empirical-SE MCSE is .5. Checks both eager and jit with unchanged
ordinary tolerances. No production formula/threshold changes.
Final combined performance/scenario/fit checks: 36 passed in 26.65s.
Ruff format/check and git diff --check clean after the test fix.
Reviewer round-1 focused checks: 35 passed in 26.54s. Raw through-A10 gate
exited 1 with own suites/doctests passing, x64 spec 2328 passed/27 skipped/5
SURVJAX_WHEEL setup errors; x32 spec not run. Our fresh-wheel gate above
passed both modes. No reason to repeat unchanged core suites for this test fix.

N-1 tracked explicitly for full-run metadata: CPU model, JAX version, thread
settings, calibrated c values, realized censoring and exact run command.
S-1 recorded as D-034 and asked of the maintainer for the later full-run
label/cache decision. No action on the spec question: first_call_seconds and
cache behavior remain unchanged, and no table/metadata label exists yet.
The question does not block SIM.6 per the reviewer; stop after this review.

---

# Preserved incoming main review (963c7fc, verbatim)

Independent review: round 1/3

# A9.close — Reconcile completed A9 records

Read PLAN.md's A9.close section, plans/A9.md's closure note and retained
A9.d1 gate map, then plans/README.md's A9 row. Previous current review is
preserved in Git at a351952:REVIEW.md (CI.integration).

Invariants:
- Original plans/A9.md text and the integrated PLAN.md history remain
  verbatim after new wrappers. No unfinished unrelated work is discarded.
- The gate/narrative map is copied verbatim from 14155ed:REVIEW.md. Its
  historical pending/attribution language is retained, not a new run claim.
- A9.4 round 1 and A9.d1 round 2 both have independent merge verdicts.
  A9.d1's F-1/N-1/N-2 are fixed. Maintainer requested this bookkeeping.
- A9.3 remains withdrawn. A9 closure does not accept all Slice A/release
  work, resolve deviations or change other workstream ownership.
- No source/docs pages/tests/spec/dependency/protected paths are changed.
  No new test is appropriate for these record-only edits.

Spec context: 7.3, 10.3, 12.1–12.3. The retained gate map names each test,
artifact and docs narrative item with its original validation provenance.
The newer combined through-A10 result is cited from a351952:REVIEW.md,
not rerun. The final A9.d1 verdict was read from the independent report.

Validation: preservation/reference/scope assertions pass: both historical
plans and the gate map are verbatim; all three artifact paths and four
commit references exist; exactly the four intended record files differ.
Whitespace check: git diff --check exits 0. No pytest, numerical gate, benchmark, R, docs build or accelerator
execution in this bookkeeping step. No uncertainty about implementation is
introduced; overall Slice A/release acceptance remains maintainer-owned.
