# B4.2 — Untruncated exact/right likelihood path

Independent review: round 1/3

Base / previous step: 7ddec637227982785347880906ee15e1bcfe8036
(B4.1.COMMENTS); implementation dependency B4.1, f6a9bf3.
Original worktree base: cc0e30f. Cumulative gate: A12 with existing observation
features; no family-feature activation in this step.

## Read first

1. [Plan](PLAN.md); spec 2's closed-at-infinity convention and Z=1 sentence,
   then 3.2, 4.2 and 7.1–7.3.
2. src/survjax/likelihood.py: untruncated_exact_right guard and its two uses.
3. tests/unit/test_likelihood_untruncated_pruning.py: hook omission, individual
   disabling flags, cure-model algebra, inherited flags and invalid-field checks.
4. Measurements below, using unchanged scripts/benchmark_likelihood_pruning.py.
   tests/slow_tests.txt is a mechanical timing refresh.

## Invariants and evidence

Only _evaluate changes in production. The guard requires absence of flags 2–5:
left censoring, interval censoring, left truncation and right truncation. With
only exact/right rows and no truncation, exact values are log f(t), right values
log S(c), and log Z is zero. Both interval hooks are omitted. Inactive or bad
right-row inputs use zero, the lower endpoint of the untruncated window;
inactive density inputs still use the stopped family reference time. Both
absent-kind guards from B4.1 remain, including empty data behavior.

Spec 2 includes T=+inf in (c,+inf], so cure mass remains in S(c) and Z=1 for
improper as well as proper families. This resolves B4.1's earlier conservative
concern about retaining log Z without truncation; no assumption about S(inf)=0
is introduced. Correct interval overrides satisfy the same identities. All
other data continue through the existing conditional hook, without family-type
dispatch. General-path active arithmetic and substitutes are unchanged.

Current-field checks, stored issues, parameter preparation, broadcast shapes,
common dtype and final bad-row NaNs still surround both paths. Traced
construction retains all-true metadata. Slices retain parent flags. No new
host transfer, callback, public API, library module, dependency, tolerance,
protected-path edit or metadata scheme.

- test_untruncated_exact_right_omits_both_interval_hooks rejects either hook
  being called eagerly or during tracing; unit-rate values are independently -t.
- test_each_disqualifying_flag_retains_conditional_hook tests each of the four
  disabling flags alone, with an independent unit-rate conditional mass ratio.
  Thus a wrong flag index cannot be hidden by another enabling flag.
- test_cure_mass_and_joint_derivatives_match_independent_algebra uses
  S(t)=c+(1-c)exp(-rt), default autodiff density, and nonzero log S(inf).
  Exact at 1, right at 2 and uninformative right at zero have hand-derived
  values, Jacobians and full (rate,cure) Hessians. Both Jacobian modes and all
  four Hessian compositions are checked, eagerly at the first point and under
  jit(vmap) at two points, for concrete and conservative metadata. Valid eager
  values run under debug_nans. Whole-window values/derivatives are exactly zero.
- test_inherited_truncation_metadata_keeps_general_path checks that untruncated
  rows sliced from a truncated parent still use the hook and give the same values.
- test_fast_path_checks_invalid_current_fields_and_stored_issues verifies NaN
  for invalid rebuilt windows whose stored issue bits remain zero, and retained
  raw-indicator errors. Existing core/pruning tests cover invalid-parameter
  isolation, dtype promotion, scalar/broadcast/empty shapes and compilation reuse.

The existing family/wrapper suites retain tail, boundary, finite-difference and
transformation coverage; no new guarantee for custom nonsmooth primitives.

## Performance evidence

Run separately:

```sh
JAX_ENABLE_X64=1 uv run python scripts/benchmark_likelihood_pruning.py
JAX_ENABLE_X64=0 uv run python scripts/benchmark_likelihood_pruning.py
```

Same Apple M4 Max (16 cores, 48 GiB) environment recorded in B4.1; script reports
macOS 26.7.2 arm64, Python 3.12.9, JAX 0.11.2, CpuDevice(0). Seed 20261112;
uniform times in [0.5,4], alternating exact/right for mixed batches, no truncation;
shared Exponential rate=.75 or Weibull shape=1.5/scale=2. Both metadata forms
hold identical prepared observations. The conservative path is the general
calculation in the same run, not a timing comparison to another day's run.

The unchanged benchmark times jit(value_and_grad(sum(log_likelihood))). Every
output is synchronized; three warmups precede 30 measured calls per route,
interleaved with alternating order. Objective and gradient comparisons pass
the unchanged implementation tolerances. Construction is outside evaluation;
this is not a benchmark of construction inside jit or per-observation parameters.

Median warmed microseconds, concrete / all-true metadata:

| N | Kind | Family | float64 | float32 |
|---|---|---|---:|---:|
| 1,000 | exact | Exponential | 81.0 / 205.5 | 41.4 / 149.6 |
| 1,000 | exact | Weibull | 128.3 / 349.6 | 83.7 / 398.9 |
| 1,000 | censored | Exponential | 40.0 / 189.8 | 34.8 / 167.6 |
| 1,000 | censored | Weibull | 70.8 / 354.7 | 45.8 / 374.4 |
| 1,000 | mixed | Exponential | 70.1 / 191.3 | 53.2 / 162.2 |
| 1,000 | mixed | Weibull | 137.5 / 344.0 | 99.1 / 385.8 |
| 10,000 | exact | Exponential | 137.1 / 529.5 | 127.8 / 333.6 |
| 10,000 | exact | Weibull | 272.4 / 1335.7 | 208.4 / 945.6 |
| 10,000 | censored | Exponential | 113.3 / 547.1 | 102.0 / 309.3 |
| 10,000 | censored | Weibull | 151.1 / 1307.2 | 116.2 / 948.2 |
| 10,000 | mixed | Exponential | 129.2 / 526.7 | 105.7 / 281.8 |
| 10,000 | mixed | Weibull | 245.0 / 1287.7 | 186.0 / 915.0 |
| 100,000 | exact | Exponential | 355.4 / 1728.8 | 483.8 / 1841.9 |
| 100,000 | exact | Weibull | 826.6 / 6980.8 | 535.9 / 2549.0 |
| 100,000 | censored | Exponential | 289.3 / 1745.2 | 197.5 / 774.0 |
| 100,000 | censored | Weibull | 478.6 / 6982.9 | 305.5 / 2341.0 |
| 100,000 | mixed | Exponential | 373.4 / 1850.3 | 249.7 / 755.2 |
| 100,000 | mixed | Weibull | 865.6 / 7226.1 | 542.4 / 2484.6 |

Mixed-batch ratios are 2.50–8.35 in this local run. Timings remain noisy and
workload-specific; no fixed speed guarantee or CI timing gate is introduced.
First call (compilation plus execution): float64 concrete 0.047–0.137s,
all-true 0.160–0.626s; float32 concrete 0.052–0.115s, all-true 0.153–0.467s.
Preparation/placement plus traced-constructor compilation, outside evaluation:
float64 0.028–0.581s; float32 0.027–0.455s.
Input arrays occupy 37,008–3,700,016 bytes float64, 21,004–2,100,008 float32;
output objective/gradient occupy 16/24 bytes float64 or 8/12 float32. Peak memory
was not measured; these are logical array sizes only.

## Validation

- Focused Ruff format/lint passed; mypy passed on 25 source files.
- JAX_ENABLE_X64=1 uv run --group test pytest -n 6
  tests/unit/test_likelihood_untruncated_pruning.py
  tests/unit/test_likelihood_pruning.py tests/unit/test_likelihood_core.py
  tests/unit/test_likelihood_intervals.py tests/unit/test_likelihood_truncation.py
  --durations=15: 52 passed in 23.60s. Same command with X64=0:
  52 passed in 24.64s. The cure derivative test took about 21–23s.
- Both benchmark commands exited 0; all value/gradient comparisons passed.
- PYTEST_WORKERS=6
  SPEC_FEATURES=observation.censoring,observation.truncation
  scripts/update_slow_tests.sh --through A12: exit 0; 522 slow tests written,
  including the new cure-derivative test. The generated list shrinks from 803
  entries (56 added/337 deleted lines), isolated in a mechanical commit.
  All underlying pytest summaries were retained:
  float64 implementation 1514 passed/8 skipped (294.69s);
  float64 acceptance 3867 passed/32 skipped/2 warnings (117.55s);
  float32 implementation 1513 passed/9 skipped (266.65s);
  float32 acceptance 3813 passed/82 skipped/2 warnings (98.48s).
  The script tolerates pytest failures; these summaries, not its exit status
  alone, supply the full-suite evidence. Timings are local measurements, not
  controlled before/after proof for the whole suite.
- Mechanical timing commit: 6aabc98.
- scripts/check.sh --fast --through A12: exit 0, all gates passed. Guard,
  formatting (270 files), lint and mypy (25 source files) clean.
  Implementation: float64 1206 passed/8 skipped; float32 1208 passed/6 skipped.
  Doctests: 4 passed per mode. Both acceptance modes passed; float32 reported
  3634 passed/51 skipped/2 existing PH reference-arithmetic warnings (35.42s).
  The float64 acceptance summary was truncated from the captured tool output;
  no exact fast-suite count is claimed for that mode. Full timing-run counts
  above were retained separately. Changed fast counts follow the generated list.
- git diff --check: clean. No R regeneration, docs build or full nonfast
  check.sh invocation; full test runs were supplied by the timing refresh.

## Remaining

Family-feature activation, likelihood documentation, final Slice B evidence
audit/integration. B4.1 and B4.2 provide kind pruning and the dominant
untruncated exact/right path; no additional pruning or full Slice B completion
claim is made. Reviewer-workflow investigation remains deferred.
