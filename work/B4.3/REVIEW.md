# B4.3 — Argument-form metadata for traced construction

Independent review: round 1/3

## Read first

1. PLAN.md and spec rev 62 sections 3.2, 7.2 and 12.3 (decision 0060).
2. src/survjax/data.py: only the two metadata selection sites change.
3. tests/unit/test_data_metadata.py and the independently synced
   tests/spec/test_observation_traced_metadata.py (51 acceptance tests).
4. scripts/spec-features.txt and scripts/benchmark_likelihood_pruning.py.
5. The benchmark records below and DEVIATIONS.md's historical test discrepancy.

## Invariants and evidence

Only None versus supplied determines metadata when any observation input is
traced, including nested inputs or concrete arguments beside a tracer.
Flag-form construction retains exact; event/right-only forms retain right;
supplying left or interval keeps all four kinds. Bounds form retains all
censored kinds and retains exact only when supplied. Each truncation flag is
independently true only when that bound is supplied. time2 adds no kind.
No traced or captured observation values are inspected for specialization.

Concrete metadata still comes from normalized rows, including invalid ones.
Normalization, stored issues, current-field validation, dtype policy, slicing,
PyTree reconstruction and likelihood arithmetic are unchanged. No new host
transfer, callback or synchronization is introduced in the compiled library
path: metadata selection uses Python identity checks on argument presence.
Benchmark synchronization is explicit caller-side timing, not library behavior.

The updated unit metadata suite covers vmap and nested-tracer forms, supplied
false indicators, supplied default windows, omitted exact, unchanged leaves,
invalid-indicator preservation and compilation reuse across changed values
(spec 3.2, 6.2, 7.2). Existing construction/tracing tests retain validation
coverage. Both pruning suites retain hook omission, invalid-row isolation and
parameter derivative checks. The 51 independent tests cover every argument
form, slices, jit/vmap, compilation reuse and raw likelihood value/derivative
agreement (spec 3.2, 6.2, 7.2, 12.1). All ran without a slow-test filter.

D-B4.3-001 recorded a protected test that changed shape while expecting one
compilation. The maintainer supplied the independent author's correction via
3d9704a / merge b4b6700 and instructed work to resume. The corrected test now
passes. No protected test was changed by the implementation; the historical
entry's formal status remains maintainer-owned.

## Validation

Focused command (also run with JAX_ENABLE_X64=0):

```bash
JAX_ENABLE_X64=1 SPEC_THROUGH=A12 SPEC_FEATURES=observation.censoring,observation.truncation,observation.traced_metadata,family.lognormal,family.loglogistic uv run --group test pytest -n 6 -q --durations=15 tests/unit/test_data_metadata.py tests/unit/test_data_tracing.py tests/unit/test_data_construction.py tests/unit/test_likelihood_pruning.py tests/unit/test_likelihood_untruncated_pruning.py tests/spec/test_observation_traced_metadata.py
```

- Before the independent test correction: 124 passed, 1 failed in each dtype
  (19.63s float64, 19.60s float32), solely D-B4.3-001.
- After correction: 125 passed float64 (21.46s), 125 passed float32 (22.03s),
  both exit 0. All 51 acceptance tests are included. New unit tests were below
  the top-15 call-duration cutoffs of 0.48s / 0.45s; no slow-list refresh needed.
- First fast gate stopped at mypy: fixed-versus-variable-length tuple inference
  at data.py:147. Added a local tuple[bool, ...] annotation, with no runtime
  change; focused tests and benchmark runs predate only this annotation.
- Final scripts/check.sh --fast --through A12: exit 0, all gates passed.
  Formatting: 275 files already formatted; Ruff: all checks passed; mypy:
  no issues in 25 source files. Implementation tests: float64 1212 passed /
  8 skipped (14.94s), float32 1214 passed / 6 skipped (15.02s). Source doctests:
  4 passed each. Acceptance tests: float64 4654 passed / 46 skipped (40.82s),
  float32 4597 passed / 93 skipped (37.11s). Each acceptance run emitted the
  two existing NumPy warnings at test_a11_ph_direct_methods.py:78 (log zero
  and invalid multiply in its expected-value expression).
- git diff --check: clean.
- No full non-fast cumulative gate, docs build or R fixture regeneration run.

## Benchmark method and results

Run sequentially, after focused tests completed:

```bash
JAX_ENABLE_X64=1 uv run python scripts/benchmark_likelihood_pruning.py > work/B4.3/benchmark-float64.jsonl
JAX_ENABLE_X64=0 uv run python scripts/benchmark_likelihood_pruning.py > work/B4.3/benchmark-float32.jsonl
```

Both exited 0. The committed JSONL files retain every case's metadata,
first-call cost, warmed median, input/output bytes and compilation-reuse result.
Reported host: macOS 26.7.2 arm64, Python 3.12.9, JAX 0.11.2, CpuDevice(0),
processor arm. B4.1/B4.2 identify this host as Apple M4 Max, 16 cores, 48 GiB;
a fresh sysctl model query was sandbox-denied, so this run's automatic hardware
record is limited to the reported platform/processor/device fields.

Seed 20261112; uniform times in [0.5,4), exact-only, right-only and alternating
mixed rows, without truncation. Shared Exponential rate=.75 or Weibull
shape=1.5/scale=2; N=1,000/10,000/100,000. Each route returns summed likelihood,
its parameter gradient and data.issues.ok, so diagnostic work is consumed.
Three warmups precede 30 interleaved repetitions, with rotating/reversed route
order; every measured output is synchronized. All route values/gradients agree
at unchanged implementation tolerances and all returned diagnostic checks pass.

Prepared concrete metadata derives from observed kinds. Prepared all-true
metadata now comes from traced bounds with exact and both windows supplied;
the benchmark asserts all six flags. Traced construction calls the ordinary
time/event constructor inside jit and records its actual auxiliary flags:
exact/right true, left/interval and both truncations false, even for single-kind
values. Changed parameters, reversed prepared data, and changed traced times
and events reuse compilation (trace counters checked in every case).

Median warmed microseconds: prepared concrete / prepared all-true / traced
construction plus evaluation and diagnostics.

| N | Kind | Family | float64 | float32 |
|---|---|---|---:|---:|
| 1,000 | exact | exponential | 63.8 / 199.8 / 46.5 | 46.6 / 159.1 / 43.2 |
| 1,000 | exact | weibull | 138.0 / 347.7 / 122.9 | 92.7 / 370.1 / 85.4 |
| 1,000 | censored | exponential | 50.9 / 207.9 / 51.0 | 35.4 / 157.4 / 36.5 |
| 1,000 | censored | weibull | 72.8 / 377.6 / 112.9 | 48.2 / 362.5 / 82.3 |
| 1,000 | mixed | exponential | 65.5 / 177.3 / 41.9 | 49.8 / 147.3 / 39.1 |
| 1,000 | mixed | weibull | 146.3 / 364.0 / 123.6 | 105.7 / 371.6 / 78.4 |
| 10,000 | exact | exponential | 130.1 / 548.7 / 130.0 | 114.2 / 333.4 / 126.4 |
| 10,000 | exact | weibull | 270.1 / 1281.4 / 230.2 | 198.1 / 925.1 / 207.0 |
| 10,000 | censored | exponential | 118.0 / 578.6 / 125.5 | 87.6 / 312.9 / 131.6 |
| 10,000 | censored | weibull | 173.3 / 1347.5 / 270.1 | 117.8 / 915.9 / 209.3 |
| 10,000 | mixed | exponential | 132.3 / 498.0 / 117.4 | 124.5 / 325.2 / 120.8 |
| 10,000 | mixed | weibull | 254.0 / 1302.7 / 244.2 | 195.0 / 903.4 / 215.9 |
| 100,000 | exact | exponential | 345.6 / 1789.2 / 306.0 | 234.9 / 787.0 / 227.2 |
| 100,000 | exact | weibull | 810.7 / 7146.8 / 746.0 | 509.5 / 2449.4 / 480.3 |
| 100,000 | censored | exponential | 254.6 / 1696.8 / 311.5 | 237.6 / 1198.0 / 284.0 |
| 100,000 | censored | weibull | 477.9 / 6567.3 / 705.4 | 311.3 / 2452.1 / 476.0 |
| 100,000 | mixed | exponential | 384.7 / 1909.1 / 337.0 | 262.2 / 788.8 / 214.9 |
| 100,000 | mixed | weibull | 844.4 / 7394.3 / 744.9 | 542.7 / 2411.5 / 477.0 |

For mixed rows, all-true/prepared-concrete median ratios are 2.49–8.76 in
float64 and 2.61–4.63 in float32. Traced/prepared-concrete ratios are
0.64–0.96 and 0.74–1.11 respectively. The traced route has fewer input arrays
and can fuse normalization/validation with evaluation; these observations
measure whole routes, not isolated construction overhead or a guaranteed
pruning speedup. Right-only traced construction retains a possible exact
branch and is slower than prepared right-only data in several cases. This is
expected conservative form specialization, not a reason to inspect event values.

First-call ranges (compilation plus execution), concrete / all-true / traced:
float64 0.050–0.189 / 0.163–0.703 / 0.080–0.197s;
float32 0.055–0.114 / 0.147–0.509 / 0.073–0.136s.
Data preparation/placement plus control-constructor compilation, outside the
evaluation timers: float64 0.030–0.615s, float32 0.035–0.505s. Parameter
preparation/placement: 0.000082–0.003313s and 0.000079–0.003334s respectively.
Prepared input bytes: 37,008–3,700,016 float64; 21,004–2,100,008 float32.
Traced input bytes: 9,008–900,016 float64; 5,004–500,008 float32.
Output bytes (including diagnostic bool): 17/25 float64; 9/13 float32.
Peak memory was not measured; array sizes are not peak-memory evidence.

Limits: one local CPU run per dtype, shared parameters only, no accelerator
claim, fixed small workload matrix, no statistical confidence intervals or
performance gate. Unlike B4.2, these timings also return data.issues.ok;
comparisons across those runs do not isolate the metadata change. Per-row
parameters and the wider performance baseline are outside this narrow rerun.
No general Slice B completion claim is made.

## Commits and remaining work

Step starting commit: aaba08f6b42c00433792a5984d53faa85a78d7fd.
Resumed on maintainer merge b4b6700c24626e53463dec65430587be378bf5ae.
Previous branch step final: 82c1826435c065bfcd4b13e05a77be26f35e0a89
(B5.FAMILIES); accepted pruning dependency: f568b9b (B4.2).
Original worktree base: cc0e30f.

Both requested spec-since commands ran. Rev 62 acknowledgement was already
committed by 71cadbf (.spec-seen = a5c1dea); --mark changed no content, so no
redundant acknowledgement commit was made.

Likelihood documentation (including argument-form versus value-derived
metadata), final Slice B evidence audit and maintainer integration remain.
Reviewer-workflow investigation stays deferred. After review, attempt the
workspace diff-display hook using the saved step starting SHA and the final
reviewed SHA, as required by the updated workflow.
