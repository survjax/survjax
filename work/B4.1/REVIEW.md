# B4.1 — Prune unused exact and censored pieces

Independent review: round 2/3

Base / previous step: da4dd733aa92bf7040e918a5fc0e8ef82ba0b85f (B3.5).
Original worktree base: cc0e30f. Cumulative gate: A12 with the existing
observation.censoring and observation.truncation features.

## Read first

1. [Plan](PLAN.md), spec 3.2, 7.1–7.3, then src/survjax/likelihood.py.
2. tests/unit/test_likelihood_pruning.py: omitted-hook test, independent
   delayed-entry algebra, metadata reuse and validation tests.
3. scripts/benchmark_likelihood_pruning.py and measurements below.
4. tests/slow_tests.txt: generated timing refresh, separate mechanical commit.

## Invariants and evidence

Only _evaluate changes. Absence of any censored kind omits the conditional
interval calculation; absence of exact rows omits the evaluator's reference,
density and exact normalizer work. Retained family hooks may still call
reference_time internally. Empty data has neither branch and returns the
correct broadcast empty shape. Mixed or all-true metadata retains both pieces.
This does not yet optimize the mixed exact/right case (B4.2).

Metadata are the existing constructor-derived flags and remain unchanged across
slicing. Traced construction still sets all six flags true. All current-field,
stored-issue and parameter checks precede pruning; no runtime host inspection,
callback, unchecked constructor or new specialization is introduced. Active
arithmetic and safe inactive substitutions are unchanged. Exact rows still
compute log Z even without truncation, respecting custom family boundary mass.
The common dtype and broadcasting logic are unchanged. No protected path,
public API, library module, dependency, tolerance or feature-selection change.

- The omitted-hook test fails immediately if a statically absent evaluator
  piece is invoked eagerly or during JIT tracing (spec 7.2). It checks exact,
  censored and empty batches. It does not forbid reference use inside a retained
  family hook.
- Delayed-entry identities check log(r)-r(t-L) or -r(c-L), first/second
  derivatives, both Jacobian modes and all four Hessian compositions, eager
  and jit(vmap), at r=0.5 and 0.75. Both concrete and traced-built data use the
  independent algebra. Valid values run under debug_nans; right at entry is
  included. This is parameter differentiation, not endpoint differentiation.
- Censored-only left/interval/right rows with finite right truncation use the
  independent exponential mass ratio. Both metadata forms are checked.
- Slices of a mixed parent reuse one compilation as values/kinds change within
  the parent flags. Separately constructed exact-only and censored-only data
  each compile once despite new values (spec 3.2/7.2).
- Invalid stored status codes, invalid rebuilt truncation fields and invalid
  rates retain NaN. A valid row's gradient/Hessian excludes an invalid rate.
  Scalar, broadcast draw and empty shapes remain correct (spec 3.3/7.1).

Existing family/wrapper and custom-family suites retain broader derivative,
finite-difference, improper-family, tail and zero-mass coverage. No new public
derivative guarantee is claimed.

## Measurements

Commands, each in a separate process:

```sh
JAX_ENABLE_X64=1 uv run python scripts/benchmark_likelihood_pruning.py
JAX_ENABLE_X64=0 uv run python scripts/benchmark_likelihood_pruning.py
```

Apple M4 Max, 16 CPU cores, 48 GiB RAM; macOS 26.7.2 arm64; Python 3.12.9,
JAX 0.11.2, CpuDevice(0). Seed 20261112, synthetic uniform times in [0.5,4],
shared Exponential rate=0.75 or Weibull shape=1.5/scale=2. No truncation. Mixed
means alternating exact/right rows. Concrete metadata is compared with the
same fields reconstructed from traced inputs (all-true metadata), which keeps
the unpruned computation. Neither timed path constructs data inside evaluation.

The measured function is jit(value_and_grad(sum(log_likelihood))). Each output
is synchronized. Three warmups follow the first call; 30 measured calls per
route interleave and alternate order. Both objective and gradient agree within
the unchanged implementation tolerances. These agreement checks are benchmark
controls; the tests above supply independent mathematical expectations.

Median warmed microseconds, concrete / all-true metadata:

| N | Kind | Family | float64 | float32 |
|---|---|---|---:|---:|
| 1,000 | exact | Exponential | 139.1 / 185.8 | 98.3 / 168.4 |
| 1,000 | exact | Weibull | 196.8 / 345.1 | 205.3 / 385.6 |
| 1,000 | censored | Exponential | 105.5 / 175.1 | 103.3 / 177.0 |
| 1,000 | censored | Weibull | 234.8 / 329.4 | 290.0 / 364.3 |
| 1,000 | mixed | Exponential | 191.8 / 192.9 | 159.6 / 161.9 |
| 1,000 | mixed | Weibull | 346.8 / 344.4 | 349.1 / 405.7 |
| 10,000 | exact | Exponential | 420.3 / 483.2 | 270.0 / 324.7 |
| 10,000 | exact | Weibull | 926.8 / 1220.8 | 588.9 / 924.5 |
| 10,000 | censored | Exponential | 494.0 / 515.2 | 293.4 / 326.3 |
| 10,000 | censored | Weibull | 1016.3 / 1210.0 | 862.3 / 909.0 |
| 10,000 | mixed | Exponential | 501.4 / 501.2 | 316.8 / 328.5 |
| 10,000 | mixed | Weibull | 1184.9 / 1205.0 | 931.5 / 897.7 |
| 100,000 | exact | Exponential | 1046.9 / 1670.8 | 689.6 / 980.9 |
| 100,000 | exact | Weibull | 2951.4 / 6379.2 | 1466.2 / 2291.5 |
| 100,000 | censored | Exponential | 1314.7 / 1657.9 | 673.0 / 831.1 |
| 100,000 | censored | Weibull | 3053.5 / 6303.3 | 1582.5 / 2302.4 |
| 100,000 | mixed | Exponential | 1635.6 / 1684.2 | 781.9 / 772.0 |
| 100,000 | mixed | Weibull | 6196.4 / 6179.1 | 2747.8 / 2715.7 |

First-call cost (compilation plus execution), across these cases: float64
concrete 0.109–0.531s, all-true 0.155–0.561s; float32 concrete 0.095–0.456s,
all-true 0.143–0.473s. Preparation/placement plus traced-constructor compilation,
outside evaluation: 0.027–0.601s float64, 0.027–0.451s float32.
Input arrays occupy 37,008–3,700,016 bytes float64, 21,004–2,100,008 float32;
output objective/gradient occupy 16/24 bytes float64 or 8/12 float32. These
are logical array sizes; peak memory was not measured.

At N=100,000 homogeneous batches are 1.23–2.16 times faster in this run;
mixed controls are 0.99–1.03. Across smaller mixed controls ratios range
0.96–1.16, showing timing noise. An initial sequential-route pilot had larger
control variation; the interleaved run above supersedes it. No timing threshold,
claim of universal speedup, or before/after improvement for mixed data is made.
This narrow benchmark supplements, rather than replaces, Slice A's baseline
(which includes per-observation parameters and construction inside jit).

## Validation

- Initial focused run: 40 passed, 1 failed because the new omission test also
  prohibited references inside the retained Exponential CDF. Corrected that
  overbroad test assertion; no production adjustment was needed.
- JAX_ENABLE_X64=1 uv run --group test pytest -n 6
  tests/unit/test_likelihood_pruning.py tests/unit/test_likelihood_core.py
  tests/unit/test_likelihood_intervals.py tests/unit/test_likelihood_truncation.py
  --durations=12: 41 passed in 14.55s. Float32: 41 passed in 19.02s.
- After adding the explicit invalid-parameter gradient/Hessian assertions,
  the complete new file passed again: float64 9 passed in 8.48s; float32
  9 passed in 8.49s. Slow-list refresh is required by these timings.
- Focused Ruff formatting/lint and mypy on src: passed; formatting final
  gate will verify the subsequent test/benchmark edits too.
- Both benchmark commands above: exit 0, all value/gradient comparisons passed.
- PYTEST_WORKERS=6
  SPEC_FEATURES=observation.censoring,observation.truncation
  scripts/update_slow_tests.sh --through A12: exit 0; 803 slow tests written,
  including 4 of the 9 new tests. All four underlying summaries were retained:
  float64 implementation 1503 passed/8 skipped (470.06s);
  float64 acceptance 3867 passed/32 skipped/2 warnings (386.00s);
  float32 implementation 1502 passed/9 skipped (385.09s);
  float32 acceptance 3813 passed/82 skipped/2 warnings (293.17s).
  The script tolerates pytest failures, so these summaries, rather than its
  exit status alone, establish the full-suite test results.
  Mechanical timing commit: 2f3e131 (127 additions/48 deletions).
- The fast A12 gate first reported a missing blank line in the final test
  addition. Ruff format fixed it; no behavior changed.
- scripts/check.sh --fast --through A12 rerun: exit 0, all gates passed.
  Guard clean; Ruff formatting (265 files), lint and mypy (25 source files)
  clean. Implementation: float64 1189 passed/8 skipped, float32 1190 passed/
  7 skipped. Doctests: 4 passed per mode. Acceptance: float64 3385 passed/
  32 skipped, float32 3373 passed/40 skipped. Each acceptance mode reports the
  two existing PH reference-arithmetic warnings in test_a11_ph_direct_methods.py.
- git diff --check: clean. No R regeneration, docs build or full nonfast
  check.sh invocation; the timing refresh supplied full test runs above.

## Remaining

B4.2 mixed exact/right optimization and measurements; family feature activation;
likelihood documentation. No Slice B completion claim. Reviewer-workflow
investigation remains deferred. No new spec ambiguity identified.

## Round 1 findings and follow-up

Reviewed commit: 8dde3ffb1a919fe5975244b40cb3a6ca867b9fab.
Verdict: fix-first. F-1 (should-fix) identified a surviving mutation that
omitted the interval-kind flag; the original mass-ratio test mixed all censored
kinds, so other flags still activated the branch. N-1 required no action.

F-1: extend the analytical mass-ratio test with right-only, left-only and
interval-only batches, each checked with concrete and conservative metadata.
The existing mixed case remains. No production code or tolerance change.
Complete new-file runs: float64 12 passed in 8.44s; float32 12 passed in 8.57s.
Each new single-kind case took at most 0.46s; no further timing-list refresh
was needed for these additions. Ruff formatting and lint passed.

Negative controls, float64, separate Python processes: replace only _evaluate
in memory using its inspected source, then invoke pytest without xdist on the
matching single-kind test. The `[1:3]` mutant fails the interval-only assertion
(actual zeros versus negative analytical log probabilities); `[1:2]` fails
the left-only assertion. Each run exits pytest with 1 as expected; no repository
or reviewer checkout is mutated. The tests now detect both omitted-kind faults.
Fast A12 gate after the test extension: exit 0, all gates passed. Formatting
(265 files), Ruff, mypy (25 files) and guard clean. Implementation: float64
1192 passed/8 skipped; float32 1193 passed/7 skipped. Doctests: 4 per mode.
Acceptance: float64 3385 passed/32 skipped; float32 3373 passed/40 skipped;
the same two PH reference-arithmetic warnings per mode. git diff --check clean.

The round-1 reviewer did not rerun the benchmark; the performance table remains
implementer-measured evidence. Reviewer mutation-checkout cleanup was blocked
by its permissions; no implementer cleanup or reviewer configuration change.
