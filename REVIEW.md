Independent review: round 1/3

# MNIST.2 — Digit outcomes and calibrated censoring

Read first:
1. PLAN.md MNIST.2; use-cases/03 details Precision, Splits and keys, Outcome
   generator; 00-common and spec 11.2.
2. examples/survival_mnist/model.py; unchanged shared generator study.py.
3. tests/unit/test_mnist_outcomes.py; small data.py/test_mnist_data.py changes.
4. calibration.json and README preparation recipe.

Scope/invariants:
- This is outcome preparation only: no CNN, training, reference fit, metrics,
  docs, dependency, core API or shared-generator change. MNIST.1 was approved
  by maintainer after round-2 merge. Same branch and sub-slice; inherited
  unfinished workstream plans retained. About 276 changed source/test/artifact
  lines, excluding plan/review records.
- NumPy float64 k/log-scale table obeys k=1.38 and log b=log 10 - 0.2*d.
  Calibration explicitly requires the 60000 training-source integer labels;
  official test rows and smoke subsets cannot be passed as a whole. Every
  label frequency is retained by passing per-row scales to the unchanged
  shared float64 host quadrature solver, target .5. No test-data calibration.
- Float32 shape/scales/c go to the shared JAX generator, even in x64 mode.
  Per-image keys are fold_in(fold_in(fold_in(key(20261025),3),0),global_index).
  Shared generator splits each key U-then-C; events use inverse CDF and
  independent Uniform(0,c) censoring. No new random sampling implementation.
- Default indices 0..N-1 require official global order. Explicit global
  indices preserve rows under subsets/reordering. Numeric value domains are
  documented preconditions as in the shared JIT-compatible generator; the
  verified loader supplies digits and split_indices supplies valid indices.
  Shape mismatches and nonscalar censor limits raise explicitly.
- Event-generation precision is float32, with float64 oracle parameters and
  calibration saved in calibration.json. That file records aggregates/hashes,
  no image or label rows. Later oracle likelihood evaluation will use float32
  per the design; no test-model metrics are reported now.
- Prior optional N-2 is addressed: train/validation host indices now int64 in
  either mode, matching test indices. Prior N-3 is addressed by a labelled
  JAX 0.11.2 prefix anchor measured by the MNIST.1 independent reviewer.
  Neither change alters cohort membership.

Tests → requirements:
- test_oracle_parameters_follow_digit_rule: float64 table compared with
  independent 40-digit mpmath digit formula (03 Precision/Outcome generator).
- test_calibration_matches_independent_weighted_integral: synthetic 60k
  digits in a 3:2:1 mixture, shared calibration checked by mpmath integration;
  catches incorrect equal-digit weighting (03 Outcome generator/00-common).
- test_calibration_rejects_wrong_source_cohorts_or_digits: rejects 70k,
  smoke size, float labels and out-of-range digits (03 training-only setup).
- test_outcome_keys_match_explicit_uniform_algebra: first/source-boundary/
  last-global index, independent scalar U/C inverse-CDF algebra, float32
  output fields in both processes (03 Splits and keys/Precision).
- test_outcomes_are_stable_under_subsets_jit_and_vmap: default official
  indices vs explicit reordered selection, jit/vmap agreement; public
  SurvivalData accepts generated exact/right rows and oracle likelihood is
  finite (03 outcomes, minibatch readiness). No derivative promises added:
  generation here has fixed true parameters and is outside the training loss.
- test_outcome_shapes_must_align: explicit shape errors. Existing split test
  now checks prefix anchor and host dtype in both modes.

Actual evidence:
- JAX_ENABLE_X64=1 uv run pytest -q tests/unit/test_mnist_data.py
  tests/unit/test_mnist_outcomes.py: 22 passed in 5.54s.
- Same command JAX_ENABLE_X64=0: 22 passed in 5.00s.
- Ruff lint: All checks passed! Mypy: no issues in 20 source files.
  Initial formatting check found one README code-block blank line; formatted
  it. First fresh-wheel gate stopped there, before tests; no code failure.
- Warm-cache real-data probe, JAX_ENABLE_X64=0 uv run python, loaded the
  MD5/SHA-verified four files from /private/tmp/survjax-mnist-verification.
  It called calibrate_mnist(labels[:60000]), generated all 70k once, blocked
  time array for timing, constructed public SurvivalData and explicitly
  raised on invalid issues (none). For each config/name it recorded
  mean(~outcomes.event[split_indices(config)[name]]). Recorded JSON is the
  actual probe output. Censor calibration: 2.201846s; draw including compile:
  .364040s on CPU. No dataset transfer/network was needed.
- Measured c = 6.9890429382458406. Full train/validation/test censored
  fractions .50036/.5079/.4995, smoke .5022/.499/.5045. NumPy float64 oracle
  parameters and training source counts/hash, versions, seed and timings
  recorded in calibration.json. Generated rows were not written to Git.
- Independent uv run python mpmath probe at 40 digits: sum_d count_d/60000
  * integral_0^c exp(-(t/(10*exp(-d/5)))**1.38) dt/c =
  0.500000000000193332962136588715 (within existing 1e-10 tolerance).
- scripts/check_wheel.sh --through A10: exit 0, all gates passed through A10.
  Guard, Ruff (123 formatted)/lint and mypy (20 files) passed. Own suites:
  1087 passed/7 skipped (83.64s) x64, 1093 passed/1 skipped (93.75s) x32.
  Source doctests: 4 passed per mode (1.22s/1.15s). Acceptance suites:
  2332 passed/28 skipped (100.36s) x64, 2322 passed/38 skipped (83.54s) x32.
  Fresh artifact: /var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/
  survjax-wheel.PwJRKN/survjax-0.0.1.dev0-py3-none-any.whl.

Environment: Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, macOS arm64 CPU.
No optional demo dependency is needed for this preparation step. No docs
build (no docs edits), training run, fitted-reference/evaluation or full demo.
CNN/loss/finite-gradient/checkpoint selection are the next separate steps.
Independent review follows after committing; stop after this step's verdict.
