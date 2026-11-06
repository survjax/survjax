# Instruction: rework the survival MNIST example as a showcase

From the maintainer, 2026-11-01. Scope: only the survival MNIST example
(`examples/survival_mnist/`, its README, its tests, and
`docs/use-cases/survival-mnist.md`), plus the spec change in section 1.

## Goal

This example is a showcase that should make readers want to try survjax. It is
not reference code. A reader should be able to read `model.py` in two minutes
and think "I can put my own network in there." The library is already
hardened, so trust it. Do not re-validate in the example what survjax or JAX
already checks, and do not guard against misuse of demo code.

Keep the spec's acceptance checks (finite gradients at initialization and
after step 1, CNN beats intercept-only, learned k within 5% of 1.38, cohort
sizes stated) as **tests**, not as code in the narrative.

Size targets: `model.py` about 100 lines, `data.py` about 80, `run.py` about
120 (currently 356, 222 and 343).

Maintainer decisions behind this instruction:

- **Everything runs in float32**, which is what users expect in a neural
  network demo. The example never enables x64.
- **No censoring calibration.** Use the fixed limit c = 7. The calibrated
  value was 6.989, and both give about 50% censoring.

## 1. Spec change (first commit, on its own, in the design repo)

The maintainer asks you to make this change. Follow the spec-change procedure
in `CLAUDE.md`.

`spec/use-cases/03-survival-mnist-details.md`:

- **Precision** (lines 7–10): replace the paragraph with "Everything runs in
  JAX's default float32, as is usual for neural training. This is an explicit
  exception to the float64 default in 00-common.md. The example never enables
  x64." No mention of calibration or float64 may remain.
- **Outcome generator** (lines 46–48): replace the calibration bullets with
  "Censoring C ~ Uniform(0, 7), which gives about 50% censoring." Delete
  "Record c". Keep "Report the realized censored fraction for each split".
- **References and metrics** (line 85): "fitted ... by Optimistix BFGS in
  float64. Evaluate it in float32." becomes "fitted ... by Optimistix BFGS."
- **Implementation steps** (line 130): drop "and the calibrated `c`".

`spec/use-cases/03-survival-mnist.md`, Design table, "Outcomes" row: "about
50% Uniform(0, c) censoring" becomes "Uniform(0, 7) censoring (about 50%
censored)".

Bump the `SPECIFICATION.md` revision line, because a use-case requirement
changed. Commit with `git -C spec ...`, run `../scripts/spec-since.sh dev --mark`,
and commit `.spec-seen` here.

## 2. `data.py`

- `_fetch`: if the file is cached, read it; otherwise download it and write
  it. Then check SHA-256 and raise one clear error on mismatch. Drop the MD5
  path (it served its purpose on the first fetch) and the tempfile/atomic-rename
  code.
- `_parse_idx`: replace with
  `np.frombuffer(gzip.decompress(payload), np.uint8, offset=16 or 8).reshape(...)`.
  The hash already guarantees the format. Drop the header and label checks.
- Delete `calibrate_mnist` and the import of `calibrate_censoring`. Add
  `CENSOR_LIMIT = 7.0` next to `TRUE_SHAPE` and `TRUE_LOG_SCALES`.
- The oracle constants become float32 JAX arrays. No float64 host constants,
  and no "compute in float64, cast" comments.
- `generate_outcomes(labels)` uses `CENSOR_LIMIT`. Remove the `censor_limit`
  and `global_indices` parameters: it always generates all 70k rows in
  official order, which is how the example uses it.
- `split_indices`: drop the config validation.
- The example still imports `generate_weibull` from `weibull_simulation`.
  Leave `weibull_simulation` itself untouched.
- Delete `examples/survival_mnist/calibration.json`.

## 3. `model.py`

- Delete `_require_finite`, `_value_grad`, `_batch_loss`, the `first_step`
  logic, and every `FloatingPointError`/`ValueError`/`RuntimeError` raise.
- `train()`: remove the config validation, the shape/dtype/cohort checks, the
  intercept finiteness check and all `np.isfinite` checks. Replace
  `config: str` with `epochs: int`. Remove `log_every_batches`, `batch_nll`,
  `validation_progress` and `progress.csv` completely. `TrainingResult` keeps
  only `params`, `training_nll`, `validation_nll` and `selected_epoch`. The
  loop should read as: per epoch, permute; for each batch, step; then epoch
  mean, validation NLL, keep the best. About 25 lines.
- Add a plain `predict(cnn, images)` that runs the CNN in chunks of 1000 and
  concatenates the results. Validation NLL then becomes
  `-Weibull.from_log(log_shape=log_k, log_scale=predict(cnn, images)).log_likelihood(data).mean()`,
  which shows the library in use. `_validation_nll` goes away, and `run.py`
  uses `predict` for the test set too.
- `fit_intercept()`: plain float32 BFGS. Remove the x64 check, the shape
  checks, the event-count check, the solver-result check and the post-hoc
  finiteness check. Use the default `throw=True` of `optx.minimise`. About 8
  lines. Keep the same rtol/atol unless the fit fails to converge in float32.
  If it fails, report back; do not add workarounds.
- `evaluate()`: remove all validation and nonfinite checks. Remove
  `_CONCORDANCE_WARM_REPETITIONS` and the median-of-3 warm timing. Call C once
  to compile, time one further call, and report that as
  `concordance_seconds`. Return `n`, `mean_log_likelihood`, `shape`,
  `concordance` and `concordance_seconds`.
- `CNN`: drop the "use fold_in(fold_in(...))" key instruction from the
  docstring. Keep everything else.
- Docstrings: one-line summaries. No Args/Returns/Raises blocks in example
  code (that convention is for the library's public API). Remove
  process-citation comments such as `# 03 Network dimensions require…` and
  `# 03 Splits and keys…`. Keep a comment only if a curious user would want
  it.

## 4. `run.py`

- Delete the two-subprocess orchestration, `_prepare` and the preparation
  JSON. There is no `JAX_ENABLE_X64` anywhere. One `main()` reads top to
  bottom: load, generate outcomes, split, fit the intercept, train, predict
  the test set, evaluate cnn/intercept/oracle, write the outputs, draw the
  figures.
- `metadata.json` keeps `config`, `sizes`, `realized_censored`,
  `selected_epoch`, training seconds, concordance seconds per model,
  versions, and python/platform/cpu. Drop `censor_limit` (it is a constant
  in `data.py`), `keys`, `timing_notes`, `concordance_benchmark`,
  `precision`, `architecture`, `source_sha256`, `selected_epoch_indexing`
  and `drop_partial_batch`.
- fig3: training and validation NLL by epoch only, with the selected epoch
  marked. No minibatch trace.
- Delete the SVG trailing-whitespace rewrite. Instead, exclude
  `examples/survival_mnist/results/*.svg` from the whitespace hooks in
  `.pre-commit-config.yaml`.
- Drop the "figures require a test image for every digit" check.

## 5. Tests

- Delete the tests that only exercise removed guards: `check_mnist_progress.py`
  entirely, plus the raise/validation cases in the other `check_mnist_*`
  files. Drop the x64-mode variants: the example is float32 only, and the
  mpmath/float64 checks belong to the library.
- Keep: download hash and parsing, split sizes and determinism, outcome
  determinism, the loss on a tiny batch, the finite-gradient acceptance test
  (gradients at initialization and after one `_train_step` on a real smoke
  batch, in `check_mnist_training.py`), the hand-derived `evaluate()` cases,
  and the smoke run.

## 6. `README.md`

Rewrite it from scratch, about 25 lines: what the example shows (one
sentence), the one command to run it, what lands in `--out`, where the
committed results are, and the MNIST attribution. Remove "(implementation in
progress)" and all process history: MD5 dates, calibration.json provenance,
per-mode pytest commands, and any mention of float64 or calibration.

## 7. `docs/use-cases/survival-mnist.md` (a prose rewrite, not a light edit)

- Write a new intro that leads with the payoff. Draft to adapt:

  > Any model that can output a parameter per row can be trained with a
  > survjax likelihood. On this page a small CNN looks at handwritten digits
  > and learns how long each one "survives", with half the outcomes censored
  > and without ever being told which digit it is looking at. Because we
  > invented the survival times, we know the true curve for every image and
  > can see exactly how close the network gets.

  Then show the three-line loss right there as a teaser.
- Keep the spec's section order (generating rule, model and loss, live smoke
  run, full results, takeaways). Write connected prose that explains why each
  part is interesting, instead of strings of short declarative sentences
  stacked with qualifiers.
- In the generating rule, write "censoring is Uniform(0, 7), so about half
  the outcomes are censored", and give the realized fractions per split in
  one sentence. Delete the "Calibrated censoring limit c = …" line and every
  mention of calibration and float64.
- Cut the implementation-process caveats: "No global precision setting is
  changed in the library", tie-breaking, "Extra within-epoch validation is
  diagnostic", "Gradients must be finite…", "independent of this notebook's
  parent dtype", "writes only temporary output…", "C = 0.5 when comparable
  pairs exist", the step-s/update-s+1 CSV semantics, blank CSV cells, and
  timings "collected while verification suites also ran".
- Delete the minibatch fig3 figure and its caption. The epoch-summary plot is
  the only training plot: show the new `fig3.svg` with a `{figure}` directive
  and delete the inline `history.csv` plotting cell. Remove the
  `progress.csv` link.
- Collapse "Evaluation timing and reproducibility" into two sentences: the
  time for Harrell's C on 10,000 rows, then the reproduce command and the
  metadata download link.
- Tables: 3 decimals, not 6 or 8.
- Add the missing takeaways section (item 5 of the page order in the spec
  details): swap in your own network or features; any family's `from_log`
  works; minibatching is `data[idx]`; held-out likelihood and C come from the
  same library. Link the likelihood and concordance pages.
- Keep the attribution, shortened to two lines.

## Process

This is well over 300 lines, so split it into steps:

1. the spec change (design repo commit only);
2. `data.py` and its tests;
3. `model.py`, `run.py` and their tests; re-run the full experiment and
   recommit the results and figures. Expect the numbers to shift a little,
   but the acceptance checks must still pass. If k misses the 5% target,
   report it; do not tune;
4. `README.md` and the docs page, checking the rendered HTML.

Write the `PLAN.md` sections as usual and review each step.
