Independent review: round 1/3

# MNIST.10 — Within-epoch convergence trace

Read PLAN.md MNIST.10 and the maintainer request for more frequent losses;
model.py train/TrainingResult, run.py progress.csv and Figure 3, then the new
progress tests and existing runner/figure checks. User explicitly authorized
denser logging; epoch summaries and checkpoint-selection semantics stay intact.

Invariants:
- Same optimizer, step count, epoch count, keys, batches and selected checkpoint
  rule. Additional validation is diagnostic only. Test data/labels never enter
  training or checkpoint selection. No tuning based on test results.
- Each batch_nll is the existing pre-update minibatch loss. Integer step s means
  s completed updates; its training value is evaluated before update s+1.
- Full-validation diagnostics are at initialization, every requested interval
  within an epoch and the epoch boundary, even if the interval does not divide
  the epoch. Reuse that final validation for the existing epoch checkpoint rule.
- CLI interval is ceil(batches_per_epoch/10), at least one. Smoke has 39 training
  points and 11 validation points (steps 0, 4, ..., 36, 39); full has 3900 training
  and 101 validation points (0, then every 39 updates through 3900).
- history.csv preserves epoch means and validation values. progress.csv records
  integer step, raw training_batch_nll and sparse validation_nll; blank cells
  mean not evaluated, not zero. The last row has validation but no training.
- Figure 3 uses optimizer updates, labels noisy minibatch loss separately from
  full validation, and marks selected_epoch*batches_per_epoch. Epoch-only results
  without validation_progress still have the old figure fallback.
- Fail on invalid intervals or nonfinite diagnostics; no clipping or smoothing.
  New optional fields have empty defaults for existing result construction.
- No new dependency/module/core API/protected path/spec edit. D-031's timing
  behavior stays labelled first-call-inclusive; no warm timing decision made.
  Files grew modestly for this user-requested instrumentation (model.py 340,
  run.py 325 lines); the change itself is within about 300 implementation lines.

Tests:
- Real Adam with and without extra validation produces identical parameter
  leaves, epoch summaries and selected epoch on the same fixed synthetic cohort.
- Controlled three-batch run pins initial/interval/final logging (0, 2, 3),
  batch losses (1, 2, 3) and epoch mean 2. Intermediate validation .1 is better
  than endpoint 3 but must not select the intermediate model.
- Reject zero/negative/bool/fractional intervals and nonfinite initial/periodic
  validation. Existing epoch failure and gradient checks remain.
- Runner check pins logging interval, progress.csv steps/blank cells and metadata.
  Figure test pins dense x/y series and selected update 4, rather than epoch 2.
  Analytic prediction/truth, grouping and headless/closure checks remain.

Validation and visual QA:
- Initial combined run had four figure-fixture setup errors because its old
  SimpleNamespace lacked new trace fields. Added actual dense fixture values;
  focused figure/runner rerun: 9 passed. No production fallback or skip added.
- Nine-file optional suites: x64 90 passed, 1 skipped (60.42 s); float32
  68 passed, 23 skipped (49.24 s). Ruff lint passed; cumulative gate below.
- Actual fixed smoke at /private/tmp/survjax-mnist-convergence-smoke10:
  verified 40 CSV rows, 39 batch losses, 11 full-validation samples, and exact
  epoch-end CSV agreement. Validation starts at 1.2610878 and ends at 1.2119510.
  Smoke metrics reproduce the prior version exactly. No optimization change.
- Actual fixed full diagnostic at /private/tmp/survjax-mnist-convergence-full10:
  3901 CSV rows, 3900 batch losses, 101 full-validation points; every epoch-end
  value equals history.csv, and earliest best epoch is 7 in history and metadata.
  Ten epochs on 50k train/10k validation, then selected model on 10k test.
- Full validation starts at 1.2262992, falls to 1.1085411 after epoch 2, reaches
  best endpoint 1.1009218 at epoch 7, then rises to 1.1097243 at epoch 10.
  Epoch-mean training NLL continues falling from 1.1490074 to 1.0936650.
- Selected test metrics: CNN mean LL -1.12626493, k 1.36612070, C .68943512;
  intercept mean LL -1.23686779, k 1.18767083, C .5; oracle mean LL -1.11036825,
  k 1.38, C .69751966. Shape is about 1.01% below truth; CNN beats intercept.
  These are one fixed diagnostic run, not tuned/committed release artifacts.
- Rendered smoke Figure 3 and all three full SVGs with rsvg-convert and inspected
  with view_image. Full headline curves are much closer to truth. Dense Figure 3
  shows batch noise and the validation plateau/deterioration with selected epoch
  marked. Boxplots/markers are legible. No further training chosen from plots.
- Full training took 84.79 s under concurrent checks; C first-call-inclusive
  timings at 10k are .6724 s CNN, .3565 s intercept, .3711 s oracle. D-031 remains
  open before any committed full results. Smoke/full outputs stay in /tmp.

No R generation or docs build. Remaining: settle full-run timing, committed
release artifacts and executable docs after this step's review/handoff.
Previous step final: aadbb14b2e75bb2a0589691c75614b236e3322d5.
Original worktree base: ee72c837549d14cb26e9983bf6d63ac54563f189.

Cumulative check: scripts/check_wheel.sh --through A10 exited 0, all gates
passed. Ruff format: 138 files already formatted; Ruff lint passed; mypy:
no issues in 21 source files. Own suites: x64 1119 passed, 7 skipped (108.03 s);
float32 1125 passed, 1 skipped (92.63 s). Doctests: 4 passed per mode.
Acceptance: x64 2333 passed, 27 skipped (85.30 s); float32 2324 passed,
36 skipped (72.91 s). Final format and git diff --check clean.
