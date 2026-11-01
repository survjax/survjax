# MNIST.15 — Readable neural demo and refreshed full experiment

Independent review: round 1/3 (MNIST.15)
MNIST.14 follow-up: round 2/3

## Read first

1. INSTRUCTION.md sections 2–5 and Process, then PLAN.md MNIST.15 entries.
2. examples/survival_mnist/model.py: three-line loss, chunked prediction,
   epoch-only training, plain float32 intercept fitting and public evaluation.
3. examples/survival_mnist/run.py: single-process pipeline and three figures.
4. Optional tests/unit/check_mnist_*.py, test_mnist_data.py and
   test_mnist_outcomes.py; results/metrics.csv, history.csv and metadata.json.

The maintainer authorized completion of all remaining steps without stops,
and subsequently approved the specific BFGS rtol=atol=3e-4 change. The fixed
full cohort stalled at 1e-5; 3e-4 succeeds at the identical float32 parameters
in six steps. PLAN.md records the failed runs and independent analytical
profile comparison (mean NLL gap 2.624e-12). No model or test tolerance tuning.

## Invariants and evidence

No src/, public library API, dependency or shared simulation changes. Fixed
seeds/cohorts/architecture/Adam/batch size/epoch count remain unchanged. All
outcomes use float32 and c=7. Only validation chooses the checkpoint; only the
oracle uses labels for predictions. The fitted intercept initializes both
CNN output bias and log shape as already approved in D-030.

- test_mnist_data/outcomes: SHA verification, IDX parsing, cohorts,
  float32 constants, independent inverse-CDF/key algebra and determinism.
- check_mnist_model: tiny-batch analytical likelihood/derivatives and CNN.
- check_mnist_training: real first smoke minibatch has finite gradients at
  initialization and after one actual _train_step (03 acceptance).
- check_mnist_metrics: hand-derived likelihood/risk/ties, final prediction
  chunk, and synchronization/warm-only timing (03 References and metrics).
- check_mnist_run: real smoke CLI and committed full acceptance (03 Acceptance).
- check_mnist_figures: first image per digit, analytical survival curves,
  digit grouping, epoch means and selected epoch (03 Figures).

Full run command: MPLCONFIGDIR=/tmp/survjax-mnist-mpl uv run --group examples
python -m examples.survival_mnist.run --config full --out
examples/survival_mnist/results. Exit 0. Selected epoch 7; CNN shape
1.368050814, within 0.87% of 1.38; CNN test LL -1.126855135 and C .689008653
beat intercept -1.237543344/.5. Oracle -1.110933185/.697534919. Cohorts
50k/10k/10k. Censoring .4998/.5074/.4994. No checkpoint/seed/test tuning.
Training 19.267s; warmed C .348/.314/.342s. These are local measurements.

An earlier full run also passed but exposed an Equinox wrapper warning;
predict now uses eqx.filter_vmap and final artifacts were regenerated.
Subsequent plotting cleanup removes redundant casts/imports; figure tests
verify the unchanged formulas. Generated SVG whitespace is intentionally
untouched; both whitespace hooks exclude results/*.svg as requested.

## MNIST.14 findings addressed

F-1: the data-only commit 7821028 knowingly leaves the old runner callers,
check_mnist_run and docs smoke broken until this step's runner rewrite.
Its eight focused data tests were not evidence that the whole demo ran.
This step repairs those callers; the real smoke CLI now passes. The docs
page's old prose/metadata readers still await MNIST.16 and are not yet usable.
F-2: new test_download_checksum_mismatch checks rejection of a bad fresh
payload, without asserting that the cache entry is absent. N-1: unused
enumerate removed. N-2: loader now names payloads rather than nesting calls.
S-1: INSTRUCTION.md explicitly directs dropping metadata fields; follow that
maintainer instruction over the older descriptive spec file listing. No
unrequested spec edits. N-3: deterministic-repeat check retained per brief.

## Checks and remaining work

Ruff check: All checks passed. Ruff format: 139 files already formatted.
mypy: Success, no issues in 21 source files. Diff whitespace passes excluding
generated SVGs. Focused checks before the final combined run: 24 passed;
smoke CLI/full-artifact acceptance: 2 passed. A temporary figure fixture
warning was removed by making its inputs float32 as intended.
No full core/wheel gate by implementer: no packaged code changed. The prior
reviewer's broad gate reported missing SURVJAX_WHEEL setup, not a code failure.
Remaining: final combined optional checks (record below), independent review,
README/docs rewrite and rendered HTML verification. No release/publishing.
Starting/previous-step commit 7821028; original worktree base ee72c83.

Final combined command: JAX_ENABLE_X64=0 MPLCONFIGDIR=/tmp/survjax-mnist-mpl
uv run --group examples pytest -q tests/unit/test_mnist_data.py
tests/unit/test_mnist_outcomes.py tests/unit/check_mnist_*.py.
Result: 28 passed in 19.71s, no warnings. All tests use the revised code.
