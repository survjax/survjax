# Survival MNIST (implementation in progress)

MNIST by Y. LeCun, C. Cortes and C. J. C. Burges, under CC BY-SA 3.0.
Dataset: http://yann.lecun.com/exdb/mnist/.
Mirror and first-fetch MD5s: [torchvision source](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py).
SHA-256 values in `data.py` were measured from all four MD5-verified compressed
IDX downloads on 30 October 2026. Every cached file is checked against SHA-256;
a mismatch raises instead of silently downloading a replacement.

From the repository root:

```python
from examples.survival_mnist.data import load_mnist, split_indices

images, labels = load_mnist()
indices = split_indices("smoke")
train_images = images[indices["train"]]
```

Downloads need network on first use. Set `SURVJAX_EXAMPLES_DATA` to the cache
directory, or use the default `~/.cache/survjax-examples/mnist`. Data stays
outside Git; do not point the cache into the repository. Images are float32
NCHW, scaled by 255 with no fitted preprocessing. Labels are reserved for
synthetic outcomes and oracle evaluation; the network will see pixels alone.
The fixed validation permutation and smoke prefixes follow use-case 03.

Outcome preparation (run before choosing smoke/full cohorts):

```python
from examples.survival_mnist.data import calibrate_mnist, generate_outcomes

c = calibrate_mnist(labels[:60000])  # float64 host computation, all source digits
outcomes = generate_outcomes(labels, c)  # all 70k, float32 in either JAX mode
# Later: SurvivalData(outcomes.time, outcomes.event)[indices["train"]]
```

`calibration.json` records the measured c, float64 oracle parameters, aggregate
source digit frequencies, source-label hash, versions, timings and realized
full/smoke censoring fractions. It is outcome-preparation evidence, before any
training or evaluation. It contains no MNIST image or label rows. Reproduce
with `JAX_ENABLE_X64=0 uv run python` and the recipes above, then compute each
fraction as `(~outcomes.event[rows]).mean()` using `split_indices(config)`.
Every image has its own fixed global-index key; generate once before slicing.
When generating reordered rows explicitly, supply their original global indices.

The float32 CNN and likelihood loss are in `model.py` (requires `examples`).
Run its offline checks explicitly in each mode:
`JAX_ENABLE_X64=0 uv run --group examples pytest -q tests/unit/check_mnist_model.py`
and repeat with `JAX_ENABLE_X64=1`. Ordinary tests need no demo dependencies.
`model.train(train_images, train_data, validation_images, validation_data,
initial_intercept=fitted_log_parameters, config="smoke")` now returns the selected parameters, NLL histories and selected
epoch. Use float32 outcome data and pixels; batch size is 128, with 1/10 epochs
for smoke/full. Check training separately with the same command above using
`tests/unit/check_mnist_training.py`. Full results and
the executable page remain to be implemented.

The intercept-only reference is `model.fit_intercept(training_time, training_event)`.
Run it in a separate `JAX_ENABLE_X64=1 uv run --group examples python` process.
It returns a NumPy float64 `[log_shape, log_scale]` array from Optimistix BFGS;
only supplied training rows enter the fit. Save the returned values, then cast
with `jnp.asarray(values, dtype=jnp.float32)` in the float32 evaluation process.
Pass the saved fit as `initial_intercept=values` to `train`; it initializes
both the CNN final bias and shared log shape, explicitly cast to float32.
The CNN constructor takes the initial log scale. No JAX global setting is changed. The fitter raises on invalid cohorts, failed
optimization or nonfinite results. Run its explicit checks with
`JAX_ENABLE_X64=1 uv run --group examples pytest -q tests/unit/check_mnist_reference.py`;
repeat with `JAX_ENABLE_X64=0` to check refusal of unavailable float64 fitting.

Evaluate a selected CNN with `model.evaluate(log_k, predicted_log_scales, data)`;
use the fitted intercept pair or the oracle's true parameters for the other rows.
The result contains `n`, `mean_log_likelihood`, `shape`, `concordance` and
`concordance_seconds` (first invocation) and `concordance_warm_seconds` (median
of three repeats). Predictions are explicitly cast to float32; observations
must already be float32. Every row is included. Higher risk is minus log scale;
a constant intercept gives C=0.5 when comparable pairs exist. Undefined or
nonfinite metrics raise. Both C timings wait for completion. All calls use one
jitted public concordance callable; the first invocation per model can include
compilation or reuse code already compiled for an earlier model. No caches are
cleared. Warm timings repeat the same inputs three times and report the median. Run offline checks
with the examples group using `tests/unit/check_mnist_metrics.py` in both modes.

Run the pipeline from the repository root:

```bash
uv run --group examples python -m examples.survival_mnist.run --config smoke --out /tmp/survjax-mnist-smoke
```

The parent launches CPU-only float64 preparation/fit, then float32 training and
evaluation, independent of its own precision setting. It writes `metrics.csv`,
`history.csv` and `metadata.json`, and prints calibration/fit values, timings,
the selected epoch and three metric rows. Use `--config full` for the fixed
ten-epoch configuration. The runner also writes `fig1.svg` (first test image
per digit with predicted/true survival), `fig2.svg` (log-scale boxplots and
truth by digit) and `fig3.svg` (NLL history and selected epoch). Plotting uses
the headless Agg backend; Matplotlib's version is recorded in metadata.
Committed full results are in `results/`; the executable
[documentation page](../../docs/use-cases/survival-mnist.md) runs smoke live.

The metadata records versions, verified source SHA-256 hashes, CPU environment,
root seed and key derivations, actual cohort sizes/censoring fractions, settings,
selected epoch (one-based), calibrated c, fitted intercept logs and float64
oracle constants.
Preparation time excludes loading MNIST; training includes compilation,
per-batch loss logging and diagnostic validation at the recorded interval.
Prediction waits for completion and includes compilation; concordance records
both first-invocation and median warm timings with the protocol above. Concordance timing uses the entire evaluation
cohort (2k smoke, 10k full). The metadata contains no image, digit-label or
prediction rows.

Convergence logging records training NLL for every minibatch and full validation
NLL about ten times per epoch, including initialization and epoch boundaries.
`progress.csv` uses completed optimizer updates: training at step s is the
minibatch loss before update s+1, while validation at s uses the model after s
updates. Empty cells mean that loss was not evaluated there. Figure 3 plots
these traces against updates. `history.csv` retains the epoch means, and only
end-of-epoch validation selects checkpoints. Extra logging does not change
training updates or make the one-epoch smoke run evidence of convergence.

## Full-run artifacts

The committed [results](results/metadata.json) were generated from implementation
commit `222341684e877153122e3a6674250987cfdab7c9`, with the locked examples
environment. CSV line endings and SVG whitespace were normalized afterwards,
as the current runner now does; numerical fields and rendered plots are unchanged.
Reproduce from the repository root:

```bash
uv run --group examples python -m examples.survival_mnist.run --config full --out examples/survival_mnist/results
```

The cohorts are 50,000 training, 10,000 validation and 10,000 official test
images. Validation
selected epoch 7 of the fixed ten epochs; test results did not choose settings.

| Model | Mean test log likelihood | Shape k | Harrell's C |
| --- | ---: | ---: | ---: |
| cnn | -1.12626493 | 1.36612070 | 0.68943512 |
| intercept | -1.23686779 | 1.18767083 | 0.50000000 |
| oracle | -1.11036825 | 1.38000000 | 0.69751966 |

The CNN beats the intercept on both metrics. Its shape is 1.01% below
1.38, within the 5% target. The oracle gaps are 0.01589668 in mean log likelihood
and 0.00808454 in C; closing those gaps is not required.

| C timing on 10,000 test rows (seconds) | First invocation | Warm median (3 repeats) |
| --- | ---: | ---: |
| cnn | 0.342211 | 0.229402 |
| intercept | 0.188547 | 0.206041 |
| oracle | 0.191947 | 0.220775 |

These timings were collected while verification suites also ran; they are not
an isolated performance comparison. Compiled code is reused across models,
so first invocation does not mean a cold process. Full environment, settings,
source hashes, censoring fractions and timing boundaries are in metadata.json.

[Metrics](results/metrics.csv), [epoch history](results/history.csv) and
[within-epoch trace](results/progress.csv) accompany the [survival panels](results/fig1.svg),
[log-scale boxplots](results/fig2.svg) and [convergence plot](results/fig3.svg).
Figure 1 includes the MNIST examples credited under CC BY-SA 3.0 above.
