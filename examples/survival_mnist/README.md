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
`tests/unit/check_mnist_training.py`. CLI, oracle/held-out metrics, full results and
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
