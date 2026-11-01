---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  render_markdown_format: myst
---

# Neural survival analysis with MNIST

A CNN reads a handwritten digit image and predicts its Weibull log scale.
One shared shape parameter is learned with it. The training loss is a survjax
log likelihood, so right-censored outcomes enter the same minibatch loss as
observed events. This is an illustrative synthetic survival task.

## Known generating curves

Digit $d$ determines $\log b_d=\log 10-0.2d$, with shared shape $k=1.38$.
Event times are Weibull; independent censoring is Uniform$(0,c)$. Calibrating
$c$ against the digit frequencies in the 60,000 source training images gives
about 50% censoring. Labels generate outcomes and define the oracle reference;
the CNN receives pixels alone.

```{code-cell} ipython3
:tags: [remove-input]
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

import survjax

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT))
from examples.survival_mnist.data import TRUE_LOG_SCALES, TRUE_SHAPE

FULL = ROOT / "examples/survival_mnist/results"
HEADER = "| Model | Mean log likelihood | Shape k | Harrell's C |\n| --- | ---: | ---: | ---: |\n"
```

```{code-cell} ipython3
times = jnp.linspace(0, 20, 201, dtype=jnp.float32)
oracle = survjax.Weibull.from_log(
    log_shape=jnp.log(jnp.float32(TRUE_SHAPE)),
    log_scale=jnp.asarray(TRUE_LOG_SCALES, dtype=jnp.float32),
)
fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
ax.plot(times, oracle.survival_grid(times), label=[str(d) for d in range(10)])
ax.set(xlabel="Time", ylabel="True survival", ylim=(0, 1))
ax.legend(title="Digit", ncols=5)
plt.show()
plt.close(fig)
```

Each image has a fixed outcome key derived from its official global index.
Outcomes therefore stay the same across the smoke and full splits. Pixels are
float32 in [0, 1], shaped `(N, 1, 28, 28)`, with no fitted normalization.

## Model and likelihood

The Equinox CNN below maps one image to one scalar log scale. `jax.vmap`
applies it to a minibatch. A float64 BFGS intercept-only Weibull fit on training
rows initializes the output bias and shared log shape; both are cast to
float32 for neural training and evaluation. Calibration and stored oracle
constants use float64. No global precision setting is changed in the library.

```{literalinclude} ../../examples/survival_mnist/model.py
:language: python
:pyobject: CNN
```

The likelihood needs per-image parameters and the aligned observations;
it does not need an adapter for the network. Exact events and right censoring
are handled by the public [likelihood API](../likelihood.md).
Minibatches are just `data[idx]`: the training loop passes `images[idx]` and
`data[idx]` directly to `loss`, keeping pixels and observations aligned.

```{literalinclude} ../../examples/survival_mnist/model.py
:language: python
:pyobject: loss
```

Adam uses learning rate 0.001 and batches of 128, dropping a partial training
batch. Validation covers every row. After each epoch, the lowest validation
NLL selects the checkpoint; ties keep the earlier epoch. Extra within-epoch
validation is diagnostic and never selects a checkpoint. Gradients must be
finite initially and after the first update or the run fails.

## Live smoke run

This cell executes one epoch on **5,000 training, 1,000 validation and 2,000
test images** during each docs build. It uses the same keys and prefix subsets
as the full experiment. The runner launches float64 preparation/fitting and
float32 training/evaluation in separate CPU processes, independent of this
notebook's parent dtype. First use needs a download; a hash-verified warm cache
avoids network access. Set `SURVJAX_EXAMPLES_DATA` to choose the cache directory.

```{code-cell} ipython3
with tempfile.TemporaryDirectory(prefix="survjax-mnist-docs-") as temporary:
    smoke = Path(temporary)
    completed = subprocess.run(
        [sys.executable, "-m", "examples.survival_mnist.run", "--config", "smoke", "--out", str(smoke)],
        cwd=ROOT, env=dict(os.environ, MPLCONFIGDIR=str(smoke / "matplotlib")),
        check=False, capture_output=True, text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"MNIST smoke failed ({completed.returncode}):\n{completed.stdout}\n{completed.stderr}")
    with (smoke / "metrics.csv").open() as stream:
        smoke_rows = list(csv.DictReader(stream))
display(Markdown("**Live smoke result — one epoch, 2,000 test images.**"))
display(Markdown(HEADER + "".join(
    f"| {r['model']} | {float(r['mean_log_likelihood']):.6f} | {float(r['shape']):.6f} | {float(r['concordance']):.6f} |\n"
    for r in smoke_rows
)))
```

Smoke checks that the pipeline runs; one epoch is not evidence of convergence
or the full-run shape target. It writes only temporary output and leaves the
committed full experiment untouched.

## Committed full experiment

These results were measured separately on **50,000 training, 10,000 validation
and 10,000 official test images**, with ten fixed epochs and no tuning or seed
selection on test results. The intercept-only reference uses no image information;
the oracle uses the true parameters associated with the hidden digit label.
Higher mean log likelihood and C are better. With shared shape, minus log scale
ranks risk in the same direction at every time, so it supplies the score for C.
The constant intercept receives C = 0.5 when comparable pairs exist.
All three C values use the public
[`survjax.concordance_index`](../concordance.md), alongside public test likelihoods.

```{code-cell} ipython3
:tags: [remove-input]
full_meta = json.loads((FULL / "metadata.json").read_text())
display(Markdown(
    f"Calibrated censoring limit **c = {full_meta['censor_limit']:.6f}**. Realized censoring: "
    + ", ".join(f"{split} {100 * fraction:.2f}%" for split, fraction in full_meta["realized_censored"].items()) + "."
))
with (FULL / "metrics.csv").open() as stream:
    full_rows = list(csv.DictReader(stream))
display(Markdown(HEADER + "".join(
    f"| {r['model']} | {float(r['mean_log_likelihood']):.8f} | {float(r['shape']):.8f} | {float(r['concordance']):.8f} |\n"
    for r in full_rows
)))
cnn, reference, truth = full_rows
shape_error = abs(float(cnn["shape"]) / full_meta["oracle"]["shape"] - 1)
display(Markdown(
    f"Validation selected **epoch {full_meta['selected_epoch']}**. Learned shape is "
    f"{100 * shape_error:.2f}% from 1.38 (target: within 5%). The oracle gaps are "
    f"{float(truth['mean_log_likelihood']) - float(cnn['mean_log_likelihood']):.6f} "
    f"in mean log likelihood and {float(truth['concordance']) - float(cnn['concordance']):.6f} in C."
))
```

```{figure} ../../examples/survival_mnist/results/fig1.svg
:alt: First official test image per digit beside predicted and dashed true survival curves.

Committed full run: first official test image for each digit, with the selected
CNN's survival curve and the dashed generating curve. These are examples, not
images chosen for agreement.
```

```{figure} ../../examples/survival_mnist/results/fig2.svg
:alt: CNN log-scale boxplots for every true digit, with true generating log-scale diamonds.

Committed full run: all test predictions grouped by true digit. Diamonds mark
the generating log scales. The network learned this ordering from pixels.
```

```{figure} ../../examples/survival_mnist/results/fig3.svg
:alt: Per-minibatch training loss and periodic full-validation loss across 3900 updates.

Committed full run: each training minibatch and about ten full validations per
epoch. The dashed line marks the selected epoch boundary. Minibatch noise is
substantial; the validation trace is the orange line.
```

The same run's epoch summaries give a quieter view of the training trend:

```{code-cell} ipython3
:tags: [remove-input]
with (FULL / "history.csv").open() as stream:
    history = list(csv.DictReader(stream))
fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
epochs = [int(r["epoch"]) for r in history]
ax.plot(epochs, [float(r["training_nll"]) for r in history], "o-", label="Training epoch mean")
ax.plot(epochs, [float(r["validation_nll"]) for r in history], "s-", label="Full validation")
ax.axvline(full_meta["selected_epoch"], color="black", linestyle="--", label="Selected epoch")
ax.set(xlabel="Epoch", ylabel="Negative log likelihood", xticks=epochs)
ax.legend()
plt.show()
plt.close(fig)
```

Training loss can keep falling after validation stops improving. Keeping the
best validation checkpoint avoids using the final epoch automatically. The CNN
beats the intercept on both test metrics and meets the full-run shape target;
the remaining oracle gap is reported rather than required to disappear.

### Evaluation timing and reproducibility

The full run measures C on 10,000 test rows. First invocation can include
compilation, or reuse compiled code from an earlier model. Warm time is the
median of three synchronized repetitions on the same inputs. Timings were
collected while verification suites also ran, so they are not an isolated
speed comparison between models.

```{code-cell} ipython3
:tags: [remove-input]
timing_header = "| Model | First C call (s) | Warm median (s) |\n| --- | ---: | ---: |\n"
display(Markdown(timing_header + "".join(
    f"| {r['model']} | {float(r['concordance_seconds']):.6f} | {float(r['concordance_warm_seconds']):.6f} |\n"
    for r in full_rows
)))
```

The {download}`run metadata <../../examples/survival_mnist/results/metadata.json>` records
versions, CPU environment, verified source hashes, root seed 20261025, key
derivations, settings, cohort sizes and realized censoring. The
{download}`epoch history <../../examples/survival_mnist/results/history.csv>` and
{download}`within-epoch trace <../../examples/survival_mnist/results/progress.csv>` retain
the scalar losses. At step s, training loss is before update s+1; validation
is after s completed updates. Blank CSV cells mean not evaluated there.

Reproduce from the repository root:

```bash
uv run --group examples python -m examples.survival_mnist.run --config full --out /tmp/survjax-mnist-full
```

MNIST is by Y. LeCun, C. Cortes and C. J. C. Burges, under CC BY-SA 3.0.
[Dataset](http://yann.lecun.com/exdb/mnist/); the loader verifies the four raw
IDX downloads from its stable mirror against recorded SHA-256 hashes.
No raw MNIST data is committed. This page uses synthetic outcomes and provides
no evidence about clinical prediction performance.
