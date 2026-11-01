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

# Teach an image model to predict survival

Any model that outputs a parameter per row can be trained with a survjax
likelihood. Here a small CNN looks at handwritten digits and learns how long
each one “survives”, with half the outcomes censored and without ever being
told which digit it is looking at. Because we invented the survival times,
we know the true curve for every image and can see how close the network gets.

The connection between the network and the survival model is just three lines:

```python
cnn, log_k = params
dist = survjax.Weibull.from_log(log_shape=log_k, log_scale=jax.vmap(cnn)(images))
return -dist.log_likelihood(data).mean()
```

## A world where we know the answer

Digit $d$ determines the Weibull scale through $\log b_d=\log 10-0.2d$,
while every digit shares shape $k=1.38$. A zero tends to survive longer than
a nine, so recognizing the image gives the network useful information about
its outcome. Censoring is Uniform$(0,7)$, so about half the outcomes are
censored: we know that they survived until observation ended, but not their
event time. The likelihood uses that information alongside the observed events.

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
full_meta = json.loads((FULL / "metadata.json").read_text())
HEADER = "| Model | Mean log likelihood | Shape k | Harrell's C |\n| --- | ---: | ---: | ---: |\n"
display(Markdown("Realized censoring is " + ", ".join(
    f"{100 * fraction:.1f}% in {split}" for split, fraction in full_meta["realized_censored"].items()
) + "."))
```

```{code-cell} ipython3
times = jnp.linspace(0, 20, 201, dtype=jnp.float32)
oracle = survjax.Weibull.from_log(
    log_shape=jnp.log(TRUE_SHAPE), log_scale=TRUE_LOG_SCALES,
)
fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
ax.plot(times, oracle.survival_grid(times), label=[str(d) for d in range(10)])
ax.set(xlabel="Time", ylabel="True survival", ylim=(0, 1))
ax.legend(title="Digit", ncols=5)
plt.show()
plt.close(fig)
```

These are the curves we hope to recover from pixels. Digit labels generate
the outcomes and supply an oracle for comparison; they never enter training.
Images are scaled to [0, 1], and each image keeps the same synthetic outcome
in both the smoke and full experiments.

## Your network, a survival likelihood

The CNN maps one image to one log scale. Two convolutional layers extract
features, and two dense layers turn those features into a scalar. To use your
own network, this scalar output is the part to preserve.

```{literalinclude} ../../examples/survival_mnist/model.py
:language: python
:pyobject: CNN
```

`jax.vmap` applies the network to a batch. The resulting vector of log scales
and a learned shared log shape go straight into `Weibull.from_log`, whose
likelihood supplies the loss:

```{literalinclude} ../../examples/survival_mnist/model.py
:language: python
:pyobject: loss
```

Minibatching is `data[idx]`: pass `images[idx]` and `data[idx]` to the loss to
keep predictions and observations aligned. Adam updates the network and shape
together, using batches of 128 and a learning rate of 0.001. After each epoch,
the validation likelihood selects the best checkpoint.

We also fit an intercept-only Weibull with BFGS. Its two shared parameters
provide both a starting point for training and a reference that cannot use
image information. All fitting, training and evaluation use float32.

## Try one epoch

The smoke run trains on **5,000 images**, selects using **1,000 validation
images**, and evaluates **2,000 test images**. The first run downloads MNIST; subsequent runs use the checked
cache. Set `SURVJAX_EXAMPLES_DATA` to choose its location.

```bash
uv run --group examples python -m examples.survival_mnist.run \
  --config smoke --out /tmp/survjax-mnist-smoke
```

The table below comes from a live execution of the smoke run.

```{code-cell} ipython3
:tags: [remove-input]
with tempfile.TemporaryDirectory(prefix="survjax-mnist-docs-") as temporary:
    smoke = Path(temporary)
    subprocess.run(
        [sys.executable, "-m", "examples.survival_mnist.run", "--config", "smoke", "--out", str(smoke)],
        cwd=ROOT,
        env=dict(os.environ, JAX_ENABLE_X64="0", JAX_PLATFORMS="cpu", MPLBACKEND="Agg"),
        check=True, capture_output=True, text=True,
    )
    with (smoke / "metrics.csv").open() as stream:
        smoke_rows = list(csv.DictReader(stream))
display(Markdown(HEADER + "".join(
    f"| {r['model']} | {float(r['mean_log_likelihood']):.3f} | {float(r['shape']):.3f} | {float(r['concordance']):.3f} |\n"
    for r in smoke_rows
)))
```

One epoch makes it easy to try the whole pipeline. The longer experiment
below shows what the network learns with more data and training.

## What the full experiment learned

The committed run uses **50,000 training, 10,000 validation and 10,000 official
test images**, with ten epochs. The test set measures the selected model;
it does not choose the checkpoint. Higher held-out log likelihood and higher
Harrell's C are better. The intercept-only reference measures performance
without image information, while the oracle knows each image's true parameters.

```{code-cell} ipython3
:tags: [remove-input]
with (FULL / "metrics.csv").open() as stream:
    full_rows = list(csv.DictReader(stream))
display(Markdown(HEADER + "".join(
    f"| {r['model']} | {float(r['mean_log_likelihood']):.3f} | {float(r['shape']):.3f} | {float(r['concordance']):.3f} |\n"
    for r in full_rows
)))
cnn, reference, truth = full_rows
display(Markdown(
    f"Validation selected **epoch {full_meta['selected_epoch']}**. "
    f"The CNN's shape is {100 * abs(float(cnn['shape']) / 1.38 - 1):.1f}% from the generating value. "
    f"Its remaining gaps to the oracle are "
    f"{float(truth['mean_log_likelihood']) - float(cnn['mean_log_likelihood']):.3f} in mean log likelihood "
    f"and {float(truth['concordance']) - float(cnn['concordance']):.3f} in C."
))
```

```{figure} ../../examples/survival_mnist/results/fig1.svg
:alt: First test image of each digit next to predicted and dashed true survival curves.

The first test image of each digit, with the CNN's predicted survival curve
and the dashed generating curve. The network has learned distinct survival
patterns from the images, even though many event times were hidden by censoring.
```

```{figure} ../../examples/survival_mnist/results/fig2.svg
:alt: Predicted log scales grouped by digit, with diamonds marking the true log scales.

All test predictions grouped by digit. The diamonds mark the generating log
scales; the spread shows how predictions vary among different images of the
same digit.
```

```{figure} ../../examples/survival_mnist/results/fig3.svg
:alt: Training and validation negative log likelihood by epoch, with the selected epoch marked.

Epoch training and validation losses. The dashed line marks the selected
checkpoint: further improvement on training data need not improve predictions
on new images.
```

```{code-cell} ipython3
:tags: [remove-input]
display(Markdown(
    "Harrell's C on 10,000 test rows took "
    + ", ".join(f"{r['model']} {float(r['concordance_seconds']):.3f} s" for r in full_rows)
    + " after compilation."
))
```

Reproduce with the command below; the {download}`run metadata <../../examples/survival_mnist/results/metadata.json>`
records versions, CPU, cohort sizes, censoring fractions and timings.

```bash
uv run --group examples python -m examples.survival_mnist.run \
  --config full --out /tmp/survjax-mnist-full
```

## Bring your own features

Swap the CNN for your own network, or start with tabular features instead of
images. A family's `from_log` constructor connects its parameters to the
network outputs; Weibull is one choice. The [likelihood API](../likelihood.md)
handles censored observations, and minibatches remain ordinary `data[idx]`
slices. Held-out likelihood and [concordance](../concordance.md) come from the
same library, so you can compare models without building a separate evaluation
pipeline. Here, shared shape makes minus log scale a risk score for Harrell's C.

MNIST is by Y. LeCun, C. Cortes and C. J. C. Burges, under CC BY-SA 3.0.
[Dataset](http://yann.lecun.com/exdb/mnist/); survival outcomes in this example are synthetic.
