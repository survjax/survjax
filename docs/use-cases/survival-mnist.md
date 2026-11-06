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

# Neural network survival regression on MNIST

Survival analysis models the time until an event. For many subjects, follow-up
ends before the event happens, so their outcome is *censored*. In this tutorial
a small convolutional neural network learns a survival distribution from
handwritten digits with simulated outcomes. Because we simulated the data, we
can compare the network's predictions with the true distributions.

The same tutorial is available as a
{download}`runnable Jupyter notebook <../../examples/survival_mnist/survival-mnist.ipynb>`;
see [Reproduce](#reproduce) for how to launch it.

```{code-cell} ipython3
:tags: [remove-input]
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

import survjax

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT))
from examples.survival_mnist import data

FULL = ROOT / "examples/survival_mnist/results"
COLUMNS = {"mean_log_likelihood": "mean log likelihood", "shape": "shape k", "concordance": "Harrell's C"}
MODELS = {"cnn": "CNN", "intercept": "Intercept only", "oracle": "Oracle"}


def metrics_table(path):
    table = pd.read_csv(path, index_col="model").rename(index=MODELS)
    return table[list(COLUMNS)].rename(columns=COLUMNS).rename_axis(None).round(3)
```

## Censored data and the likelihood

For subject $i$ we observe $t_i=\min(T_i, C_i)$, the earlier of the event time
$T_i$ and the censoring time $C_i$, and the event indicator
$\delta_i=\mathbb{1}[T_i\le C_i]$. An observed event contributes the density
$f(t_i)$. A censored observation tells us only that $T_i>t_i$, and contributes
the survival function $S(t_i)=P(T_i>t_i)$. With independent, non-informative
censoring, the log likelihood is

$$
\ell(\theta)=\sum_{i}\Big[\delta_i\log f_\theta(t_i)+(1-\delta_i)\log S_\theta(t_i)\Big].
$$

In survjax, `dist.log_likelihood(data)` returns these terms, one per
observation. Here `data` is a [`SurvivalData`](../survival-data.md) container
that can be sliced like an array: `data[idx]` stays aligned with `images[idx]`.
`SurvivalData` can represent more types of survival data (left and interval
censoring, combined with left and right truncation). For right-censored data,
`dist.log_likelihood(time=time, event=event)` also works without constructing
a `SurvivalData` object first.

## A simulated world where we know the answer

Event times follow a Weibull distribution, $S(t)=\exp[-(t/b)^k]$, with shared
shape $k=1.38$ and a scale depending on the digit $d$:
$\log b_d=\log 10-0.2\,d$. Zeros survive longest and nines shortest. Censoring
times are Uniform$(0,7)$, which censors about half of the observations. The
digit labels are used only to simulate outcomes and to define an *oracle* that
knows each image's true parameters. The network sees only the pixels and the
(possibly censored) outcomes.

```{code-cell} ipython3
times = jnp.linspace(0, 20, 201)
truth = survjax.Weibull.from_log(log_shape=jnp.log(data.TRUE_SHAPE), log_scale=data.TRUE_LOG_SCALES)
fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
ax.plot(times, truth.survival_grid(times), label=[str(d) for d in range(10)])  # grid shape: (time, digit)
ax.set(xlabel="Time", ylabel="True survival $S(t)$", ylim=(0, 1))
ax.legend(title="Digit", ncols=5)
plt.show()
```

The full experiment uses all of MNIST: the official 10,000 test images, and the
60,000 training images split into 50,000 for training and 10,000 for
validation.

```{code-cell} ipython3
:tags: [remove-input]
full_meta = json.loads((FULL / "metadata.json").read_text())
pd.DataFrame(
    {"images": full_meta["sizes"], "censored fraction": full_meta["realized_censored"]}
).round(3)
```

## A Weibull model with a neural network scale

The network $g_\phi$ maps each image to its log scale, $\log b_i=g_\phi(x_i)$,
and one shape $k$ is shared by all images. On the log scale both parameters are
unconstrained, so `Weibull.from_log` takes the network output directly. The
loss is the mean negative log likelihood:

```python
def loss(params, images, data):
    cnn, log_k = params
    dist = survjax.Weibull.from_log(log_shape=log_k, log_scale=jax.vmap(cnn)(images))
    return -dist.log_likelihood(data).mean()
```

The CNN ({download}`model.py <../../examples/survival_mnist/model.py>`) has two
convolution and pooling layers followed by two dense layers. We first fit an
intercept-only Weibull by maximum likelihood. It is a baseline without image
information, and it provides the network's starting values. Adam then trains
the network and $\log k$ jointly, in minibatches of 128. The epoch with the
lowest validation loss is kept.

## Evaluation on held-out images

We compare the CNN with the intercept-only baseline and the oracle on the test
images, using two metrics where higher is better. The **mean log likelihood**
scores the whole predicted distribution, including the censored observations.
[**Harrell's C**](../concordance.md) scores only the ranking: it is the
fraction of comparable pairs in which the subject with the earlier event got
the higher risk score. Because the shape is shared, the predicted curves do
not cross, and $-\log b_i$ is a valid risk score.

### A quick run

This table is computed while the documentation is built. The run uses one
epoch on 5,000 training, 1,000 validation and 2,000 test images, so its numbers
are not representative of the full experiment.

```{code-cell} ipython3
:tags: [remove-input]
with tempfile.TemporaryDirectory(prefix="survjax-mnist-docs-") as smoke:
    subprocess.run(
        [sys.executable, "-m", "examples.survival_mnist.run", "--config", "smoke", "--out", smoke],
        cwd=ROOT,
        env=dict(os.environ, JAX_ENABLE_X64="0", JAX_PLATFORMS="cpu", MPLBACKEND="Agg"),
        check=True, capture_output=True, text=True,
    )
    smoke_metrics = metrics_table(Path(smoke) / "metrics.csv")
smoke_metrics
```

### The full experiment

Ten epochs on all training images, evaluated on the official test set; the
committed results:

```{code-cell} ipython3
:tags: [remove-input]
metrics_table(FULL / "metrics.csv")
```

```{figure} ../../examples/survival_mnist/results/fig1.svg
:alt: First test image of each digit next to predicted and dashed true survival curves.

The first test image of each digit, with the CNN's predicted survival curve and
the dashed true curve.
```

```{figure} ../../examples/survival_mnist/results/fig2.svg
:alt: Predicted log scales grouped by digit, with diamonds marking the true log scales.

Predicted log scales of all test images, grouped by the digit the network never
saw. The diamonds mark the true log scales.
```

```{figure} ../../examples/survival_mnist/results/fig3.svg
:alt: Training and validation negative log likelihood by epoch, with the selected epoch marked.

Training and validation loss by epoch. The dashed line marks the epoch kept.
```

## Discussion

In the full run, the CNN reached a mean log likelihood of −1.127 and $C=0.689$,
close to the oracle's −1.111 and 0.698, and well above the intercept-only
baseline's −1.238 and 0.5. The learned shape was 1.368, against a true value of
1.38. The intercept-only model ranks every image the same, so its $C$ is
exactly 0.5. It also underestimates the shape (1.19): short-lived digits leave
the risk set first, which makes the hazard of the mixed population rise more
slowly than that of any single digit. Once the network explains the
differences between images, the shared shape returns to its true value.

To go further, let the network predict $\log k_i$ as well; the curves may then
cross, so $C$ needs a different risk score. Or swap the CNN for any network or
tabular model that outputs the distribution's parameters. See the
[likelihood](../likelihood.md) and [concordance](../concordance.md) pages.

(reproduce)=
## Reproduce

Open the notebook from the repository root, then **Restart Kernel and Run All
Cells** in a fresh Python 3 kernel:

```bash
uv run --group docs --group examples jupyter lab examples/survival_mnist/survival-mnist.ipynb
```

Or run the full experiment from the command line. It takes about half a minute
on a laptop CPU, and writes the metrics, figures and
{download}`run metadata <../../examples/survival_mnist/results/metadata.json>`
(versions, seeds, timings):

```bash
uv run --group examples python -m examples.survival_mnist.run \
  --config full --out /tmp/survjax-mnist-full
```

The first run downloads MNIST into `~/.cache/survjax-examples/mnist`; set
`SURVJAX_EXAMPLES_DATA` to use another location.

## References

Harrell, F. E., Califf, R. M., Pryor, D. B., Lee, K. L. and Rosati, R. A.
(1982). Evaluating the yield of medical tests. *JAMA* 247(18), 2543–2546.

Kalbfleisch, J. D. and Prentice, R. L. (2002). *The Statistical Analysis of
Failure Time Data*, 2nd ed. Wiley.

MNIST is by Y. LeCun, C. Cortes and C. J. C. Burges, under CC BY-SA 3.0
([dataset](http://yann.lecun.com/exdb/mnist/)). The survival outcomes are
synthetic.
