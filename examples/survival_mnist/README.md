# Neural survival analysis with MNIST

Train an image model with a survjax likelihood, then compare its predicted
survival curves with the known truth behind synthetic, censored outcomes.

From the repository root:

```bash
uv run --group examples python -m examples.survival_mnist.run \
  --config full --out /tmp/survjax-mnist-full
```

Use `--config smoke` to try one epoch on a smaller subset. The first run
downloads MNIST; later runs use the SHA-checked cache at
`~/.cache/survjax-examples/mnist` (override with `SURVJAX_EXAMPLES_DATA`).

The `--out` directory receives `metrics.csv`, `history.csv`, `metadata.json`
and three figures: predicted survival curves (`fig1.svg`), log scales by
digit (`fig2.svg`) and epoch training/validation losses (`fig3.svg`).

The committed full experiment is in [results/](results/metadata.json).
Start with the loss in [model.py](model.py), or read the tutorial on the
[docs page](../../docs/use-cases/survival-mnist.md), which runs smoke live and
shows the committed full results.

The same tutorial runs cell by cell in the [Jupyter notebook](survival-mnist.ipynb):

```bash
uv run --group docs --group examples jupyter lab examples/survival_mnist/survival-mnist.ipynb
```

Choose Python 3 and restart/run all cells. It trains the full configuration
(about half a minute on a laptop CPU); set `CONFIG = "smoke"` for a one-epoch
check. Run it inside this repository checkout.

MNIST is by Y. LeCun, C. Cortes and C. J. C. Burges, under CC BY-SA 3.0.
[Dataset](http://yann.lecun.com/exdb/mnist/); survival outcomes are synthetic.
