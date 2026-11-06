# Weibull survival regression and simulation

Start with the [tutorial](../../docs/use-cases/weibull-simulation.md) or open
[the notebook](weibull-simulation.ipynb) from the repository root:

```bash
uv run --group docs --group examples jupyter lab examples/weibull_simulation/weibull-simulation.ipynb
```

Restart the Python 3 kernel and run all cells. The notebook simulates one
trial, fits a regression, computes a confidence interval, and runs a study.
Choose `CONFIG = "smoke"` for a quick run or `"full"` for 12,000 fits.
The notebook writes its study to a temporary directory and displays fresh
figures. The docs build runs smoke and displays the saved full study.

To save a study in a new directory:

```bash
uv run --group examples python -m examples.weibull_simulation.run \
  --config full --out /tmp/my-weibull-study
```

- `study.py`: likelihood and fitting, plus generation helpers shared by the
  Bayesian and MNIST examples.
- `run.py`: generate batches, fit both models, summarize estimates, and save
  tables and metadata. Rejected fits remain counted and listed.
- `_plots.py`: the survival, sampling-distribution and coverage figures.
- `check_demo.py`: two small checks against an analytic exponential fit and
  a trial with no events in one arm.
- `results/`: the saved full study used by the documentation.

All data are synthetic. Fitting uses float64 and requires the optional
`examples` dependencies; these are not dependencies of survjax itself.
