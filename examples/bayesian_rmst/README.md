# Bayesian survival demo

A Bayesian Weibull proportional hazards model of a synthetic two-arm trial,
fitted with NumPyro, with the treatment effect reported as a difference in
restricted mean survival time. All data are synthetic. Read the tutorial on
the [docs page](../../docs/use-cases/bayesian-rmst.md), or run it cell by cell
in the [Jupyter notebook](bayesian-rmst.ipynb), which samples the posterior
itself:

```sh
uv run --group docs --group examples jupyter lab examples/bayesian_rmst/bayesian-rmst.ipynb
```

- `model.py`: the simulated trial (`make_trial`) and the NumPyro model.
- `run.py`: samples the posterior and saves the draws the docs page uses.
- `results/`: the committed run, `posterior.npz` (trial and chain-grouped
  draws) and `metadata.json` (priors, seed, versions, diagnostics, truth).

To sample again, from the repository root:

```sh
JAX_ENABLE_X64=1 uv run --group examples python -m examples.bayesian_rmst.run /tmp/survjax-bayes
```

The output directory must be new. The run checks the likelihood factor, the
agreement of the two RMST routes, the convergence diagnostics and that the true
hazard ratio and ΔRMST(5) lie inside their 95% credible intervals; it saves its
results before checking. survjax itself does not depend on the `examples`
group.
