# survjax documentation

survjax is a JAX library for survival analysis. The current implementation
can describe survival observations (`SurvivalData`), provides the exponential
and Weibull distributions, and lets you write your own survival
distributions, from which survjax derives survival, cumulative hazard,
hazard, density and CDF. Every distribution can be evaluated pointwise or on
a shared time grid. Exact/right-censored likelihoods without truncation are
available; other likelihood modes, families, summaries and evaluation metrics
are planned; the Slice A completion gate has not yet been
met.

[Log likelihoods](likelihood.md) explains how survjax computes likelihood
contributions and how to call it, illustrated by fitting exponential and
Weibull models with JAX.
The [SurvivalData guide](survival-data.md) walks a synthetic cohort through
both constructors, diagnostics and R input conventions.
[Exponential and Weibull](builtin-families.md) introduces the built-in
families, their behaviour at time zero and infinity, and how their parameters
map to R's `survival` and `flexsurv` and to NumPyro.
[Survival curves and time grids](grids.md) evaluates batches of
distributions, including posterior draws, at many times at once. The
[distributions guide](distributions.md) builds constant-hazard, Gompertz,
piecewise-constant and proportional-hazards models from their survival
functions. The [playground](playground.ipynb) uses only synthetic data and the
current public API. The [API reference](api.md) lists every public class and
function with its docstring. Runnable examples execute afresh when the site
is built.

```{toctree}
:maxdepth: 1

playground
survival-data
builtin-families
likelihood
grids
distributions
api
```

## Local commands

From the repository root, run `scripts/build_docs.sh` to build the site and
execute its notebook separately with JAX float32 and float64. The generated
pages are in `docs/_build/html-x32/` and `docs/_build/html-x64/`. The build
fails on warnings and notebook errors. The examples need no network access,
R installation or external data; dependency installation uses the lockfile.

To view the rendered site on macOS, run `open
docs/_build/html-x64/index.html` (or use `html-x32` for float32). On other
systems, open that `index.html` file in a browser.

For an editable notebook, run `uv run --locked --group docs jupyter lab
docs/playground.ipynb`. JupyterLab uses the same locked environment as the
build. Notebook outputs and generated pages are local artifacts.
