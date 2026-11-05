# survjax documentation

survjax is a JAX library for survival analysis. The current implementation
can describe survival observations (`SurvivalData`), provides the exponential
and Weibull distributions and a proportional-hazards wrapper, and lets you
write your own survival
distributions, from which survjax derives survival, cumulative hazard,
hazard, density and CDF. Every distribution can be evaluated pointwise or on
a shared time grid. Exact/right-censored likelihoods without truncation are
available, along with Harrell's concordance for caller-supplied risk scores.
Restricted mean survival time (`rmst`) is available for the exponential and
Weibull families and their PH wrappers. Additional likelihood modes, families,
summaries and evaluation metrics are planned.

[Verifying the Slice A subset](slice-a-verification.md) covers compiled
derivatives, installed-wheel checks and reproducing the CPU baseline.
[Log likelihoods](likelihood.md) explains how survjax computes likelihood
contributions and how to call it, illustrated by fitting exponential and
Weibull models with JAX.
[Likelihoods in the tails](numerical-behaviour.md) explains underflow, overflow,
boundary gradients and checking that a fitting loss is finite.
[Harrell's concordance](concordance.md) evaluates risk rankings with right
censoring, including tied observations, score draws and cohort diagnostics.
[Restricted mean survival time](rmst.md) summarizes expected time before a
finite cutoff, with explicit cutoff axes and staged derivative limits.
[Survival MNIST](use-cases/survival-mnist.md) trains an image-based survival
model and compares a live smoke run with committed full results.
The [SurvivalData guide](survival-data.md) walks a synthetic cohort through
both constructors, diagnostics and R input conventions.
[Exponential and Weibull](builtin-families.md) introduces the built-in
families, their behaviour at time zero and infinity, and how their parameters
map to R's `survival` and `flexsurv` and to NumPyro.
[Proportional hazards](proportional-hazards.md) introduces the built-in wrapper,
log hazard ratios, batch and time-grid shapes, likelihoods and the
initial-release derivative limits.
[Survival curves and time grids](grids.md) evaluates batches of
distributions, including posterior draws, at many times at once. The
[custom-distribution guide](distributions.md) builds constant-hazard, Gompertz,
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
proportional-hazards
rmst
likelihood
numerical-behaviour
slice-a-verification
concordance
use-cases/survival-mnist
grids
distributions
api
```

## Local commands

From the repository root, run `scripts/build_docs.sh` to build the site and
execute its notebooks separately with JAX float32 and float64. The generated
pages are in `docs/_build/html-x32/` and `docs/_build/html-x64/`. The build
fails on warnings and notebook errors. Dependencies use the locked docs and
examples groups. Core examples use synthetic data and need no external data or
R. The MNIST page downloads hash-checked images on first use; a warm cache at
`SURVJAX_EXAMPLES_DATA` (default `~/.cache/survjax-examples/mnist`) avoids network
access on later builds.

To view the rendered site on macOS, run `open
docs/_build/html-x64/index.html` (or use `html-x32` for float32). On other
systems, open that `index.html` file in a browser.

For an editable notebook, run `uv run --locked --group docs jupyter lab
docs/playground.ipynb`. JupyterLab uses the same locked environment as the
build. Notebook outputs and generated pages are local artifacts.
