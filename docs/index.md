---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Survival analysis in JAX

survjax supplies a survival data container, survival distributions, censored-data
likelihoods, survival curves and evaluation tools that compose with JAX. You
supply the model's parameters—directly, through a regression, from a neural network or as
posterior draws—and choose how to fit them.

`SurvivalData` and distributions are JAX pytrees, so you can pass them through
transformations such as `jit` and `vmap`. survjax is lightweight on purpose:
its only core dependencies are JAX and NumPy. Optimizers, neural networks and
Bayesian inference libraries are optional tools you compose with it.

```{code-cell} ipython3
import survjax

data = survjax.SurvivalData([4.0, 12.0, 7.0], event=[True, False, True])
model = survjax.Weibull(shape=1.5, scale=12.0)
model.log_likelihood(data)  # one contribution per observation
```

Start with [survival data](survival-data.md) and
[working with distributions](working-with-distributions.md), then use
[log likelihoods](likelihood.md) to build a fitting objective.
Three documented use cases show how to [fit a Weibull distribution](use-cases/weibull-simulation.md) - or a whole lot of them in a simulation study,
using survival likelihoods in neural networks on [MNIST](use-cases/survival-mnist.md), and how to do [Bayesian survival analysis](use-cases/bayesian-rmst.md) with survjax.
Each of the use-cases include executable notebooks that you can run in your own environment.

This pre-release supports exact events and right censoring without truncation
for likelihoods. Exponential and Weibull distributions, proportional hazards,
Harrell's concordance and RMST for those built-in families are available.
Left/interval censoring and truncation can be represented as data, but their
likelihoods are not implemented. Each guide states its current limitations.

```{toctree}
:maxdepth: 1
:caption: Survival data

survival-data
```

```{toctree}
:maxdepth: 1
:caption: Survival distributions

working-with-distributions
likelihood
builtin-families
proportional-hazards
distributions
numerical-behaviour
```

```{toctree}
:maxdepth: 1
:caption: Derived quantities and metrics

grids
rmst
concordance
```

```{toctree}
:maxdepth: 1
:caption: Use cases

use-cases/weibull-simulation
use-cases/survival-mnist
use-cases/bayesian-rmst
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
```

```{toctree}
:maxdepth: 1
:caption: Technical notes

technical-notes/parameter-conventions
technical-notes/custom-distributions
technical-notes/likelihood-diagnostics
technical-notes/rmst
technical-notes/mathematics
technical-notes/verification
```

For exact signatures, see the [API reference](api.md). The technical notes
contain extension details, numerical checks and reproducibility instructions;
they are optional reading for using the main guides.

To build these docs from the repository, run `scripts/build_docs.sh`.
See [Verification and reproducibility](technical-notes/verification.md#build-and-explore-the-documentation)
for output locations, data caching and the JupyterLab command.
