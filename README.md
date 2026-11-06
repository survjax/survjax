# survjax

<!-- [![GPU CI on NVIDIA T4 / CUDA 12: latest manual run on main](https://github.com/survjax/survjax/actions/workflows/gpu.yml/badge.svg?branch=main&event=workflow_dispatch)](https://github.com/survjax/survjax/actions/workflows/gpu.yml) -->

**Survival analysis in JAX**

`survjax` is a lightweight, extensible JAX library for survival likelihoods,
predictions, and evaluation. Use it to build objectives for classical,
neural-network, and Bayesian survival models, or to run simulation studies
with JAX's automatic differentiation, compilation, and batching.

The library provides survival-data containers, distributions, and metrics.
Fitting routines, neural-network architectures, and priors belong to external
libraries such as Optimistix, Optax, Equinox, and NumPyro.

## Current scope

`survjax` is an alpha pre-release. Its first built-in distributions are
`Exponential` and `Weibull`, and its first evaluation metric is Harrell's
`concordance_index`. More distribution families and metrics will follow.

Currently available:

- **Survival data:** `SurvivalData` represents mixtures of exact, right-, left-,
  and interval-censored observations, with left and/or right truncation.
- **Likelihoods:** `model.log_likelihood(data)` returns per-observation log
  likelihoods for exact and right-censored observations **without truncation**.
  Left/interval censoring and truncation are representable but their likelihoods
  are not yet implemented; unsupported rows return NaN.
- **Survival and hazard calculations:** survival probabilities, log survival,
  log density, log CDF, hazard, log hazard, and cumulative hazard, with grid
  methods for evaluation across times.
- **Proportional hazards:** `ProportionalHazards` wraps a baseline with a
  caller-computed log hazard ratio; see the [PH guide](https://github.com/survjax/survjax/blob/main/docs/proportional-hazards.md)
  for broadcasting, likelihoods and derivative limits.
- **Evaluation:** Harrell's concordance for exact and right-censored data
  without truncation.
- **Custom distributions:** register a family with parameter declarations,
  `log_survival(t)`, and `reference_time()`; the base supplies derived methods,
  including an autodiff-based log density.

Restricted mean survival time, `model.rmst(tau)`, is available for Exponential,
Weibull and their PH wrappers; see the [RMST guide](https://github.com/survjax/survjax/blob/main/docs/rmst.md) for cutoff
broadcasting, derivatives and numerical-failure limits. Unrestricted means,
medians and quantiles remain planned.

## Installation

- using uv:

```sh
uv add survjax
```

- using PyPI:

```sh
pip install survjax
```

- from source:

From a local checkout, install the core package with:

```sh
pip install .
```

Python 3.12 or later is required. The core depends only on JAX and NumPy;
install optimization and inference libraries separately when needed.

## Minimal example

Construct observations, evaluate a likelihood, and differentiate it with
respect to a model parameter:

```python
import jax
import jax.numpy as jnp
import survjax

data = survjax.SurvivalData(
    time=jnp.array([1.0, 2.0, 1.5, 5.0]),
    event=jnp.array([True, False, False, True]),
)
# True denotes an observed event; False denotes right censoring.


def loss(log_rate):
    model = survjax.Exponential.from_log(log_rate=log_rate)
    return -model.log_likelihood(data).sum()


value_and_grad = jax.jit(jax.value_and_grad(loss))
log_rate = jnp.log(0.2)
value, gradient = value_and_grad(log_rate)
# value ≈ 5.118876; gradient ≈ -0.1

model = survjax.Exponential.from_log(log_rate=log_rate)
survival = model.survival(jnp.array([0.0, 1.0, 5.0]))
# survival ≈ [1.0, 0.818731, 0.367879]
```

The scalar `loss` can be passed to an external optimizer. The example uses
fixed parameters; it does not fit a model.

For exact and right-censored observations, likelihoods also accept raw arrays:
`model.log_likelihood(time=time, event=event)`. Concordance accepts either
`survjax.concordance_index(scores, data)` or keyword `time` and `event` arrays;
higher scores indicate higher risk.

## Guiding principles

- **Composable APIs:** data and distributions work with JAX transformations
  and external inference tools.
- **Numerical accuracy:** stable log-space calculations, without probability
  floors, clipping, or added epsilons.
- **Differentiable likelihoods:** first and second parameter derivatives on
  valid, smooth interior cases within the documented derivative guarantees.
- **Performance:** support for `jax.jit` and `jax.vmap`, with correctness checked
  in eager and transformed execution.

## Documentation

The [documentation sources](https://github.com/survjax/survjax/tree/main/docs)
contain the user guide, API reference and examples. Build the executable site
locally with `scripts/build_docs.sh`. Read the Docs hosting is not set up yet;
`https://survjax.readthedocs.io` is the intended Community address, subject to
project-slug availability and a successful hosted build.

## Development

`survjax` is under active development. Contributions and co-maintenance are welcome.
If you're interested in helping, please open an issue or reach out directly.

The GPU badge shows the latest manually started Modal GPU run on `main` for an
NVIDIA T4 with CUDA 12. Each run is attached to the selected commit in GitHub
Actions. Run GPU CI before tagging a release that claims support for this
configuration; CPU-only releases need only pass CPU CI.
