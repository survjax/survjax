---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Working with distributions

A distribution describes event times; `SurvivalData` describes what was observed.
Parameters can be scalars, patient-specific predictions, or posterior draws.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import survjax
```

## One function describes the whole distribution

Let $T$ be the time from diagnosis to disease progression. Everything we want
to know about $T$ follows from its *survival function*, the probability of
still being progression free at time $t$:

$$
S(t) = P(T > t).
$$

The other familiar quantities are transformations of it:

| Quantity | Formula | survjax method |
| --- | --- | --- |
| log survival | $\log S(t)$ | `log_survival(t)` |
| survival | $S(t)$ | `survival(t)` |
| cumulative hazard | $H(t) = -\log S(t)$ | `cumulative_hazard(t)` |
| hazard | $h(t) = H'(t) = -\dfrac{d}{dt}\log S(t)$ | `hazard(t)`, `log_hazard(t)` |
| density | $f(t) = h(t)\,S(t)$ | `log_prob(t)` (log density) |
| CDF | $F(t) = 1 - S(t)$ | `log_cdf(t)` |

```{admonition} The circle of survival
:class: note

For a visual introduction to how survival, cumulative incidence, density,
hazard and cumulative hazard connect, read
[The circle of survival](https://wvanamsterdam.com/posts/261029-survival-circle/index.html),
an illustrated teaching article by Wouter van Amsterdam.
```

Built-in and custom families share this interface. `log_prob` means log density,
not the likelihood of a censored observation: use `log_likelihood(data)` for that.

## Constructing a model

```{code-cell} ipython3
model = survjax.Weibull(shape=1.5, scale=12.0)
model.survival(6.0), model.hazard(6.0), model.log_prob(6.0)
```

Shape is dimensionless; scale and evaluation times must use the same units.
Constructors store parameters without fitting them. `from_log` accepts
unconstrained log parameters, useful for optimizers and neural networks.

## Many distributions at once

Parameters can be arrays. Each entry along the leading axes is a separate
distribution, and `batch_shape` reports those axes. Here a trial has a
control arm and a treated arm whose progression rate is 30% lower:

```{code-cell} ipython3
arms = survjax.Exponential(rate=jnp.array([0.1, 0.07]))
arms.batch_shape
```

Indexing selects distributions, just as indexing an array selects entries:

```{code-cell} ipython3
treated = arms[1]
treated.rate, treated.batch_shape
```

Times broadcast against the batch with the usual NumPy rules, matching
trailing axes. Two times and two arms therefore pair up: arm 0 at month 6 and
arm 1 at month 12.

```{code-cell} ipython3
arms.survival(jnp.array([6.0, 12.0]))
```

To evaluate every arm at every time, insert an axis with `None`. The result
has one row per arm and one column per time: two survival curves.

```{code-cell} ipython3
months = jnp.array([0.0, 6.0, 12.0, 24.0])
arms[:, None].survival(months)
```

Parameters broadcast against each other at construction, so a scalar
parameter is shared across the batch. Every family also has *grid* methods,
such as `survival_grid`, which put the time axis first and need no manual
axis; see [Survival curves and time grids](grids.md).

```{code-cell} ipython3
arms.survival_grid(months)
```

## Using JAX

Distributions are pytrees: parameters are array leaves, while static settings
belong to the structure. Pass distributions into compiled functions, or build
them inside a differentiable objective.

```{code-cell} ipython3
jax.jit(lambda dist: dist.survival(12.0))(model)
jax.grad(lambda scale: survjax.Weibull(shape=1.5, scale=scale).survival(12.0))(12.0)
```

Changing parameter values with the same shapes and dtypes reuses compiled code;
changing static settings, such as a custom family's cut points, can recompile it.
See [Log likelihoods](likelihood.md) for an objective built from observations.

## Diagnostics and limits

```{code-cell} ipython3
bad = survjax.Weibull(shape=1.5, scale=jnp.array([12.0, -1.0]))
bad.parameter_issues().invalid_entries, bad.survival(6.0)
```

Invalid numerical parameters yield NaN for the affected entries. Inspect
`parameter_issues()` and call `raise_if_invalid()` outside compiled code when
you want an exception. Malformed types and incompatible shapes raise errors.

Direct methods accept zero, negative and infinite times with defined boundary
policies; this is separate from observation validation, which requires exact
event times to be positive. See [built-in boundaries](builtin-families.md#what-happens-at-time-zero-and-at-infinity)
and [numerical behaviour](numerical-behaviour.md).

Parameter arrays follow the [data dtype policy](survival-data.md#floating-point-precision),
but distributions have no `dtype=` argument: cast parameters before construction.
Nested distributions retain their own dtypes. Time arguments follow ordinary
JAX conversion and promotion. Detailed extension rules are in
[Advanced custom distributions](technical-notes/custom-distributions.md#details).

`rmst` is currently available for Exponential, Weibull and their PH wrappers.
Other families have no generic integration fallback. Mean, median, quantiles
and sampling are not implemented. See [RMST](rmst.md) and the [API reference](api.md).
