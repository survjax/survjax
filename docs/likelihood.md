---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Likelihood contributions

A survival likelihood combines observed events with follow-up that ended
before an event was observed. survjax currently evaluates exact events and
right censoring without truncation. Both built-in families,
[Exponential and Weibull](builtin-families.md), and registered custom
families have this interface. Other censoring modes and truncation can be
represented in `SurvivalData`, but their likelihood contributions are NaN
in Slice A.

## Events contribute density; censoring contributes survival

For an event observed at time $t$, the contribution is $\log f(t)$. For a
subject still event free when follow-up ends at $t$, it is $\log S(t)$.
With exponential rate $\lambda$, these become

$$
\ell_i = d_i \log\lambda - \lambda t_i,
$$

where $d_i=1$ denotes an event and $d_i=0$ right censoring. Exact event times
must be finite and positive; a right-censored time of zero is valid and
contributes zero. The direct method `log_prob(0)` has a separate boundary
policy and does not make an exact observation at zero valid.

Consider three synthetic subjects, with events at times 1 and 3 and
censoring at time 2. At rate 1 the contributions are exactly -1, -2 and -3:

```{code-cell} ipython3
import jax
import jax.numpy as jnp

import survjax

time = jnp.array([1.0, 2.0, 3.0])
event = jnp.array([True, False, True])
data = survjax.SurvivalData(time, event)
dist = survjax.Exponential(rate=1.0)
values = dist.log_likelihood(data)
assert jnp.array_equal(values, jnp.array([-1.0, -2.0, -3.0]))
values
```

The result is an array of **per-observation** contributions. Choose your own
reduction, such as `-values.sum()` for a negative total log likelihood or
`-values.mean()` for a negative average. There is no implicit weighting.
A density can exceed one, so a positive log contribution is not itself an
error. A true zero density or survival probability gives `-inf`; invalid
inputs and detected numerical failures give NaN. Multiplying an invalid
contribution by zero does not remove its NaN.

## Prepared data or keyword arrays

The method and top-level function are equivalent, including parameter
derivatives. Both accept either prepared data or the two keyword arrays:

```{code-cell} ipython3
raw = dist.log_likelihood(time=time, event=event)
functional = survjax.log_likelihood(dist, data)
functional_raw = survjax.log_likelihood(dist, time=time, event=event)
assert jnp.array_equal(raw, values)
assert jnp.array_equal(functional, values)
assert jnp.array_equal(functional_raw, values)
```

Do not combine `data` with `time` or `event`; raw arrays are keyword-only
and both are required. Incorrect call forms, types or incompatible shapes
raise `TypeError`, including during tracing. Invalid runtime values instead
produce NaN contributions. Use `SurvivalData` for constructor options such
as dtype; the likelihood entry points do not accept those options.

## Batches use trailing-axis broadcasting

Data and distribution batches broadcast along their trailing axes. A scalar
model against data of shape `(n,)` returns `(n,)`. Parameters of shape `(n,)`
assign a model to each observation. To evaluate `m` models against all `n`
observations, give the parameters shape `(m, 1)`:

```{code-cell} ipython3
models = survjax.Exponential(rate=jnp.array([1.0, 2.0])[:, None])
by_model = models.log_likelihood(data)
assert by_model.shape == (2, 3)
assert jnp.array_equal(by_model[0], values)
by_model.sum(axis=-1)
```

The last expression totals over observations and keeps the model axis.
Unlike [time-grid methods](grids.md), likelihood evaluation adds no grid
axis. Calculations use the common floating dtype of parameters and data,
subject to JAX's configured 64-bit mode; survjax does not enable it globally.

## Compose with JAX

Construction and likelihood evaluation can occur inside `jax.jit`. Here the
parameter is log rate, so exponentiation keeps the rate positive:

```{code-cell} ipython3
@jax.jit
def negative_log_likelihood(log_rate, time, event):
    model = survjax.Exponential(rate=jnp.exp(log_rate))
    return -model.log_likelihood(time=time, event=event).sum()

eta = jnp.array(0.0)
loss = negative_log_likelihood(eta, time, event)
gradient = jax.grad(negative_log_likelihood)(eta, time, event)
curvature = jax.grad(jax.grad(negative_log_likelihood))(eta, time, event)
# Algebra: loss = 6*exp(eta) - 2*eta, so at eta=0 these are 6, 4, 6.
assert loss == 6.0
assert gradient == 4.0
assert curvature == 6.0
loss, gradient, curvature
```

`jax.vmap` can also map likelihood computations over model parameters.
First and second parameter derivatives are supported on valid smooth
interior cases. Derivatives of invalid or unsupported contributions are
unspecified; do not treat a NaN-containing objective as a valid fitting
problem. survjax supplies contributions; optimization and the choice of
objective belong to the caller.

## Diagnose NaN contributions

Inspect `data.issues` for observation problems and
`dist.parameter_issues()` for declared parameter violations. Their boolean
masks use `True` to mark a problem. At an explicit host checkpoint, call
`data.issues.raise_if_invalid()` or
`dist.parameter_issues().raise_if_invalid()` to raise `InvalidDataError` or
`InvalidParameterError`. These raising methods belong outside JAX transforms;
the diagnostic arrays themselves work in compiled code.

```{code-cell} ipython3
bad_data = survjax.SurvivalData(time, jnp.array([1, 2, 0]))
assert jnp.array_equal(bad_data.issues.invalid_rows, jnp.array([False, True, False]))
assert jnp.array_equal(jnp.isnan(dist.log_likelihood(bad_data)),
                       jnp.array([False, True, False]))

bad_model = survjax.Exponential(rate=jnp.array([1.0, -1.0, 1.0]))
assert jnp.array_equal(bad_model.parameter_issues().invalid_entries,
                       jnp.array([False, True, False]))
assert jnp.array_equal(jnp.isnan(bad_model.log_likelihood(data)),
                       jnp.array([False, True, False]))
```

The invalid indicator remains recorded even after construction normalizes
it. Declared parameter violations produce NaN even if a family's formula
happens to return a finite value. For custom families, an empty parameter
report only means no declared violation was found.

Unsupported observation modes are a separate cause: structurally valid
interval censoring has no data issue, but its likelihood is not implemented.
This example mixes an event, an interval and right censoring:

```{code-cell} ipython3
mixed = survjax.SurvivalData.from_bounds(
    lower=jnp.array([1.0, 1.0, 2.0]),
    upper=jnp.array([1.0, 3.0, jnp.inf]),
    exact=jnp.array([True, False, False]),
)
assert mixed.issues.ok
ll = dist.log_likelihood(mixed)
bad_data_rows = jnp.broadcast_to(mixed.issues.invalid_rows, ll.shape)
bad_params = jnp.broadcast_to(dist.parameter_issues().invalid_entries, ll.shape)
supported_mode = (
    (mixed.left_trunc == 0)
    & jnp.isposinf(mixed.right_trunc)
    & (mixed.exact | jnp.isposinf(mixed.upper))
)
unsupported = jnp.broadcast_to(~supported_mode, ll.shape)
unexplained = jnp.isnan(ll) & ~bad_data_rows & ~bad_params
unsupported_nan = unexplained & unsupported
other = unexplained & ~unsupported
assert jnp.array_equal(unsupported_nan, jnp.array([False, True, False]))
assert not jnp.any(other)
assert ll[0] == -1.0 and ll[2] == -2.0
ll
```

For data constructed as above, check both diagnostics and mode support
before investigating remaining NaNs as numerical failures. This recipe is
for data whose normalized fields have not been edited after construction.
Unsupported rows also yield NaN under `jit` and `vmap`; supported rows keep
their contributions. All truncation, including otherwise valid left
truncation, is currently unsupported. See the [API reference](api.md) for
signatures and the [data guide](survival-data.md) for observation rules.
