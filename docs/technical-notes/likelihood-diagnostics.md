---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Diagnosing likelihood NaNs

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import survjax
```

```{code-cell} ipython3
model = survjax.Exponential(rate=0.05)
```

A NaN likelihood contribution can arise from invalid data, invalid parameters,
zero model mass in the inclusion window or a numerical failure. The [likelihood guide](../likelihood.md#when-a-contribution-is-nan)
introduces these cases; this note separates them in one batch. Take a valid event at month 4, an exact
time of $-1$ (invalid: event times must be positive), patient 5 from the
[data guide](../survival-data.md), who progressed between scans at months 6 and
9 (interval censored), and a patient censored at month 12:

```{code-cell} ipython3
nan = jnp.nan
mixed = survjax.SurvivalData(
    [4.0, -1.0, 6.0, 12.0],
    right_censored=[False, False, False, True],
    interval_censored=[False, False, True, False],
    time2=[nan, nan, 9.0, nan],
)
ll = model.log_likelihood(mixed)
assert jnp.array_equal(jnp.isnan(ll), jnp.array([False, True, False, False]))
ll
```

Only the negative exact time gives NaN; the interval-censored row now has a
finite contribution. All four observation kinds and their truncation
combinations are evaluated. To explain a NaN, first identify rows flagged by
data and parameter diagnostics. Broadcast their masks to the result shape,
since a batch of distributions can enlarge it:

```{code-cell} ipython3
bad_data = jnp.broadcast_to(mixed.issues.invalid_rows, ll.shape)
bad_params = jnp.broadcast_to(model.parameter_issues().invalid_entries, ll.shape)
other = jnp.isnan(ll) & ~bad_data & ~bad_params
assert jnp.array_equal(bad_data, jnp.array([False, True, False, False]))
assert not jnp.any(bad_params | other)
bad_data, bad_params, other
```

The second row is bad data and nothing is left over. A NaN in `other` needs
further investigation: check that the model assigns positive mass to the
inclusion window, then inspect its interval methods and numerical behavior.
A geometrically valid window can have zero mass under a bounded-support
model, without a data or parameter issue. A zero numerator with positive
window mass instead gives `-inf`, so it is not selected by `jnp.isnan`.

This recipe is for data as constructed; it does not cover `SurvivalData`
whose fields were edited afterwards. For raw `time`/`event` input, build
`SurvivalData(time, event)` to read its `issues`. Likelihood evaluation also
preserves stored raw-input issues even if normalization made fields look valid.

See the [API reference](../api.md) for the full signatures.
