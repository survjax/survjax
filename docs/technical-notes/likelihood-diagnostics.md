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
an unsupported observation mode or a numerical failure. The [likelihood guide](../likelihood.md#when-a-contribution-is-nan)
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
ll
```

Two rows are NaN, for different reasons. To explain each NaN, first remove
those flagged by the data and parameter diagnostics. Then check which of
the rest use a mode that the current version supports: exact or right
censored ($\text{upper} = \infty$), with no truncation. The diagnostic masks
are broadcast to the shape of the result, since a batch of distributions
can enlarge it:

```{code-cell} ipython3
bad_data = jnp.broadcast_to(mixed.issues.invalid_rows, ll.shape)
bad_params = jnp.broadcast_to(model.parameter_issues().invalid_entries, ll.shape)
supported_mode = (
    (mixed.left_trunc == 0)
    & jnp.isposinf(mixed.right_trunc)
    & (mixed.exact | jnp.isposinf(mixed.upper))
)
unsupported = jnp.broadcast_to(~supported_mode, ll.shape)
unexplained = jnp.isnan(ll) & ~bad_data & ~bad_params
unsupported_nan = unexplained & unsupported
other = unexplained & ~unsupported  # zero normalization or numerical failure
bad_data, unsupported_nan, other
```

The second row is bad data, the third an unsupported mode, and nothing is
left over. A NaN in `other` would point to a numerical problem worth
investigating. This recipe is for data as constructed; it does not cover
`SurvivalData` whose fields were edited afterwards. For raw `time`/`event` input, build
`SurvivalData(time, event)` to read its `issues`. As later versions
implement more modes, `supported_mode` grows.

See the [API reference](../api.md) for the full signatures.
