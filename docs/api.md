# API reference

This reference covers public distributions, their methods, data and diagnostic
types. Each entry is generated from the docstring you also see with `help()`.
For worked examples, see the guides:
[SurvivalData](survival-data.md), [Exponential and Weibull](builtin-families.md),
[likelihoods](likelihood.md), [time grids](grids.md) and [writing a distribution](distributions.md).

## Likelihood contributions

```{eval-rst}
.. automethod:: survjax.SurvivalDistribution.log_likelihood
```

## Observation data

```{eval-rst}
.. autoclass:: survjax.SurvivalData
   :members: from_bounds, issues
```

```{eval-rst}
.. autoclass:: survjax.diagnostics.DataIssues
   :members: raise_if_invalid
```

`DataIssues` is not exported itself; you get it from `SurvivalData.issues`.

```{eval-rst}
.. autoexception:: survjax.InvalidDataError
```

```{eval-rst}
.. autoexception:: survjax.InvalidParameterError
```

## Parameter diagnostics

```{eval-rst}
.. automethod:: survjax.SurvivalDistribution.parameter_issues
.. autoclass:: survjax.diagnostics.ParameterIssues
   :members: raise_if_invalid
```

`ParameterIssues` is obtained from `dist.parameter_issues()` and is not a
top-level export.

## Built-in distributions

Both families also have every method of `SurvivalDistribution` listed under
[writing your own family](#writing-your-own-family), including the seven
`_grid` methods.

```{eval-rst}
.. autoclass:: survjax.Exponential
   :members: from_log, log_survival, cumulative_hazard, log_cdf, log_prob, log_hazard, hazard, reference_time
```

```{eval-rst}
.. autoclass:: survjax.Weibull
   :members: from_log, log_survival, cumulative_hazard, log_cdf, log_prob, log_hazard, hazard, reference_time
```

## Writing your own family

```{eval-rst}
.. automodule:: survjax.constraints
   :members: real, positive, nonnegative, greater_than, interval, closed_interval, unit_interval, ordered, simplex
```

```{eval-rst}
.. autoclass:: survjax.SurvivalDistribution
```

### Methods every family implements

```{eval-rst}
.. automethod:: survjax.SurvivalDistribution.log_survival
.. automethod:: survjax.SurvivalDistribution.reference_time
```

### Limit at infinity

Proper families inherit the default $\log S(+\infty) = -\infty$. A cure
family overrides this method with its log cure fraction.

```{eval-rst}
.. automethod:: survjax.SurvivalDistribution.log_survival_at_infinity
```

### Shape and indexing

```{eval-rst}
.. autoproperty:: survjax.SurvivalDistribution.batch_shape
.. automethod:: survjax.SurvivalDistribution.__getitem__
```

### Direct methods, derived from log survival unless a family overrides them

```{eval-rst}
.. automethod:: survjax.SurvivalDistribution.survival
.. automethod:: survjax.SurvivalDistribution.cumulative_hazard
.. automethod:: survjax.SurvivalDistribution.log_cdf
.. automethod:: survjax.SurvivalDistribution.log_prob
.. automethod:: survjax.SurvivalDistribution.log_hazard
.. automethod:: survjax.SurvivalDistribution.hazard
```

### Grid methods, one shared vector of times against the whole batch

```{eval-rst}
.. automethod:: survjax.SurvivalDistribution.survival_grid
.. automethod:: survjax.SurvivalDistribution.log_survival_grid
.. automethod:: survjax.SurvivalDistribution.cumulative_hazard_grid
.. automethod:: survjax.SurvivalDistribution.log_cdf_grid
.. automethod:: survjax.SurvivalDistribution.log_prob_grid
.. automethod:: survjax.SurvivalDistribution.log_hazard_grid
.. automethod:: survjax.SurvivalDistribution.hazard_grid
```

### Declaring a family

```{eval-rst}
.. autofunction:: survjax.register_family
```

```{eval-rst}
.. autofunction:: survjax.param
```

```{eval-rst}
.. autofunction:: survjax.static
```

```{eval-rst}
.. autofunction:: survjax.distribution
```
