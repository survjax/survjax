---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Registering a family

Flat family construction and PyTree registration are available. Nested fields,
batch indexing and generic direct methods are planned for later A4 steps;
built-in families and time-first survival grids are planned for A5. The base's
`log_survival(t)` and `reference_time()` hooks raise `NotImplementedError`
unless supplied by a family. The direct-method boundary policy is planned;
this example demonstrates construction only.

Put `register_family` above a frozen dataclass. Construction uses keyword
arguments, and parameters are required. `kw_only=True` allows a static field
with a default before required parameters.

```{code-cell} ipython3
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import survjax

@survjax.register_family
@dataclass(frozen=True, kw_only=True)
class Recipe(survjax.SurvivalDistribution):
    method: str = survjax.static(default="sum")
    rate: jax.Array = survjax.param()
    weights: jax.Array = survjax.param(component_ndim=1)

recipe = Recipe(rate=[[1.0], [2.0]], weights=jnp.ones((3, 4)))
assert recipe.batch_shape == (2, 3)
assert recipe.rate.shape == (2, 3)
assert recipe.weights.shape == (2, 3, 4)
assert jax.tree.map(lambda x: x[:1], recipe).batch_shape == (1, 3)
assert jax.jit(lambda x: x)(recipe).method == "sum"
```

Parameter arrays are dynamic leaves. Their leading batch axes broadcast;
`component_ndim` trailing axes remain components. `batch_shape` follows current
leaves, including under tree mapping and vmap. The base has no `shape` alias;
a family can use `shape` as a parameter name. Hashable static configuration is
part of the tree structure. Reconstruction preserves leaves without running
the constructor. Subclasses require explicit registration.

Own parameters share the [floating dtype policy](survival-data.md): at least
float32, with the configured default for integer-only or scalar-only inputs.
Float64 arrays warn at the call site when x64 is disabled. Nonreal inputs are
rejected; negative, NaN and infinite values are stored without domain checks.
There is no constructor `dtype=` keyword; cast inputs before construction.
