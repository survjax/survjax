---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Registering a family

Family construction, nested fields, batch indexing and PyTree registration are
available, with survival, log-CDF, cumulative-hazard, density and hazard defaults;
built-in families and time-first survival grids are planned for A5. The base's
`log_survival(t)` and `reference_time()` hooks raise `NotImplementedError`
unless supplied by a family. The base handles negative and NaN times; family
code supplies limits at zero and positive infinity.

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

    def log_survival(self, t):
        return -self.rate * jnp.sum(self.weights, axis=-1) * t

    def reference_time(self):
        return 1.0 / (self.rate * jnp.sum(self.weights, axis=-1))

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

Nested fields retain their own parameter dtypes and static configuration. A
scalar baseline broadcasts to the wrapper batch; `dist[idx]` selects batch
axes while retaining trailing components. Ellipsis spans only batch axes.
Out-of-bounds integer indices clamp to the nearest batch entry, following JAX.

```{code-cell} ipython3
@survjax.register_family
@dataclass(frozen=True)
class ScaledRecipe(survjax.SurvivalDistribution):
    base: survjax.SurvivalDistribution = survjax.distribution()
    multiplier: jax.Array = survjax.param()

    def log_survival(self, t):
        return self.multiplier * self.base.log_survival(t)

    def reference_time(self):
        return self.base.reference_time() / self.multiplier

scaled = ScaledRecipe(base=Recipe(rate=2.0, weights=[1.0, 2.0]),
                      multiplier=[1.0, 2.0, 3.0])
assert scaled.base.batch_shape == (3,)
assert scaled[1].base.weights.shape == (2,)
assert scaled.cumulative_hazard(1.0).tolist() == [6.0, 12.0, 18.0]
assert scaled.survival(-1.0).tolist() == [1.0, 1.0, 1.0]
assert scaled.log_survival(jnp.ones(3)).shape == (3,)  # paired entries
assert scaled[:, None].log_survival(jnp.ones(4)).shape == (3, 4)  # every pair
```

Negative times give log survival zero, survival one, log-CDF minus infinity,
log density and log hazard minus infinity, and cumulative hazard and hazard
zero. NaN times give NaN. These pointwise methods use
trailing-axis broadcasting; shared time-first grids remain planned for A5.
The log-CDF default uses `log(-expm1(log_survival(t)))` and retains small CDFs
until log survival underflows to zero; override it for smaller lower tails.

```{code-cell} ipython3
single = Recipe(rate=2.0, weights=[1.0, 2.0])  # constant hazard 6
assert jnp.allclose(single.log_prob(jnp.array([0.0, 1.0])),
                    jnp.log(6.0) - jnp.array([0.0, 6.0]))
assert jnp.allclose(single.log_hazard(1000.0), jnp.log(6.0))
assert jnp.allclose(jax.jit(single.hazard)(1.0), 6.0)
assert single.log_prob(jnp.inf) == -jnp.inf
assert jnp.isnan(single.hazard(jnp.inf))
```

The density default is `log(-d log S/dt) + log S`, computed by forward-mode
autodiff independently of log-hazard overrides. The log-hazard default uses
the derivative directly, avoiding cancellation at large cumulative hazards.
Both defaults require the hazard to be representable in linear space.
Zero time uses the primitive's boundary derivative; density at positive
infinity is always minus infinity, including for overrides.

Generic log hazard and hazard give NaN where log survival is exactly minus
infinity, including at finite support ends. Finite log survival still permits
a hazard when linear survival underflows. Override `log_hazard` for analytic
limits where survival is zero; the hazard default calls that override.

Direct-method times follow JAX conversion and promotion. Boolean, complex,
string and object times raise `TypeError`, including during tracing. A float64
NumPy time array converts silently to float32 when x64 is disabled; times are
transient arguments, unlike stored observation fields and parameters.

For negative or NaN times, NaN family values at the substituted input remain
NaN. Finite invalid parameters can give NaN or the policy value, depending on
the family's formula at that input; the base infers no parameter validity domain.
