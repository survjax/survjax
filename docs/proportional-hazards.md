---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Proportional hazards

`survjax.ProportionalHazards` combines a baseline distribution with a
caller-computed log hazard ratio, `eta`. Covariate mapping stays in your code:
for example, compute `eta = X @ beta`, then pass it as `log_hazard_ratio`.
All values on this page are synthetic, with time measured in months.

For baseline hazard $h_0$, cumulative hazard $H_0$ and survival $S_0$,

$$
h(t)=e^\eta h_0(t),\qquad H(t)=e^\eta H_0(t),\qquad
S(t)=S_0(t)^{e^\eta}.
$$

**Positive eta means higher hazard** thus shorter survival.
The hazard ratio $e^\eta$ is dimensionless and constant over time; time
units come from the baseline.
See the [API reference](api.md#proportional-hazards) for signatures.

## A hazard ratio of two

The baseline below has constant hazard 0.5 per month. With `eta=log(2)`, the
PH model has hazard 1 per month and survival $e^{-t}$.
The baseline may be positional or keyword; `log_hazard_ratio` is keyword only.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import survjax

baseline = survjax.Exponential(rate=0.5)
ph = survjax.ProportionalHazards(baseline, log_hazard_ratio=jnp.log(2.0))
months = jnp.array([0.0, 1.0, 2.0])
assert ph.survival(0.0) == 1.0
print("dtype:", ph.log_hazard_ratio.dtype)
print("columns: baseline survival, PH survival, PH hazard")
jnp.column_stack((baseline.survival(months), ph.survival(months), ph.hazard(months)))
```

```{code-cell} ipython3
plot_months = jnp.linspace(0.0, 6.0, 121)
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(plot_months, baseline.survival_grid(plot_months), label="Baseline: hazard 0.5")
ax.plot(plot_months, ph.survival_grid(plot_months), label="PH: hazard ratio 2")
ax.set(xlabel="Months", ylabel="Survival probability", ylim=(0, 1))
ax.legend()
fig.tight_layout()
plt.show()
```

The wrapper also provides `log_survival`, `log_cdf`, `log_prob`,
`cumulative_hazard` and `log_hazard`, with the same direct-method boundary
policy as the [built-in families](builtin-families.md). In particular, direct
survival at zero is 1; an **exact observation** at zero is still invalid.

## A piecewise-constant baseline

Proportional hazards does not require a constant baseline hazard. Suppose the
baseline hazard is 0.08 per month for months 0–6, 0.05 for months 6–12, then
0.02. The hazard ratio still multiplies every interval's hazard by the same
factor. This is a parametric PH model with chosen cut points, not a Cox fit.

The [custom-distribution guide](distributions.md#fixed-settings-and-vector-parameters-a-piecewise-constant-hazard)
explains the baseline below, including its static cut points and vector of rates.
We repeat the definition so this page runs independently.

```{code-cell} ipython3
from dataclasses import dataclass

@survjax.register_family
@dataclass(frozen=True)
class PiecewiseHazard(survjax.SurvivalDistribution):
    starts: tuple = survjax.static()  # interval start times, first is 0
    rates: jax.Array = survjax.param(component_ndim=1)  # one rate per interval

    def log_survival(self, t):
        starts = jnp.asarray(self.starts)
        ends = jnp.append(starts[1:], jnp.inf)
        t = jnp.asarray(t)[..., None]  # trailing axis lines up with the intervals
        # Time spent in each interval by time t; a later interval wins at a
        # cut point, so the hazard is right-continuous.
        exposure = jnp.where(
            t < starts, 0.0, jnp.where(t < ends, t - starts, ends - starts)
        )
        return -jnp.sum(self.rates * exposure, axis=-1)

    def reference_time(self):
        # Middle of the first interval, one reference per batch entry.
        return jnp.full(self.batch_shape, self.starts[1] / 2, dtype=self.rates.dtype)


relapse = PiecewiseHazard(starts=(0.0, 6.0, 12.0), rates=jnp.array([0.08, 0.05, 0.02]))
```

Use the same built-in wrapper as in the exponential example:

```{code-cell} ipython3
patients = survjax.ProportionalHazards(
    relapse, log_hazard_ratio=jnp.log(jnp.array([1.0, 0.7, 1.5]))
)
# At month 12, baseline cumulative hazard is 6*0.08 + 6*0.05 = 0.78.
assert jnp.allclose(patients.survival(12.0), jnp.exp(-0.78 * jnp.array([1.0, 0.7, 1.5])))
fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
times = jnp.linspace(0.0, 24.0, 241)
labels = ["HR 1", "HR 0.7", "HR 1.5"]
axes[0].plot(times, patients.hazard_grid(times), label=labels)
axes[1].plot(times, patients.survival_grid(times), label=labels)
axes[0].set(xlabel="Months", ylabel="Hazard per month")
axes[1].set(xlabel="Months", ylabel="Survival probability", ylim=(0, 1))
axes[1].legend()
plt.show()
```

The baseline hazard changes at the cut points while each patient's hazard
ratio stays constant. For regression, compute `log_hazard_ratio = X @ beta`;
survjax handles the distribution and likelihood, while you choose and fit
`beta`. RMST is currently supported only for Exponential and Weibull baselines,
not this custom piecewise family.

## Batches of draws and arms

The baseline and predictor broadcast at construction. Here three hypothetical
parameter draws have baseline shape `(3, 1)` and predictor shape `(3, 2)`:
three draws, two arms. The first arm has hazard ratio 1; the second has
ratios 2, 1.5 and 3 across the draws. These are supplied values, not fitted draws.

```{code-cell} ipython3
scales = jnp.array([2.0, 3.0, 4.0])[:, None]
eta = jnp.log(jnp.array([[1.0, 2.0], [1.0, 1.5], [1.0, 3.0]]))
draws = survjax.ProportionalHazards(
    survjax.Weibull(shape=2.0, scale=scales), log_hazard_ratio=eta
)
assert draws.batch_shape == (3, 2)
assert draws[0].batch_shape == (2,)
curves = draws.survival_grid(months)
assert curves.shape == (3, 3, 2)
assert jnp.all(curves[0] == 1.0)
draws.batch_shape, draws.survival(2.0).shape, curves.shape
```

Pointwise calls use ordinary trailing-axis broadcasting. A scalar time applies
to every draw and arm. The `_grid` methods instead add a leading time axis:
`(K, draws, arms)`, even when `K` equals the number of draws, as it does here.
Indexing `draws[0]` selects the first draw and retains its two arms. See
[Survival curves and time grids](grids.md) for the pointwise/grid distinction.

```{note}
**Initial-release extreme-scaling limit:** first and second parameter
derivatives of PH direct methods and likelihoods can be inaccurate, zero or
nonfinite even when values and mathematical derivatives are finite. This
includes baseline parameters, eta and mixed Hessians. For example, in float64,
Weibull `shape=2, scale=1`, `eta=-720`, at `t=1e150` can give zero reverse
baseline gradients although the true gradients are nonzero. Overflow or
underflow inside differentiation is the issue; an accurate log value alone
does not ensure an accurate gradient. There is no fully mapped safe boundary
or blanket exemption for all tails. Stabilization is post-initial-release
work; ordinary derivatives, value/boundary policies and supported valid-entry
isolation remain required.
```
