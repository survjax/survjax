---
orphan: true
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

**Positive eta means higher hazard.** It does not mean longer survival.
The hazard ratio $e^\eta$ is dimensionless and constant over time; time
units come from the baseline. The survival-probability ratio need not be
constant. See the [API reference](api.md#proportional-hazards) for signatures.

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

## Exact events, right censoring and derivatives

Use `log_likelihood` for per-observation contributions. The current likelihood
subset is exact events and right censoring without truncation; representing
other observation kinds in `SurvivalData` does not make their likelihoods
available. Unsupported likelihood rows yield NaN. See
[Log likelihoods](likelihood.md).

For one exact event at month 1 and censoring at month 2, with baseline rate 1,
the negative sum of contributions is $L(\eta)=3e^\eta-\eta$.
At zero its value, first derivative and second derivative are 3, 2 and 3.

```{code-cell} ipython3
data = survjax.SurvivalData(
    jnp.array([1.0, 2.0]), event=jnp.array([True, False])
)


def loss(eta):
    model = survjax.ProportionalHazards(
        survjax.Exponential(rate=1.0), log_hazard_ratio=eta
    )
    return -model.log_likelihood(data).sum()

value, gradient = jax.jit(jax.value_and_grad(loss))(0.0)
curvature = jax.jit(jax.grad(jax.grad(loss)))(0.0)
assert jnp.all(jnp.isfinite(jnp.array([value, gradient, curvature])))
value, gradient, curvature
```

With the built-in baselines, ordinary finite, valid, smooth interior parameter
derivatives support `jit`, `vmap`, forward and reverse AD to second order. Covariate design, objective
reduction and optimization remain with the caller.

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
isolation remain required. This limit does **not** relax the RMST contract below.

## Restricted mean survival time

`rmst(tau)` returns $\int_0^\tau S(t)\,dt$, in the baseline's time units.
For the rate-1 PH model above, $M(2)=1-e^{-2}\approx0.864665$ months:

```{code-cell} ipython3
restricted_mean = ph.rmst(2.0)
assert jnp.isfinite(restricted_mean)
restricted_mean, -jnp.expm1(-2.0)
```

RMST delegates through the built-in same-family PH closure:

| Baseline | Equivalent distribution used for RMST |
| --- | --- |
| `Exponential(rate=r)` | `Exponential(rate=r*exp(eta))` |
| `Weibull(shape=k, scale=b)` | `Weibull(shape=k, scale=b*exp(-eta/k))` |

Direct methods and likelihoods use the PH expressions; they do not delegate
through this summary closure. The closure hook is private, not an extension
API. RMST on a custom baseline without a supported closure raises
`NotImplementedError`. Only `rmst` is available among summaries in this release;
`mean`, `median` and `quantile` raise `NotImplementedError`.

Cutoffs broadcast against the batch. There is no `rmst_grid`: add the leading
cutoff axis explicitly for the `(draws, arms)` batch:

```{code-cell} ipython3
cutoffs = months[:, None, None]
restricted_means = draws.rmst(cutoffs)
assert restricted_means.shape == (3, 3, 2)
assert jnp.all(restricted_means[0] == 0.0)
assert jnp.all(jnp.isfinite(restricted_means))
restricted_means.shape, restricted_means[-1]
```

Negative, NaN or infinite cutoffs yield NaN, as do declared parameter issues.
At `tau=0`, RMST is exactly 0, parameter derivatives are 0 and the cutoff
derivative is $S(0)=1$. First derivatives at positive cutoffs include all
baseline parameters and eta, with $\partial M/\partial\tau=S(\tau)$.
Exponential and PH-Exponential support higher derivatives on smooth interiors.
For Weibull and PH-Weibull, **second or higher derivatives involving shape
are unsupported** and generally raise `NotImplementedError`; some orders or
tail branches can instead return the correct result. Higher derivatives in
scale, eta and positive cutoff remain supported; keep shape outside the
arguments being differentiated for these nonshape Hessians. At zero, the Weibull
guarantee covers only values and first derivatives; do not infer higher
boundary calculus.

## Diagnostics and numerical failures

Construction stores parameter values without automatically raising on numerical
violations. Inspect `parameter_issues()`; nested fields use the `baseline.`
prefix. Malformed types or incompatible shapes still raise `TypeError`.

```{code-cell} ipython3
bad = survjax.ProportionalHazards(
    survjax.Weibull(shape=2.0, scale=jnp.array([2.0, -2.0])),
    log_hazard_ratio=0.0,
)
issues = bad.parameter_issues()
assert issues.invalid_entries.tolist() == [False, True]
assert "baseline.scale" in issues.fields
values = bad.survival(1.0)
assert jnp.isfinite(values[0]) and jnp.isnan(values[1])
issues.fields, values
```

Use `issues.raise_if_invalid()` at a concrete host checkpoint when you want an
exception. Also check the evaluated values: clean diagnostics do not guarantee
that a numerical summary can be represented.

In particular, an equivalent PH scale can overflow although the true RMST is
finite. This example explicitly uses float32 in **both** documentation builds:

```{code-cell} ipython3
fragile = survjax.ProportionalHazards(
    survjax.Weibull(shape=jnp.float32(0.2), scale=jnp.float32(1.0)),
    log_hazard_ratio=jnp.float32(-18.0),
)
failed = fragile.rmst(jnp.float32(1.0))
assert fragile.parameter_issues().ok
assert jnp.isnan(failed)
fragile.parameter_issues().ok, failed
```

Here the equivalent scale is $e^{90}$, beyond float32, while the true
restricted mean through month 1 is close to 1. The cutoff/equivalent-scale
ratio is about $e^{-90}$, outside the promised RMST accuracy range
$[10^{-2},10^5]$ (with Weibull shape in $[0.2,50]$). NaN is the initial
release's detectable numerical-failure result in this regime, even with valid
original parameters. Zero cutoff still has its exact boundary value. This
limitation does not weaken accuracy inside the specified range or isolation
of other valid entries. Wider dtype can delay overflow; no universal safe
eta boundary is claimed. Never replace a failed summary with zero or a
probability floor. For interpreting infinities and NaNs, see
[Likelihoods in the tails](numerical-behaviour.md).
