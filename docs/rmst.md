---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Restricted mean survival time

Restricted mean survival time (RMST) is the expected time accumulated before
a finite cutoff $\tau$. For a nonnegative event time $T$,

$$
M(\tau)=\operatorname{E}[\min(T,\tau)]=\int_0^\tau S(t)\,dt.
$$

RMST has the same units as time. It is an area under a model's survival
curve, computed from that model's parameters; calling `rmst` does not fit
censored observations. Compare models at the same cutoff and time origin.
The runnable examples here use synthetic parameters and time in months.

## A finite horizon, in months

An exponential model with rate 0.2 per month gives
$M(12)=(1-e^{-0.2\times12})/0.2\approx4.5464$ months. This is less than
12 months because some events occur before the cutoff.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import survjax

model = survjax.Exponential(rate=0.2)
tau = 12.0
print("RMST at 12 months:", float(model.rmst(tau)))
```

The shaded area is RMST; survival beyond 12 months does not add to it.

```{code-cell} ipython3
time = jnp.linspace(0.0, 18.0, 181)
before_cutoff = jnp.linspace(0.0, tau, 121)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(time, model.survival(time), label="S(t)")
ax.fill_between(before_cutoff, model.survival(before_cutoff), alpha=0.3,
                label="area = RMST(12)")
ax.axvline(tau, color="grey", linestyle=":")
ax.set(xlabel="time (months)", ylabel="survival probability", ylim=(0, 1))
ax.legend()
fig.tight_layout()
```

The exponential expression uses `expm1` to retain small-cutoff accuracy.
For Weibull shape $k$ and scale $b$,

$$
M(\tau)=b\,\Gamma(1+1/k)\,P(1/k,(\tau/b)^k),
$$

where $P$ is the regularized lower incomplete gamma function. Shape is
dimensionless, scale is in months, and `shape=1, scale=b` is equivalent to
`Exponential(rate=1/b)`.

```{code-cell} ipython3
weibull = survjax.Weibull(shape=1.0, scale=5.0)
assert jnp.allclose(weibull.rmst(tau), model.rmst(tau))
print("Weibull shape 1:", float(weibull.rmst(tau)))
```

## Proportional-hazards models

For a log hazard ratio $\eta$, the equivalent exponential rate is
$r_{\mathrm{PH}}=r e^\eta$; the equivalent Weibull scale is
$b_{\mathrm{PH}}=b_0 e^{-\eta/k}$, with shape $k$ unchanged.
The native `Exponential` and `Weibull` methods evaluate their own RMST
formulas directly.
`ProportionalHazards.rmst` constructs the equivalent built-in distribution
and delegates to its `rmst` method. Positive eta increases hazard and reduces
RMST at a fixed positive cutoff. See the [PH guide](proportional-hazards.md)
for constructing predictors and interpreting hazard ratios.

```{code-cell} ipython3
eta = jnp.log(2.0)
ph_exp = survjax.ProportionalHazards(baseline=model, log_hazard_ratio=eta)
equivalent_exp = survjax.Exponential(rate=0.2 * jnp.exp(eta))
assert jnp.allclose(ph_exp.rmst(tau), equivalent_exp.rmst(tau))
ph_weibull = survjax.ProportionalHazards(
    baseline=survjax.Weibull(shape=2.0, scale=10.0), log_hazard_ratio=eta,
)
equivalent_weibull = survjax.Weibull(shape=2.0, scale=10.0 * jnp.exp(-eta / 2.0))
assert jnp.allclose(ph_weibull.rmst(tau), equivalent_weibull.rmst(tau))
print("PH exponential and Weibull RMST:", ph_exp.rmst(tau), ph_weibull.rmst(tau))
```

## Cutoff axes and model batches

`rmst(tau)` follows ordinary trailing-axis broadcasting against
`model.batch_shape`. There is no `rmst_grid` method: add cutoff axes explicitly.
Here three synthetic parameter draws each describe two arms. A scalar cutoff
returns `(draws, arms)`; a cutoff grid needs shape `(cutoffs, 1, 1)`.

```{code-cell} ipython3
models = survjax.Weibull(
    shape=2.0,
    scale=jnp.array([[8.0, 10.0], [9.0, 11.0], [10.0, 12.0]]),
)
cutoffs = jnp.array([0.0, 6.0, 12.0])
at_twelve = models.rmst(12.0)
by_cutoff = jax.jit(models.rmst)(cutoffs[:, None, None])
assert at_twelve.shape == (3, 2)
assert by_cutoff.shape == (3, 3, 2)
assert jnp.allclose(by_cutoff[-1], at_twelve)
assert jnp.allclose(jax.vmap(models.rmst)(cutoffs), by_cutoff)
gain_months = by_cutoff[..., 1] - by_cutoff[..., 0]
print("cutoffs, draws, arms:", by_cutoff.shape)
print("arm 2 minus arm 1, at 12 months:", gain_months[-1])
```

Each draw retains its own result. A positive contrast means more predicted
months before that cutoff in arm 2. Averaging or summarizing draws is a caller
choice. Here `models.rmst(cutoffs)` raises `TypeError`: `(3,)` cutoffs cannot
broadcast against the trailing arm dimension of the `(3, 2)` batch. A cutoff
vector of shape `(2,)` instead pairs one cutoff with each arm, returning
`(3, 2)` without a new cutoff axis:

```{code-cell} ipython3
per_arm = models.rmst(jnp.array([6.0, 12.0]))
assert per_arm.shape == (3, 2)
assert jnp.allclose(per_arm[:, 0], by_cutoff[1, :, 0])
assert jnp.allclose(per_arm[:, 1], by_cutoff[2, :, 1])
try:
    models.rmst(cutoffs)
except TypeError:
    print("Three cutoffs cannot broadcast against two arms.")
else:
    raise AssertionError("Expected incompatible cutoff/batch shapes")
print("One cutoff per arm:", per_arm.shape)
```

PH models use the same cutoff broadcasting rules.

## Supported inputs and derivatives

`rmst(tau)` is available for Exponential, Weibull and their PH wrappers.
Other custom families raise `NotImplementedError`; there is no generic
quadrature fallback. Mean, median and quantile methods are not implemented.

For valid parameters, `rmst(0)` is zero. Negative, infinite or NaN cutoffs,
and invalid parameters, return NaN. Malformed types and incompatible shapes
raise `TypeError`. A NaN can also indicate numerical failure; clean parameter
diagnostics do not guarantee a finite result, especially when extreme PH
scaling makes the equivalent rate or scale unrepresentable.

| Family / location | Derivative support |
| --- | --- |
| Exponential | All orders on valid inputs |
| Weibull, smooth interior with `tau > 0` | First derivatives in shape, scale and cutoff; higher derivatives that do not involve shape |
| Weibull at `tau = 0` | Value and first derivatives only: parameter derivatives 0, cutoff derivative 1 |

PH-Exponential inherits Exponential's all-order support. PH-Weibull inherits
both Weibull restrictions in the table. The extreme-scaling derivative
exception for PH direct methods and likelihoods does not relax RMST's
separate requirements.

Weibull and PH-Weibull higher derivatives involving shape are unsupported:
JAX has no derivative rule for its incomplete-gamma shape gradient. They raise
`NotImplementedError`; some orders or tail branches may return the correct
value instead, but this is not a general higher-shape guarantee.
For supported scale/cutoff Hessians, keep shape outside the differentiated
argument. Packing a constant shape with scale into one differentiated array
can materialize a zero shape tangent and invoke that unsupported shape rule.


At zero, Weibull and PH-Weibull guarantees stop at first order. Use a valid
smooth interior point for supported higher derivatives. See
[RMST numerics and derivatives](technical-notes/rmst.md) for accuracy domains,
invalid-entry isolation, PH scaling failures and executable derivative checks.
