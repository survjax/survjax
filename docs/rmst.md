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

The implementation uses a series in the lower tail and the mean in the far
upper tail, with the gamma expression between them. Value comparisons target
relative accuracy of `1e-8` in float64 and `1e-4` in float32 over
$k\in[0.2,50]$ and $\tau/b\in[0.01,10^5]$, including underflow and overflow
of the power $(\tau/b)^k$. Accuracy outside that domain is not promised;
numerical failure can produce NaN even for positive parameters.
The mathematical approximation bounds and safe arithmetic are explained in the
{download}`Weibull RMST appendix <../appendices/weibull-rmst.pdf>`.

## Proportional-hazards models

For a log hazard ratio $\eta$, the equivalent exponential rate is
$r_{\mathrm{PH}}=r e^\eta$; the equivalent Weibull scale is
$b_{\mathrm{PH}}=b_0 e^{-\eta/k}$, with shape $k$ unchanged.
The native `Exponential` and `Weibull` methods compute RMST independently.
`ProportionalHazards.rmst` constructs the equivalent built-in distribution
and delegates to its `rmst` method. Positive eta increases hazard and reduces
RMST at a fixed positive cutoff. See the [PH guide](proportional-hazards.md)
for constructing predictors and interpreting hazard ratios.

```{code-cell} ipython3
eta = jnp.log(2.0)
ph_exp = survjax.ProportionalHazards(model, log_hazard_ratio=eta)
equivalent_exp = survjax.Exponential(rate=0.2 * jnp.exp(eta))
assert jnp.allclose(ph_exp.rmst(tau), equivalent_exp.rmst(tau))
ph_weibull = survjax.ProportionalHazards(
    survjax.Weibull(shape=2.0, scale=10.0), log_hazard_ratio=eta,
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

PH uses the same cutoff rules. Here baseline scales have shape `(3, 1)`
and eta has shape `(3, 2)`: three draws, two hazard ratios per draw.
The wrapper's batch is `(3, 2)`; the explicit cutoff axis gives `(3, 3, 2)`.

```{code-cell} ipython3
baseline_scales = jnp.array([[8.0], [9.0], [10.0]])
predictors = jnp.broadcast_to(jnp.array([0.0, eta]), (3, 2))
ph_batch = survjax.ProportionalHazards(
    survjax.Weibull(shape=2.0, scale=baseline_scales),
    log_hazard_ratio=predictors,
)
ph_grid = jax.jit(ph_batch.rmst)(cutoffs[:, None, None])
closed_batch = survjax.Weibull(
    shape=2.0, scale=baseline_scales * jnp.exp(-predictors / 2.0),
)
assert ph_batch.rmst(12.0).shape == (3, 2)
assert ph_grid.shape == (3, 3, 2)
assert jnp.allclose(ph_grid, closed_batch.rmst(cutoffs[:, None, None]))
assert jnp.allclose(ph_grid, jax.vmap(ph_batch.rmst)(cutoffs))
print("PH cutoffs, draws, arms:", ph_grid.shape)
```

## Zero, invalid values and argument errors

Rates, shapes and scales must be finite and strictly positive.
For valid parameters, `rmst(0)` is zero. Negative, NaN or infinite cutoffs
return NaN. Parameter-constraint violations also give NaN, including when the
cutoff is zero. Use `parameter_issues()` to inspect parameter constraints.
For PH, eta must also be finite; baseline issues have `baseline.` key prefixes.
There is no summary-info result in this staged API.

```{code-cell} ipython3
cutoff_values = jnp.array([-1.0, jnp.nan, jnp.inf, -jnp.inf, 0.0])
values = weibull.rmst(cutoff_values)
assert jnp.isnan(values[:-1]).all() and values[-1] == 0
bad_parameters = survjax.Weibull(shape=jnp.array([2.0, 0.0]), scale=10.0)
print("invalid cutoffs followed by zero:", values)
print("valid and invalid shapes at cutoff zero:", bad_parameters.rmst(0.0))
```

A NaN result's own derivatives are unspecified. Invalid or failed entries
must not contaminate derivatives of other valid entries, including through
shared parameters. The following PH example shares baseline parameters across
a valid predictor and an invalid predictor; selecting the valid result keeps
its derivatives equal to those of the model evaluated alone.

```{code-cell} ipython3
def beside_invalid(k, b, e):
    batch = survjax.ProportionalHazards(
        survjax.Weibull(shape=k, scale=b), log_hazard_ratio=jnp.array([e, jnp.nan]),
    )
    return batch.rmst(6.0)


def ph_restricted(k, b, e, t):
    return survjax.ProportionalHazards(
        survjax.Weibull(shape=k, scale=b), log_hazard_ratio=e,
    ).rmst(t)


mixed = beside_invalid(2.0, 10.0, eta)


def valid_entry(k, b, e):
    return beside_invalid(k, b, e)[0]


isolated = jax.jit(jax.grad(valid_entry, argnums=(0, 1, 2)))
alone = jax.grad(ph_restricted, argnums=(0, 1, 2))(2.0, 10.0, eta, 6.0)
assert jnp.isfinite(mixed[0]) and jnp.isnan(mixed[1])
assert jnp.allclose(jnp.array(isolated(2.0, 10.0, eta)), jnp.array(alone))
print("PH valid entry beside invalid eta:", mixed)
```

NaN signals invalid input, invalid parameters or numerical
failure; it does not mean a true zero RMST.

Boolean, complex, string and object cutoffs raise `TypeError`, as do shapes
that cannot broadcast against the model batch. These are argument errors;
negative and infinite numeric values instead follow the NaN policy above.
A cutoff of infinity cannot be used to request an unrestricted mean.

PH can also fail numerically when its equivalent rate or scale cannot be
represented, even though the original parameters pass `parameter_issues()`.
The accuracy domain above uses the **equivalent scale** $b_{\mathrm{PH}}$.
Outside that domain, an unrepresentable closure can give NaN even when the
mathematical RMST is finite. Clean original diagnostics do not rule this out.
For example, the scale $e^{1000}$ overflows in both dtypes below; zero cutoff
still returns exactly zero. These failures do not relax the accuracy domain
or valid-entry isolation requirements.

```{code-cell} ipython3
extreme = survjax.ProportionalHazards(
    survjax.Weibull(shape=0.2, scale=1.0), log_hazard_ratio=-200.0,
)
assert extreme.parameter_issues().ok
assert jnp.isnan(extreme.rmst(1.0))
assert extreme.rmst(0.0) == 0
print("Clean original diagnostics, failed PH closure:", extreme.rmst(1.0))
```

## Derivatives and their limits

First derivatives in parameters and cutoff use native JAX rules in forward
and reverse mode. For the exact integral, $dM/d\tau=S(\tau)$. Far-tail
approximations can return zero for a negligible cutoff derivative: accuracy
is measured at absolute scale 1, not relative to an arbitrarily tiny derivative.
Parameter-derivative error is measured in log-parameter units
($k\,dM/dk$, $b\,dM/db$, or $r\,dM/dr$) at scale $|M|$;
$r$ denotes the exponential rate.

```{code-cell} ipython3

def restricted(k, b, t):
    return survjax.Weibull(shape=k, scale=b).rmst(t)


first = jax.jit(jax.grad(restricted, argnums=(0, 1, 2)))(2.0, 10.0, 6.0)
assert jnp.all(jnp.isfinite(jnp.array(first)))
forward = jax.jacfwd(restricted, argnums=(0, 1, 2))(2.0, 10.0, 6.0)
assert jnp.allclose(jnp.array(first), jnp.array(forward))
assert jnp.allclose(first[2], survjax.Weibull(shape=2.0, scale=10.0).survival(6.0))
print("shape, scale, cutoff derivatives:", first)
```

PH derivatives include eta and all baseline parameters through the chain
rule. For Weibull, writing $M_b$ for the equivalent distribution's scale
derivative, $\partial M/\partial\eta=-b_{\mathrm{PH}}M_b/k$.
The next example checks all four first derivatives in both AD modes and
checks that eta identity. It also evaluates a supported scale/eta/cutoff
Hessian with shape held outside the differentiated arguments.

```{code-cell} ipython3
ph_first = jax.jit(jax.grad(ph_restricted, argnums=(0, 1, 2, 3)))(2.0, 10.0, eta, 6.0)
ph_forward = jax.jacfwd(ph_restricted, argnums=(0, 1, 2, 3))(2.0, 10.0, eta, 6.0)
assert jnp.allclose(jnp.array(ph_first), jnp.array(ph_forward))
b_ph = 10.0 * jnp.exp(-eta / 2.0)
equivalent_db = jax.grad(restricted, argnums=1)(2.0, b_ph, 6.0)
assert jnp.allclose(ph_first[2], -b_ph * equivalent_db / 2.0)
assert jnp.allclose(ph_first[3], ph_weibull.survival(6.0))


def ph_fixed_shape(b, e, t):
    return ph_restricted(2.0, b, e, t)


ph_hessian = jax.jit(jax.hessian(ph_fixed_shape, argnums=(0, 1, 2)))
assert jnp.isfinite(jnp.array(ph_hessian(10.0, eta, 6.0))).all()
print("PH shape, baseline scale, eta, cutoff derivatives:", ph_first)
```

For PH-Exponential, differentiating $M=(1-e^{-r_{\mathrm{PH}}\tau})/r_{\mathrm{PH}}$
gives $M_{r_{\mathrm{PH}}}=((1+r_{\mathrm{PH}}\tau)e^{-r_{\mathrm{PH}}\tau}-1)/r_{\mathrm{PH}}^2$.
Chain-rule factors are $e^\eta$ for the baseline rate and $r_{\mathrm{PH}}$ for eta.

```{code-cell} ipython3
def ph_constant(r, e, t):
    return survjax.ProportionalHazards(
        survjax.Exponential(rate=r), log_hazard_ratio=e,
    ).rmst(t)


exp_first = jax.jit(jax.grad(ph_constant, argnums=(0, 1, 2)))(0.2, eta, 12.0)
exp_forward = jax.jacfwd(ph_constant, argnums=(0, 1, 2))(0.2, eta, 12.0)
r_ph = 0.2 * jnp.exp(eta)
surv_at_cutoff = jnp.exp(-r_ph * 12.0)
dr_ph = ((1 + r_ph * 12.0) * surv_at_cutoff - 1) / r_ph**2
expected = jnp.array([jnp.exp(eta) * dr_ph, r_ph * dr_ph, surv_at_cutoff])
assert jnp.allclose(jnp.array(exp_first), expected)
assert jnp.allclose(jnp.array(exp_first), jnp.array(exp_forward))
assert ph_constant(0.2, eta, 0.0) == 0
assert jnp.array_equal(
    jnp.array(jax.grad(ph_constant, argnums=(0, 1, 2))(0.2, eta, 0.0)),
    jnp.array([0.0, 0.0, 1.0]),
)
print("PH exponential rate, eta, cutoff derivatives:", exp_first)
```

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

```{code-cell} ipython3

def fixed_shape(b, t):
    return restricted(2.0, b, t)


hessian = jax.jit(jax.hessian(fixed_shape, argnums=(0, 1)))(10.0, 6.0)
assert jnp.all(jnp.isfinite(jnp.array(hessian)))
try:
    jax.hessian(lambda k: restricted(k, 10.0, 6.0))(2.0)
except NotImplementedError:
    print("Higher Weibull shape derivatives are unsupported.")
else:
    raise AssertionError("Expected the native shape-Hessian restriction")
try:
    jax.hessian(lambda k: ph_restricted(k, 10.0, eta, 6.0))(2.0)
except NotImplementedError:
    print("Higher PH-Weibull shape derivatives are unsupported.")
else:
    raise AssertionError("Expected the PH shape-Hessian restriction")
```

At zero, the Weibull and PH-Weibull guarantee stops at first order. Higher right
cutoff derivatives can be nonfinite or differ from autodiff's result there.
Use a valid smooth interior point for supported higher derivatives.

```{code-cell} ipython3
assert restricted(2.0, 10.0, 0.0) == 0
assert jax.grad(restricted, argnums=0)(2.0, 10.0, 0.0) == 0
assert jax.grad(restricted, argnums=1)(2.0, 10.0, 0.0) == 0
assert jax.grad(restricted, argnums=2)(2.0, 10.0, 0.0) == 1
assert ph_restricted(2.0, 10.0, eta, 0.0) == 0
ph_zero = jax.jit(jax.grad(ph_restricted, argnums=(0, 1, 2, 3)))(2.0, 10.0, eta, 0.0)
assert jnp.array_equal(jnp.array(ph_zero), jnp.array([0.0, 0.0, 0.0, 1.0]))
print("PH zero shape, scale, eta, cutoff derivatives:", ph_zero)
```

## Available summaries

`rmst(tau)` is available on `Exponential`, `Weibull` and their PH wrappers.
On other registered families and PH wrappers of them it raises
`NotImplementedError`, even if they supply a differentiable survival function.
The type decides support at call or trace time; there is no generic quadrature
fallback yet. `mean()`, `median()` and `quantile(p)` raise `NotImplementedError`
on every family. `rmst(tau)` has no `options` argument.
Sampling, inverse cumulative hazard, `rmst_with_info`,
summary options and summary-info types are not provided.

For example, a custom constant-hazard family can supply direct methods
without gaining RMST support. The PH closure is private, outside the public
extension interface; custom families cannot rely on it.

```{code-cell} ipython3
from dataclasses import dataclass
from survjax import constraints


@survjax.register_family
@dataclass(frozen=True)
class ConstantHazard(survjax.SurvivalDistribution):
    rate: jax.Array = survjax.param(constraint=constraints.positive)

    def log_survival(self, t: jax.Array) -> jax.Array:
        return -self.rate * t

    def reference_time(self) -> jax.Array:
        return 1.0 / self.rate


custom = ConstantHazard(rate=0.2)
for candidate in (custom, survjax.ProportionalHazards(custom, log_hazard_ratio=eta)):
    try:
        jax.jit(candidate.rmst)(6.0)
    except NotImplementedError:
        print(type(candidate).__name__, "RMST is unsupported.")
    else:
        raise AssertionError("Expected unsupported custom-baseline RMST")
```
