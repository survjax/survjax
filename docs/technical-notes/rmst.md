---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# RMST numerics and derivatives

For interpretation and basic calls, see [RMST](../rmst.md).

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import survjax
```

```{code-cell} ipython3
model = survjax.Exponential(rate=0.2)
weibull = survjax.Weibull(shape=1.0, scale=5.0)
eta = jnp.log(2.0)
ph_weibull = survjax.ProportionalHazards(
    survjax.Weibull(shape=2.0, scale=10.0), log_hazard_ratio=eta,
)
```

The implementation uses a series in the lower tail and the mean in the far
upper tail, with the gamma expression between them. Value comparisons target
relative accuracy of `1e-8` in float64 and `1e-4` in float32 over
$k\in[0.2,50]$ and $\tau/b\in[0.01,10^5]$, including underflow and overflow
of the power $(\tau/b)^k$. Accuracy outside that domain is not promised;
numerical failure can produce NaN even for positive parameters.
The mathematical approximation bounds and safe arithmetic are explained in the
{download}`Weibull RMST appendix <../../appendices/weibull-rmst.pdf>`.

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
