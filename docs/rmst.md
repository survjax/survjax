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
choice. Passing `cutoffs` directly would follow broadcasting rules, which can
pair cutoffs with a batch axis rather than creating a new grid axis.

## Zero, invalid values and argument errors

Rates, shapes and scales must be finite and strictly positive.
For valid parameters, `rmst(0)` is zero. Negative, NaN or infinite cutoffs
return NaN. Parameter-constraint violations also give NaN, including when the
cutoff is zero. Use `parameter_issues()` to inspect parameter constraints.
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
shared parameters. NaN signals invalid input, invalid parameters or numerical
failure; it does not mean a true zero RMST.

Boolean, complex, string and object cutoffs raise `TypeError`, as do shapes
that cannot broadcast against the model batch. These are argument errors;
negative and infinite numeric values instead follow the NaN policy above.
A cutoff of infinity cannot be used to request an unrestricted mean.

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

| Family / location | Derivative support |
| --- | --- |
| Exponential | All orders on valid inputs |
| Weibull, smooth interior with `tau > 0` | First derivatives in shape, scale and cutoff; higher derivatives that do not involve shape |
| Weibull at `tau = 0` | Value and first derivatives only: parameter derivatives 0, cutoff derivative 1 |

Weibull second- and higher-order derivatives involving shape are unsupported:
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
```

At zero, the Weibull boundary guarantee stops at first order. Higher right
cutoff derivatives can be nonfinite or differ from autodiff's result there.
Use a valid smooth interior point for supported higher derivatives.

```{code-cell} ipython3
assert restricted(2.0, 10.0, 0.0) == 0
assert jax.grad(restricted, argnums=0)(2.0, 10.0, 0.0) == 0
assert jax.grad(restricted, argnums=1)(2.0, 10.0, 0.0) == 0
assert jax.grad(restricted, argnums=2)(2.0, 10.0, 0.0) == 1
```

## Scope of this release step

`rmst(tau)` currently works on the merged `Exponential` and `Weibull` families.
On other registered families it raises `NotImplementedError`, even if they
supply a differentiable survival function; there is no generic quadrature
fallback yet. `mean()`, `median()` and `quantile(p)` raise `NotImplementedError`
on every family. `rmst(tau)` has no `options` argument.
Sampling, inverse cumulative hazard, `rmst_with_info`,
summary options and summary-info types are not provided.

PH-wrapper RMST is a separate pending release integration. Its closure
examples and joint verification will be added after that implementation
merges; this guide does not require or import an unmerged wrapper.
