---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Likelihoods in the tails

An extremely unlikely observation can have a useful log likelihood even when
its density or survival probability is too small to store. This page shows how to read those
values, distinguish them from numerical overflow and invalid inputs, and
check a loss before using its gradient in a fit.

We use the current likelihood subset: exact events and right censoring,
without truncation. For call signatures and fitting examples, see
[Log likelihoods](likelihood.md).

## Keep the logarithm

For a Weibull with shape $k$ and scale $b$, the cumulative hazard is
$H(t)=(t/b)^k$, and $S(t)=e^{-H(t)}$. The observation's log contribution is

$$
\ell_i =
\begin{cases}
\log(k/b)+(k-1)\log(t_i/b)-H(t_i), & \text{exact event},\\
-H(t_i), & \text{right censored}.
\end{cases}
$$

Consider three synthetic observations in months: no follow-up (censored at
zero), an event at month 1, and censoring at month 100. The hypothetical
model has shape 2 and scale 1 month:

```{code-cell} ipython3
import jax
import jax.numpy as jnp

import survjax

months = jnp.array([0.0, 1.0, 100.0])
event = jnp.array([False, True, False])
data = survjax.SurvivalData(months, event)
model = survjax.Weibull(shape=2.0, scale=1.0)

print("dtype:", months.dtype)
print("columns: survival, log survival, log contribution")
jnp.column_stack(
    (model.survival(months), model.log_survival(months), model.log_likelihood(data))
)
```

At month 100, $H=10000$. Survival is mathematically positive, but
$e^{-10000}$ underflows to zero in both float32 and float64. The log
contribution stays finite at $-10000$. Computing a logarithm *after*
evaluating survival has already lost that information:

```{code-cell} ipython3
jnp.log(model.survival(100.0)), model.log_survival(100.0)
```

The first calculation returns $-\infty$; `log_survival` stays close to the
mathematical value $-10000$. Float32 prints about `-10000.001` for the latter
because of floating-point rounding.
Use `log_survival` for a right-censored
log contribution, `log_prob` for an exact event, or `log_likelihood` to select
the appropriate quantity from data.

The other tail has the opposite rounding problem. Censoring at $c=10^{-10}$
has $H(c)=10^{-20}$, so $S(c)$ rounds to 1 but $\log S(c)=-10^{-20}$ is
still representable:

```{code-cell} ipython3
early = jnp.asarray(1e-10)
model.survival(early), model.log_likelihood(time=early, event=False)
```

The small negative log contribution carries information that would disappear
if it were rounded to zero. survjax does not add an epsilon or floor a
likelihood. See
[precision in both tails](builtin-families.md#precision-in-both-tails) for
the direct methods, including `log_cdf`.

## When the log contribution also overflows

Log space extends the useful range; the log value itself still has to fit
in the dtype. For shape 50, scale 1 and censoring at $10^{10}$ months,

$$
H(c)=(10^{10})^{50}=10^{500},\qquad \log S(c)=-10^{500}.
$$

That finite mathematical log value is outside the finite range of both
float32 and float64. The returned value is $-\infty$, with no data or
parameter issue:

```{code-cell} ipython3
extreme = survjax.Weibull(shape=50.0, scale=1.0)
late = survjax.SurvivalData(1e10, event=False)

extreme.log_likelihood(late), late.issues.ok, extreme.parameter_issues().ok
```

This is overflow of the log contribution, rather than a claim that the
model assigns exactly zero mass. Both can return $-\infty$:

| Result | Meaning in the supported likelihood subset |
| --- | --- |
| Finite log contribution | A representable log density or log survival, even if the corresponding density/probability underflows |
| $-\infty$ from true zero | The model has zero density or mass for the observation, for example outside a custom family's bounded support |
| $-\infty$ from overflow | The mathematical log contribution is below the dtype's finite range |
| NaN | Invalid data/declared parameters, an unsupported mode, or a numerical failure |

The scalar value alone does not identify the cause. Use data and parameter
diagnostics and the model's support. A nonzero contribution representable in
the dtype's normal range must not be relabelled as zero. Float64 can delay overflow, but it
cannot represent $-10^{500}$ either. survjax follows JAX's configured dtype;
it does not turn on 64-bit mode itself.

## Zero censoring is an exact boundary value

Censoring at $c=0$ tells us only that $T>0$. For these continuous event-time
models, $S(0)=1$, so the contribution is exactly zero and its parameter
derivatives are zero. This remains true for a Weibull with $k<1$, whose
density at zero is infinite:

```{code-cell} ipython3
def no_follow_up(shape):
    candidate = survjax.Weibull(shape=shape, scale=1.0)
    return candidate.log_likelihood(time=0.0, event=False)


no_follow_up(0.5), jax.grad(no_follow_up)(0.5)
```

The likelihood substitutes a safe reference time for the inactive density
evaluation. Computing an unsafe density at zero and multiplying it by zero
would not protect gradients. This also matters for
[custom families](distributions.md): right censoring at zero does not require
their generic density default to have a finite value there.
An **exact event** at zero is a different observation and is invalid.

## Check the loss as well as the gradient

For an exponential rate $r$, an exact event at month 2 contributes
$\log r-2r$, with derivative $1/r-2$. Its negative log contribution has
derivative 1 at $r=1$. Now give a second, censored observation an invalid
rate of $-1$:

```{code-cell} ipython3
def negative_log_likelihood(rates):
    candidate = survjax.Exponential(rate=rates)
    contributions = candidate.log_likelihood(
        time=jnp.array([2.0, 1.0]), event=jnp.array([True, False])
    )
    return -contributions.sum()


proposal = jnp.array([1.0, -1.0])
loss, gradient = jax.value_and_grad(negative_log_likelihood)(proposal)
loss, gradient, jnp.isfinite(loss), jnp.all(jnp.isfinite(gradient))
```

The loss is NaN, yet the gradient is finite: `[1, 0]`. The second row has a
declared parameter violation, and the likelihood stops its parameter-leaf
gradients before evaluating family code. A gradient-only update could
therefore continue while silently omitting that row.

At a host checkpoint in a fitting loop, check the loss before using an update
and inspect diagnostics when it is not finite:

```{code-cell} ipython3
if not bool(jnp.isfinite(loss)):
    print("Reject this proposal: the loss is not finite.")
    print(survjax.Exponential(rate=proposal).parameter_issues().invalid_entries)
```

This reports the invalid second rate. A nonfinite loss can also arise from
the overflow above; a clear issue report alone does not certify a finite
objective. Choosing the next proposal or terminating the fit is the caller's
decision. Inside compiled code, keep the finite-loss check as a JAX boolean
and handle it with compiled control flow; the Python checkpoint above is
for concrete results outside transformations.

## The limits of derivative isolation

First and second parameter derivatives are guaranteed for the built-in
likelihoods on finite, valid, smooth interior cases, including `jit` and
`vmap`. Boundary derivatives require the relevant derivative to exist;
the zero-censoring example above is constant in the parameters.
These guarantees concern parameters, not observation-time derivatives,
support-change boundaries or derivative orders above two.

At invalid or infinite likelihoods, derivatives are outside that guarantee.
With **shared parameters**, an infinite row can also contaminate reverse-mode
derivatives of finite rows in the same call through $0\times\infty$.
Forward-mode derivatives and per-row parameters do not have that shared
reverse-path contamination. Do not assume that selecting a finite row after
evaluating an overflowing batch repairs its reverse derivative.

Isolation acts at the distribution's parameter leaves. A caller's earlier
transformation such as `rate=jnp.exp(X @ beta)`, including the exponential
inside `from_log`, is outside its reach. If that step overflows, reverse
derivatives for shared upstream parameters can still become NaN. Keep those
transformations in a representable range when fitting.

For NaN contributions, follow the [likelihood guide's diagnostic recipe](likelihood.md#telling-the-causes-apart).
Its exact attribution applies to data as constructed; edited normalized
fields can introduce invalidity beyond the stored issue bits.
