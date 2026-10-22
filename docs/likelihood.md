---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Log likelihoods

This page explains how survjax computes log likelihoods for survival data:
which quantity each observation contributes, and how to call the likelihood
with prepared `SurvivalData` or with plain time and event arrays. To show the
contributions at work, we use them to fit [exponential and
Weibull](builtin-families.md) models to a small synthetic cohort by maximum
likelihood. survjax supplies the contributions; the fitting loop is ordinary
JAX code that you write yourself.

```{note}
survjax currently evaluates likelihoods for exact events and right
censoring, without truncation. Interval and left censoring and truncation
can already be described with `SurvivalData`, but their likelihood
contributions are NaN for now (see the last section).
```

## The cohort

Ten patients were followed from diagnosis. Time is in months and the event
is disease progression. Six patients progressed; four were still progression
free when follow-up ended, so their event time is only known to be later:
they are *right censored*.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import survjax

months = jnp.array([4.0, 12.0, 7.0, 9.0, 15.0, 2.0, 20.0, 11.0, 5.0, 18.0])
progressed = jnp.array([1, 0, 1, 0, 1, 1, 0, 1, 1, 0])
cohort = survjax.SurvivalData(months, progressed)
cohort
```

## The likelihood of a censored sample

Suppose $T$ has density $f$ and survival function $S(t) = P(T > t)$. A
patient who progressed at time $t_i$ tells us the density there, $f(t_i)$.
A patient censored at $t_i$ tells us only that $T > t_i$, which has
probability $S(t_i)$. With $d_i = 1$ for an event and $d_i = 0$ for
censoring, and patients independent, the likelihood of parameters $\theta$
is

$$
L(\theta) = \prod_i f(t_i)^{d_i}\, S(t_i)^{1 - d_i}.
$$

We work with its logarithm. Each patient adds one *contribution*,

$$
\ell_i(\theta) = d_i \log f(t_i) + (1 - d_i) \log S(t_i),
\qquad
\ell(\theta) = \sum_i \ell_i(\theta).
$$

For the exponential with rate $\lambda$, $\log f(t) = \log\lambda - \lambda
t$ and $\log S(t) = -\lambda t$, so

$$
\ell_i(\lambda) = d_i \log\lambda - \lambda t_i,
\qquad
\ell(\lambda) = D \log\lambda - \lambda \sum_i t_i,
$$

where $D = \sum_i d_i$ is the number of events.

## Contributions in survjax

`log_likelihood` returns the contributions $\ell_i$, one per patient. Each
event row uses the distribution's log density and each censored row its log
survival. Here is the exponential with rate 0.05 per month:

```{code-cell} ipython3
model = survjax.Exponential(rate=0.05)
contributions = model.log_likelihood(cohort)
contributions
```

Patient 1 progressed at month 4: $\log 0.05 - 0.05 \cdot 4 \approx -3.196$.
Patient 2 was censored at month 12: $-0.05 \cdot 12 = -0.6$.

The result has one entry per observation; survjax does not sum for you. The
log likelihood is the sum, and how you reduce (a sum, a mean, a weighted
sum) is your choice:

```{code-cell} ipython3
contributions.sum()
```

## Two entry points, two input forms

The likelihood is available both as a method of every distribution and as a
top-level function. They compute the same thing, including derivatives:

```python
model.log_likelihood(cohort)  # method
survjax.log_likelihood(model, cohort)  # function: distribution first
```

Use whichever reads better in your code. The function form is convenient
when the distribution is one argument among several, for example in a
generic objective `objective(dist, data)`.

Either entry point accepts the data in one of two forms. The first is a
`SurvivalData`, passed positionally as above. It is the general form: it
holds every kind of observation that survjax can describe, and it records
[data diagnostics](survival-data.md) at construction.

The second form is for plain time/status data, the common case of exact
events and right censoring. Pass the two arrays as the keywords `time` and
`event`, with `event` 1 (or `True`) for an event and 0 (or `False`) for
censoring, as in R's `Surv(time, status)`. survjax builds the `SurvivalData`
internally:

```{code-cell} ipython3
from_arrays = model.log_likelihood(time=months, event=progressed)
from_function = survjax.log_likelihood(model, time=months, event=progressed)
from_arrays.sum(), from_function.sum(), contributions.sum()
```

All three agree. `time` and `event` are keyword-only and must be given
together, and they cannot be combined with a `SurvivalData`. `event`
broadcasts against `time`, so `event=1` marks every observation as an event. A raw array in
the data position is a mistake in the call, so it raises `TypeError`
immediately, also while JAX traces a function:

```{code-cell} ipython3
:tags: [raises-exception]

model.log_likelihood(months)
```

Constructor options, such as a dtype or a richer censoring type, are not
accepted by the likelihood; build a `SurvivalData` for those.

## The exponential in closed form

Setting the derivative of $\ell(\lambda)$ to zero,

$$
\frac{d\ell}{d\lambda} = \frac{D}{\lambda} - \sum_i t_i = 0
\quad\Longrightarrow\quad
\hat\lambda = \frac{D}{\sum_i t_i}:
$$

events divided by total time at risk. Our cohort has $D = 6$ events in 103
months of follow-up, so $\hat\lambda = 6/103 \approx 0.0583$ per month.

To see the likelihood as a function of $\lambda$, evaluate many rates at
once. A distribution with parameters of shape `(m, 1)` broadcasts against
the `(10,)` patients to give an `(m, 10)` array of contributions; summing
over the last axis gives one log likelihood per rate.

```{code-cell} ipython3
rates = jnp.linspace(0.02, 0.15, 200)
profile = survjax.Exponential(rate=rates[:, None]).log_likelihood(cohort).sum(axis=-1)

fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(rates, profile)
ax.axvline(6 / 103, color="grey", linestyle=":")
ax.set(xlabel="rate λ (per month)", ylabel="log likelihood ℓ(λ)")
fig.tight_layout()
```

The maximum sits at $\hat\lambda = 6/103$ (dotted line).

## Using the likelihood to fit a model

Most models have no closed-form maximum, so we fit numerically. Rates must
be positive; optimising the log rate $\eta = \log\lambda$ removes that
constraint. In $\eta$,

$$
\ell(\eta) = D\eta - e^{\eta} \sum_i t_i,
\qquad
\ell'(\eta) = D - e^{\eta} \sum_i t_i,
\qquad
\ell''(\eta) = -e^{\eta} \sum_i t_i.
$$

Newton's method updates $\eta \leftarrow \eta - \ell'(\eta)/\ell''(\eta)$.
We let JAX compute both derivatives of the negative log likelihood:

```{code-cell} ipython3
def exponential_nll(log_rate):
    model = survjax.Exponential.from_log(log_rate=log_rate)
    return -model.log_likelihood(cohort).sum()


gradient = jax.jit(jax.grad(exponential_nll))
hessian = jax.jit(jax.hessian(exponential_nll))

log_rate = jnp.log(1 / months.mean())  # a rough starting value
for _ in range(6):
    log_rate = log_rate - gradient(log_rate) / hessian(log_rate)

jnp.exp(log_rate)
```

Newton lands on $6/103$. The curvature at the maximum also gives a standard
error: the observed information is $-\ell''(\hat\eta) = \hat\lambda \sum_i
t_i = D$, so $\operatorname{se}(\hat\eta) = 1/\sqrt{D} = 1/\sqrt 6 \approx
0.408$.

```{code-cell} ipython3
1 / jnp.sqrt(hessian(log_rate))
```

## A Weibull fit

The Weibull with shape $k$ and scale $b$ has

$$
\log f(t) = \log\frac{k}{b} + (k - 1)\log\frac{t}{b} - \left(\frac{t}{b}\right)^{k},
\qquad
\log S(t) = -\left(\frac{t}{b}\right)^{k}.
$$

Setting both partial derivatives of $\ell(k, b)$ to zero has no closed-form
solution, but the numerical recipe is the same with two parameters, $(\log k,
\log b)$. With a parameter vector $\theta$, the Newton step divides by the
Hessian matrix instead of a scalar second derivative,
$\theta \leftarrow \theta - \left[\nabla^2 \ell(\theta)\right]^{-1} \nabla \ell(\theta)$,
which the code computes by solving the 2×2 linear system with
`jnp.linalg.solve` rather than forming the inverse. We start at the
exponential fit, which is the Weibull with $k = 1$ and $b = 1/\hat\lambda$:

```{code-cell} ipython3
def weibull_nll(log_params):
    model = survjax.Weibull.from_log(log_shape=log_params[0], log_scale=log_params[1])
    return -model.log_likelihood(cohort).sum()


gradient = jax.jit(jax.grad(weibull_nll))
hessian = jax.jit(jax.hessian(weibull_nll))

log_params = jnp.array([0.0, -log_rate])
for _ in range(6):
    log_params = log_params - jnp.linalg.solve(hessian(log_params), gradient(log_params))

jnp.exp(log_params)
```

Because JAX supplies exact gradients and Hessians of the likelihood, a
hand-written Newton method like this takes only a few lines. For larger
models, a dedicated optimizer is the better tool; a later version of these
guides shows fits with JAX-based optimizer libraries such as
[Optimistix](https://docs.kidger.site/optimistix/).

The fitted shape is about 1.29, a hazard that rises slowly with time, and
the scale is about 16 months. Is the extra parameter worth it? The
exponential is the Weibull with $k = 1$, so the likelihood-ratio statistic
$2(\ell_{\text{Weibull}} - \ell_{\text{exp}})$ is approximately
$\chi^2_1$ distributed under the exponential:

```{code-cell} ipython3
2 * (exponential_nll(log_rate) - weibull_nll(log_params))
```

A statistic of about 0.47 is far below the 5% critical value of 3.84, so
this sample provides little evidence against a constant hazard.

## When a contribution is NaN

survjax never raises inside a likelihood for a bad *value*, because it must
work under `jit` and `vmap`. Instead, that patient's contribution is NaN and
the others are unaffected. Two diagnostics explain why. Suppose the last
status was mistyped as 2:

```{code-cell} ipython3
typo = survjax.SurvivalData(months, progressed.at[-1].set(2))
model.log_likelihood(typo)[-3:], typo.issues
```

A negative rate is reported by `parameter_issues()`:

```{code-cell} ipython3
survjax.Exponential(rate=-0.05).parameter_issues()
```

Outside JIT, at a point where you want to stop on bad input, call
`raise_if_invalid()`:

```{code-cell} ipython3
:tags: [raises-exception]

typo.issues.raise_if_invalid()
```

A NaN is different from $-\infty$. A contribution of $-\infty$ means the
model gives the observation zero probability: the data are valid, the model
just rules them out. NaN means the contribution could not be computed. Since
$0 \times \text{NaN}$ is still NaN, a NaN cannot be masked away by a zero
weight; fix or drop the row first.

The third cause of NaN is an observation that is valid but whose likelihood
is not implemented yet, such as interval censoring, left censoring or any
truncation. An unsupported mode is not bad data, so it adds no data or
parameter issue.

### Telling the causes apart

A real batch can mix all of these. Take a valid event at month 4, an exact
time of $-1$ (invalid: event times must be positive), patient 5 from the
[data guide](survival-data.md), who progressed between scans at months 6 and
9 (interval censored), and a patient censored at month 12:

```{code-cell} ipython3
nan = jnp.nan
mixed = survjax.SurvivalData(
    [4.0, -1.0, 6.0, 12.0],
    right_censored=[False, False, False, True],
    interval_censored=[False, False, True, False],
    time2=[nan, nan, 9.0, nan],
)
ll = model.log_likelihood(mixed)
ll
```

Two rows are NaN, for different reasons. To explain each NaN, first remove
those flagged by the data and parameter diagnostics. Then check which of
the rest use a mode that the current version supports: exact or right
censored ($\text{upper} = \infty$), with no truncation. The diagnostic masks
are broadcast to the shape of the result, since a batch of distributions
can enlarge it:

```{code-cell} ipython3
bad_data = jnp.broadcast_to(mixed.issues.invalid_rows, ll.shape)
bad_params = jnp.broadcast_to(model.parameter_issues().invalid_entries, ll.shape)
supported_mode = (
    (mixed.left_trunc == 0)
    & jnp.isposinf(mixed.right_trunc)
    & (mixed.exact | jnp.isposinf(mixed.upper))
)
unsupported = jnp.broadcast_to(~supported_mode, ll.shape)
unexplained = jnp.isnan(ll) & ~bad_data & ~bad_params
unsupported_nan = unexplained & unsupported
other = unexplained & ~unsupported  # zero normalization or numerical failure
bad_data, unsupported_nan, other
```

The second row is bad data, the third an unsupported mode, and nothing is
left over. A NaN in `other` would point to a numerical problem worth
investigating. This recipe is for data as constructed; it does not cover
`SurvivalData` whose fields were edited afterwards. For raw `time`/`event` input, build
`SurvivalData(time, event)` to read its `issues`. As later versions
implement more modes, `supported_mode` grows.

See the [API reference](api.md) for the full signatures.
