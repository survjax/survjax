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
contributions at work, we use them to fit an [exponential](builtin-families.md) model to a small synthetic cohort by maximum
likelihood. survjax supplies the contributions; the fitting loop is ordinary
JAX code that you write yourself.

```{note}
Likelihoods support exact events, right, left and interval censoring, with
optional left, right or two-sided truncation on each row. These use the same
evaluator for the built-in families and registered custom families, including
PH and AFT wrappers. The examples below start with untruncated exact/right
data, then show how the contributions change when inclusion is restricted.
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

`dist.log_likelihood(...)` returns the contributions $\ell_i$, one per patient. Each
event row in this cohort uses the distribution's log density and each
right-censored row its log survival. Here is the exponential with rate 0.05 per month:

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

## Two input forms

Call the `log_likelihood` method on a distribution with data in one of two
forms. The first is a
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
from_arrays.sum(), contributions.sum()
```

Both forms agree. `time` and `event` are keyword-only and must be given
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

## Left and interval censoring, with truncation

Censoring tells us which event times remain possible. Truncation determines
which individuals can enter the sample: only those with $L<T\le U$. The
likelihood therefore conditions on inclusion, whose probability is
$Z=P(L<T\le U)$. Omitted bounds mean $L=0$, $U=+\infty$ and $Z=1$.
Under ignorable censoring and this inclusion mechanism, contributions are:

| Observation | Conditional likelihood contribution |
| --- | --- |
| Exact at $t$ | $f(t)/Z$, requiring $L<t\le U$ |
| Right censored at $c$ | $P(\max(c,L)<T\le U)/Z$ |
| Left censored at $c$ | $P(L<T\le\min(c,U))/Z$ |
| Interval censored in $(a,b]$ | $P(\max(a,L)<T\le\min(b,U))/Z$ |

survjax evaluates the logarithms directly. It does not subtract small
probabilities or floor a tiny positive $Z$. These are conditional event-time
likelihoods, not a joint model of informative censoring or arbitrary selection.
Ordinary predictions such as `model.survival(t)` remain unconditional.

Consider four observations with inclusion window $(1,4]$ months: an exact
event at 2, right censoring at 3, left censoring at 3, and an interval $(2,5]$.
The last two sets overlap the window only partly; the numerator uses their
intersection with it. Both observation constructors can describe these rows;
here the [bounds form](survival-data.md#other-observation-encodings) makes the sets explicit:

```{code-cell} ipython3
windowed = survjax.SurvivalData.from_bounds(
    [2.0, 3.0, 0.0, 2.0],
    [2.0, jnp.inf, 3.0, 5.0],
    exact=[True, False, False, False],
    left_trunc=1.0,
    right_trunc=4.0,
)
unit_rate = survjax.Exponential(rate=1.0)
windowed.issues.raise_if_invalid()
windowed_ll = unit_rate.log_likelihood(windowed)

# Independent algebra: P(a<T<=b) = exp(-a) * (1-exp(-(b-a))).
log_z = -1.0 + jnp.log(-jnp.expm1(-3.0))
expected = jnp.array([
    -2.0,                                     # density at 2
    -3.0 + jnp.log(-jnp.expm1(-1.0)),           # mass in (3,4]
    -1.0 + jnp.log(-jnp.expm1(-2.0)),           # mass in (1,3]
    -2.0 + jnp.log(-jnp.expm1(-2.0)),           # mass in (2,4]
]) - log_z
assert jnp.allclose(windowed_ll, expected)
assert jnp.allclose(jax.jit(unit_rate.log_likelihood)(windowed), expected)
windowed_ll
```

Bounds can vary by row and follow the constructor's broadcasting rules.
Right censoring before entry ($c<L$), an exact event outside $(L,U]$, and an
empty intersection are invalid observations and produce NaN. An empty
observation is different from a valid set to which the model assigns zero mass.

A censored set covering the entire inclusion window is `uninformative`, not
invalid. For example, right censoring exactly at entry gives no information
beyond inclusion and contributes exactly zero when $Z>0$:

```{code-cell} ipython3
at_entry = survjax.SurvivalData(1.0, event=False, left_trunc=1.0, right_trunc=4.0)
assert at_entry.issues.ok & at_entry.issues.uninformative
assert unit_rate.log_likelihood(at_entry) == 0.0
unit_rate.log_likelihood(at_entry)
```

## The exponential in closed form

Returning to the original untruncated exact/right cohort, we can fit the rate
in closed form. The calculation below does not apply unchanged to the general
censored/truncated example above.

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

This fixed six-step loop illustrates the derivatives; it is not a general
optimizer and has no line search, convergence test or failure reporting.

Newton lands on $6/103$. The curvature at the maximum also gives a standard
error: the observed information is $-\ell''(\hat\eta) = \hat\lambda \sum_i
t_i = D$, so $\operatorname{se}(\hat\eta) = 1/\sqrt{D} = 1/\sqrt 6 \approx
0.408$.

```{code-cell} ipython3
1 / jnp.sqrt(hessian(log_rate))
```

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

A NaN is different from $-\infty$. A contribution of $-\infty$ can mean true
zero model density or mass, or a log contribution below the dtype's finite
range. The [numerical guide](numerical-behaviour.md) explains that distinction
and its derivative limits. NaN means the contribution could not be computed. Since
$0 \times \text{NaN}$ is still NaN, a NaN cannot be masked away by a zero
weight; fix or drop the row first.

Even with clear data and parameter diagnostics, conditioning on zero model
mass ($Z=0$) gives NaN, including for a structurally uninformative row.
Detected numerical failures can also give NaN. The diagnostics do not promise
to identify every numerical failure. By contrast, a valid observation with
zero numerator and positive $Z$ contributes $-\infty$.

For a mixed-batch triage recipe, see [Diagnosing likelihood NaNs](technical-notes/likelihood-diagnostics.md).
