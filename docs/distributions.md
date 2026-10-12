---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Writing a survival distribution

```{note}
This page shows the extension interface: how to describe your own
distribution and evaluate it. Built-in families (`Exponential`, `Weibull`),
shared time grids and likelihoods for `SurvivalData` are planned for later
steps of Slice A. Until then, every distribution on this page is one we write
ourselves.
```

## One function describes the whole distribution

Let $T$ be the time from diagnosis to disease progression. Everything we want
to know about $T$ follows from its *survival function*, the probability of
still being progression free at time $t$:

$$
S(t) = P(T > t).
$$

The other familiar quantities are transformations of it:

| Quantity | Formula | survjax method |
| --- | --- | --- |
| log survival | $\log S(t)$ | `log_survival(t)` |
| survival | $S(t)$ | `survival(t)` |
| cumulative hazard | $H(t) = -\log S(t)$ | `cumulative_hazard(t)` |
| hazard | $h(t) = H'(t) = -\dfrac{d}{dt}\log S(t)$ | `hazard(t)`, `log_hazard(t)` |
| density | $f(t) = h(t)\,S(t)$ | `log_prob(t)` (log density) |
| CDF | $F(t) = 1 - S(t)$ | `log_cdf(t)` |

So a family in survjax needs to supply only $\log S(t)$. survjax derives the
rest, using JAX's automatic differentiation for the hazard and density. We
work with $\log S$ rather than $S$ because survival probabilities in the far
tail underflow to zero long before their logarithm becomes unrepresentable.

## A first family: constant hazard

Suppose progression happens at a constant rate of $\lambda = 0.1$ per month,
whatever the time since diagnosis. Then $H(t) = \lambda t$ and

$$
\log S(t) = -\lambda t.
$$

In survjax a family is a frozen dataclass with one field per parameter.

```{code-cell} ipython3
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import survjax


@survjax.register_family
@dataclass(frozen=True)
class ConstantHazard(survjax.SurvivalDistribution):
    rate: jax.Array = survjax.param()

    def log_survival(self, t):
        return -self.rate * t

    def reference_time(self):
        # The mean time to event, 1 / rate: a typical time on this scale.
        return 1.0 / self.rate
```

Four things make this a family:

- it subclasses `survjax.SurvivalDistribution`;
- `@dataclass(frozen=True)` sits directly above the class, and
  `@survjax.register_family` above that;
- every field is declared with `survjax.param()` (and later, `static()` or
  `distribution()`);
- it implements `log_survival(t)` and `reference_time()`.

`reference_time()` returns some positive, finite time where the distribution
is well behaved. survjax uses it as a safe place to evaluate the family when a
requested time cannot be used directly, for example a negative or NaN time.
A natural time scale such as the mean or median is a good choice.

Now ask the model some questions. Parameters are passed by keyword.

```{code-cell} ipython3
progression = ConstantHazard(rate=0.1)

months = jnp.array([6.0, 12.0, 24.0])
progression.survival(months)
```

The probability of being progression free at one year is
$e^{-0.1 \cdot 12} = e^{-1.2} \approx 0.301$. The median, where $S(t) = 1/2$,
is $\log 2 / \lambda \approx 6.93$ months:

```{code-cell} ipython3
progression.survival(jnp.log(2.0) / 0.1)
```

The hazard and density we never wrote are available too. With a constant
hazard, $h(t) = 0.1$ at every time and $f(t) = 0.1\,e^{-0.1 t}$:

```{code-cell} ipython3
progression.hazard(months), progression.cumulative_hazard(months)
```

```{code-cell} ipython3
progression.log_prob(months), jnp.log(0.1) - 0.1 * months
```

## The hazard comes from autodiff: Gompertz mortality

A constant hazard is too simple for many questions. Adult human mortality
famously follows the *Gompertz law*: the hazard of death grows exponentially
with age,

$$
h(t) = b\,e^{c t},
\qquad
H(t) = \frac{b}{c}\left(e^{c t} - 1\right),
\qquad
\log S(t) = -\frac{b}{c}\left(e^{c t} - 1\right).
$$

Take $t$ as years since age 30, a hazard of $b = 0.0005$ per year at age 30,
and $c = 0.085$, so that the hazard doubles every $\log 2 / c \approx 8.2$
years. We write only $\log S$, using `jnp.expm1` so that small $c t$ keeps
full precision.

```{code-cell} ipython3
@survjax.register_family
@dataclass(frozen=True)
class Gompertz(survjax.SurvivalDistribution):
    baseline_hazard: jax.Array = survjax.param()  # b, hazard at t = 0
    growth: jax.Array = survjax.param()  # c, per unit time

    def log_survival(self, t):
        return -(self.baseline_hazard / self.growth) * jnp.expm1(self.growth * t)

    def reference_time(self):
        # One hazard e-folding time, 1 / c.
        return 1.0 / self.growth


mortality = Gompertz(baseline_hazard=0.0005, growth=0.085)
years_since_30 = jnp.array([20.0, 40.0, 50.0])  # ages 50, 70, 80
mortality.survival(years_since_30)
```

The hazard is computed as $-\frac{d}{dt}\log S(t)$ by forward-mode autodiff.
It agrees with the closed form $b\,e^{c t}$ that we never gave survjax:

```{code-cell} ipython3
mortality.hazard(years_since_30), 0.0005 * jnp.exp(0.085 * years_since_30)
```

## Many distributions at once

Parameters can be arrays. Each entry along the leading axes is a separate
distribution, and `batch_shape` reports those axes. Here a trial has a
control arm and a treated arm whose progression rate is 30% lower:

```{code-cell} ipython3
arms = ConstantHazard(rate=jnp.array([0.1, 0.07]))
arms.batch_shape
```

Indexing selects distributions, just as indexing an array selects entries:

```{code-cell} ipython3
treated = arms[1]
treated.rate, treated.batch_shape
```

Times broadcast against the batch with the usual NumPy rules, matching
trailing axes. Two times and two arms therefore pair up: arm 0 at month 6 and
arm 1 at month 12.

```{code-cell} ipython3
arms.survival(jnp.array([6.0, 12.0]))
```

To evaluate every arm at every time, insert an axis with `None`. The result
has one row per arm and one column per time: two survival curves.

```{code-cell} ipython3
months = jnp.array([0.0, 6.0, 12.0, 24.0])
arms[:, None].survival(months)
```

Parameters broadcast against each other at construction, so a scalar
parameter is shared across the batch. Shared time-first grids, which put the
time axis first and need no manual axis, are planned with the built-in
families.

## Fixed settings and vector parameters: a piecewise-constant hazard

Progression risk is often highest soon after diagnosis. A
*piecewise-constant hazard* (the model behind piecewise exponential
regression) splits follow-up at fixed cut points and gives each interval its
own rate. Over months 0–6, 6–12 and 12 onward:

$$
h(t) = \lambda_j \text{ for } t \text{ in interval } j,
\qquad
H(t) = \sum_j \lambda_j \times (\text{time spent in interval } j \text{ by } t).
$$

This family needs two new kinds of field:

- the cut points are a fixed design choice, not something to fit or
  differentiate. Declare them with `survjax.static()`; they must be hashable,
  so use a tuple.
- the rates form a *vector* per distribution. `param(component_ndim=1)`
  says the last axis holds the components, so `rates` of shape `(3,)` is one
  distribution with three rates, and shape `(2, 3)` is a batch of two.

```{code-cell} ipython3
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
relapse.batch_shape
```

The batch shape is `()`: one distribution, whose three rates are components.
At one year, $H(12) = 6 \times 0.08 + 6 \times 0.05 = 0.78$, so
$S(12) = e^{-0.78} \approx 0.458$. The hazard steps down at each cut point:

```{code-cell} ipython3
relapse.survival(12.0), relapse.hazard(jnp.array([3.0, 9.0, 18.0]))
```

Two hospitals with different rates form a batch of two. The component axis
stays at the end of the parameter, and indexing the batch keeps it:

```{code-cell} ipython3
hospitals = PiecewiseHazard(
    starts=(0.0, 6.0, 12.0),
    rates=jnp.array([[0.08, 0.05, 0.02], [0.06, 0.04, 0.03]]),
)
hospitals.batch_shape, hospitals[1].rates
```

Static settings are part of the distribution's structure rather than its
numbers. JAX treats different cut points as a different function and compiles
it again, while new rates reuse the compiled code.

## Building on another distribution: proportional hazards

A *proportional hazards* model multiplies a baseline hazard by a hazard ratio
$\theta$ for each patient: $h(t) = \theta\,h_0(t)$. Integrating gives
$H(t) = \theta H_0(t)$, so

$$
\log S(t) = \theta \log S_0(t).
$$

A family can hold another distribution as a field, declared with
`survjax.distribution()`. Here the baseline is the piecewise hazard above,
shared by three patients with different hazard ratios:

```{code-cell} ipython3
@survjax.register_family
@dataclass(frozen=True)
class ProportionalHazards(survjax.SurvivalDistribution):
    baseline: survjax.SurvivalDistribution = survjax.distribution()
    hazard_ratio: jax.Array = survjax.param()

    def log_survival(self, t):
        return self.hazard_ratio * self.baseline.log_survival(t)

    def reference_time(self):
        return self.baseline.reference_time()


patients = ProportionalHazards(baseline=relapse, hazard_ratio=jnp.array([1.0, 0.7, 1.5]))
patients.batch_shape
```

The single baseline is broadcast to the three patients, keeping its rate
components and cut points. One-year survival is $e^{-0.78\,\theta}$ for each
patient, and each hazard at month 3 is $0.08\,\theta$:

```{code-cell} ipython3
patients.survival(12.0), patients.hazard(3.0)
```

```{code-cell} ipython3
patients[2].baseline.rates.shape, patients[2].baseline.starts
```

Built-in proportional hazards and accelerated failure time wrappers are
planned for a later slice; this example only shows how nesting works.

## Distributions are JAX values

A registered family is a JAX *PyTree*: its parameters are leaves and its
static settings are structure. So distributions can be passed through
`jax.jit`, `jax.vmap` and `jax.grad` like arrays.

To see this, take the first four patients from the
[SurvivalData guide](survival-data.md): progression at months 4 and 7, and
censoring at months 12 and 9. With event indicator $d_i$, the log-likelihood
for a constant hazard is

$$
\ell(\lambda) = \sum_i \bigl[d_i \log h(t_i) + \log S(t_i)\bigr]
= \Bigl(\sum_i d_i\Bigr)\log\lambda - \lambda \sum_i t_i,
$$

which is maximised at $\hat\lambda = \sum_i d_i / \sum_i t_i = 2/32 = 0.0625$
per month. survjax will compute likelihoods for `SurvivalData` directly in a
later step; here we write the formula out to see the pieces.

```{code-cell} ipython3
time = jnp.array([4.0, 12.0, 7.0, 9.0])
event = jnp.array([1.0, 0.0, 1.0, 0.0])


def log_likelihood(dist):
    return jnp.sum(event * dist.log_hazard(time) + dist.log_survival(time))


log_likelihood(ConstantHazard(rate=0.0625))
```

`jax.grad` differentiates with respect to the parameters. The score
$\ell'(\lambda) = 2/\lambda - 32$ is zero at the maximum, and at
$\lambda = 1/24$ it is $48 - 32 = 16$:

```{code-cell} ipython3
score = jax.grad(lambda rate: log_likelihood(ConstantHazard(rate=rate)))
score(0.0625), score(1 / 24)
```

`jax.vmap` maps over a batch of distributions; here, a profile of the
log-likelihood over a grid of rates. The largest value is at 0.0625:

```{code-cell} ipython3
rates = jnp.array([0.04, 0.05, 0.0625, 0.08, 0.1])
jax.vmap(log_likelihood)(ConstantHazard(rate=rates))
```

and `jax.jit` compiles a function that takes a distribution:

```{code-cell} ipython3
jax.jit(log_likelihood)(ConstantHazard(rate=0.0625))
```

## The edges of the time axis

Times are measured from a time origin such as diagnosis, so $T > 0$. survjax
answers for any real time without raising, and your `log_survival` only needs
to handle $t \ge 0$, including $t = \infty$:

| Time | Meaning | What survjax returns |
| --- | --- | --- |
| $t < 0$ | before the time origin | $S = 1$, $H = 0$, $h = 0$, log density and log CDF $-\infty$ |
| $t = 0$ | the origin | your family's values, $S(0) = 1$ |
| $t = \infty$ | never | your family's values, usually $S = 0$; log density is $-\infty$, or NaN if log survival is NaN |
| NaN | missing | NaN |

survjax does not pass negative or NaN times to your code. It substitutes the
reference time, evaluates, and then replaces the result.

```{code-cell} ipython3
edge_times = jnp.array([-1.0, 0.0, jnp.inf, jnp.nan])
progression.survival(edge_times), progression.log_prob(edge_times)
```

At infinity, survjax checks your `log_survival` even if you override
`log_prob`. A NaN survival limit gives NaN log density; a defined survival
limit gives minus infinity, including when the density formula itself has
an undefined infinity-minus-infinity expression.

```{code-cell} ipython3
invalid_progression = ConstantHazard(rate=jnp.nan)
assert jnp.isnan(invalid_progression.log_prob(jnp.inf))
assert progression.log_prob(jnp.inf) == -jnp.inf
```

For the generic `survival` and `log_cdf`, a raw log-survival limit of
minus infinity gives zero with zero parameter gradient. survjax evaluates
the default at the reference time before selecting that boundary value.
A finite limit stays active: for a cure fraction $p$, $S(\infty)=p$ and
$\log F(\infty)=\log(1-p)$ retain their derivatives in $p$.

Your primitive must still use safe operands wherever differentiation is
needed. For example, this cure family substitutes before multiplying by
infinity, then selects the remaining susceptible fraction. It preserves NaN
from the parameter calculation rather than hiding it behind the limit.

```{code-cell} ipython3
@survjax.register_family
@dataclass(frozen=True)
class CureHazard(ConstantHazard):
    fraction: jax.Array = survjax.param()

    def log_survival(self, t):
        time = jnp.asarray(t)
        infinite = jnp.isposinf(time)
        safe_time = jnp.where(infinite, jax.lax.stop_gradient(jnp.ones_like(time)), time)
        remaining = jnp.exp(-self.rate * safe_time)
        remaining = jnp.where(infinite & ~jnp.isnan(remaining), 0.0, remaining)
        return jnp.log(self.fraction + (1.0 - self.fraction) * remaining)


assert jax.grad(lambda r: ConstantHazard(rate=r).survival(jnp.inf))(0.1) == 0.0
assert jax.grad(lambda r: ConstantHazard(rate=r).log_cdf(jnp.inf))(0.1) == 0.0
assert jax.grad(lambda p: CureHazard(rate=0.1, fraction=p).survival(jnp.inf))(0.25) == 1.0
assert jnp.isclose(
    jax.grad(lambda p: CureHazard(rate=0.1, fraction=p).log_cdf(jnp.inf))(0.25),
    -1.0 / 0.75,
)
assert jnp.isnan(CureHazard(rate=jnp.nan, fraction=0.25).survival(jnp.inf))
```

This guard is specific to the generic methods and a zero survival limit at
positive infinity. Family overrides remain responsible for their own boundary
operands; finite times, even in an underflowing tail, are evaluated as given.

Parameter values are not checked. A negative rate is stored as given and
produces whatever the formula gives; validating parameters is up to the
caller.

## Why work on the log scale

The derived methods avoid computing $S$, and then subtracting or dividing,
where that would lose precision. Two cases come up in practice.

**Very early events.** Just after diagnosis, $F(t) = 1 - S(t)$ is tiny. With
$\lambda t = 10^{-17}$, $S(t)$ rounds to exactly 1 in floating point, so the
naive $\log(1 - S)$ is $-\infty$. `log_cdf` computes
$\log(-\operatorname{expm1}(\log S))$ instead and returns
$\log(10^{-17}) \approx -39.1$:

```{code-cell} ipython3
t_early = 1e-16
naive = jnp.log(1.0 - progression.survival(t_early))
naive, progression.log_cdf(t_early)
```

**The far tail.** At 150 years past age 30, Gompertz survival is about
$e^{-2000}$, which underflows to zero, so the density over survival,
$f/S$, would be $0/0$. `log_hazard` uses the derivative of $\log S$ directly
and stays exact. The closed form is $\log b + c t = \log 0.0005 + 12.75$:

```{code-cell} ipython3
mortality.survival(150.0), mortality.log_hazard(150.0), jnp.log(0.0005) + 0.085 * 150
```

The hazard itself must still be representable as an ordinary number; it is
the survival probability that may underflow. If survival is *exactly* zero
(log survival is $-\infty$), the hazard is undefined and `hazard` and
`log_hazard` return NaN. A family that knows the limit, for example at the
end of a bounded support, can override `log_hazard`; `hazard` then uses the
override.

## Checklist for a new family

- Return $\log S(t)$, not $S(t)$, and use log-space functions such as
  `jnp.expm1`, `jnp.log1p` and `jnp.logaddexp` to avoid subtracting nearly
  equal numbers.
- Give $\log S(0) = 0$ and $\log S(\infty) = -\infty$ (or the right limit for
  your family). Negative and NaN times never reach your code.
- Make $\log S$ differentiable in $t$: the hazard and density come from its
  derivative. At a kink, the derivative JAX returns is the hazard you get.
- Return a positive, finite `reference_time()` of exactly `batch_shape`, in
  the parameter dtype and the smooth interior of the distribution.
- Write `log_survival` for a single distribution and let broadcasting handle
  batches: parameters have the batch shape (plus any component axes) and `t`
  broadcasts against it.
- Override `log_cdf` if your family has a more accurate lower tail, and
  `log_hazard` if you know its value where survival is exactly zero.

## Details

**Registration.** A subclass of a registered family is a new family and needs
its own `@dataclass(frozen=True)` and `@survjax.register_family`. Each field
must use `param()`, `static()` or `distribution()`. Parameters are required;
a static field may have a default, and `@dataclass(frozen=True, kw_only=True)`
allows such a default before required parameters. Instances accept keyword
arguments only.

**Shapes.** Leading parameter axes broadcast to a common `batch_shape`;
`component_ndim` trailing axes stay as components. `batch_shape` follows the
current parameters, including inside `jax.tree.map` and `jax.vmap`. Indexing
follows JAX rules on batch axes only: `...` spans batch axes, `None` adds one,
and out-of-bounds integers clamp to the nearest entry. The base class has no
`shape` attribute, so `shape` is free as a parameter name.

**Coming from NumPyro.** `batch_shape` means the same as in NumPyro. A
survjax distribution describes one scalar event time per batch entry, so in
NumPyro terms its `event_shape` is always `()`. A parameter's component axes
play the role of its constraint's `event_dim`, as with the `probs` of
`Categorical`: survjax calls them components rather than event axes so they
are not mistaken for a vector-valued event time.

**Parameter dtypes.** A family's own parameters share one floating dtype,
following the [same policy as SurvivalData](survival-data.md): at least
float32, and JAX's default float for integer-only or Python-scalar inputs. A
float64 array warns when JAX's 64-bit mode is off. Nested distributions keep
their own dtypes. There is no `dtype=` argument; cast inputs before
construction. Nonreal parameters raise `TypeError`.

**Time dtypes.** Times passed to the methods follow ordinary JAX conversion
and promotion. Boolean, complex, string and object times raise `TypeError`.
A float64 NumPy array of times converts silently to float32 when 64-bit mode
is off, because times are transient arguments rather than stored data.

**NaN from the family.** If your formula gives NaN at the substituted
reference time, for example because a parameter is invalid, the result stays
NaN even for a negative time.
