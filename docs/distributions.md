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
distribution and evaluate it. survjax also ships the
[exponential and Weibull families](builtin-families.md), which work the same
way but use closed forms. Likelihoods for `SurvivalData` are planned for a
later step of Slice A.
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
parameter is shared across the batch. Every family also has *grid* methods,
such as `survival_grid`, which put the time axis first and need no manual
axis; see [Survival curves and time grids](grids.md).

```{code-cell} ipython3
arms.survival_grid(months)
```

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

Times are measured from a time origin such as diagnosis, so $T > 0$. Still,
a survival curve is often drawn from $t = 0$, administrative censoring is
sometimes coded as $t = \infty$, and a bad row can carry a negative or
missing time. survjax answers for any real time without raising. Your
`log_survival` only needs to handle $t \ge 0$, including $t = \infty$; survjax
takes care of the rest:

| Time | Meaning | What survjax returns |
| --- | --- | --- |
| $t < 0$ | before the time origin | $S = 1$, $H = 0$, $h = 0$, log density and log CDF $-\infty$ |
| $t = 0$ | the origin | your family's values, $S(0) = 1$ |
| $t = \infty$ | never | your family's values, usually $S = 0$; log density $-\infty$ |
| NaN | missing | NaN |

```{code-cell} ipython3
edge_times = jnp.array([-1.0, 0.0, jnp.inf, jnp.nan])
progression.survival(edge_times), progression.log_prob(edge_times)
```

survjax never passes a negative or NaN time to your code. It evaluates your
family at the reference time instead, discards that value and returns the
answer from the table. If your family gives NaN even at the reference time,
typically because a parameter is NaN, that NaN is returned instead, so invalid
parameters do not hide behind the table's constants:

```{code-cell} ipython3
broken = ConstantHazard(rate=jnp.nan)
broken.survival(-1.0), broken.log_prob(jnp.inf)
```

### Gradients at the boundaries

Fitting a model means differentiating with respect to the parameters, and
boundary times are where naive code produces NaN gradients. Two examples show
what survjax guarantees.

First, a proper distribution has $S(\infty) = 0$ and $\log F(\infty) = 0$
whatever its parameters, so their derivatives are zero. Computing $e^{-\lambda
\cdot \infty}$ naively and differentiating would multiply zero by infinity;
survjax returns the correct 0 (JAX prints a signed zero, `-0.`):

```{code-cell} ipython3
def survival_at_infinity(rate):
    return ConstantHazard(rate=rate).survival(jnp.inf)


jax.grad(survival_at_infinity)(0.1)
```

Second, in a batch where some entries are infinite, the infinite entries must
not spoil the derivatives of the finite ones. Here the cumulative hazard at
$t = 2$ is $2\lambda$, so its derivative is 2, and the entry at infinity is
masked out of the sum. This gives 2, not NaN:

```{code-cell} ipython3
def finite_cumulative_hazard(rate):
    H = ConstantHazard(rate=rate).cumulative_hazard(jnp.array([2.0, jnp.inf]))
    return jnp.where(jnp.isfinite(H), H, 0.0).sum()


jax.grad(finite_cumulative_hazard)(0.1)
```

survjax does this by evaluating the default methods at a harmless time
wherever the limit is known to be $S = 0$, and only then filling in the
limit. Derivatives of the infinite values themselves are not defined, and
survjax promises nothing about them.

Not every distribution has $S(\infty) = 0$. In a *cure model*, a fraction
$p$ of patients never progresses, so $S(\infty) = p$ and
$\log F(\infty) = \log(1 - p)$. These limits depend on $p$, and their
derivatives, 1 and $-1/(1 - p)$, must survive. survjax leaves such finite
limits to your `log_survival`, which therefore has to compute them with safe
operands itself: replace the time before multiplying it by a parameter, then
select the limit.

```{code-cell} ipython3
@survjax.register_family
@dataclass(frozen=True)
class CureHazard(ConstantHazard):
    fraction: jax.Array = survjax.param()  # p, never progresses

    def log_survival(self, t):
        time = jnp.asarray(t)
        infinite = jnp.isposinf(time)
        # Substitute before multiplying: rate * inf would poison the gradient.
        safe_time = jnp.where(infinite, jax.lax.stop_gradient(jnp.ones_like(time)), time)
        remaining = jnp.exp(-self.rate * safe_time)
        # At infinity no susceptible patient remains; keep NaN from bad rates.
        remaining = jnp.where(infinite & ~jnp.isnan(remaining), 0.0, remaining)
        return jnp.log(self.fraction + (1.0 - self.fraction) * remaining)


def cure_limits(p):
    cured = CureHazard(rate=0.1, fraction=p)
    return cured.survival(jnp.inf), cured.log_cdf(jnp.inf)


cure_limits(0.25), jax.jacobian(cure_limits)(0.25)
```

With $p = 0.25$ the limits are $0.25$ and $\log 0.75 \approx -0.288$, with
derivatives $1$ and $-1/0.75 \approx -1.33$. The `jnp.where` before the
multiplication is the important line: without it, the derivative of
$-\lambda t$ at $t = \infty$ is infinite, and the masked-out branch turns the
gradient into NaN even though the selected value is fine.

Parameter values are not checked. A negative rate is stored as given and
produces whatever the formula gives; validating parameters is up to the
caller.

## When the formula breaks down at zero or infinity

The families so far evaluate cleanly at $t = 0$ and $t = \infty$. Many
formulas do not: they contain $\log t$, $t^{k}$ or a ratio that becomes
$0 \cdot \infty$ or $\infty - \infty$ exactly at a boundary. Take the
[log-logistic](https://en.wikipedia.org/wiki/Log-logistic_distribution)
distribution, with shape $k > 0$ and scale $b > 0$ (survjax will ship one
later; writing it here shows the technique):

$$
S(t) = \frac{1}{1 + (t/b)^{k}},
\qquad
\log S(t) = -\log\bigl(1 + (t/b)^{k}\bigr).
$$

Because $k > 0$, $(t/b)^k$ grows from 0 to $\infty$, so $S(0) = 1$ and
$S(\infty) = 0$, as a survival function requires. In code we write
$(t/b)^k = e^{z}$ with $z = k(\log t - \log b)$, so that
$\log S = -\operatorname{softplus}(z)$, where
$\operatorname{softplus}(z) = \log(1 + e^{z})$ is JAX's numerically stable
`jax.nn.softplus`: it stays accurate for large $z$, where computing $e^{z}$
first would overflow.

Written directly, it gives the right *values* at the boundaries, because
$\log 0 = -\infty$ and softplus handles infinite arguments. But the hazard
at zero and infinity, and the gradient of $S(0)$, are NaN:

```{code-cell} ipython3
@survjax.register_family
@dataclass(frozen=True)
class NaiveLogLogistic(survjax.SurvivalDistribution):
    shape: jax.Array = survjax.param()
    scale: jax.Array = survjax.param()

    def log_survival(self, t):
        return -jax.nn.softplus(self.shape * (jnp.log(t) - jnp.log(self.scale)))

    def reference_time(self):
        return self.scale  # the median


boundaries = jnp.array([0.0, jnp.inf])
naive = NaiveLogLogistic(shape=2.0, scale=10.0)
naive.survival(boundaries), naive.hazard(boundaries)
```

```{code-cell} ipython3
jax.grad(lambda k: NaiveLogLogistic(shape=k, scale=10.0).survival(0.0))(2.0)
```

With $z$ as above, the derivatives of $\log S$ are
$-\sigma(z)\,k/t$ in $t$ and $-\sigma(z)(\log t - \log b)$ in $k$, where
$\sigma$ is the logistic function. At $t = 0$ both are $0 \cdot \infty$,
which JAX evaluates literally as NaN.

### Use the limits you already know

You rarely need the formula at the boundaries, because the limits are known
in advance:

- $S(0) = 1$, so $\log S(0) = 0$, for every distribution of a positive time;
- $S(\infty) = 0$, so $\log S(\infty) = -\infty$, unless some individuals
  never have the event (a cure fraction $p$ gives $\log p$, as above).

Neither depends on the parameters except through such a cure fraction. So
select the limit, and make sure the formula never sees the boundary time:
detect the boundary, substitute a harmless time *before* the operation that
fails, compute, then choose. This is the same double-`where` pattern as in
the cure model.

```{code-cell} ipython3
@survjax.register_family
@dataclass(frozen=True)
class LogLogistic(survjax.SurvivalDistribution):
    shape: jax.Array = survjax.param()
    scale: jax.Array = survjax.param()

    def log_survival(self, t):
        time = jnp.asarray(t)
        boundary = (time == 0) | jnp.isposinf(time)
        safe_time = jnp.where(boundary, jnp.ones_like(time), time)
        interior = -jax.nn.softplus(self.shape * (jnp.log(safe_time) - jnp.log(self.scale)))
        limit = jnp.where(time == 0, 0.0, -jnp.inf)
        # A NaN parameter makes the safe interior value NaN: keep it.
        limit = jnp.where(jnp.isnan(interior), interior, limit)
        return jnp.where(boundary, limit, interior)

    def reference_time(self):
        return self.scale


jax.grad(lambda k: LogLogistic(shape=k, scale=10.0).survival(0.0))(2.0)
```

The gradient of $S(0)$ is now 0, as it should be. Survival, cumulative hazard
and `log_cdf` need only the *value* of $\log S$, so they are now right at both
boundaries.

The second `where` on `limit` matters too. The limits are constants, so
without it a NaN shape or scale would still give $S(0) = 1$ and
$S(\infty) = 0$: an undefined distribution would report valid-looking
boundary values, and survjax cannot recover the NaN once your
`log_survival` has dropped it. The interior formula, evaluated at the safe
time 1, is NaN exactly when a parameter is NaN, so passing that NaN on is
enough. survjax does not check parameter ranges for you; the formula is the
judge of what is undefined.

### The hazard and density need more

The hazard and density are different: survjax derives them from the
*derivative* of $\log S$ in $t$. Selecting a constant at $t = 0$ makes that
derivative 0 there, so the default hazard at zero becomes 0 for every
shape. That is right for $k > 1$ but silently wrong for $k \le 1$, whose
hazard at zero is $\infty$ ($k < 1$) or $1/b$ ($k = 1$), just as for the
Weibull:

```{code-cell} ipython3
three_shapes = LogLogistic(shape=jnp.array([0.5, 1.0, 2.0]), scale=10.0)
three_shapes.hazard(0.0)
```

At infinity, where $S = 0$, the default hazard is undefined and returns NaN.
So when you substitute at a boundary, also give the hazard and density their
limits by overriding `log_hazard` and `log_prob`. Here the hazard is

$$
h(t) = \frac{k}{t}\,\frac{(t/b)^{k}}{1 + (t/b)^{k}},
\qquad
\log h(t) = \log k - \log t + \log\sigma\bigl(k(\log t - \log b)\bigr),
$$

with $\sigma$ the logistic function again. Near zero it behaves like the Weibull
hazard $(k/b)(t/b)^{k-1}$; near infinity like $k/t$, which goes to 0.
`hazard` follows a `log_hazard` override automatically, but `log_prob` does
not: by default it takes its own derivative of $\log S$, so override it too.

```{code-cell} ipython3
@survjax.register_family
@dataclass(frozen=True)
class LogLogisticWithLimits(LogLogistic):
    def log_hazard(self, t):
        time = jnp.asarray(t)
        boundary = (time == 0) | jnp.isposinf(time)
        safe_time = jnp.where(boundary, jnp.ones_like(time), time)
        z = self.shape * (jnp.log(safe_time) - jnp.log(self.scale))
        interior = jnp.log(self.shape) - jnp.log(safe_time) + jax.nn.log_sigmoid(z)
        # Near zero h behaves like the Weibull hazard; near infinity like k / t.
        at_zero = jnp.where(self.shape < 1, jnp.inf, -jnp.inf)
        at_zero = jnp.where(self.shape == 1, -jnp.log(self.scale), at_zero)
        limit = jnp.where(time == 0, at_zero, -jnp.inf)
        # As in log_survival: a NaN parameter stays NaN at the boundaries.
        limit = jnp.where(jnp.isnan(interior), interior, limit)
        return jnp.where(boundary, limit, interior)

    def log_prob(self, t):
        return self.log_hazard(t) + self.log_survival(t)


fixed = LogLogisticWithLimits(shape=jnp.array([0.5, 1.0, 2.0])[:, None], scale=10.0)
fixed.hazard(boundaries), fixed.log_prob(boundaries)
```

One row per shape, one column per boundary: the hazard at zero is $\infty$,
$1/b = 0.1$ and 0, and at infinity it is 0 for every shape. At $t = 5$, in
the interior, the override and the autodiff default agree, so the override
changes only the boundaries:

```{code-cell} ipython3
fixed.log_prob(5.0)[:, 0], three_shapes.log_prob(5.0)
```

Both overrides keep NaN parameters visible at the boundaries. A batch with
one valid and one NaN shape gives the limits in the first row and NaN in the
second, for every method:

```{code-cell} ipython3
mixed = LogLogisticWithLimits(shape=jnp.array([2.0, jnp.nan])[:, None], scale=10.0)
mixed.survival(boundaries), mixed.log_cdf(boundaries), mixed.hazard(boundaries), mixed.log_prob(boundaries)
```

A few rules of thumb:

- **Values first.** If only survival, cumulative hazard and `log_cdf` matter
  to you, selecting $\log S(0) = 0$ and $\log S(\infty) = -\infty$ is enough.
- **Derivatives in $t$ at a boundary need an override.** Once you substitute
  in `log_survival` at a time, override `log_hazard` and `log_prob` to supply
  their limits there; otherwise the autodiff default sees a constant.
- **Keep NaN parameters NaN.** A selected constant hides a NaN parameter.
  Pass on the NaN of the safe interior value, as above.
- **Find limits from the leading term.** Near zero and near infinity, keep
  only the dominant term of $h(t)$; a family often looks like a Weibull or an
  exponential there.
- **If you cannot work out a limit, leave NaN.** NaN says "undefined";
  survjax keeps it visible rather than replacing it with a plausible number.
- **Check the boundaries.** Evaluate every method at `jnp.array([0.0,
  jnp.inf])`, with parameters below, at and above any critical value (here
  $k = 1$), and take `jax.grad` of $S(0)$ and $\log F(\infty)$. A family
  checker that automates this is planned.

## Why work on the log scale

Four related quantities appear on this page and are easy to mix up:

| Symbol | Name | survjax method | Typical range |
| --- | --- | --- | --- |
| $H(t)$ | cumulative hazard | `cumulative_hazard(t)` | $0$ to $\infty$ |
| $\log H(t)$ | log cumulative hazard | used inside the built-ins | $-\infty$ to $\infty$ |
| $h(t)$ | hazard, the instantaneous rate | `hazard(t)` | $0$ to $\infty$ |
| $\log h(t)$ | log hazard | `log_hazard(t)` | $-\infty$ to $\infty$ |

$H$ is the integral of $h$, and $\log S = -H$. The derived methods avoid
computing $S$ and then subtracting or dividing where that would lose
precision. Three cases come up in practice.

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

The generic `log_cdf` is accurate as long as $\log S$ itself is not rounded
to zero. The built-in families go further by working from $\log H$, which
stays accurate even when $H$ is too small to store; see [precision in both
tails](builtin-families.md#precision-in-both-tails).

**Late events.** Late in follow-up, $S$ is tiny and $\log F = \log(1 - S)
\approx -S$ is a tiny negative number. `log_cdf` returns it with relative
precision as long as $S$ is a normal floating-point number (here
$S = e^{-30}$, so about $-9.4 \times 10^{-14}$),
where the naive $\log(1 - S)$ rounds it away: to zero in float32, and to a
value with only a few correct digits in float64.

```{code-cell} ipython3
t_late = 300.0
jnp.log(1.0 - progression.survival(t_late)), progression.log_cdf(t_late)
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
override. The built-in families do this: the exponential's hazard at
infinity is its rate.

```{code-cell} ipython3
progression.hazard(jnp.inf), survjax.Exponential(rate=0.1).hazard(jnp.inf)
```

## Checklist for a new family

- Return $\log S(t)$, not $S(t)$, and use log-space functions such as
  `jnp.expm1`, `jnp.log1p` and `jnp.logaddexp` to avoid subtracting nearly
  equal numbers.
- Give $\log S(0) = 0$ and $\log S(\infty) = -\infty$ (or the right limit for
  your family). Negative and NaN times never reach your code.
- If your limit at infinity is finite and depends on parameters, as in a cure
  model, substitute a safe time before any operation that would meet
  infinity, so that the limit's derivatives stay finite.
- Make $\log S$ differentiable in $t$: the default hazard and density come
  from its derivative. At a kink, the derivative JAX returns is the hazard you
  get. Where it is not differentiable, or where you select a boundary limit
  (see [when the formula breaks
  down](#when-the-formula-breaks-down-at-zero-or-infinity)), override both
  `log_hazard` and `log_prob`: `hazard` follows a `log_hazard` override, but
  `log_prob` does not. `log_cdf` needs no override for this, because it uses
  only the value of $\log S$.
- Return a positive, finite `reference_time()` of exactly `batch_shape`, in
  the parameter dtype and the smooth interior of the distribution.
- Write `log_survival` for a single distribution and let broadcasting handle
  batches: parameters have the batch shape (plus any component axes) and `t`
  broadcasts against it.
- Override `log_cdf` if your family has a more accurate lower tail, and
  `log_hazard` and `log_prob` where you know limits that the autodiff
  default cannot see, such as the hazard where survival is exactly zero.

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
