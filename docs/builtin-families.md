---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Built-in families

survjax provides the
[exponential](https://en.wikipedia.org/wiki/Exponential_distribution) and the
[Weibull](https://en.wikipedia.org/wiki/Weibull_distribution) distributions,
and the log-normal and log-logistic families described below.
They have the same methods as the families you write yourself (see
[Writing a survival distribution](distributions.md) for what each method
means), with family-specific formulas for the direct methods.

Throughout this page, $T$ is the time from diagnosis to disease progression,
in months unless stated otherwise. We use the usual quantities of survival
analysis: the [survival function](https://en.wikipedia.org/wiki/Survival_function)
$S(t) = P(T > t)$, the cumulative hazard $H(t) = -\log S(t)$ and the
[hazard](https://en.wikipedia.org/wiki/Failure_rate) $h(t) = H'(t)$.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import survjax
```

## The exponential: a constant hazard

The simplest survival model says the risk of progression is the same every
month, whatever has happened before. With hazard (rate) $\lambda$,

$$
h(t) = \lambda,
\qquad
H(t) = \lambda t,
\qquad
S(t) = e^{-\lambda t},
\qquad
f(t) = \lambda e^{-\lambda t}.
$$

The rate has units of one over time, and the mean time to progression is
$1/\lambda$. A mean of 20 months is therefore `rate=1/20`:

```{code-cell} ipython3
progression = survjax.Exponential(rate=1 / 20)

months = jnp.array([0.0, 12.0, 20.0, 60.0])
progression.survival(months)
```

At one year $S(12) = e^{-0.6} \approx 0.549$; at the mean, $S(20) = e^{-1}
\approx 0.368$, which is less than a half because the distribution has a long
right tail. The median is $\log 2/\lambda \approx 13.9$ months.

A constant hazard means the distribution is *memoryless*: having been
progression free for a year does not change the outlook for the next year,
$S(12 + 12)/S(12) = S(12)$.

```{code-cell} ipython3
progression.survival(24.0) / progression.survival(12.0), progression.survival(12.0)
```

The hazard and log density come from the closed forms rather than from
autodiff. The log density at time zero is $\log \lambda = \log 0.05 \approx
-3.00$:

```{code-cell} ipython3
progression.hazard(months), progression.log_prob(0.0)
```

In R, the same model is `pexp(t, rate = 1/20, lower.tail = FALSE)` for the
survival function and `dexp(t, rate = 1/20, log = TRUE)` for the log density.

## The Weibull: a hazard that rises or falls

A constant hazard is rarely realistic. Progression risk is often highest soon
after treatment, while the risk of a wear-out failure grows with time. The
Weibull distribution adds a dimensionless *shape* $k$ to a *scale* $b$, which
has units of time:

$$
H(t) = \left(\frac{t}{b}\right)^{k},
\qquad
S(t) = \exp\!\left[-\left(\frac{t}{b}\right)^{k}\right],
\qquad
h(t) = \frac{k}{b}\left(\frac{t}{b}\right)^{k-1}.
$$

The shape decides how the hazard changes over time:

- $k < 1$: the hazard falls, as for early complications after surgery;
- $k = 1$: the hazard is constant, $1/b$, and the Weibull *is* the
  exponential with `rate=1/b`;
- $k > 1$: the hazard rises, as for ageing or wear-out.

The scale is the time at which $H(b) = 1$, so $S(b) = e^{-1} \approx 0.368$
for every shape. Here are three Weibull models with $b = 10$ months, built as
one batch of three distributions:

```{code-cell} ipython3
shapes = survjax.Weibull(shape=jnp.array([0.5, 1.0, 2.0]), scale=10.0)
shapes.batch_shape, shapes.survival(10.0)
```

To draw their hazards we evaluate all three on one shared vector of times with
`hazard_grid`. Its result has one row per time and one column per
distribution; the [time grids page](grids.md) explains grids in detail.

```{code-cell} ipython3
t = jnp.linspace(0.0, 25.0, 201)  # k = 0.5 has an infinite hazard at 0, left out of the plot
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2))
for column, k in enumerate([0.5, 1.0, 2.0]):
    left.plot(t, shapes.hazard_grid(t)[:, column], label=f"k = {k}")
    right.plot(t, shapes.survival_grid(t)[:, column], label=f"k = {k}")
left.set(xlabel="months", ylabel="hazard h(t)", ylim=(0, 0.5))
right.set(xlabel="months", ylabel="survival S(t)")
right.axvline(10.0, color="grey", linestyle=":")
left.legend()
fig.tight_layout()
```

All three survival curves cross at the scale, $t = b = 10$ (dotted line).

## The log-normal: normally distributed log time

For a log-normal event time, $\log T$ follows a normal distribution with
location $\mu$ and standard deviation $\sigma$:

$$
\log T \sim \operatorname{Normal}(\mu, \sigma^2),
\qquad
S(t) = \Phi\!\left(\frac{\mu - \log t}{\sigma}\right),
$$

where $\Phi$ is the standard normal CDF. `meanlog` is the finite real
location $\mu$ of log time in the chosen time unit; `sdlog` is the positive
finite, dimensionless standard deviation $\sigma$. Neither is the mean or
standard deviation of $T$ itself. These names match R's `lnorm` functions
and flexsurv's `lnorm` family; NumPyro uses `loc=meanlog`, `scale=sdlog`.
The mathematical median is $\exp(\mu)$, so $\mu=0$ gives $S(1)=1/2$:

```{code-cell} ipython3
lognormal = survjax.LogNormal(meanlog=0.0, sdlog=0.5)
assert round(float(lognormal.survival(1.0)), 6) == 0.5
lognormal.survival(1.0)
```

`from_log` exponentiates only `log_sdlog`; `meanlog` is already unconstrained
and stays unchanged. The result stores the natural `sdlog`, just as the
ordinary constructor does. Overflow or underflow in the exponential can
produce an invalid parameter; inspect `parameter_issues()` when diagnosing
model outputs.

```{code-cell} ipython3
groups = survjax.LogNormal.from_log(
    meanlog=jnp.array([0.0, 1.0]), log_sdlog=jnp.log(0.5)
)
times = jnp.array([0.0, 1.0, 3.0, jnp.inf])
curves = groups.survival_grid(times)
assert curves.shape == (4, 2)  # time first, then the distribution batch
assert bool(jnp.all(curves[0] == 1.0))
assert bool(jnp.all(curves[-1] == 0.0))
curves
```

With valid parameters, density and hazard are zero at both time zero and
infinity, so `log_prob` and `log_hazard` return `-inf` at those boundaries.
As with the other families, direct methods accept negative times as before
the origin and propagate NaN times. Exact event observations still require
strictly positive times.

Use `log_survival` and `log_cdf` directly when working in the tails. They
use symmetric normal-tail formulas to retain small negative log probabilities
that would disappear if a rounded probability were logged afterward:

```{code-cell} ipython3
log_tails = jnp.stack([
    lognormal.log_survival(jnp.exp(-3.5)),
    lognormal.log_cdf(jnp.exp(3.5)),
])
assert bool(jnp.all(log_tails < 0.0))
log_tails
```

Both float32 and float64 are supported; float64 is recommended for tail and
reference work. See [Numerical behaviour](numerical-behaviour.md#precision-in-both-tails).

## The log-logistic: logistic log time

The log-logistic family has dimensionless shape $\beta>0$ and time scale
$\alpha>0$, both finite. Its survival and hazard are

$$
S(t) = \frac{1}{1+(t/\alpha)^\beta},
\qquad
h(t) = \frac{\beta}{\alpha}
       \frac{(t/\alpha)^{\beta-1}}{1+(t/\alpha)^\beta}.
$$

At $t=\alpha$, survival is one half for every shape: the scale is the
mathematical median. With $\beta\leq1$ the hazard decreases; with $\beta>1$
it first rises, then falls. These parameters match flexsurv's
`llogis(shape=beta, scale=alpha)`. Equivalently, log time has a logistic
distribution with location $\mu=\log\alpha$ and scale $s=1/\beta$.
For example, log-time location $\log 10$ and scale $0.5$ give
`LogLogistic(shape=2, scale=10)`:

```{code-cell} ipython3
loglogistic = survjax.LogLogistic(shape=2.0, scale=10.0)
assert round(float(loglogistic.survival(10.0)), 6) == 0.5
assert round(float(loglogistic.survival(20.0)), 6) == 0.2
loglogistic.survival(jnp.array([0.0, 10.0, 20.0, jnp.inf]))
```

`from_log(log_shape=..., log_scale=...)` exponentiates both inputs and stores
ordinary shape and scale leaves. It does not retain a log-space representation.
As with other families, invalid values are stored without automatic raising;
inspect `parameter_issues()` to diagnose them. Exponentiation can overflow or
underflow to an invalid natural parameter.

```{code-cell} ipython3
loglogistic_groups = survjax.LogLogistic.from_log(
    log_shape=jnp.log(jnp.array([0.5, 1.0, 2.0])), log_scale=jnp.log(10.0)
)
loglogistic_times = jnp.array([0.0, 10.0, 20.0, jnp.inf])
loglogistic_curves = loglogistic_groups.survival_grid(loglogistic_times)
assert loglogistic_curves.shape == (4, 3)  # time first, then batch
assert bool(jnp.all(loglogistic_curves[0] == 1.0))
assert bool(jnp.all(loglogistic_curves[-1] == 0.0))
assert bool(jnp.all(jnp.round(loglogistic_curves[1], 6) == 0.5))
loglogistic_curves
```

At zero, hazard and density have limits $+\infty$, $1/\alpha$ or $0$ for
$\beta<1$, $\beta=1$ or $\beta>1$, respectively. The hazard tends to zero
at infinity for every valid shape. The zero-time log density changes
discontinuously in shape at $\beta=1$, so its shape derivative there is outside
the guarantee. Its scale derivative is defined. Direct times can include zero;
exact event observations must remain strictly positive.

Use `log_survival` and `log_cdf` directly in the tails. They evaluate opposite
softplus tails of $z=\beta(\log t-\log\alpha)$, preserving small negative log
probabilities that would disappear by logging a probability rounded to one.
The following tails remain negative in both float32 and float64:

```{code-cell} ipython3
loglogistic_tails = jnp.stack([
    loglogistic.log_survival(1e-10),
    loglogistic.log_cdf(1e10),
])
assert bool(jnp.all(loglogistic_tails < 0.0))
loglogistic_tails
```

## Unconstrained parameters: `from_log`

Rates, shapes and scales must be positive. When parameters come from an
optimiser or a neural network, it is convenient to work with their logarithms,
which can take any real value. `from_log` exponentiates them and stores the
ordinary (natural) parameters, so the result is an ordinary distribution:

```{code-cell} ipython3
from_logs = survjax.Weibull.from_log(log_shape=jnp.log(2.0), log_scale=jnp.log(10.0))
from_logs.shape, from_logs.scale
```

Gradients flow through the exponential as you would expect. For the
exponential family there is one log parameter, the log rate $a$ with
$\lambda = e^{a}$. A model written in terms of the log mean $m$ uses
$\lambda = e^{-m}$, so `from_log(log_rate=-m)`:

```{code-cell} ipython3
log_mean = jnp.log(20.0)
survjax.Exponential.from_log(log_rate=-log_mean).rate
```

`from_log` keeps no log-scale copy. If a log parameter is so large that its
exponential overflows (above about 88 in float32, 709 in float64), the stored
parameter is infinite, and results are whatever the formulas give for an
infinite parameter, as for any invalid parameter. Constructors never check
parameter values: a negative rate or a NaN shape is stored as given and shows
up as NaN or meaningless results, not as an error.

## What happens at time zero and at infinity

survjax methods accept any time, including $t = 0$ and $t = \infty$, and
never raise for a value (see [the edges of the time
axis](technical-notes/custom-distributions.md#the-edges-of-the-time-axis)). For the Weibull, time
zero is interesting because the hazard there depends on the shape. Taking the
limit of $h(t)$ as $t \to 0$:

| Shape | $h(0)$ | $\log f(0)$ | $h(\infty)$ |
| --- | --- | --- | --- |
| $k < 1$ | $\infty$ | $\infty$ | $0$ |
| $k = 1$ | $1/b$ | $\log(1/b)$ | $1/b$ |
| $k > 1$ | $0$ | $-\infty$ | $\infty$ |

survjax returns exactly these limits rather than NaN:

```{code-cell} ipython3
shapes.log_prob(0.0), shapes.hazard(0.0), shapes.hazard(jnp.inf)
```

Every proper survival function has $S(0) = 1$ and $S(\infty) = 0$. These
values do not depend on the parameters, so their parameter derivatives are
zero, and survjax returns exactly that rather than NaN. This matters when a
grid of times starts at zero inside a model you differentiate:

```{code-cell} ipython3
def survival_at_zero(k):
    return survjax.Weibull(shape=k, scale=10.0).survival(0.0)


jax.grad(survival_at_zero)(1.0)
```

One boundary derivative is deliberately left out. At $k = 1$, the log density
at zero jumps from $+\infty$ (just below $k=1$) through $\log(1/b)$ to
$-\infty$ (just above), so it has no derivative in $k$ there:

```{code-cell} ipython3
near_one = survjax.Weibull(shape=jnp.array([0.99, 1.0, 1.01]), scale=10.0)
near_one.log_prob(0.0)
```

The derivative in the scale *is* defined, $\partial_b \log(1/b) = -1/b =
-0.1$, and survjax returns it:

```{code-cell} ipython3
jax.grad(lambda b: survjax.Weibull(shape=1.0, scale=b).log_prob(0.0))(10.0)
```

Observed event times are positive, so in a likelihood this only affects
evaluation on grids that start at zero.


For precision in extreme tails, see [Numerical behaviour](numerical-behaviour.md#precision-in-both-tails).
For R and NumPyro mappings, see [Parameter conventions](technical-notes/parameter-conventions.md).
