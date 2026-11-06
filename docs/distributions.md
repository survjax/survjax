---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Defining your own distribution

Use a custom family when the built-in distributions do not describe your
event-time model. Implement log survival and a reference time; survjax derives
the other direct methods. For the common interface, start with
[Working with distributions](working-with-distributions.md).

## A first family: constant hazard

Suppose progression happens at a constant rate of $\lambda = 0.1$ per month,
whatever the time since diagnosis. Then $H(t) = \lambda t$ and

$$
\log S(t) = -\lambda t.
$$

This is the *exponential distribution* with rate $\lambda$, the simplest
survival model. survjax already ships it as
[`survjax.Exponential`](builtin-families.md#the-exponential-a-constant-hazard).
We write it by hand here anyway, because its formula is short enough to show
the mechanics of a family without distraction. We call our version
`ConstantHazard` to keep it apart from the built-in.

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

Use the built-in `Exponential` in applications. This custom version illustrates
the extension mechanism; the built-in uses more accurate closed forms.

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

## Fixed settings and vector parameters: a piecewise-constant hazard

In many medical settings, progression risk is highest soon after diagnosis. A
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

Use this baseline with the built-in wrapper in
[Proportional hazards](proportional-hazards.md#a-piecewise-constant-baseline).

## Checklist for a new family

- Return $\log S(t)$, not $S(t)$, and use log-space functions such as
  `jnp.expm1`, `jnp.log1p` and `jnp.logaddexp` to avoid subtracting nearly
  equal numbers.
- Write `log_survival` for finite $t > 0$; the base supplies its zero and
  infinity values. Negative and NaN times never reach your code. Only the
  generic density and log-hazard defaults call the raw primitive at zero.
- If your limit at infinity is finite and depends on parameters, as in a cure
  model, return it from `log_survival_at_infinity()` in the parameter dtype,
  with exactly `batch_shape`. Its parameter derivatives stay active.
- Make $\log S$ differentiable in $t$: the default hazard and density come
  from its derivative. At a kink, the derivative JAX returns is the hazard you
  get. If you need hazard or density at a boundary where the formula or its
  derivative breaks down
  (see [advanced boundary examples](technical-notes/custom-distributions.md#when-the-formula-breaks-down-at-zero-or-infinity)), override both
  `log_hazard` and `log_prob`: `hazard` follows a `log_hazard` override, but
  `log_prob` does not. `log_cdf` needs no override for this, because it uses
  only the value of $\log S$.
- Return a positive, finite `reference_time()` of exactly `batch_shape`, in
  the parameter dtype and the smooth interior of the distribution.
- Write `log_survival` for a single distribution and let broadcasting handle
  batches: parameters have the batch shape (plus any component axes) and `t`
  broadcasts against it.
- Override `log_cdf` if your family has a more accurate lower tail. If you
  need boundary hazard or density limits that the autodiff default cannot
  supply, override `log_hazard` and `log_prob` there too.


Registration rules, nested fields, dtypes, cure fractions and endpoint overrides
are covered in [Advanced custom distributions](technical-notes/custom-distributions.md).
