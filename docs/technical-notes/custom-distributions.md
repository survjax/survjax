---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Advanced custom distributions

Read [Defining your own distribution](../distributions.md) first. This note
covers endpoint values, cure fractions, hazard overrides and registration details.

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

```{code-cell} ipython3
progression = ConstantHazard(rate=0.1)
```

## The edges of the time axis

Times are measured from a time origin such as diagnosis, so $T > 0$. Still,
a survival curve is often drawn from $t = 0$, administrative censoring is
sometimes coded as $t = \infty$, and a bad row can carry a negative or
missing time. survjax answers for any real time without raising. Your
`log_survival` only needs to handle finite $t > 0$ for survival, CDF and
cumulative hazard. survjax supplies their boundary values; density and hazard
limits need more, as shown below:

| Time | Meaning | What survjax returns |
| --- | --- | --- |
| $t < 0$ | before the time origin | $S = 1$, $H = 0$, $h = 0$, log density and log CDF $-\infty$ |
| $t = 0$ | the origin | from the base: $S = 1$, $H = 0$, log CDF $-\infty$; density and hazard limits from your family |
| $t = \infty$ | never | from the base: $\log S =$ `log_survival_at_infinity()` (default $-\infty$), $S$, log CDF and $H$ from it; log density $-\infty$; hazard limit from your family |
| NaN | missing | NaN |

```{code-cell} ipython3
edge_times = jnp.array([-1.0, 0.0, jnp.inf, jnp.nan])
progression.survival(edge_times), progression.log_prob(edge_times)
```

survjax never passes a negative or NaN time to your code. For these and the
base-supplied boundary values, it evaluates your family at the reference time
instead, discards that value and returns the answer from the table. If your
family gives NaN even at the reference time, typically because a parameter is
NaN, that NaN is returned instead, so invalid
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
survjax returns the correct 0 (JAX may print a signed zero):

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

survjax does this by evaluating the family at the reference time before
filling in the base-supplied boundary values. This applies even to overrides
of survival, log survival, log CDF and cumulative hazard. Derivatives of the
infinite values themselves are not defined, and survjax promises nothing
about them.

Not every distribution has $S(\infty) = 0$. In a *cure model*, a fraction
$p$ of patients never progresses, so $S(\infty) = p$ and
$\log F(\infty) = \log(1 - p)$. These limits depend on $p$, and their
derivatives, 1 and $-1/(1 - p)$, must survive. Declare the limit by overriding
`log_survival_at_infinity()`. The default returns $-\infty$ for a proper
family; a cure family returns $\log p$, with exactly `batch_shape` in the
parameter dtype. survjax differentiates through this value:

```{code-cell} ipython3
@survjax.register_family
@dataclass(frozen=True)
class CureHazard(ConstantHazard):
    fraction: jax.Array = survjax.param()  # p, never progresses

    def log_survival(self, t):
        remaining = jnp.exp(-self.rate * t)
        return jnp.log(self.fraction + (1.0 - self.fraction) * remaining)

    def log_survival_at_infinity(self):
        return jnp.log(self.fraction)


def cure_limits(p):
    cured = CureHazard(rate=0.1, fraction=p)
    return cured.survival(jnp.inf), cured.log_cdf(jnp.inf)


cure_limits(0.25), jax.jacobian(cure_limits)(0.25)
```

With $p = 0.25$ the limits are $0.25$ and $\log 0.75 \approx -0.288$, with
derivatives $1$ and $-1/0.75 \approx -1.33$. The primitive never needs to
multiply a parameter by infinity: survjax uses the declared limit instead.
A NaN from that limit makes the base-supplied infinity values NaN, including
log density. An analytic hazard override still supplies its own limit.

These examples do not declare parameter constraints. Declare constraints with
`param(constraint=...)` and inspect `parameter_issues()` when your family needs
parameter validation; see the [API reference](../api.md#parameter-diagnostics).

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
$S(\infty) = 0$, as a proper survival function requires. In code we write
$(t/b)^k = e^{z}$ with $z = k(\log t - \log b)$, so that
$\log S = -\operatorname{softplus}(z)$, where
$\operatorname{softplus}(z) = \log(1 + e^{z})$ is JAX's numerically stable
`jax.nn.softplus`: it stays accurate for large $z$, where computing $e^{z}$
first would overflow.

Written directly, the formula has troublesome derivatives at the boundaries.
survjax supplies the survival values and parameter gradients there, so the
plain formula works for survival; the default hazard still gives NaN:

```{code-cell} ipython3
@survjax.register_family
@dataclass(frozen=True)
class LogLogistic(survjax.SurvivalDistribution):
    shape: jax.Array = survjax.param()
    scale: jax.Array = survjax.param()

    def log_survival(self, t):
        return -jax.nn.softplus(self.shape * (jnp.log(t) - jnp.log(self.scale)))

    def reference_time(self):
        return self.scale  # the median


boundaries = jnp.array([0.0, jnp.inf])
plain = LogLogistic(shape=2.0, scale=10.0)
plain.survival(boundaries), plain.hazard(boundaries)
```

```{code-cell} ipython3
jax.grad(lambda k: LogLogistic(shape=k, scale=10.0).survival(0.0))(2.0)
```

With $z$ as above, the derivatives of $\log S$ are
$-\sigma(z)\,k/t$ in $t$ and $-\sigma(z)(\log t - \log b)$ in $k$, where
$\sigma$ is the logistic function. At $t = 0$ both are $0 \cdot \infty$,
which JAX evaluates literally as NaN if the raw formula is called there.
The base avoids that call for survival, log survival, log CDF and cumulative
hazard, so the gradient of $S(0)$ above is 0.

### Use the limits you already know

You rarely need the formula at the boundaries, because the limits are known
in advance:

- $S(0) = 1$, so $\log S(0) = 0$, for every distribution of a positive time;
- $S(\infty) = 0$, so $\log S(\infty) = -\infty$, unless some individuals
  never have the event (a cure fraction $p$ gives $\log p$, as above).

survjax supplies these limits for every registered family. Your
`log_survival` can use the plain formula above, without testing for zero or
infinity. The base substitutes a finite positive reference time before the
formula runs, then selects the limit; the substituted time carries no
parameter derivative. A cure family's limit comes from
`log_survival_at_infinity()`, whose parameter derivatives stay active.

A NaN shape or scale still gives NaN at the boundaries: the base preserves a
NaN from the formula evaluated at the reference time. the unconstrained example does not declare parameter ranges.

### The hazard and density need more

You usually do not need hazard or density values exactly at zero or infinity.
Likelihoods need density at finite, positive event times; censoring uses
survival or CDF values. Mean and RMST integrals use survival. Even an integral
of the hazard or density starting at zero does not depend on the value at
that single endpoint, though numerical quadrature must handle any singularity
there.

The boundary overrides below are an optional extension, useful for displaying
an initial or limiting hazard, or plotting on a grid that includes zero.
Starting a hazard plot at a positive time is often enough. The plain family
above already works for ordinary evaluations at finite, positive times.

If you decide to implement hazard and density values at the limits,
it's important to understand how they are different from survival:
survjax derives them from the *derivative* of $\log S$ in $t$.
At $t = 0$, the generic defaults differentiate
the raw primitive, the one exception to keeping it away from the boundaries.
There is no value or gradient guarantee for those defaults at zero. Here
they give NaN, although the correct hazard is $\infty$ ($k < 1$), $1/b$
($k = 1$) or 0 ($k > 1$), just as for the Weibull:

```{code-cell} ipython3
three_shapes = LogLogistic(shape=jnp.array([0.5, 1.0, 2.0]), scale=10.0)
three_shapes.hazard(0.0)
```

At infinity the default hazard returns NaN, including for cure families.
If you need these boundary limits, override `log_hazard` and `log_prob`:
these overrides are called at zero, and `log_hazard` also at
infinity. Log density at infinity is always supplied by the base. Here the
hazard is

$$
h(t) = \frac{k}{t}\,\frac{(t/b)^{k}}{1 + (t/b)^{k}},
\qquad
\log h(t) = \log k - \log t + \log\sigma\bigl(k(\log t - \log b)\bigr),
$$

with $\sigma$ the logistic function again. Near zero it behaves like the Weibull
hazard $(k/b)(t/b)^{k-1}$; near infinity like $k/t$, which goes to 0.
`hazard` follows a `log_hazard` override automatically, but `log_prob` does
not: by default it takes its own derivative of $\log S$. To supply the density
limit at zero as well, override it too.

```{code-cell} ipython3
@survjax.register_family
@dataclass(frozen=True)
class LogLogisticWithLimits(LogLogistic):
    def log_hazard(self, t):
        time = jnp.asarray(t)
        boundary = (time == 0) | jnp.isposinf(time)
        reference = jax.lax.stop_gradient(self.reference_time())
        safe_time = jnp.where(boundary, reference, time)
        z = self.shape * (jnp.log(safe_time) - jnp.log(self.scale))
        interior = jnp.log(self.shape) - jnp.log(safe_time) + jax.nn.log_sigmoid(z)
        # Near zero h behaves like the Weibull hazard; near infinity like k / t.
        at_zero = jnp.where(self.shape < 1, jnp.inf, -jnp.inf)
        at_zero = jnp.where(self.shape == 1, -jnp.log(self.scale), at_zero)
        limit = jnp.where(time == 0, at_zero, -jnp.inf)
        # Keep a NaN from the safe interior value at the boundaries.
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

The overrides and the base keep NaN parameters visible at the boundaries. A
batch with one valid and one NaN shape gives the limits in the first row and
NaN in the second, for every method:

```{code-cell} ipython3
mixed = LogLogisticWithLimits(shape=jnp.array([2.0, jnp.nan])[:, None], scale=10.0)
mixed.survival(boundaries), mixed.log_cdf(boundaries), mixed.hazard(boundaries), mixed.log_prob(boundaries)
```

A few rules of thumb:

- **Values first.** The base supplies survival, log survival, cumulative
  hazard and `log_cdf` at zero and infinity. A proper family needs no boundary
  code for these methods.
- **Endpoint limits are optional.** Override `log_hazard` and `log_prob` if
  you need boundary limits that the raw formula's derivative cannot supply.
  `hazard` follows `log_hazard`; `log_prob` does not.
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
following the [same policy as SurvivalData](../survival-data.md): at least
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
