---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Exponential and Weibull

survjax ships two parametric families so far: the
[exponential](https://en.wikipedia.org/wiki/Exponential_distribution) and the
[Weibull](https://en.wikipedia.org/wiki/Weibull_distribution) distribution.
They have the same methods as the families you write yourself (see
[Writing a survival distribution](distributions.md) for what each method
means), but compute every quantity from a closed form instead of from
automatic differentiation. Log-normal and log-logistic families are planned.

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
axis](distributions.md#the-edges-of-the-time-axis)). For the Weibull, time
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

## Precision in both tails

The probability of progression by time $t$, $F(t) = 1 - S(t)$, is needed
for left- and interval-censored data, and it is easy to lose. survjax
computes $\log F$ from $\log H$, which keeps relative precision at both
ends of the time axis, within the limits of floating point: in the lower tail
wherever $\log H$ itself is representable, even when $H$ and $F$ underflow,
and in the upper tail wherever $S = e^{-H}$ is a normal floating-point number
(down to about $10^{-38}$ in float32 and $10^{-308}$ in float64). Beyond
that, $S$ underflows and $\log F \approx -S$ rounds to 0; for example,
`Exponential(rate=1.0).log_cdf(1000.0)` is $-0.0$ in both modes, although
the exact value $-e^{-1000} \approx -10^{-434}$ is negative.

**Very early.** With $k = 2$ and $b = 10$, at $t = 10^{-30}$ months
$H = 10^{-62}$, which is too small even to store in float32. The naive
$\log(1 - S)$ is $-\infty$, while $\log F \approx \log H = 2\log(10^{-31})
\approx -142.8$:

```{code-cell} ipython3
wearout = survjax.Weibull(shape=2.0, scale=10.0)
early = 1e-30
jnp.log(1.0 - wearout.survival(early)), wearout.log_cdf(early)
```

**Very late.** At $t = 60$ months, $H = 36$ and $S = e^{-36} \approx 2.3
\times 10^{-16}$. Then $\log F = \log(1 - S) \approx -S$, a tiny negative
number. Forming $1 - S$ first rounds it away (to exactly 0 in float32, and
to the wrong value in float64); `log_cdf` keeps it:

```{code-cell} ipython3
late = 60.0
jnp.log(1.0 - wearout.survival(late)), wearout.log_cdf(late), -wearout.survival(late)
```

## Coming from R and NumPyro

The exponential has one common parameterisation, the rate (R's `rate`). The
Weibull has several, and translating between them is a frequent source of
error. survjax uses the shape $k$ and the scale $b$, the same as base R and
ordinary `flexsurv`:

| Library and convention | Parameters | In survjax |
| --- | --- | --- |
| base R `dweibull`, `pweibull` | `shape`, `scale` | `shape=k`, `scale=b` |
| `flexsurv` `dist = "weibull"` | `shape`, `scale` | `shape=k`, `scale=b` |
| `flexsurv` `dist = "weibullPH"` | `shape` $k$, `scale` $m$, with $S(t) = e^{-m t^{k}}$ | `shape=k`, `scale=m ** (-1 / k)` |
| `survival::survreg`, `dist = "weibull"` | `Intercept` $\mu$, `Scale` $\sigma$ | `shape=1 / sigma`, `scale=exp(mu)` |
| NumPyro `Weibull` | `scale`, `concentration` | `shape=concentration`, `scale=scale` |
| an inverse time scale $r$, $S(t) = e^{-(rt)^k}$ | $r$ | `scale=1 / r` |

The links point to the
[R `Weibull` help page](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Weibull.html),
[flexsurv](https://cran.r-project.org/package=flexsurv),
[`survreg`](https://rdrr.io/cran/survival/man/survreg.html) and its
[distribution conventions](https://rdrr.io/cran/survival/man/survreg.distributions.html),
and [NumPyro's distributions](https://num.pyro.ai/en/stable/distributions.html).
The [R comparisons below](#comparisons-with-r) describe the committed
reference outputs used to check these parameter mappings numerically.

### `survreg`: an accelerated failure time model

`survreg` models log time, $\log T = \mu + \sigma W$ with a standard
[extreme value](https://en.wikipedia.org/wiki/Gumbel_distribution) error $W$.
This is the Weibull with $b = e^{\mu}$ and $k = 1/\sigma$. Suppose an R fit
reports

```r
fit <- survreg(Surv(time, status) ~ 1, dist = "weibull")
coef(fit)   # (Intercept) 2.4849  -> mu = log(12)
fit$scale   # 0.6667             -> sigma = 1/1.5
```

then the survjax model has shape 1.5 and scale 12. Because $\mu$ and
$\log \sigma$ are already on the log scale, `from_log` takes them directly:

```{code-cell} ipython3
mu, sigma = jnp.log(12.0), 2.0 / 3.0
direct = survjax.Weibull(shape=1 / sigma, scale=jnp.exp(mu))
logged = survjax.Weibull.from_log(log_shape=-jnp.log(sigma), log_scale=mu)
direct.survival(12.0), logged.survival(12.0)
```

With covariates, `survreg` adds a linear predictor to the log scale,
$\log b_i = \mu + x_i^\top \beta$. This is an
[accelerated failure time](https://en.wikipedia.org/wiki/Accelerated_failure_time_model)
model: a negative $\beta$ shortens times by the factor $e^{\beta}$. For the
exponential, `survreg(..., dist = "exponential")` has no scale and its
intercept is the log mean, so `Exponential.from_log(log_rate=-mu)`.

### Proportional hazards coefficients

When all individuals share the shape $k$, the Weibull is also a
[proportional hazards](https://en.wikipedia.org/wiki/Proportional_hazards_model)
model. Scaling time by $e^{\beta}$ multiplies the hazard by $e^{-k\beta}$, so
the AFT coefficient $\beta$ corresponds to the log hazard ratio $\gamma =
-k\beta$. With $k = 1.5$ and $\beta = -0.6$ (a group with shorter times), the
hazard ratio is $e^{0.9} \approx 2.46$ at every time:

```{code-cell} ipython3
beta = -0.6
groups = survjax.Weibull(shape=1.5, scale=12.0 * jnp.exp(jnp.array([0.0, beta])))
times = jnp.array([1.0, 12.0, 36.0])
groups[1].hazard(times) / groups[0].hazard(times), jnp.exp(-1.5 * beta)
```

If the shape varies between individuals, hazards are no longer proportional
and there is no such simple conversion.

### flexsurv: natural estimates and `coef`

`flexsurvreg(..., dist = "weibull")` reports estimates on the natural scale
(`fit$res`: shape $k$, scale $b$), but `coef(fit)` returns the baseline shape
and scale *logged*, followed by the covariate effects on the log scale. So a
model rebuilt from `coef(fit)` uses `from_log`:

```r
fit <- flexsurvreg(Surv(time, status) ~ group, dist = "weibull")
fit$res[c("shape", "scale"), "est"]   # natural k and b
coef(fit)                             # log(k), log(b), then beta per covariate
```

```{code-cell} ipython3
log_shape, log_scale, beta_group = jnp.log(1.5), jnp.log(12.0), -0.6
x = jnp.array([0.0, 1.0])  # reference group, comparison group
survjax.Weibull.from_log(log_shape=log_shape, log_scale=log_scale + beta_group * x).scale
```

`dist = "weibullPH"` is the same distribution written as $S(t) = e^{-m t^k}$
with $m = b^{-k}$; its covariates act on $\log m$, as log hazard ratios. For
$k = 1.5$, $b = 12$, $m = 12^{-1.5} \approx 0.0241$, and back again
$b = m^{-1/k}$:

```{code-cell} ipython3
k, m = 1.5, 12.0 ** -1.5
survjax.Weibull(shape=k, scale=m ** (-1 / k)).scale
```

### NumPyro

NumPyro's `Weibull(scale, concentration)` is the same distribution, with
`concentration` $= k$. Its `batch_shape` means the same as survjax's;
survjax distributions additionally carry survival-specific methods such as
`log_cdf`, `cumulative_hazard` and the time-first grids.

## Comparisons with R

The committed synthetic fixtures compare exponential and Weibull likelihoods
and predictions with R's **flexsurv 2.3.2** and **survival 3.7-0**, generated
with **R 4.4.2**. Only simulated observations and their R outputs are stored.
The reference tests read those files without running R or accessing a network.
They evaluate fixed parameter vectors; they do not fit models in survjax.

| Fixture | Observations and parameter cases | R quantities compared |
| --- | --- | --- |
| Exponential | 16 rows, seed 8101; rates 0.1, 0.35 and 1.2 per year | Every row's log likelihood, survival and log survival; separate total log likelihood |
| Weibull AFT | 686 rows, seed 8201; three common-shape, group-scale vectors | Every row's log likelihood, survival and log survival; separate totals; a time-first survival grid |

Both fixtures contain exact events and right censoring without truncation.
The Weibull fixture uses Good/Medium/Poor groups of sizes 229/229/228,
with scales $b_g = b_{\mathrm{Good}}\exp(\beta_g)$. Its three vectors are
named `fitted`, `unit_shape` and `alternative`. `fitted` identifies the
recorded vector used to simulate the data, rather than a fit to these
synthetic rows. The rows are newly simulated; no patient observations from
`flexsurv::bc` are loaded or committed. Fixture times are in **years**,
while the earlier teaching examples on this page use months.

The Weibull grid has shape `(141, 3)`: times from 0 to 7 years in rows,
Good/Medium/Poor in columns. Tests use the stored R times; rebuilding R's
`seq` with JAX `arange` or `linspace` can shift a time by one ULP. The survreg
grid route evaluates `psurvreg` directly at supplied AFT locations and is
cross-checked against flexsurv. Singleton-fit and full-data likelihood checks
cover the observed-time contributions, not individual grid cells.

The comparisons exercise eager execution, `jax.jit` and `jax.vmap` in separate
float64 and float32 processes. Weibull checks include both natural and
`from_log` constructors, a common shape with batched scales, and raw and
prepared likelihood inputs through the function and distribution method.
Expected R values retain their stored float64 precision; only inputs are cast
to the active JAX dtype. The central test tolerances are:

| Active dtype | Absolute tolerance | Relative tolerance |
| --- | --- | --- |
| float64 | `1e-10` | `1e-8` |
| float32 | `1e-5` | `1e-4` |

These finite reference cases complement the independent mathematical and
[numerical boundary tests](numerical-behaviour.md). The two R package paths
are related: flexsurv uses survreg on this route, so their agreement is not
fully independent numerical evidence. The fixtures provide no evidence for
unimplemented censoring or truncation likelihoods, or for fitting accuracy.

### Integrity and regeneration

In a source checkout, `tests/reference/test_exponential_r.py` and
`tests/reference/test_weibull_bc_like_r.py` use one shared fixture loader.
It verifies all required payloads against each fixture's `SHA256SUMS` before
parsing them, then checks provenance, row alignment, dimensions and labels.
The Weibull loader also checks that the acceptance-schema export preserves
the original observations, parameters, source results and grid.

Regenerate only with the pinned R/package versions above. The generators in
`scripts/fixtures/` support `--check-only` without writing. A normal run
**overwrites payload files** in its destination; without a destination it
writes into the committed fixture directory. To inspect fresh outputs first,
pass a new scratch directory, for example from the repository root:

```sh
Rscript --vanilla scripts/fixtures/generate_bc_like.R --check-only
Rscript --vanilla scripts/fixtures/generate_bc_like.R /tmp/bc-like-review-run1
```

Follow the full two-run comparison and checksum recipes in the fixture
READMEs: {download}`exponential regeneration <../tests/fixtures/exponential/README.md>`
and {download}`Weibull regeneration <../tests/fixtures/bc_like/README.md>`.
Compare fresh outputs with each other and with committed payloads before
replacing anything. Exports use 17 significant digits and record seed,
versions, generating scripts and checksums. During fixture validation,
Python separately checked decimal read-back against original R hexadecimal
doubles; Python did not generate expected numerical values.

The directory also holds the same R outputs in the acceptance-test schema
(`bc_synthetic_weibull.json`); ordinary regeneration never writes to
`tests/spec/`.
