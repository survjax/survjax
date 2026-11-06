---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# R and NumPyro parameter conventions

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import survjax
```

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
The [R comparisons](verification.md#comparisons-with-r) describe the committed
reference outputs used to check these parameter mappings numerically.

## `survreg`: an accelerated failure time model

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

## Proportional hazards coefficients

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

## flexsurv: natural estimates and `coef`

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

## NumPyro

NumPyro's `Weibull(scale, concentration)` is the same distribution, with
`concentration` $= k$. Its `batch_shape` means the same as survjax's;
survjax distributions additionally carry survival-specific methods such as
`log_cdf`, `cumulative_hazard` and the time-first grids.

