# survjax

**Survival Analysis in JAX**

`survjax` is a lightweight JAX library for survival analysis.
It provides `SurvivalData`: an object for representing
arbitrary mixtures of observations
that can be exact, censored (right, left, interval),
and truncated (right and left).

It provides `SurvivalDistribution`s that allow for calculating
the `log_likelihood` of observations (currently implemented: exact and right-censored).
In addition to built-in families, users can define their own family
by specifying only `log_survival`; the base can take care of derived functions and quantities such as
`log_cdf`, `log_prob` (density for exact observations), `log_hazard`, `log_cumulative_hazard`,
and predictions like `survival`, `mean` and `rmst`

And `metric`s useful in survival analysis, such as Harrell's `concordance` index.

Among `survjax`s use-cases are large-scale simulation studies,
neural-network based survival analysis (e.g. with Optax and Equinox)
and Bayesian survival analysis (e.g. with NumPyro).

What `survjax` does **not** provide are fitting routines, as the library is intended
to be used in conjunction with other inference libraries, such as
Optax, optimistix or NumPyro (see also the use-cases)

## Minimal Example

```{python}
import survjax
from survjax import SurvivalData, Exponential
import optimistix as otpx

data = SurvivalData(
  time=jnp.array([1., 2., 1.5, 5.]),
  event=jnp.array([True, False, False, True])
)

def loss(log_rate, *):
  model = Exponential.from_log(log_rate=log_rate)
  return -model.log_likelihood(data).sum() # log_likelihood(time=time, event=event) works too

# get maximum likelihood solution
sol = optx.minimize(loss, optx.BFGS(), jnp.zeros(()))
sol.value # estimate of log_rate

# get confidence intervals
hessian = jax.hessian(loss)

```

All distributions and metrics have a default data first-argument API (`model.log_likelihood(data)`) and - for exact right-censored data only - `model.log_likelihood(time=time, event=event)`.

## Goals and guiding principles

- consistent API
- numerical accuracy (no floors, clamps, added epsilons) and speed (functions tested with and without `jit` and `vmap`)
- speed
- safe log-space calculations to prevent underflow and overflow (as survival analysis often works with exponentials of exponentials)
- gradients come 'for free' by using `JAX`

## Documentation

`survjax` comes with documentation which can be found at [https://survjax.readthedocs.io](https://survjax.readthedocs.io).
The documentation contains a user guide, API reference, and examples.

## Development status

This is an alpha pre-release.
