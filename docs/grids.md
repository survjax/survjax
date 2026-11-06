---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Survival curves and time grids

A fitted survival model is usually reported as curves: survival over a range
of times, one curve per patient, group or posterior draw. This page shows how
to evaluate a batch of distributions at many times at once, and the one
broadcasting trap to avoid on the way.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import survjax
```

## Three prognostic groups

The flexsurv paper's breast-cancer example (`flexsurv::bc`) models recurrence
in three prognostic groups, Good, Medium and Poor, with a Weibull
accelerated failure time model: a common shape $k$ and a group scale
$b_g = b_{\text{Good}}\, e^{\beta_g}$, in years. We take illustrative
parameter values close to that fit, $k = 1.5$, $b_{\text{Good}} = 12$,
$\beta_{\text{Medium}} = -0.6$ and $\beta_{\text{Poor}} = -1.2$; see
[Exponential and Weibull](technical-notes/parameter-conventions.md) for
how these parameters relate to R's output.

```{code-cell} ipython3
beta = jnp.array([0.0, -0.6, -1.2])  # Good, Medium, Poor
groups = survjax.Weibull(shape=1.5, scale=12.0 * jnp.exp(beta))
groups.batch_shape
```

## Pointwise methods pair times with distributions

The direct methods, `survival(t)` and its siblings, evaluate pointwise:
time and parameters broadcast against each other with the usual
[NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html),
matching trailing axes. A scalar time applies to every group, so this is
five-year survival in each group:

```{code-cell} ipython3
groups.survival(5.0)
```

A vector of times, however, is lined up *element by element* with the batch.
Three times against three groups pair up: Good at year 1, Medium at year 3,
Poor at year 5. That is the right thing when each patient has their own
time, as in a likelihood, but it is not a set of survival curves:

```{code-cell} ipython3
groups.survival(jnp.array([1.0, 3.0, 5.0]))
```

The pairing above happened without complaint, only because there were as
many times as groups. When the lengths differ and neither is one,
broadcasting fails and survjax raises `TypeError`, so the mistake is caught;
the equal-length case is the one to watch for.

```{code-cell} ipython3
try:
    groups.survival(jnp.array([1.0, 2.0, 3.0, 4.0]))
except TypeError as error:
    print("TypeError:", error)
```

## Grids put time first

Every pointwise method has a *grid twin* with a `_grid` suffix. It takes one
vector of $K$ times, shared by the whole batch, and returns an array of shape
`(K, *batch_shape)`: one row per time, the time axis first.

```{code-cell} ipython3
years = jnp.array([1.0, 3.0, 5.0])
groups.survival_grid(years)
```

Row $i$ belongs to `years[i]`, whatever its length, and the times need not be
sorted or distinct. Under the hood, `survival_grid(t)` is `survival` with `t`
reshaped to `(K, 1)` here: one trailing axis of length 1 per batch axis.

```{code-cell} ipython3
groups.survival(years[:, None])
```

The twins are `survival_grid`, `log_survival_grid`, `log_prob_grid`,
`log_cdf_grid`, `cumulative_hazard_grid`, `hazard_grid` and
`log_hazard_grid`. Every registered family has them, including the ones you
write yourself; there is nothing to implement. The time argument must be
one-dimensional; a scalar or a 2-D array raises `TypeError`, since the
pointwise methods already cover those shapes.

Grids make plotting curves a single call. Here are survival and hazard for
the three groups over seven years:

```{code-cell} ipython3
t = jnp.linspace(0.0, 7.0, 141)
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2))
for column, name in enumerate(["Good", "Medium", "Poor"]):
    left.plot(t, groups.survival_grid(t)[:, column], label=name)
    right.plot(t, groups.hazard_grid(t)[:, column], label=name)
left.set(xlabel="years", ylabel="survival S(t)", ylim=(0, 1.02))
right.set(xlabel="years", ylabel="hazard h(t)")
left.legend()
fig.tight_layout()
```

With a common shape the hazards are proportional: Medium's is
$e^{1.5 \times 0.6} \approx 2.5$ times Good's at every time.

## Grids that start at zero

Curves usually start at $t = 0$, a boundary of the time axis. Grids follow the
same boundary rules as the pointwise methods: survival is exactly 1 at time
zero, and its derivative in the parameters is exactly 0, not NaN. The
following uses the shape $k = 1$, the case where the Weibull hazard at zero is
finite but its log density is discontinuous in $k$ (see [time zero and
infinity](builtin-families.md#what-happens-at-time-zero-and-at-infinity)):

```{code-cell} ipython3
def mean_survival(shape):
    curves = survjax.Weibull(shape=shape, scale=10.0 * jnp.exp(jnp.array([0.0, -0.5, -1.0])))
    return curves.survival_grid(jnp.array([0.0, 1.0, 2.0])).mean()


mean_survival(1.0), jax.grad(mean_survival)(1.0)
```

The value and gradient are finite because every term is: the time-zero row
contributes exactly 1 to the curves and 0 to the gradient.

## Posterior draws: a curve per draw and patient

A Bayesian regression gives $D$ posterior draws of the shape $k$ and the
coefficients $\beta$. For $N$ patients with covariates $X$ (shape `(N, P)`),
draw $d$ and patient $n$ have scale $b_{dn} = \exp(x_n^\top \beta_d)$, so
`jnp.exp(beta @ X.T)` has shape `(D, N)`. Here are synthetic draws for four
patients:

```{code-cell} ipython3
draw_key, shape_key = jax.random.split(jax.random.key(0))
D = 500
# Columns: intercept, Medium, Poor. Draws scatter around the values above.
beta_draws = jnp.array([jnp.log(12.0), -0.6, -1.2]) + 0.1 * jax.random.normal(draw_key, (D, 3))
k_draws = 1.5 * jnp.exp(0.05 * jax.random.normal(shape_key, (D,)))
X = jnp.array(
    [
        [1.0, 0.0, 0.0],  # Good
        [1.0, 1.0, 0.0],  # Medium
        [1.0, 0.0, 1.0],  # Poor
        [1.0, 0.0, 1.0],  # Poor
    ]
)
scale = jnp.exp(beta_draws @ X.T)
scale.shape
```

The shape draws have shape `(D,)` and the scales `(D, N)`. Broadcasting
matches *trailing* axes, so `Weibull(shape=k_draws, scale=scale)` would try to
pair the $D$ draws of $k$ with the $N$ patients. Here that fails, because
$D \ne N$; if they happened to be equal it would silently pair draw $n$'s
shape with patient $n$:

```{code-cell} ipython3
try:
    survjax.Weibull(shape=k_draws, scale=scale)
except TypeError as error:
    print("TypeError:", error)
```

Give the shape an explicit patient axis of length 1, `k_draws[:, None]`, so
that each draw's shape applies to all patients:

```{code-cell} ipython3
posterior = survjax.Weibull(shape=k_draws[:, None], scale=scale)
posterior.batch_shape
```

The grid then has shape `(K, D, N)`: for every time, every draw and every
patient. Summaries over draws reduce axis 1. Below is the posterior median
survival curve with a 90% band for each of the first three patients:

```{code-cell} ipython3
t = jnp.linspace(0.0, 7.0, 71)
curves = posterior.survival_grid(t)
print(curves.shape)

lower, middle, upper = jnp.quantile(curves, jnp.array([0.05, 0.5, 0.95]), axis=1)
fig, ax = plt.subplots(figsize=(5, 3.2))
for n, name in enumerate(["Good", "Medium", "Poor"]):
    (line,) = ax.plot(t, middle[:, n], label=name)
    ax.fill_between(t, lower[:, n], upper[:, n], color=line.get_color(), alpha=0.2)
ax.set(xlabel="years", ylabel="survival S(t)", ylim=(0, 1.02))
ax.legend()
fig.tight_layout()
```

## Grids inside `jit` and `vmap`

Grids are ordinary JAX code. They compile with `jax.jit`, and under
`jax.vmap` over draws each mapped distribution has a batch of `N` patients,
so the grid per draw has shape `(K, N)` and the mapped result `(D, K, N)`:

```{code-cell} ipython3
years = jnp.array([1.0, 3.0, 5.0])
by_draw = jax.vmap(lambda dist: dist.survival_grid(years))(posterior)
compiled = jax.jit(lambda dist: dist.survival_grid(years))(posterior)
by_draw.shape, compiled.shape
```

The two layouts hold the same numbers, with the draw axis in a different
place:

```{code-cell} ipython3
jnp.allclose(jnp.moveaxis(by_draw, 0, 1), compiled)
```
