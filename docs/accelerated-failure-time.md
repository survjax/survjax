---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Accelerated failure time

An accelerated failure time (AFT) model describes how covariates stretch or
compress the time to an event. Let $T_0$ denote a baseline event time and
write

$$
T = e^\eta T_0, \qquad \log T = \log T_0 + \eta.
$$

The dimensionless factor $e^\eta$ is the *time ratio*: with $\eta=\log(1.5)$,
each quantile of the event-time distribution is 50% later than its baseline
counterpart. **Positive eta means longer event times.** In
[proportional hazards](proportional-hazards.md), positive eta instead means
higher hazard. The two predictors describe different statistical quantities.

`survjax.AcceleratedFailureTime` takes a baseline distribution and a
caller-computed `log_time_ratio`. We will use synthetic progression times in
months, beginning with a Weibull baseline.

## Stretching a survival curve

The event $T>t$ is equivalent to $T_0>t e^{-\eta}$. Thus

$$
S(t\mid\eta)=S_0(t e^{-\eta}).
$$

For the same survival probability, the AFT curve reaches it $e^\eta$ times
later. Consider a baseline with Weibull shape $k=1.5$ and scale $b_0=12$
months, shared by two groups with time ratios 1.0 and 1.5:

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import survjax

baseline = survjax.Weibull(shape=1.5, scale=12.0)
aft = survjax.AcceleratedFailureTime(
    baseline=baseline, log_time_ratio=jnp.log(jnp.array([1.0, 1.5]))
)
months = jnp.array([6.0, 12.0, 18.0])
print("distribution batch:", aft.batch_shape)
print("columns: month, group 1 survival, group 2 survival")
jnp.column_stack((months, aft.survival_grid(months)))
```

The scalar baseline broadcasts across the two log time ratios, giving
`aft.batch_shape == (2,)`: one distribution per group. Group 1 has time ratio
1 and follows the baseline. `survival_grid(months)` evaluates both groups at
every month, with shape `(3, 2)`: time first, then group.

At 18 months group 2's survival equals group 1's survival at 12 months:
both are $e^{-1}\approx0.368$. In fact, substituting the Weibull
survival into the AFT expression gives

$$
S(t\mid\eta)
=\exp\!\left[-\left(\frac{t}{b_0 e^\eta}\right)^k\right].
$$

Each group therefore has a Weibull distribution with the same shape and
scale $b=b_0e^\eta$: 12 months for group 1 and 18 for group 2. Group 2's
median is also 1.5 times as large. We can calculate both medians from the
Weibull formula $m=b(\log 2)^{1/k}$:

```{code-cell} ipython3
baseline_median = 12.0 * jnp.log(2.0) ** (1 / 1.5)
group_medians = baseline_median * jnp.exp(aft.log_time_ratio)
print("median months, groups 1 and 2:", group_medians)

times = jnp.linspace(0.0, 36.0, 181)
fig, ax = plt.subplots(figsize=(6, 3.5))
curves = aft.survival_grid(times)
ax.plot(times, curves, label=["Group 1: time ratio 1", "Group 2: time ratio 1.5"])
ax.axhline(0.5, color="grey", linestyle=":")
ax.scatter(group_medians, [0.5, 0.5], color=["C0", "C1"])
ax.set(xlabel="Months", ylabel="Survival probability", ylim=(0, 1))
ax.legend()
fig.tight_layout()
plt.show()
```

The marked medians are about 9.4 and 14.1 months. AFT rescales the whole
time axis, so the same time ratio applies to other quantiles too. The median
calculation here is analytical; the wrapper's summary methods are not yet
implemented.

## What does this imply for the hazard?

Differentiating the AFT cumulative hazard $H(t\mid\eta)=H_0(t e^{-\eta})$
gives the hazard:

$$
h(t\mid\eta)=e^{-\eta}h_0(t e^{-\eta}).
$$

The density then follows from $f(t)=h(t)S(t)$. Substituting both the AFT
hazard above and $S(t\mid\eta)=S_0(t e^{-\eta})$ gives

$$
\begin{aligned}
f(t\mid\eta)
&=h(t\mid\eta)S(t\mid\eta) \\
&=e^{-\eta}h_0(t e^{-\eta})S_0(t e^{-\eta}) \\
&=e^{-\eta}f_0(t e^{-\eta}).
\end{aligned}
$$

The last step uses the same density identity for the baseline,
$f_0(u)=h_0(u)S_0(u)$.

The factor $e^{-\eta}$ also has a probability interpretation. A density is
probability *per unit of time*. Stretching the time axis spreads the same
probability over a wider interval. With a time ratio of 1.5, a short baseline
interval becomes 1.5 times wider, so the corresponding density per month
must be divided by 1.5.

More formally, the baseline time corresponding to $t$ is $u=t e^{-\eta}$.
A short interval of width $dt$ corresponds to baseline width
$du=e^{-\eta}dt$. Both intervals contain the same probability, so

$$
f(t\mid\eta)\,dt=f_0(u)\,du,
\qquad
f(t\mid\eta)=f_0(u)\frac{du}{dt}=f_0(u)e^{-\eta}.
$$

The derivative $du/dt=e^{-\eta}$ is called the Jacobian of this inverse
time transformation. Taking its logarithm gives the $-\eta$ adjustment in
`log_prob`; it also enters the likelihood for exact events below. survjax
includes it automatically. Survival probabilities such as $P(T>t)$ already
refer to an entire set of times, rather than a probability per unit time,
so they need only the transformed threshold $u$, with no extra multiplier.

A hazard compares event rates at a fixed time, whereas the time ratio
compares times at a fixed quantile. In general, the hazard ratio

$$
\frac{h(t\mid\eta)}{h_0(t)}
=e^{-\eta}\frac{h_0(t e^{-\eta})}{h_0(t)}
$$

varies with $t$. A Weibull baseline is a special case: its power-law hazard
makes this ratio $e^{-k\eta}$, constant over time. With a common shape $k$,
a Weibull AFT predictor $\eta$ consequently corresponds to a PH predictor
$-k\eta$. In our example group 2's hazard ratio relative to group 1 is
$1.5^{-1.5}\approx0.544$, even though the time ratio is 1.5. For an
exponential baseline ($k=1$), the rescaled rate is simply $r e^{-\eta}$.

Use `aft.log_hazard(12.0)` for the two groups' log event rates at month 12,
or `aft.log_prob(12.0)` for their log densities. The
[API reference](api.md#accelerated-failure-time) lists the remaining direct
methods. Time retains the baseline's units throughout.

## From covariates to a likelihood

For regression, let $\eta_i=x_i^\mathsf{T}\beta$. A one-unit difference in
covariate $j$, holding the others fixed, multiplies event-time quantiles by
$e^{\beta_j}$. You construct the design matrix and predictor; survjax
evaluates the resulting distributions and their likelihood contributions.

Suppose two synthetic observations belong to each of two groups. We keep
the baseline Weibull parameters fixed and encode group membership in a
single design-matrix column:

```{code-cell} ipython3
X = jnp.array([[0.0], [0.0], [1.0], [1.0]])
beta = jnp.array([jnp.log(1.5)])
data = survjax.SurvivalData(
    time=[6.0, 12.0, 10.0, 18.0], event=[True, False, True, False]
)
regression = survjax.AcceleratedFailureTime(
    baseline=baseline, log_time_ratio=X @ beta
)
regression.log_likelihood(data)
```

An exact event at $t_i$ contributes a density; right censoring at $c_i$
contributes the probability of surviving beyond $c_i$. Under independent,
noninformative censoring, and omitting factors involving only the censoring
mechanism, their combined log likelihood is

$$
\ell(\beta)=\sum_i\left\{
\delta_i\left[\log f_0(y_i e^{-\eta_i})-\eta_i\right]
+(1-\delta_i)\log S_0(y_i e^{-\eta_i})
\right\},
$$

where $y_i$ is the observed event or censoring time and $\delta_i$ indicates
an event. The $-\eta_i$ term is the log density adjustment for the change
in interval width derived above. It belongs to exact-event contributions;
a right-censored record contributes $\log S_0(y_i e^{-\eta_i})$ without
this adjustment, because it records survival beyond a threshold. Both
`log_prob` and `log_likelihood` include the appropriate adjustment for you.

The returned array has one entry per observation. Sum it to obtain an
objective for your chosen optimizer:

```{code-cell} ipython3
def negative_log_likelihood(beta):
    model = survjax.AcceleratedFailureTime(
        baseline=baseline, log_time_ratio=X @ beta
    )
    return -model.log_likelihood(data).sum()


value, gradient = jax.jit(jax.value_and_grad(negative_log_likelihood))(beta)
print("negative log likelihood:", value)
print("derivative with respect to beta:", gradient)
```

These are an objective and its derivative at the supplied coefficient,
rather than a fitted coefficient. Baseline parameters can also be arguments
to an objective. If estimating both a free baseline scale and regression
coefficients, avoid including a second free intercept in the predictor:
only their combined log scale is identifiable.

## Draws, groups and a shared time grid

The baseline and predictor broadcast at construction. For example, three
hypothetical parameter draws can each describe two groups. Store baseline
scales with shape `(3, 1)` and log time ratios with shape `(3, 2)`:

```{code-cell} ipython3
scales = jnp.array([10.0, 12.0, 14.0])[:, None]
time_ratios = jnp.array([[1.0, 1.3], [1.0, 1.5], [1.0, 1.7]])
draws = survjax.AcceleratedFailureTime(
    baseline=survjax.Weibull(shape=1.5, scale=scales),
    log_time_ratio=jnp.log(time_ratios),
)
curves = draws.survival_grid(times)
print("distribution batch:", draws.batch_shape)
print("shared-grid curves:", curves.shape)
print("12-month survival, first draw:", draws[0].survival(12.0))
```

The curves have shape `(181, 3, 2)`: time first, then draw, then group.
A scalar pointwise call such as `draws.survival(12.0)` instead returns one
value per draw and group, shape `(3, 2)`. Array-valued pointwise times follow
ordinary trailing-axis broadcasting. See [time grids](grids.md) for choosing
between paired evaluations and a shared grid.

## Current scope

The AFT wrapper supports direct distribution methods, their grid forms,
indexing and exact/right-censored likelihoods without truncation. Baselines
can be built-in or registered custom distributions. AFT RMST, mean, median,
quantile and sampling methods are not implemented yet, including for a
Weibull baseline; see [RMST](rmst.md) for the currently supported families.

The time rescaling is evaluated in log space. If a positive interior time
cannot be represented after rescaling, the current implementation returns
NaN as a numerical failure instead of substituting a zero or infinite-time
limit. See [numerical behaviour](numerical-behaviour.md) for the dtype and
diagnostic conventions.
