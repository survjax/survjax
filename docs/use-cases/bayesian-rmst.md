---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Bayesian survival analysis of a randomized trial

How much longer does a treatment keep patients event-free? In this tutorial we
fit a Bayesian Weibull proportional hazards model to a simulated two-arm trial
and answer that question with the difference in *restricted mean survival
time* (RMST), together with its full posterior uncertainty.
The prior definition and sampling are done in NumPyro,
survjax supplies the censored-data likelihood and turns the posterior
draws into survival curves and RMST. Because we simulated the trial, we can
compare the posterior with the truth.

The same tutorial is available as a
{download}`runnable Jupyter notebook <../../examples/bayesian_rmst/bayesian-rmst.ipynb>`,
which samples the posterior itself; see [Reproduce](#reproduce-bayes).

```{code-cell} ipython3
:tags: [remove-input]
import json
from pathlib import Path

import numpy as np
from IPython.display import HTML, display

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").is_file())
FULL = ROOT / "examples/bayesian_rmst/results"
full_meta = json.loads((FULL / "metadata.json").read_text())


def kaplan_meier(time, event):
    """Product-limit step coordinates; censorings tied with an event stay at risk."""
    time, event = np.asarray(time), np.asarray(event)
    event_times = np.unique(time[event])
    deaths = np.array([np.sum((time == t) & event) for t in event_times])
    at_risk = np.array([np.sum(time >= t) for t in event_times])
    survival = np.cumprod(1 - deaths / at_risk)
    return np.r_[0.0, event_times, time.max()], np.r_[1.0, survival, survival[-1]]


def posterior_table(draws):
    """Summarize per-draw values in an HTML table, with the run's truth and diagnostics."""
    header = ["", "Truth", "Mean", "SD", "95% credible interval", "R-hat", "ESS"]
    rows = []
    for key, (label, values) in draws.items():
        values = np.asarray(values)
        low, high = np.quantile(values, [0.025, 0.975])
        diag = full_meta["diagnostics"][key]
        cells = [label, f"{full_meta['truth'][key]:.3f}", f"{values.mean():.3f}",
                 f"{values.std(ddof=1):.3f}", f"[{low:.3f}, {high:.3f}]",
                 f"{diag['r_hat']:.3f}", f"{diag['ess']:.0f}"]
        rows.append("".join(f"<td>{c}</td>" for c in cells))
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join(f"<tr>{r}</tr>" for r in rows)
    return HTML(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
```

## The model

Each patient $i$ is randomized to control ($a_i=0$) or treatment ($a_i=1$).
Event times follow a Weibull proportional hazards model,

$$
S(t\mid a)=\exp\!\Big[-(t/b_0)^k\,e^{\gamma a}\Big],
$$

with shape $k$, control scale $b_0$ and log hazard ratio $\gamma$; the hazard
ratio of treatment versus control is $e^\gamma$. As is typical in survival analysis,
some patients are still event-free when follow-up ends. For patient $i$ we
observe $t_i=\min(T_i, C_i)$ and the event indicator $\delta_i$. An observed
event contributes the density $f(t_i)$, a censored observation contributes
the survival probability $S(t_i)$, and with independent, non-informative
censoring the log likelihood is

$$
\ell(k, b_0, \gamma)=\sum_{i}\Big[\delta_i\log f(t_i\mid a_i)+(1-\delta_i)\log S(t_i\mid a_i)\Big].
$$

survjax computes the terms of this sum, one per patient, with
`dist.log_likelihood(data)`; `data` is a
[`SurvivalData`](../survival-data.md) container holding the times and event
indicators.

## A simulated trial where we know the answer

The trial has 300 patients, 150 per arm. The true shape is $k=1.38$, the
control scale $b_0=11.42$ years and the hazard ratio 0.7, so treatment lowers
the hazard by 30% at every time. Censoring times are Uniform$(0, c)$ with
$c\approx19.2$ years, chosen so that about 55% of patients are censored; in
the simulated trial 164 of 300 are.

A hazard ratio is hard to translate into time gained. RMST is easier: it is
the expected event-free time within a horizon $\tau$, the area under the
survival curve up to $\tau$,

$$
\mathrm{RMST}(\tau)=\operatorname{E}[\min(T,\tau)]=\int_0^\tau S(t)\,dt.
$$

The difference between the arms, $\Delta\mathrm{RMST}(\tau)$, is measured in
years and does not rely on proportional hazards to be interpretable. We use
$\tau=5$ years. survjax evaluates RMST in closed form for the Weibull and its
proportional hazards version (see [RMST](../rmst.md)). For the true model:

```{code-cell} ipython3
import jax.numpy as jnp
import matplotlib.pyplot as plt

import survjax

both_arms = jnp.array([0.0, 1.0])  # control, treatment
truth = survjax.ProportionalHazards(
    survjax.Weibull(shape=1.38, scale=11.42), log_hazard_ratio=jnp.log(0.7) * both_arms
)
times = jnp.linspace(0, 20, 201)
curves = truth.survival_grid(times)  # grid shape: (time, arm)

fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
ax.plot(times, curves, label=["Control", "Treatment"])
up_to_5 = times <= 5
ax.fill_between(times[up_to_5], curves[up_to_5, 0], curves[up_to_5, 1], color="0.6",
                label="ΔRMST(5)")
ax.set(xlabel="Time (years)", ylabel="True survival $S(t)$", ylim=(0, 1))
ax.legend()
plt.show()

rmst = truth.rmst(5.0)
print(f"True RMST(5): control {rmst[0]:.3f}, treatment {rmst[1]:.3f}, "
      f"difference {rmst[1] - rmst[0]:.3f} years")
```

Over the first five years, treated patients gain on average 0.17 years, about
two months, of event-free time. Most patients in both arms are still
event-free at five years, which is why the difference is small compared to
the horizon.

## The likelihood as a NumPyro factor

We define a model in NumpPyro that samples the
three parameters on the log scale, builds the survjax survival distribution,
and adds the per-patient log likelihood to the target with `numpyro.factor`:

```python
def model(data, arm):
    log_k = numpyro.sample("log_k", distributions.Normal(0, 5))
    log_b0 = numpyro.sample("log_b0", distributions.Normal(0, 5))
    gamma = numpyro.sample("gamma", distributions.Normal(0, 5))

    base = survjax.Weibull.from_log(log_shape=log_k, log_scale=log_b0)
    survival = survjax.ProportionalHazards(base, log_hazard_ratio=gamma * arm)
    with numpyro.plate("observations", len(data)):
        numpyro.factor("survival", survival.log_likelihood(data))
```

The full model is in
{download}`model.py <../../examples/bayesian_rmst/model.py>`. The factor
receives the vector of 300 terms, one per patient, inside the observation
plate, which tells NumPyro that the terms are conditionally independent
and can speed up with vectorization.

NumPyro then samples the posterior with NUTS. We run four chains of 1,000 warmup
and 1,000 retained draws each, and collect the draws of all chains:

```python
from numpyro.infer import MCMC, NUTS

mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=1000, num_chains=4)
mcmc.run(jax.random.key(1), data, arm)
mcmc.print_summary()  # R-hat and effective sample size per parameter
samples = mcmc.get_samples()  # {"log_k": (4000,), "log_b0": (4000,), "gamma": (4000,)}
```

The full run had no divergent transitions, and all parameters and derived
quantities reached $\hat R<1.01$ and an effective sample size above 2,000.

## The posterior is a batch of distributions

The posterior is a set of $S=4{,}000$ draws of $(k, b_0, \gamma)$. In survjax
a distribution with array-valued parameters is a batch of distributions,
so we can create one distribution with the draws and treatment arm ($a=0, a=1$)
and get batch shape (draw, arm). The cells below run on the 4,000 draws of the
committed full run:

```{code-cell} ipython3
:tags: [remove-input]
with np.load(FULL / "posterior.npz", allow_pickle=False) as saved:
    samples = {name: jnp.asarray(saved[f"sample_{name}"]).reshape(-1)
               for name in ("log_k", "log_b0", "gamma")}
    tau_grid = jnp.asarray(saved["tau_grid"])
    observed_time, observed_event, observed_arm = saved["time"], saved["event"], saved["arm"]
```

```{code-cell} ipython3
base = survjax.Weibull.from_log(
    log_shape=samples["log_k"][:, None], log_scale=samples["log_b0"][:, None]
)
posterior = survjax.ProportionalHazards(
    base, log_hazard_ratio=samples["gamma"][:, None] * jnp.array([0.0, 1.0])
)
rmst5 = posterior.rmst(5.0)          # (S, 2)
delta5 = rmst5[:, 1] - rmst5[:, 0]   # paired within each draw, (S,)

low, high = jnp.quantile(delta5, jnp.array([0.025, 0.975]))
print(f"ΔRMST(5): mean {delta5.mean():.3f} years, 95% interval [{low:.3f}, {high:.3f}]")
print(f"P(ΔRMST(5) > 0) = {(delta5 > 0).mean():.3f}")
```

To propagate the uncertainty correctly, we compute every summary within each
draw, keep the two arms of a draw paired, and summarize only at the end: the
posterior mean, credible interval and $P(\Delta\mathrm{RMST}(5)>0)$ are all
summaries of the 4,000 values in `delta5`.

The same quantities can also be declared inside the model, as deterministic
sites. A flag keeps them out of the sampler, where they are not needed, and
NumPyro's `Predictive` fills them in afterwards for every posterior draw:

```python
def model(data, arm, compute_rmst=False, tau=5.0, tau_grid=None):
    ...  # priors and likelihood factor as above
    if compute_rmst:
        per_arm = survjax.ProportionalHazards(
            base, log_hazard_ratio=gamma * jnp.array([0.0, 1.0])
        )
        numpyro.deterministic("rmst_tau", per_arm.rmst(tau))  # (2,)
        numpyro.deterministic("rmst_curve", per_arm.rmst(tau_grid[:, None]))  # (m, 2)


predictive = Predictive(model, posterior_samples=samples)
sites = predictive(jax.random.key(0), data, arm, compute_rmst=True, tau_grid=tau_grid)
sites["rmst_tau"]  # (S, 2), the same values as rmst5 above
```

This route is convenient when the summaries belong with the model, for
example to store them with the draws. The example computes both and checks
that they agree to within $10^{-13}$ years.

## Results

The posterior summaries of these draws, with the true values and the sampling
diagnostics of the run:

```{code-cell} ipython3
:tags: [remove-input]
display(posterior_table({
    "k": ("Shape k", jnp.exp(samples["log_k"])),
    "HR": ("Hazard ratio", jnp.exp(samples["gamma"])),
    "RMST0(5)": ("RMST(5), control", rmst5[:, 0]),
    "RMST1(5)": ("RMST(5), treatment", rmst5[:, 1]),
    "delta_RMST(5)": ("ΔRMST(5)", delta5),
}))
```

The posterior survival curves per arm, with 95% pointwise credible bands, the
true curves and the observed Kaplan–Meier curves:

```{code-cell} ipython3
survival = posterior.survival_grid(times)  # (time, S, 2)
low, median, high = jnp.quantile(survival, jnp.array([0.025, 0.5, 0.975]), axis=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, layout="constrained")
for a, (ax, label) in enumerate(zip(axes, ["Control", "Treatment"], strict=True)):
    ax.fill_between(times, low[:, a], high[:, a], alpha=0.3, label="95% pointwise band")
    ax.plot(times, median[:, a], label="Posterior median")
    ax.plot(times, curves[:, a], "--", color="C1", label="Truth")
    in_arm = observed_arm == a
    ax.step(*kaplan_meier(observed_time[in_arm], observed_event[in_arm]), where="post",
            color="black", label="Observed Kaplan–Meier")
    ax.set(title=label, xlabel="Time (years)", ylim=(0, 1))
axes[0].set_ylabel("Survival probability")
axes[1].legend()
plt.show()
```

The posterior distribution of $\Delta\mathrm{RMST}(5)$, with its 95% credible
interval and the true value. Positive values favour treatment.

```{code-cell} ipython3
:tags: [remove-input]
low, high = jnp.quantile(delta5, jnp.array([0.025, 0.975]))
fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
ax.hist(delta5, bins=50, density=True, alpha=0.6)
ax.axvspan(low, high, alpha=0.15, label="95% credible interval")
ax.axvline(full_meta["truth"]["delta_RMST(5)"], color="C1", linestyle="--", label="Truth")
ax.axvline(0, color="0.5", linewidth=0.8)
ax.set(xlabel="ΔRMST(5) (years)", ylabel="Posterior density")
ax.legend()
plt.show()
```

Putting the horizons on a leading axis evaluates RMST for every horizon, draw
and arm at once. The grid runs to 13 years, about the 90th percentile of
follow-up:

```{code-cell} ipython3
rmst_curve = posterior.rmst(tau_grid[:, None, None])     # (horizon, S, 2)
delta_curve = rmst_curve[..., 1] - rmst_curve[..., 0]    # (horizon, S)
low, median, high = jnp.quantile(delta_curve, jnp.array([0.025, 0.5, 0.975]), axis=1)
true_rmst = truth.rmst(tau_grid[:, None])                # (horizon, 2)

fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
ax.fill_between(tau_grid, low, high, alpha=0.3, label="95% pointwise band")
ax.plot(tau_grid, median, label="Posterior median")
ax.plot(tau_grid, true_rmst[:, 1] - true_rmst[:, 0], "k--", label="Truth")
ax.axhline(0, color="0.5", linewidth=0.8)
ax.set(xlabel="Horizon τ (years)", ylabel="ΔRMST(τ) (years)")
ax.legend(loc="upper left")
plt.show()
```

To go further, add covariates to the linear predictor, `log_hazard_ratio=X @
beta`, try another baseline distribution, or report RMST at several horizons
from the same draws. See the [proportional hazards](../proportional-hazards.md),
[RMST](../rmst.md) and [likelihood](../likelihood.md) pages.

(reproduce-bayes)=

## Reproduce

To work through the analysis interactively, open the notebook from the
repository root, then **Restart Kernel and Run All Cells** in a fresh Python 3
kernel. It samples the posterior in a few seconds:

```bash
uv run --group docs --group examples jupyter lab examples/bayesian_rmst/bayesian-rmst.ipynb
```

This page computes everything from the simulated trial and the posterior
draws committed in `examples/bayesian_rmst/results/posterior.npz`, with the
{download}`run metadata <../../examples/bayesian_rmst/results/metadata.json>`
(priors, seed, versions and diagnostics). To sample them again, from the
repository root:

```bash
JAX_ENABLE_X64=1 uv run --group examples python -m examples.bayesian_rmst.run /tmp/survjax-bayes
```

The run takes about ten seconds on a laptop CPU and writes to a new directory.
It also checks that the likelihood factor is wired correctly, that the two
RMST routes agree, that the chains converged, and that the true hazard ratio
and $\Delta\mathrm{RMST}(5)$ lie inside their 95% credible intervals. The
example runs in float64 and needs only the optional `examples` dependencies;
survjax itself does not depend on NumPyro.

## References

Hoffman, M. D. and Gelman, A. (2014). The No-U-Turn sampler: adaptively
setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning
Research* 15, 1593–1623.

Phan, D., Pradhan, N. and Jankowiak, M. (2019). Composable effects for
flexible and accelerated probabilistic programming in NumPyro.
arXiv:1912.11554.

Royston, P. and Parmar, M. K. B. (2013). Restricted mean survival time: an
alternative to the hazard ratio for the design and analysis of randomized
trials with a time-to-event outcome. *BMC Medical Research Methodology* 13,
152.

All data on this page are synthetic.
