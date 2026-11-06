---
orphan: true
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Weibull simulation: coverage under model misspecification

A two-arm synthetic trial illustrates how to combine a survjax likelihood,
Optimistix BFGS and an exact JAX Hessian. We estimate a constant hazard ratio
with a correctly specified Weibull model and a misspecified Exponential model.
All times are in years; the hazard ratio is dimensionless.

The control distribution has shape $k=1.38$ and scale $b_0=11.42$.
With fixed equal allocation $x\in\{0,1\}$ and $\gamma=\log(0.7)$,

$$
H(t\mid x)=e^{\gamma x}(t/b_0)^k.
$$

Independent censoring is Uniform$(0,c)$, calibrated to pooled censoring
fractions 25% and 55%. We observe $\min(T,C)$ and whether $T\le C$.
Only exact events and right censoring without truncation are used.
The full design is $n\in\{100,300,1000\}$, 1000 replicates per scenario/model.

## True and fitted curves — committed full run

This illustration uses the first replicate accepted by **both** models at
$n=300$, target censoring 55% (scenario 3, replicate 0). These are committed
full-run curves; the live smoke below uses $n=100$. Survival curves are
illustrative, without confidence bands or inference on the curves.

![True and fitted survival by arm, committed n=300 example.](../../examples/weibull_simulation/results/fig1_survival.svg)

## The model and fitting code

The linear predictor is computed by the caller. The demo's actual objective
is shown here: $\theta=(\log k,\log b_0,\gamma)$ for Weibull and
$(\log\lambda_0,\gamma)$ for Exponential.

```{literalinclude} ../../examples/weibull_simulation/study.py
:language: python
:start-at:     if model == "weibull":
:end-at:     return -dist.log_likelihood(data).sum()
:dedent: 4
```

The fitter starts at shape 1, baseline scale equal to mean observed time
(or its reciprocal rate), and $\gamma=0$. BFGS uses `rtol=atol=1e-8`,
`max_steps=1000`, `throw=False`; derivatives are of the summed objective.
This is the corresponding code inside the reviewed single-replicate helper:

```{literalinclude} ../../examples/weibull_simulation/study.py
:language: python
:start-at:         solution = optx.minimise(
:end-at:         hessian = jax.hessian(objective)(theta)
:dedent: 8
```

Acceptance requires solver success, finite parameters/objective, strict
$\max|\nabla L|<10^{-5}$ and a finite positive-definite Hessian. An arm with
no events is unestimable. Accepted fits use
$\mathrm{se}(\hat\gamma)=\sqrt{[H^{-1}]_{\gamma\gamma}}$ and the Wald interval
$\exp(\hat\gamma\pm1.96\,\mathrm{se})$. Failure reasons are separate flags;
`failed` counts each rejected replicate once even if reasons overlap.

## Live smoke: 20 replicates, both censoring levels

The smoke checks execution; coverage evidence comes from the 1000-replicate
full run below. This study explicitly selects **float64**, including when the
site is built in float32 mode. Its notebook has its own kernel.

```{code-cell} ipython3
:tags: [hide-input]
import csv
import hashlib
import json
import sys
from pathlib import Path
from html import escape
from tempfile import TemporaryDirectory
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)

# Host path discovery/formatting only; simulation batches both levels and replicates.
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
from IPython.display import HTML, display
from examples.weibull_simulation import run as runner, study


def show_table(columns, rows):
    head = "".join(f"<th>{escape(str(value))}</th>" for value in columns)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    display(HTML(f'<div class="table-wrapper"><table class="docutils"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'))
```

```{code-cell} ipython3
start = perf_counter()
with TemporaryDirectory(prefix="survjax-smoke-") as folder:
    out = Path(folder)
    runner.run_config("smoke", out)
    with (out / "performance.csv").open(newline="") as handle:
        smoke = list(csv.DictReader(handle))
    smoke_metadata = json.loads((out / "metadata.json").read_text())
assert len(smoke) == 4 and all(int(row["attempted"]) == 20 for row in smoke)
assert smoke_metadata["runtime"]["x64_enabled"]
print(f"Live smoke: n=100, R=20 per model/level; float64; {perf_counter()-start:.2f} s")
# Host output formatting only; no scenario fitting loop.
show_table(["Target censoring", "Model", "Accepted", "Failed", "Coverage"], [
    [row["target_censoring"], row["model"], row["accepted"], row["failed"], row["coverage"]]
    for row in smoke
])
```

The runner uses `jit` with nested `vmap` over censoring levels and replicates.
A Python loop changes cohort size, because changing $n$ changes array shapes;
the two models have different parameter dimensions and compiled graphs.

### One live smoke-replicate fit

`prepare_scenario` generates validated observations using the same explicit
keys as the smoke. Here is a separate Weibull fit for scenario 1, replicate 0
($n=100$, target censoring 55%). This executes the objective/solver code above.

```{code-cell} ipython3
c55 = smoke_metadata["calibration"][1]["censor_limit"]
data, arm = runner.prepare_scenario(1, 100, 20, c55)
cohort = data[0]
cohort.issues.raise_if_invalid()  # concrete host checkpoint, before JIT
fit = jax.jit(lambda d: study.fit_replicate(d, arm, model="weibull"))(cohort)
fit = jax.device_get(fit)
assert fit.accepted
print(f"Accepted: {bool(fit.accepted)}; HR={float(jax.numpy.exp(fit.theta[-1])):.4f}")
print(f"Wald 95% HR interval: [{fit.hr_interval[0]:.4f}, {fit.hr_interval[1]:.4f}]")
```

## Committed full results: 1000 replicates per model/scenario

Everything in this section is loaded from the committed full run, generated
from clean source `a3792a7703ed539db777472d876bd00ac2d1568e`. It is separate
from the live smoke. File hashes are checked before displaying the results.

```{code-cell} ipython3
:tags: [hide-input]
results = ROOT / "examples/weibull_simulation/results"
manifest = json.loads((results / "evidence.json").read_text())
# Host file inspection only; the full simulation is not rerun during a docs build.
for name, digest in manifest["output_sha256"].items():
    assert hashlib.sha256((results / name).read_bytes()).hexdigest() == digest
with (results / "performance.csv").open(newline="") as handle:
    full = list(csv.DictReader(handle))
metadata = json.loads((results / "metadata.json").read_text())
assert len(full) == 12 and sum(int(r["failed"]) for r in full) == 127
assert metadata["source"]["worktree_dirty_at_start"] is False
assert metadata["source"]["commit"] == "a3792a7703ed539db777472d876bd00ac2d1568e"
assert all(int(r["gradient"]) == int(r["failed"]) for r in full)
assert all(int(r[k]) == 0 for r in full for k in ("solver", "nonfinite", "hessian", "unestimable"))
for row in full:  # Check the committed host summary, never refit the full study.
    if row["n"] == "1000" and row["model"] == "weibull":
        assert abs(float(row["coverage"]) - .95) <= 3 * float(row["coverage_mcse"])
```

### Censoring calibration

Calibration solves $\tfrac12\sum_x c^{-1}\int_0^c S_x(t)\,dt$ for each target.
The same calibrated limit applies at every cohort size; realized censoring
averages all observations across the 1000 replicates.

```{code-cell} ipython3
:tags: [hide-input]
show_table(["n", "Target %", "Limit c (years)", "Realized %"], [
    [batch["n"], f"{s['target_censoring']*100:.0f}", f"{s['censor_limit']:.8f}", f"{s['realized_censoring']*100:.3f}"]
    for batch in metadata["batches"] for s in batch["scenarios"]
])
```

### Bias and coverage

Bias and standard errors are on the **log-HR scale**. Parentheses contain
Monte Carlo SEs; coverage and its MCSE are expressed in percent/percentage
points. All performance denominators use accepted fits. Each row attempted
1000 fits; the accepted and overall failed counts are shown together.

```{code-cell} ipython3
:tags: [hide-input]
show_table(["n / target", "Model", "Accepted / failed", "Bias (MCSE)", "Coverage % (MCSE pp)"], [
    [f"{r['n']} / {float(r['target_censoring']):.0%}", r["model"], f"{r['accepted']} / {r['failed']}",
     f"{float(r['bias']):.4f} ({float(r['bias_mcse']):.4f})", f"{100*float(r['coverage']):.2f} ({100*float(r['coverage_mcse']):.2f})"]
    for r in full
])
```

All **127 failures** are gradient-only (Weibull 64, Exponential 63); solver,
nonfinite, Hessian and unestimable reason counts are zero. Their maximum
absolute gradients span $1.0008\times10^{-5}$ to $5.0971\times10^{-5}$ despite
successful solver status and positive-definite Hessians. Relative solver
stopping on a summed objective is a plausible cause;
the strict acceptance threshold and rejected classifications are retained.

### Precision and power

Empirical SE is the sample SD of accepted log-HR estimates (`ddof=1`).
Model SE is **RMS**, $\sqrt{\mathrm{mean}(\mathrm{se}^2)}$; its MCSE is not
reported. Power is the fraction with $|\hat\gamma/\mathrm{se}|>1.96$.

```{code-cell} ipython3
:tags: [hide-input]
show_table(["n / target", "Model", "Empirical SE (MCSE)", "RMS model SE", "Power % (MCSE pp)"], [
    [f"{r['n']} / {float(r['target_censoring']):.0%}", r["model"],
     f"{float(r['empirical_se']):.4f} ({float(r['empirical_se_mcse']):.4f})", f"{float(r['model_se']):.4f}",
     f"{100*float(r['power']):.2f} ({100*float(r['power_mcse']):.2f})"] for r in full
])
```

![Sampling distributions of accepted HR estimates, committed full run.](../../examples/weibull_simulation/results/fig2_sampling.svg)

![Coverage with one-MCSE error bars, committed full run.](../../examples/weibull_simulation/results/fig3_coverage.svg)

At $n=1000$, Weibull coverage is **94.34% and 93.08%**, respectively 0.89 and
2.36 observed MCSEs from 95%; both satisfy the three-MCSE check. Exponential
log-HR bias is about 0.069 and 0.035, with coverage 89.61% and 94.47%.
These results illustrate how misspecification and censoring affect inference.
The HR is the study's familiar effect measure; this demo does not establish
that it is the most useful treatment-effect measure for another question.

### CPU timings — committed full run

Each row below times **both censoring levels together**: 2000 fits, not an
individual scenario. First-call time includes compilation if uncached; warm
time is the median of three synchronized subsequent calls. Throughput is
$2000/\text{warm seconds}$ and includes the fit diagnostics/inference.
Generation, calibration, performance summaries and export are outside it.

```{code-cell} ipython3
:tags: [hide-input]
runtime = metadata["runtime"]
print(f"{runtime['cpu_model']}; {runtime['backend']}; Python {runtime['python']}; JAX {runtime['packages']['jax']}; float64")
show_table(["n", "Model", "First call (s)", "Warm (s)", "Fits/s"], [
    [batch["n"], model, f"{timing['first_call_seconds']:.3f}", f"{timing['warm_seconds']:.3f}", f"{timing['fits_per_second']:.1f}"]
    for batch in metadata["batches"] for model, timing in batch["models"].items()
])
```

Thread-setting environment variables were unset; effective worker count and
peak memory were not measured. These measurements describe this machine/run.

## Reproduce and inspect the evidence

From the repository root, prepare the approved optional environment and build:

```bash
uv sync --locked --group docs --group examples
scripts/build_docs.sh
```

The page selects float64 in each build. For a new full run or smoke output:

```bash
uv run --locked --group examples python examples/weibull_simulation/run.py --config smoke --out /tmp/survjax-smoke
uv run --locked --group examples python examples/weibull_simulation/run.py --config full --out /tmp/survjax-full
```

Root seed `20261025` uses `fold_in(scenario)`, then `split(R)`, then U/C keys.
Scenario order is increasing $n$, then censoring target. Smoke and full have
their own grids. Demo generation stays outside the public survjax API.

The approved compact evidence keeps every failed ID/reason and exact accepted
histogram counts in Git; full per-fit diagnostics stay local. Original
invocation paths in metadata are historical. Download
{download}`performance <../../examples/weibull_simulation/results/performance.csv>`,
{download}`failure ledger <../../examples/weibull_simulation/results/failures.csv>`,
{download}`metadata <../../examples/weibull_simulation/results/metadata.json>` and
{download}`hash manifest <../../examples/weibull_simulation/results/evidence.json>`.
