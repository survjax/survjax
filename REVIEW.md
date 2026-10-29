Independent review: round 1/3

# SIM.2 — Maintainer questions and approved examples environment

Branch release-simulation; start 3556d12c1abca553f77607c1e1739df2bd1911f0.
SIM.1 round 2 returned merge. Initial dirty file was study.py containing three
maintainer questions; each is addressed here. Existing PLAN history preserved.
About 300 source/test/config/lock/proposal changed lines (303 before records).

## Reading order

1. PLAN.md SIM.2 and the maintainer questions recorded below.
2. examples/weibull_simulation/study.py: optional PH input and calibration
   setup order; host calibration remains outside compiled replicate work.
3. tests/unit/test_simulation_generator.py: mpmath PH calibration, transformed
   generator identity, unusable effective scales and analytical PH derivatives.
4. pyproject.toml / uv.lock / examples/weibull_simulation/DEPENDENCIES.md:
   approved exact pins, unchanged existing versions, staged ArviZ follow-up.

## Questions and decisions

- "rules is used but not defined until below": Python resolves the closure
  when fraction is called, after initialization. Valid but unnecessarily hard
  to read; now initialize rules before defining the closure. A JAX function
  can also close over constants; this ordering was not a JAX limitation.
- "a lot of np code and python loops; will this ever be jitted?": no, this is
  concrete once-per-cohort/target float64 setup. Its result c is reused by
  compiled generation/fits. Calibrating inside each replicate would also
  change the intended fixed-c experiment. Docstring makes this split explicit.
- "explicit PH parameterization would be handy": expose optional keyword
  log_hazard_ratio=0.0 on both generation and calibration; scale remains b0.
  H(t)=exp(eta)*(t/b0)**k, equivalent to Weibull scale b0*exp(-eta/k).
  Asked baseline/log-HR versus Weibull-PH rate m asynchronously; no answer
  arrived while independent environment work ran. Proceeded with the stated
  baseline/log-HR default, matching the 01-details gamma*arm formulation and
  spec 4.4 wrapper input. No extra library wrapper or fitting implementation.

## Invariants and requirement/test mapping

- Shared 00 / 01 DGP: no-keyword calls preserve the old shape/scale generator.
  The sole key still splits U then C; output remains the same GeneratedData
  NamedTuple, sample_shape prefix and boolean event. eta participates in all
  parameter broadcasting. Positive eta means higher hazard/shorter times.
- Spec 4.4/4.6: PH transformation is a scale conversion, not a new distribution.
  Multi-axis k/b0/eta test compares ordinary/eager/jit/jit-of-vmap outputs to
  the verified equivalent Weibull law, with the identical supplied key.
- Host calibration is still the equal-row mean of integral S/c, now after
  optional PH scale conversion. Existing 40-digit mpmath checks for both
  simulation targets now use baseline b0 and arm-wise eta. Existing plain
  calibration/empirical checks remain. Invalid eta or unrepresentable positive
  effective scales raise ValueError. No clipping, floors or epsilons.
- Spec 7.3: original k/b check_grads(order=2,fwd/rev) retained. New PH test
  derives all three first partials and the full 3x3 Hessian independently
  from T=b*exp(a/k), a=log(-log(U))-eta. Checks compiled jacfwd/jacrev and
  hessian against that algebra, including all cross-partials, both dtypes.
- Spec 10.3: user explicitly approved the version proposal. The examples
  group pins Optimistix .1.0, Matplotlib 3.11.2, Equinox .13.8, Optax .2.8,
  NumPyro .22.0. Core dependencies remain jax/numpy only. All 155 prior lock
  package versions are identical; 10 new transitive/direct packages added.
  ArviZ .23.0 remains approved and isolated-probe tested but its six-package
  transitive addition is staged separately: full group preview had 295 lock
  lines alone; this stage's lock has 160 changed lines. No approval re-request.
- No public library API, protected scripts, docs/config or another agent's
  files changed. Existing demo entry points/imports remain valid. Consumer
  coordination goes through the maintainer: optional eta is the only addition.

## Updated shared import contract

From a source checkout with its root on sys.path, import from
examples.weibull_simulation.study:

- generate_weibull(key, shape, scale, censor_limit, *, sample_shape=(),
  log_hazard_ratio=0.0) -> GeneratedData(time,event,event_time,censor_time).
- calibrate_censoring(shape, scale, target, *, log_hazard_ratio=0.0) -> float.

No-keyword calls and explicit-key scheme are unchanged. For a trial, pass
lp=gamma*arm with baseline scale b0. Calibration remains host-only float64;
generation supports JIT/vmap and follows input/process precision.

## Actual checks

`uv lock --check`: Resolved 165 packages, exit 0.
`uv sync --locked --group examples`: exit 0, installed approved packages.
Compared parsed old/new lock package name/version mappings: all 155 existing
versions unchanged, 10 additions. Initial preview mistakenly used a placeholder
temporary path once (exit 2, no such directory); reran with the real temporary
path, resolved successfully. No project changes resulted from that typo.

Optional compatibility command:
`JAX_ENABLE_X64=1 MPLCONFIGDIR=/private/tmp/survjax-sim-mpl uv run --locked --group examples python /private/tmp/survjax-sim-probe.py`
Exit 0; prints JAX/jaxlib .11.2, NumPy 2.5.3, Optimistix .1.0,
Matplotlib 3.11.2, Equinox .13.8, Optax .2.8, NumPyro .22.0, Lineax .1.1;
then PASS: compiled vmapped BFGS, Equinox/Optax update, NUTS, SVG.
The temporary probe source is preserved below for reproducibility.
No PH fitting, neural training or Bayesian convergence claim follows from it.

Focused final commands in separate processes:
`JAX_ENABLE_X64=1 uv run pytest -q tests/unit/test_simulation_generator.py`:
17 passed in 10.17s.
`JAX_ENABLE_X64=0 uv run pytest -q tests/unit/test_simulation_generator.py`:
17 passed in 10.56s. No skips.
`uv run ruff format --check .`: 114 files already formatted.
`uv run ruff check .`: All checks passed!
`uv run mypy src`: Success: no issues found in 20 source files.
`git diff --check`: exit 0, no output.

Initial attempt to expand check_grads to a joint (k,b,eta) projection passed
float64 but failed the float32 second-order VJP projection: -0.00390816 versus
-0.00765991 (difference .00375175, tolerance .001). Investigated with the
full closed-form gradient/Hessian above: both AD modes match within the
stricter ordinary dtype tolerances. Restored the original k/b finite-difference
check and used the analytical PH test for the new directions. No tolerance
or finite-difference step was loosened. Final suites above pass.

`scripts/check_wheel.sh --through A10`: exit 0, final output
`check: all gates passed through A10`.
Log: /private/tmp/survjax-sim2-gate.log.
Own suites: 1064 passed, 7 skipped in 177.60s (x64); 1071 passed, 1 skipped
in 176.34s (x32). Source doctests: 4 passed per mode (2.11s/1.95s).
Independent: 2333 passed, 27 skipped in 118.46s (x64); 2323 passed,
37 skipped in 88.67s (x32). NumPyro availability enables one formerly skipped
constraint test per mode. The full gate began before the mixed derivative
test was replaced; final focused checks above verify all 17 final tests in
both modes. No implementation behavior changed after the gate started.
Wheel SHA-256: `8ca5093b02ae2ee23e85bc8feea86374f6ab2114906b5f3896a8b78c80af37f9`.


## Limits

Calibration deliberately cannot be jitted or differentiated. PH fitting still
waits for reviewed core integration. ArviZ's already-approved group addition
is a separate dependency step; no full docs/demo execution in this step.
No merge, push, publication, review-directory edits or reviewer cleanup.

## Temporary compatibility probe source

```python
"""Explicit optional-environment compatibility probe; not a regular test."""

import importlib.metadata as metadata
import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpyro
import numpyro.distributions as dist
import optax
import optimistix as optx

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

for name in (
    "jax",
    "jaxlib",
    "numpy",
    "optimistix",
    "matplotlib",
    "equinox",
    "optax",
    "numpyro",
    "lineax",
):
    print(name, metadata.version(name))
solver = optx.BFGS(rtol=1e-8, atol=1e-8)


def fit(start):
    return optx.minimise(
        lambda x, _: (x - 2) ** 2, solver, start, max_steps=1000, throw=False
    )


solutions = jax.jit(jax.vmap(fit))(jnp.array([-1.0, 0.0, 4.0]))
assert bool(jnp.all(solutions.result == optx.RESULTS.successful))
assert bool(jnp.all(jnp.abs(solutions.value - 2) < 1e-8))
linear = eqx.nn.Linear(2, 1, key=jax.random.key(0))
params = eqx.filter(linear, eqx.is_inexact_array)
optimizer = optax.adam(1e-3)
grads = eqx.filter_grad(lambda model: jnp.sum(model(jnp.ones(2)) ** 2))(linear)
updates, _ = optimizer.update(grads, optimizer.init(params), params)
assert bool(jnp.all(jnp.isfinite(eqx.apply_updates(linear, updates)(jnp.ones(2)))))


def model():
    numpyro.sample("x", dist.Normal(0.0, 1.0))


mcmc = numpyro.infer.MCMC(
    numpyro.infer.NUTS(model), num_warmup=16, num_samples=32, progress_bar=False
)
mcmc.run(jax.random.key(1))
assert bool(jnp.all(jnp.isfinite(mcmc.get_samples()["x"])))
with tempfile.TemporaryDirectory() as directory:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    fig.savefig(Path(directory) / "probe.svg")
    plt.close(fig)
print("PASS: compiled vmapped BFGS, Equinox/Optax update, NUTS, SVG")
```
