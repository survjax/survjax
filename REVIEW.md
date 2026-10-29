Independent review: round 2/3

# SIM.1 — Shared Weibull generator and censoring calibration

Branch `release-simulation`; actual launch/base commit
`8757c0fda78e34e6e818249adedd2a144a33c33d`. Reviewer checkout already existed.
Initial commit: 300 implementation/test/proposal lines, excluding PLAN/REVIEW
records. Review fix-up: 7 net added lines (307 total), same single concept.
Unfinished PLAN history is preserved verbatim below the new section.

## Round 1 response

Written review: ../review/slices/SIM.1.md, round 1 on 1e3f5fc, fix-first.
- F-1 fixed: add `test_calibration_rejects_unconverged_quadrature` for k=20,
  b=5, target=.5; it requires RuntimeError matching "quadrature". The review
  independently established that this call reaches the convergence guard and
  that removing the guard previously survived the suite. No tolerance change.
- N-1 fixed: document possible exact-zero underflow for extreme parameters;
  no floors/clipping or unsupported robust sampling claim introduced.
- N-2 fixed: c uses the same time units as scale, not multiples of scale.
- N-3 deferred as suggested: only one source-demo test currently adjusts the
  import path; a shared helper would be premature. No consumer files touched.
- Reviewer's isolated dependency probe and proposed future docs command remain
  unrerun by the reviewer; our compatibility evidence is below. A prunable
  temporary mutation-worktree administrative entry is reviewer-owned cleanup;
  no review-directory or Git worktree pruning performed by this implementer.

Fix-up checks (both modes, sequential commands as above):
focused 13 passed in 4.32s / 13 passed in 4.05s, no skips.
Ruff: 114 files already formatted / All checks passed!
mypy: Success: no issues found in 20 source files.
This adds a regression test and docstrings only; initial full-gate evidence
below remains the regression evidence. No implementation behavior changed.
Reviewer round 1's raw `scripts/check.sh --through A10` exited 1: Ruff,
mypy, both own suites and doctests passed; x64 independent selection had
2327 passed / 28 skipped / 5 errors, all five A9 wheel fixtures requiring
SURVJAX_WHEEL. The raw gate stops before x32 spec tests. Reviewer execution
of the wheel wrapper was denied. Our fresh-wheel wrapper passed both full
through-A10 selections (evidence below); no packaging/core code changed.
This is a reviewer environment limitation, not a new implementation failure.

## Reading order

1. PLAN.md's SIM.1 section; use-cases/00-common.md and both 01 files.
2. examples/weibull_simulation/study.py: explicit-key inverse CDF followed by
   independent censoring; host-only pooled numerical calibration.
3. tests/unit/test_simulation_generator.py and tests/impl_tolerances.py.
4. examples/weibull_simulation/DEPENDENCIES.md: proposal only; no dependency
   group or lockfile edits. No optional imports in the generator or tests.

## Invariants and requirement evidence

- Shared convention / 01 DGP: T=b*(-log(U))**(1/k), C=c*Uniform(0,1),
  observed min(T,C), event T<=C. `shape`, `scale`, `censor_limit` broadcast;
  sample_shape prefixes these axes. Positive finite values are preconditions.
  No arbitrary probability/time floors or clipping. JAX's representable U=0
  yields latent T=inf and finite censored observation; no repair is applied.
- 01 seeds: exactly split each supplied replicate key into U then C keys.
  Caller owns root `jax.random.key(20261025)`, `fold_in(root,s)` with full
  scenarios n ascending then censoring target, and `split(scenario_key,R)`.
  No internal/global RNG state; no automatic root or scenario derivation.
  Scenario runner is later work. Test checks the split numerically, scalar
  and multi-axis broadcasts, prefix shape, eager/jit/vmap/jit-of-vmap.
- Calibration: NumPy float64 only, equal weight per broadcast entry. Solve
  mean integral_0^c S_i(t)dt/c=target, not F(c) or censoring of a single event
  draw. 128-node Gauss-Legendre with t=c*x**8, bracket doubling/bisection,
  fraction residual <=1e-12; independently repeat at 64 nodes within 1e-10.
  These numerical stopping thresholds are solver settings, not test tolerance
  changes. Failure to bracket/converge raises, without using core RMST.
- Tests: exponential integral hand derivation at c=2b; heterogeneous 2x3
  broadcast (k=.5/2.7) and both exact simulation arms/targets versus 40-digit
  mpmath quadrature. Invalid calibration inputs fail explicitly.
- Empirical: 100,000 draws per arm, each target; compare survival at 5/10/20
  against exp(-(t/b)**k), log cumulative-hazard ratio at t=10 against log(.7),
  joint exceedances against S_T(10)/2, pooled censoring against its target,
  and observation construction exactly. Central six-MCSE bounds are fixed
  binomial or delta-method bounds, with conservative p(1-p)<=1/4 for pooled
  censoring. No fits, PH implementation or data-dependent tolerance tuning.
- Spec 7.3: check_grads(order=2, modes=(fwd,rev)) for positive k/b with one
  fixed key, latent event-time sum; current values/derivatives are O(1–10),
  central differences retain the existing derivative tolerances and dtype
  steps. No gradient claim for host calibration or discrete event flags.
- Spec 10.3/12.2: no public library changes, optional packages, shared scripts,
  docs/config/navigation, protected paths or another agent's files changed.
  Fitting awaits reviewed PH integration. No full simulation/coverage claim.

## Consumer import contract (freeze if review approves)

From a source checkout with repository root on Python's import path:

```python
from examples.weibull_simulation.study import (
    GeneratedData,
    calibrate_censoring,
    generate_weibull,
)
```

`generate_weibull(key, shape, scale, censor_limit, *, sample_shape=())`
returns `GeneratedData(time, event, event_time, censor_time)`: a NamedTuple
PyTree of equal-shaped arrays, event boolean; JAX precision follows inputs
and process configuration. Explicit float32 inputs retain float32 in x64 mode.
`calibrate_censoring(shape, scale, target) -> float` requires concrete inputs,
uses host float64 even in float32 processes, and is not jit/differentiable.
Both k/b broadcast per row; consumers compute PH scales b0*exp(-gamma*arm/k).
Consumers own their specified key derivation; scalar calls support MNIST's
per-global-index key scheme and batched parameters support posterior draws.
Not packaged/exported in the survjax wheel. Consumer coordination/integration
is through the maintainer; no messages sent to other agents/services.

Equal-arm k=1.38, b0=11.42, HR=.7 calibrated c values (years):
25%: 47.793149053990135; 55%: 19.17253543619806.
These are pilot calibrations, not committed full-run results metadata.
Float64 seeded empirical censoring: .249355/.550485; survival-based HR at
t=10: .6991430905461749/.7003407816242653 (100,000 rows per arm/target).

## Commands and actual results

Locked project: Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, CPU.
Final focused checks, separate processes:
`JAX_ENABLE_X64=1 uv run pytest -q tests/unit/test_simulation_generator.py`:
12 passed in 7.64s.
`JAX_ENABLE_X64=0 uv run pytest -q tests/unit/test_simulation_generator.py`:
12 passed in 4.82s. No skips. Earlier 9/11-test iterations also passed.
`uv run ruff format --check .`: 114 files already formatted.
`uv run ruff check .`: All checks passed!
`uv run mypy src`: Success: no issues found in 20 source files.

Initial focused collection failed because pytest's import path excludes the
repository root; fixed the source-only example path explicitly in this own
test module. Initial Ruff found import order, lambda assignment and unbound
quadrature lambda loop variables; corrected before final checks.

Cumulative gate command: `scripts/check_wheel.sh --through A10`, which builds
one fresh wheel then runs the unchanged `scripts/check.sh --through A10`.
Log: /private/tmp/survjax-sim-gate.log. Exit 0; final output:
`check: all gates passed through A10`.
Own tests: 1059 passed, 7 skipped in 176.64s (float64); 1066 passed, 1 skipped
in 200.21s (float32). Source doctests: 4 passed in each mode (1.87s/2.22s).
Independent through-A10: 2332 passed, 28 skipped in 187.91s (float64);
2322 passed, 38 skipped in 116.18s (float32).
Wheel SHA-256: `8ca5093b02ae2ee23e85bc8feea86374f6ab2114906b5f3896a8b78c80af37f9`.
Existing skips only. The x64 own-suite collection preceded addition of the
new derivative test; final focused 12-test runs above verify it in both modes.
No implementation behavior changed after the gate started.

## Isolated dependency proposal probe

Proposed exact pins: Optimistix 0.1.0, Matplotlib 3.11.2, Equinox 0.13.8,
Optax 0.2.8, NumPyro 0.22.0. Tested together against locked JAX/jaxlib 0.11.2
and NumPy 2.5.3; resolver selected transitive Lineax 0.1.1, 28 packages total.
No extra diagnostics/loader dependency proposed. Approval is required by
AGENTS.md before adding the examples group or changing the lockfile.

Command (temporary probe preserved below; initially run at its transient
source-tree path, moved to /private/tmp after Python loaded it, then rerun
successfully using the exact command below):

```sh
JAX_ENABLE_X64=1 MPLCONFIGDIR=/private/tmp/survjax-sim-mpl uv run --isolated --no-project --python 3.12 --with jax==0.11.2 --with numpy==2.5.3 --with optimistix==0.1.0 --with matplotlib==3.11.2 --with equinox==0.13.8 --with optax==0.2.8 --with numpyro==0.22.0 python /private/tmp/survjax-sim-probe.py
```

Output lists the nine versions above, then
`PASS: compiled vmapped BFGS, Equinox/Optax update, NUTS, SVG` (exit 0).
First Matplotlib import warned its default home cache was unwritable and used
a temporary cache; SVG export succeeded. Proposal validates compatibility,
not PH integration, real fit acceptance, neural training or Bayesian diagnostics.
No full demos, docs build, R generation, performance benchmark or remote work.

Probe source, independent of survjax/PH and excluded from ordinary tests:

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
