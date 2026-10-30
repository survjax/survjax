Independent review: round 1/3

# SIM.3 — Complete the approved diagnostics environment

Branch release-simulation, clean start ce68f76ffc4ee0fdcda1757232f88ee57ff13c3d.
SIM.2 round 1 returned merge. Maintainer previously approved ArviZ .23.0 and
now requests the next step. No new API, demo/source behavior or ordinary tests.
Existing PLAN history remains. Under 200 changed config/lock/proposal lines.

## Reading order

1. PLAN.md SIM.3; spec 10.3 and Bayesian details Sampling/Diagnostics.
2. pyproject.toml and uv.lock: the final approved examples pin and its six
   packages, with all 165 prior package versions unchanged.
3. examples/weibull_simulation/DEPENDENCIES.md: completed group and actual
   environment evidence; the optional diagnostics probe below.

## Invariants and verification mapping

- Spec 10.3 / use-case 00: ArviZ belongs only to the examples dependency group.
  Core runtime dependencies remain jax and numpy. No protected build scripts,
  docs config/navigation, exports or other workstream files touched.
- Bayesian details require NumPyro split R-hat/ESS and ArviZ tail ESS. The
  compatibility probe takes four independently seeded Normal chains with
  shape (4,1000), converts to InferenceData and checks named chain/draw axes.
  Both R-hat values <1.01 and all ESS values >400, using the use-case gates.
  It then shifts chain means by 0,2,4,6 and requires both R-hat implementations
  to notice nonmatching chains. This catches mistakenly pooled chains.
  It is not a posterior-convergence check or an implementation of the demo.
- No library code, shared generator/import contract, fitting, numerical
  tolerances or tests changed. Existing independent install checks establish
  core isolation; inspect the fresh-wheel hash below too.
- All 165 existing locked package versions are identical by parsed name/version
  comparison. Only ArviZ .23.0, h5netcdf 1.8.2, pandas 3.1.0, setuptools 84.2.1,
  xarray 2026.9.0, xarray-einstats .11.0 are added. Original proposal approval
  persists; no renewed dependency approval requested.
- The exact pinned ArviZ package source confirms a daily import warning stamp
  under Path.home()/arviz_data. Probe ran with required sandbox escalation;
  HOME and package code were not modified. FutureWarning was emitted normally.

## Actual commands/results

`uv lock`: resolved 171 packages, added the six packages listed above.
`uv lock --check`: exit 0, resolved 171 packages.
`uv sync --locked --group examples`: exit 0, installed six new packages and
rebuilt the local editable package. No earlier package version changed.

Optional-environment probe (source preserved below):
`uv run --locked --group examples env JAX_ENABLE_X64=1 MPLCONFIGDIR=/private/tmp/survjax-sim-mpl python /private/tmp/survjax-sim3-diagnostics.py`
Exit 0, devices [CpuDevice(id=0)]. Python 3.12.9, JAX/jaxlib .11.2,
NumPy 2.5.3; approved Optimistix .1.0, Matplotlib 3.11.2, Equinox .13.8,
Optax .2.8, NumPyro .22.0, ArviZ .23.0. Also prints the six new package
versions above. NumPyro split R-hat 1.0000270082954177, ESS 3976.5802682493636;
ArviZ R-hat 1.0001031616427534, bulk ESS 3979.3278033665047,
tail ESS 3931.288721722709. Both shifted-chain checks pass.
Final output: PASS: chain/draw axes, iid diagnostic gates and shifted-chain detection.

Existing focused generator suite (unchanged), separate processes:
`JAX_ENABLE_X64=1 uv run pytest -q tests/unit/test_simulation_generator.py`:
17 passed in 3.60s.
`JAX_ENABLE_X64=0 uv run pytest -q tests/unit/test_simulation_generator.py`:
17 passed in 3.18s, no skips.
`uv run ruff format --check .`: 114 files already formatted.
`uv run ruff check .`: All checks passed!
`uv run mypy src`: Success: no issues found in 20 source files.

`scripts/check_wheel.sh --through A10`: exit 0;
final `check: all gates passed through A10`.
Log: /private/tmp/survjax-sim3-gate.log.
Own tests: 1065 passed / 7 skipped in 63.32s (x64); 1071 passed / 1 skipped
in 63.34s (x32). Doctests: 4 passed per mode (1.13s/1.03s).
Independent through-A10: 2333 passed / 27 skipped in 72.13s (x64),
2324 passed / 36 skipped in 65.11s (x32). The one additional x32 pass versus
SIM.2 is the already-authored optional pandas float64 conversion test;
pandas now arrives transitively with ArviZ. No spec-test changes.
Wheel SHA-256: `8ca5093b02ae2ee23e85bc8feea86374f6ab2114906b5f3896a8b78c80af37f9` (identical to SIM.2).
Wheel METADATA runtime requirements: ['jax>=0.11.2', 'numpy>=2.5.3'].
`git diff --check`: exit 0, no output.

## Limits and remaining integration

No full Bayesian sampling or diagnostic acceptance claim, simulation fit,
MNIST training, docs execution, PH/RMST integration, or publication.
The shared examples environment is complete after this step. Next simulation
fitting needs maintainer integration of the reviewed PH prerequisite into this
worktree; no merge/rebase/cherry-pick is performed here. Prior reviewer-owned
mutation worktrees/administrative entries are untouched.

## Reproducible temporary diagnostics probe

```python
"""Seeded optional-environment compatibility probe, not posterior validation."""

import importlib.metadata as metadata

import arviz as az
import jax
import numpy as np
from numpyro.diagnostics import summary

for name in (
    "jax",
    "jaxlib",
    "numpy",
    "optimistix",
    "matplotlib",
    "equinox",
    "optax",
    "numpyro",
    "arviz",
    "h5netcdf",
    "pandas",
    "setuptools",
    "xarray",
    "xarray-einstats",
):
    print(name, metadata.version(name))
print("devices", jax.devices())
# Known equal-chain law and fixed seed; thresholds are from Bayesian details.
samples = {"x": np.random.default_rng(20261025).normal(size=(4, 1000))}
posterior = az.from_dict(posterior=samples)
assert posterior.posterior["x"].dims == ("chain", "draw")
result = summary(samples, group_by_chain=True)["x"]
bulk = float(az.ess(posterior, method="bulk")["x"])
tail = float(az.ess(posterior, method="tail")["x"])
rhat = float(az.rhat(posterior)["x"])
assert np.isfinite([result["r_hat"], result["n_eff"], bulk, tail, rhat]).all()
assert result["r_hat"] < 1.01 and rhat < 1.01
assert result["n_eff"] > 400 and bulk > 400 and tail > 400
# Different chain means must be noticed; accidentally pooled draws would hide it.
shifted = {"x": samples["x"] + 2 * np.arange(4)[:, None]}
assert summary(shifted, group_by_chain=True)["x"]["r_hat"] > 1.01
assert float(az.rhat(shifted)["x"]) > 1.01
print("NumPyro split Rhat", result["r_hat"], "ESS", result["n_eff"])
print("ArviZ Rhat", rhat, "bulk ESS", bulk, "tail ESS", tail)
print("PASS: chain/draw axes, iid diagnostic gates and shifted-chain detection")
```
