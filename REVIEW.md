# PH.2b — Declared-invalid row isolation and custom-baseline composition

Independent review: round 1/3

Base: 813ee36d5ec51649386b633f0c131d1e39b220b7 on release-ph, clean before planning.
Authorized by “ok on to next bit”. Same PH sub-slice; existing PLAN history
is preserved. No production changes. One new test module, 259 lines/55 cases.

## Read first

1. PLAN.md PH.2b, plans/release-agents/ph.md, spec 4.2/4.4/4.7,
   6.1, 7.1–7.3 and 12.1–12.2 (current revision 38, read-only).
2. tests/unit/test_proportional_hazards_isolation.py: templates and custom
   baseline, distribution construction, independent formulas, then four tests.
3. Existing distributions/wrappers.py, distributions/_parameter_issues.py,
   _likelihood_parameters.py and likelihood.py for the composition exercised.
   They are unchanged by this step.

## Invariants and expected values

The likelihood applies stop_gradient to every own/nested parameter leaf at
entries declared invalid, after broadcasting leaves, before any family code.
It forces those contributions to NaN. Isolation is asserted only for the
selected valid contribution outputs, never for invalid outputs' derivatives.
The cases share healthy eta or healthy baseline parameters across rows, so
reverse-mode contamination cannot hide behind independent parameter rows.
No upstream caller overflow, infinite valid contribution, support boundary or
time derivative is asserted. The invalid middle row is right-censored; the
valid rows are one exact and one right-censored observation.

Each parameter violation class gets a separate vmap input: rate/shape/scale
0, -1, NaN, +inf, -inf; eta NaN, +inf, -inf over both built-ins. Finite zero
and negative eta remain valid. The custom baseline's positive lower/upper
fields have sqrt(upper-lower) rate; equal and reversed endpoints violate only
its named condition, without a field violation. Generic baseline log_prob,
log_hazard and log_cdf defaults compose with PH here; no custom summaries.

The valid-row oracle gathers rows 0/2 before *any* parameter arithmetic.
It uses ell=event*log(h)-H. Exponential has log(h)=log(r)+eta,
H=exp(log(r)+eta)*t. Weibull has H=exp(eta+k*log(t/b)) and
log(h)=log(k)-log(b)+(k-1)*log(t/b)+eta. The custom baseline has
log(r)=0.5*log(upper-lower). Neither likelihood nor wrapper methods produce
these expected contributions. All comparisons use existing central ATOL/RTOL,
equal_nan=False, with exact masks/order checks. The derivative entries are
O(1–10); comparisons involve exact AD and analytic identities, so ordinary
rounding tolerances suffice; there is no finite difference across bad inputs.

## Requirement-to-test evidence

- test_invalid_rows_have_nan_values_and_prefixed_diagnostics (6 cases):
  spec 4.7/7.1 prefixed fields/conditions, declaration order, union/polarity,
  scalar ok (mapped per example), NaN only at the declared-invalid row,
  eager/jit/vmap and independent analytical values at valid rows.
- test_valid_contribution_derivatives_survive_invalid_neighbors (36 cases):
  spec 7.1/7.3 valid outputs' first forward/reverse Jacobians and all four
  forward/reverse second-order compositions, with vmap and jit(vmap), across
  every declared violation above. Shared parameters are included in the
  full Jacobian/Hessian tensors, as are the inactive bad coordinates.
- test_custom_baseline_grid_values_and_derivatives_match_scaled_rate (7 cases):
  spec 4.2/4.4/7.3 all seven time-first grids and first forward/reverse parameter
  derivatives over two valid parameter vectors, eager/jit/vmap, against the
  analytical equivalent Exponential(sqrt(upper-lower)*exp(eta)). Positive
  interior grid times avoid generic hazard-at-infinity contract differences.
- test_runtime_parameter_issues_and_raw_observations_reuse_one_trace (6 cases):
  spec 4.7/7.3 valid/invalid/validity-class changes plus time/event changes
  with fixed parameter shapes/dtypes/weak types and structure, one trace;
  values and masks are checked on each call, not just compile counts.

## Validation

Python 3.12.9, JAX 0.11.2, NumPy 2.5.3.

- `uv run ruff format --check .`: 122 files already formatted.
- `uv run ruff check .`: All checks passed.
- `uv run mypy src`: Success, no issues found in 21 source files.
- `git diff --check`: clean.
- `JAX_ENABLE_X64=1 uv run pytest -q tests/unit/test_proportional_hazards_isolation.py`:
  initial version 54 passed, 1 failed in 267.50s (assertion issue below).
- The targeted x64 reproduction (`-k 'prefixed and eta-weibull'`) failed
  identically: 1 failed, 54 deselected in 6.73s. After correction, the complete
  affected value/diagnostics test family (`-k prefixed`) passed: 6 passed,
  49 deselected in 19.76s. The final full gate below exercises all 55 final
  cases in x64 again, alongside all existing PH/regression tests.
- `JAX_ENABLE_X64=0 uv run pytest -q tests/unit/test_proportional_hazards_isolation.py`:
  final version 55 passed in 274.00s (concurrent with initial x64/final gate).
- `scripts/check_wheel.sh --through A10`: **exit 0, all gates passed through A10**. The
  unchanged protected-path guard, Ruff and mypy passed. Own tests passed:
  **1159 passed, 8 skipped** x64 in 169.20s; **1164 passed, 3 skipped** float32
  in 122.27s. Source doctests: **4 passed** each mode (1.49s / 1.40s).
  Independent acceptance: **2332 passed, 28 skipped** x64 in 95.54s;
  **2322 passed, 38 skipped** float32 in 58.65s.
- Fresh wheel: `/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.8SIxw4/survjax-0.0.1.dev0-py3-none-any.whl`;
  SHA256 `3af2391ef81dacf1dfadeb997fbdc9d22d53a39c3091657d0e910eeb0451ee3c`.
  Byte-identical to PH.2a, consistent with a test-only step.
  Full gate log: `/private/tmp/ph2b-wheel-gate.log`.

Initial focused x64 run exposed an overstrict bit-equality assertion comparing
an eager and compiled valid value (-0.9527152835009017 vs -0.9527152835009018).
Corrected it to compare valid entries against the analytical oracle using
unchanged central tolerances and equal_nan=False, with a separate NaN check
for the invalid entry. Exact diagnostics assertions remain exact. No tolerance
or production expression was changed.

## Scope and remaining work

No exports, docs-site, dependency, protected-path, RMST-owned or tolerance
changes. Tests import PH from its defining module until public integration.
No R fixture generation or PDF build is needed for this test-only step.
No spec ambiguity/deviation found. This is not closure of the PH release gate:
extended-tail/range evidence, RMST ownership handoff/closure, public exports
with the API-reference agreement, executable topic and final integration
remain separate. No unfinished PH history was archived. No merge/push/publish.

---

# Incoming main review record — SIM.3

Historical evidence from the incoming main parent; its validation results
apply to that step, not to the combined merge tree.

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
