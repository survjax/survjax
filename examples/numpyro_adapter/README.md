# NumPyro adapter discussion prototype — issue #1

Start with [the model](run.py), then [the adapter](adapter.py).
This is an experiment for API discussion, not an exported survjax API.

```bash
JAX_ENABLE_X64=1 uv run --with numpyro==0.22.0 python -m examples.numpyro_adapter.run
JAX_ENABLE_X64=1 uv run --with numpyro==0.22.0 --group test pytest -n 6 tests/unit/test_numpyro_adapter.py
```

Repeat with `JAX_ENABLE_X64=0` for float32. The tiny synthetic example has a
Gamma(4, rate=7) posterior; it prints the analytical mean and NUTS diagnostics.
Monte Carlo estimates are illustrative, not an acceptance threshold.

`SurvivalLikelihood(survival)` scores `SurvivalData` through `log_likelihood`
and event-time arrays through `log_prob`. Censoring stays with the observation.
Future `sample()` would draw underlying event times; currently it raises
`NotImplementedError`. Simulating censored records is a separate model.

The tests cover exact/right observations, factor equivalence, pointwise
extraction, explicit indexing, JIT/vmap, analytical parameter derivatives,
plate/mask ordering, nested masks, broadcasting, and invalid excluded rows.
SVI, generic distribution wrappers and structured `obs_mask` imputation have
not been demonstrated. The adapter inherits the core's available modes.

The core already guards likelihood evaluation numerically. Record replacement
also prevents excluded invalid observations from emitting support-validation
warnings. The tests check that behavior and finite masked derivatives; they do
not establish that replacement is necessary for gradient safety in these cases.

The example and tests explicitly enter `prototype_compatibility()`: a temporary
`SurvivalData.ndim` property, restored on exit. Do not use this shim concurrently
with other work. A production property requires coordinated spec/test changes.
NumPyro remains optional; the normal test environment may skip these tests.
