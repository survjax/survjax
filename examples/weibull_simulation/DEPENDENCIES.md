# Approved examples environment (SIM.2)

The maintainer approved these exact pins; they now form the `examples` group:

| Package | Tested pin | Consumer |
| --- | --- | --- |
| Optimistix | 0.1.0 | Simulation BFGS |
| Matplotlib | 3.11.2 | All demo figures; already in docs group |
| Equinox | 0.13.8 | MNIST model; Optimistix also needs it |
| Optax | 0.2.8 | MNIST training |
| NumPyro | 0.22.0 | Bayesian inference and built-in diagnostics |

Tested on CPU: Python 3.12.9, locked JAX/jaxlib 0.11.2/NumPy 2.5.3; transitive Lineax 0.1.1.
Compiled vmapped BFGS, Equinox/Optax update, short NUTS and Agg SVG export pass.
REVIEW.md records the isolated command/limits; full demos/other versions remain untested.
ArviZ 0.23.0 is also approved and isolated-probe tested for Bayesian tail ESS;
its lockfile addition is a separate step to keep this change reviewable.
No loader dependency is required for the standard-library IDX reader.

Install with `uv sync --locked --group examples`; use `uv sync --locked --group docs --group examples` for future demo docs builds.
Then in separate JAX_ENABLE_X64=0/1 processes: `uv run --locked --group docs --group examples sphinx-build -b html -E -n -W --keep-going docs docs/_build/html-{mode}`.
The maintainer serializes build-script changes; fitting waits for reviewed PH.
