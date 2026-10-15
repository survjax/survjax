# A5.5 review fix-up — full boundary Jacobians and tracing coverage

Follow-up to `2d0c22d` on the same `A5.5` branch, authorized by the maintainer's
review. Changes are tests and grid docstrings; executable library code is unchanged.

## Reading order

1. `tests/unit/test_survival_grid.py`, `test_mixed_boundary_grid_jacobian`:
   both AD modes differentiate all three rows at t=[0, 1, +inf].
2. The misuse tests: JAX-compatible arguments are passed dynamically;
   string/object values are bound inside zero-argument jitted calls.
3. `src/survjax/distributions/_grid_methods.py`: seven docstrings now use
   prose matching the direct-method style. Calls and signatures are unchanged.
4. `PLAN.md` records the requested review fix-up and validation scope.

## Invariants and spec-to-test mapping

- Spec 4.5/7.2: the full forward and reverse Jacobians have shape (3,), and
  every required entry must be finite and equal its analytical derivative.
  Forward mode now checks the boundary derivatives, while reverse mode also
  checks contamination of the interior entry. At rate=1, the boundary rows
  include zero derivatives for constant survival/CDF limits and unit
  derivatives for density at zero and constant log hazard/hazard.
- None in the expectation table marks nonfinite output values: log survival,
  log density and cumulative hazard at infinity, and log CDF at zero. Their
  boundary derivatives are outside spec 4.5's guarantee; the test deliberately
  asserts only the finite-valued boundary rows and the interior row.
- Spec 4.5/5: scalar/rank, boolean, complex and None misuse still reach grid
  code through dynamic JAX-compatible arguments. String/object tests use
  `jax.jit(partial(method, bad))()` so JAX cannot reject them as arguments
  before tracing. Matching the grid conversion error proves the adapter ran.
- The docstrings state the input vector, time-first output shape, component
  exclusion, input order/repeats, direct-method policy and TypeError conditions
  in prose. No implementation, tolerance, spec or independent-test change.

## Checks actually run

    JAX_ENABLE_X64=1 .venv/bin/pytest tests/unit/test_survival_grid.py -q
    JAX_ENABLE_X64=0 .venv/bin/pytest tests/unit/test_survival_grid.py -q

Both exited 0: **42 passed** per dtype mode, with no skips or warnings.

    .venv/bin/ruff format --check .
    .venv/bin/ruff check .
    .venv/bin/mypy src
    git diff --check

All passed: 46 files formatted, Ruff clean, mypy clean for 12 source files,
and no whitespace errors. The full A5 gate was not rerun for this test/docstring
follow-up. Its parent commit `2d0c22d` records the passing through-A5 gate in
both dtype modes.

## Remaining limitation and stop

D-022 remains the separate numerical follow-up. No deviation is resolved.
Commit this review fix-up on A5.5 and stop before the next step.
