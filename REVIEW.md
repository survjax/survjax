# A5.4 review fix-up — stable built-in log CDF tails

Implemented on branch `A5.4` after the maintainer resumed this review fix-up.
The methods stay private except for the existing `log_cdf` API; exports do not
change. Revision 24 of the spec is current, and `.spec-seen` matches it.

## Reading order

1. `src/survjax/distributions/weibull.py`: `_log_cumulative_hazard` evaluates
   `shape * log(t / scale)` with safe operands and analytic boundary limits;
   `log_cdf` applies the dtype-specific lower-tail switch.
2. `src/survjax/distributions/exponential.py`: the matching helper evaluates
   `log(rate) + log(t)` and uses the same `log_cdf` branches.
3. `tests/unit/test_weibull.py` and `tests/unit/test_exponential.py`: hand-derived
   underflow-tail values, continuity around the threshold and finite gradients
   with `jax.debug_nans` enabled.
4. `tests/spec/test_a5_closed_forms.py`, `test_a5_boundaries.py`,
   `test_a5_construction.py`, `test_a5_derivatives.py` and `test_a5_from_log.py`:
   read-only acceptance tests synchronized to revision 24.
5. `PLAN.md` records this fix-up; `DEVIATIONS.md` records D-021 and the
   revision-24 follow-up to D-020.

## Invariants and test mapping

- For small positive H, each family computes `log H` without forming H. It uses
  `log H` below `log(finfo(dtype).eps)`; the omitted `-H/2 + O(H^2)` is below
  dtype epsilon there (spec 4.2/4.3, 7.1). The Weibull and Exponential
  hand-derived tail tests include float32 cases where their direct H calculation
  is zero.
- Before evaluating the standard expression, `jnp.where` substitutes a finite
  `log H` on the tiny branch. The upper branch also substitutes before `exp`
  when `log H` reaches `log(finfo(dtype).max)`, so infinite limits and
  overflowing exponentiation cannot poison the selected result or gradients
  (spec 7.2). Both switch tests compare values with `log H` within epsilon and
  check finite parameter gradients under `jax.debug_nans` on both sides.
- At zero, valid positive parameters give `log F=-inf`. At positive infinity,
  valid positive parameters give `log F=0` with zero parameter gradients.
  Existing boundary acceptance tests cover both; NaN family values retain
  precedence through the family helpers and registered time policy (spec 4.5).
- D-021 remains open: this extends section 4.3's HazardFamily switch to the two
  built-ins before that base class exists. No spec or independent test was
  changed. D-020's erroneous density expectation is corrected in the synced
  revision-24 acceptance test; the entry remains open for maintainer closure.

## Checks actually run

Focused own distribution tests:

    JAX_ENABLE_X64=1 .venv/bin/pytest tests/unit/test_weibull.py tests/unit/test_exponential.py -q
    JAX_ENABLE_X64=0 .venv/bin/pytest tests/unit/test_weibull.py tests/unit/test_exponential.py -q

x64: **36 passed, 1 skipped**. x32: **37 passed**.

The repository gate was run with its cache under `/private/tmp`:

    UV_CACHE_DIR=/private/tmp/uv-cache scripts/check.sh --through A5

The guard, Ruff format check (44 files), Ruff lint, mypy (11 source files),
own test suites (x64: **373 passed, 7 skipped**; x32: **379 passed, 1 skipped**)
and source doctests (1 passed per mode) passed. The x64 independent spec run
reached **1006 passed, 26 skipped, 110 failed**. Every failure is in the newly
synced `tests/spec/test_a5_survival_grids.py`: revision 24 requires grid methods
from the later A5.5 grids step, which is not implemented on A5.4. The gate stops
after that x64 failure, so its x32 full spec run did not execute.

To check the A5 acceptance modules relevant before grids, I ran:

    JAX_ENABLE_X64=1 SPEC_THROUGH=A5 .venv/bin/pytest tests/spec/test_a5_from_log.py tests/spec/test_a5_boundaries.py tests/spec/test_a5_construction.py tests/spec/test_a5_derivatives.py tests/spec/test_a5_closed_forms.py -q
    JAX_ENABLE_X64=0 SPEC_THROUGH=A5 .venv/bin/pytest tests/spec/test_a5_from_log.py tests/spec/test_a5_boundaries.py tests/spec/test_a5_construction.py tests/spec/test_a5_derivatives.py tests/spec/test_a5_closed_forms.py -q

x64: **322 passed**. x32: **312 passed, 10 skipped**. These include the
corrected Weibull density expectation from D-020. `git diff --check` passed.

## Uncertainty and stop

The implementation tests and non-grid A5 acceptance modules pass in both dtype
modes. The full gate remains red until A5.5 implements the revision-24 grid
methods; this fix-up did not change grid scope. D-021 and D-020 stay open for
the maintainer. Commit this fix-up on `A5.4` and stop before the grids step.
