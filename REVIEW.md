# A5.5 — Seven time-first grid methods

Implemented on the maintainer-selected `A5.5` branch from `861ac71`, following
approval on 2026-10-15. All seven grid twins are inherited by built-in and
registered custom families. The full through-A5 gate now passes in both modes.

## Reading order

1. `src/survjax/distributions/_grid_methods.py`: `_reshape_grid` converts and
   checks rank, then appends one singleton axis per current batch dimension.
   The seven typed methods call the corresponding public direct method.
2. `src/survjax/distributions/base.py`: import and seven bindings only;
   registration still wraps only the direct methods.
3. `tests/unit/test_survival_grid.py`: independent Exponential formula table,
   shapes/empty dimensions, component-family dispatch, argument rules,
   transformations/trace reuse, analytical derivatives and Weibull boundaries.
4. `tests/impl_tolerances.py`: scale justification for the existing derivative
   tolerances; numerical tolerance values are unchanged.
5. `PLAN.md`: approved scope, outcome, deferred docs and the D-022 follow-up.

## Invariants and spec-to-test mapping

- Spec 5/8.1: `(K,)` becomes `(K,) + (1,) * len(batch_shape)`; outputs have
  `(K, *batch_shape)`. Shape comes from current leaves under `vmap`, excluding
  components. K=N still forms every time/batch pair. Own value and shape tests
  cover ordered/repeated rows, scalar/several batch axes, empty grids/batches
  and component reduction owned by a custom family.
- Spec 4.2/4.5: adapters call public registered direct methods. Dtype/value
  policy, NaN precedence, analytic limits and overrides are inherited without
  another wrapper or numerical formula. Component-family call evidence checks
  that an analytic density override is reached. Misuse tests cover rank and
  nonreal types in eager/jit calls; dtype tests cover integers, explicit float32
  and silent transient float64 conversion in x32.
- Spec 7.2/7.3/12.1: every grid has eager/jit/vmap algebra checks and one trace
  across changed values of fixed shape/dtype. Analytical first derivatives and
  Hessians run through `jit(vmap(...))`; order-two forward/reverse `check_grads`
  runs in both dtype modes. All expected derivatives come from the stated
  Exponential formulas. A separate mixed zero/interior/infinity test asserts
  the interior derivative in both AD modes for every grid.
- Spec 4.5/11.1: Weibull k=1 with group scales
  `10 * exp([0, -0.5, -1])` gives analytical survival, density and hazard from
  zero through infinity. NaN times/parameters stay in their rows/columns.
  The shape derivative of S(0) is asserted to be zero.
- The read-only `tests/spec/test_a5_survival_grids.py` covers the independent
  revision-24 contract (116 collected cases). Spec files, independent tests,
  exports, dependencies and docs pages were not changed.

## Checks actually run

Initial focused execution of `tests/unit/test_survival_grid.py` reported
42 passed in each dtype mode. It exposed a pytest iterator-parametrization
deprecation; the parameter values now use a list, before the final gate run.

    UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A5

Exit 0, final line: `check: all gates passed through A5`.

- Protected-path guard: `ok (A5.5)`.
- Ruff: 46 files already formatted; all lint checks passed.
- mypy: success, no issues in 12 source files.
- Own tests: x64 **415 passed, 7 skipped**; x32 **421 passed, 1 skipped**.
  These include all 42 new grid cases in each mode, with no final-run warnings.
- Source doctests: **1 passed** in each mode.
- Independent through-A5 tests: x64 **1116 passed, 26 skipped**;
  x32 **1121 passed, 21 skipped**. Existing suite skips are retained;
  no skip or xfail was added by this step.
- `git diff --check`: exit 0, no output.

## Remaining limitation and stop

D-022 remains open: `log_cdf_grid` inherits the current direct method's known
upper-tail relative-precision loss. This step adds shapes/dispatch and makes
no claim to repair that numerical issue. PLAN.md proposes its own numerical
step after A5.5, before the separate A5 docs steps. No deviation is resolved.
Docs needs are recorded in PLAN.md; no docs build was run for this code step.

Commit A5.5 and stop for review before starting another step.
