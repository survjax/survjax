# A6.4b2 — Private exact/right likelihood evaluator

Status: implemented on the existing authorized `A6` branch, based on
`523532a`. The checkout initially pointed to main at the same commit; switched
back to A6 before changes. Spec revision 32, marker `cf4fef9`; independent
snapshot `08d97a6`. Stop after this commit; A6.4c has not begun.

## Read first

1. `src/survjax/likelihood.py`: _log_likelihood, taking prepared SurvivalData
   and returning one array. No public export or new result container.
2. `tests/unit/test_likelihood_core.py`: analytical values and transforms,
   custom defaults/dtype probe, true zero mass, shapes, data evidence,
   unsupported-row isolation and traced construction.
3. `tests/unit/test_likelihood_parameters.py`: run the existing 35-case
   violation/family matrix against both the helper and the core. The core
   branch also checks the declared-invalid row is NaN.
4. PLAN.md: current approved step and the remaining public integration.

## Invariants and test/spec mapping

- Spec 3.2/6.2/7.1: data fields and issue bits have aligned observation shapes.
  Reuse row_issues on current normalized fields, including exact upper=lower,
  and OR the resulting bits with stored raw-input bits. Indexing/reconstruction
  cannot erase raw evidence, and changing normalized values cannot evade the
  core's checks. Inputs and their stored diagnostics are not mutated.
- Spec 3.3/5: compute trailing-axis output shape; incompatible data/batch shapes
  raise TypeError. Promote active time operands and every own/nested parameter
  leaf to the common data/parameter dtype, preserving static fields, component
  axes and original objects. Masks explicitly have the contribution shape.
- Spec 7.1/7.3: obtain declared violations from original parameters. Apply the
  reviewed helper to the temporary promoted tree before numerical family
  methods, including reference_time. Force NaN on data-issue, parameter-issue
  or unsupported rows. Undeclared family NaNs remain NaN; zero mass stays -inf.
- Spec 7.2: supported means exact or upper=+inf, with left_trunc=0 and
  right_trunc=+inf, determined from current arrays rather than metadata.
  Exact rows use log_prob; right-censored rows use log_survival. Inactive,
  invalid and unsupported density operands use a stopped reference; inactive
  survival uses zero via the base policy. No truncation normalization,
  interval primitive, flooring, clipping or stopped valid parameters.
- Mixed-value tests use hand-derived Exponential/Weibull formulas and include
  censoring at zero. They run eager/jit/vmap and valid eager jax_debug_nans.
  A custom family with generic autodiff density asserts common operand/leaf
  dtype before evaluation, including a nested baseline. A uniform (0,1)
  override verifies true zero density and survival outside support remain -inf.
- Shape tests cover scalar, paired and draw-column batches with analytical
  expectations, and TypeError eagerly/under tracing. Data tests preserve an
  invalid raw indicator after indexing/rebuilding and catch newly invalid
  lower/upper/window values with zero old bits. A separate zero-censoring
  test anchors the contribution, gradient and Hessian to zero.
- Unsupported cases cover left/interval censoring and left/right truncation,
  with valid data and parameters. Supported neighbors retain analytical
  values, jit forward/reverse gradients and all four second-order compositions
  for shared/per-row parameters. Traced ordinary construction works despite
  conservative all-true metadata.
- The reused matrix checks five violation classes, conditions, custom/nested
  families and components. Selected valid values and first/second derivatives
  match isolated evaluation, including jit and vmap; named failures and
  Exponential analytical anchors remain. Invalid-row derivatives are unspecified.

## Checks actually run

All uv/gate commands use UV_CACHE_DIR=/private/tmp/survjax-uv-cache.

- Initial focused core/new matrix branches, separate JAX_ENABLE_X64=1/0:
  `uv run --locked --group test pytest tests/unit/test_likelihood_core.py
  tests/unit/test_likelihood_parameters.py -k core -q --tb=short`.
  Both exited 1: 60 passed, 1 failed, 38 deselected (53.26s each).
  Failure was the test's scalar NumPy multiplication of a Python list in
  expected-value algebra. Replaced the list operand with a NumPy array;
  library code was unchanged.
- Corrected broadcasting cases in both modes:
  `uv run --locked --group test pytest tests/unit/test_likelihood_core.py
  -k trailing_axis -q --tb=short`: exit 0, 3 passed, 23 deselected
  (1.54s each). Complete corrected coverage runs in the full gate.
- Focused Ruff lint and format checks: exit 0; all checks passed, three files
  formatted. mypy src: exit 0, no issues found in 17 source files.
- `scripts/check.sh --through A5`: exit 0; final output
  `check: all gates passed through A5`. Guard: `ok (A6)`; Ruff:
  `64 files already formatted`, `All checks passed!`; mypy:
  `Success: no issues found in 17 source files`.
  Own x64: 647 passed, 7 skipped (112.02s); x32: 653 passed, 1 skipped
  (115.69s). Source doctests: 4 passed per mode (0.84s each).
  Independent through-A5 x64: 1256 passed, 26 skipped, 769 deselected
  (24.50s); x32: 1261 passed, 21 skipped, 769 deselected (24.75s).
  Full log: `/private/tmp/survjax-a64b2-check.log`.
- `../scripts/spec-since.sh dev`: exit 0, spec unchanged since cf4fef9.
- `git diff --check`: exit 0, no output.

## Limits and uncertainties

No new spec ambiguity or deviation. No docs, dependency, spec, independent-test
or public API change. This is the prepared-data private evaluator only: no
raw-form resolver, base method, export or complete A6 gate yet. Those belong
to A6.4c, including the synced public acceptance suites. No docs build, R
fixture, install or benchmark run. This step stays within the 300-line
implementation/test diff budget (297 lines); the existing matrix is reused rather than
copied. All existing tolerances are unchanged.
