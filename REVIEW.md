# A6.4b1 — Prepare parameter leaves for likelihood evaluation

Status: implemented on maintainer-selected branch `A6`, based on `9099310`.
Spec revision 32, marker `cf4fef9`; independent snapshot `08d97a6`.
The maintainer approved this next part with "ok now do the actual next step".
Stop after this commit for review. A6.4b2/c have not begun.

## Read first

1. `src/survjax/_likelihood_parameters.py`: stop_invalid_gradients, a private
   helper returning a temporary distribution with unchanged parameter values.
2. `tests/unit/test_likelihood_parameters.py`: custom families, the 35-case
   isolation matrix, then reconstruction, per-leaf vmap and draw-column checks.
3. PLAN.md: implementation scope and remaining evaluator/public-API steps.

## Invariants and test/spec mapping

- Spec 7.1: compute the outer current batch_shape once. Every own or nested
  parameter leaf broadcasts to it plus its own trailing component_shape.
  The supplied declared invalid-entry union expands only over components.
  `where(mask, stop_gradient(value), value)` stops invalid entries without
  numerical substitution. Valid entries stay differentiable.
- The outer union reaches every leaf, including unconstrained siblings and
  nested leaves whose own diagnostics report no issue. It is never narrowed
  to a nested family's mask, and the helper infers no new validity domain.
- Spec 4.2/7.2: reconstruct without constructors, dtype normalization, family
  formulas, host inspection or mutation. Preserve runtime types/static fields
  and leaf dtypes. This is revision 32's explicit stop_gradient exception;
  invalid-row gradients remain unspecified. No NaN contribution selection yet.
- `test_declared_violations_isolate_valid_values_and_derivatives`: zero,
  negative, NaN and both infinities for Exponential rate, Weibull shape/scale,
  unguarded custom/nested rate and a two-axis component field; nonfinite outer
  parameters, base conditions, and an outer-only condition with an undeclared
  NaN sibling. As functions of the same original parameter vector, selected
  valid rows match evaluation alone in values, jacfwd/jacrev, all four
  second-derivative compositions, compiled reverse Jacobian/Hessian and
  compiled vmap Jacobians. Shared valid leaves are differentiated too.
- The same test anchors all Exponential violation cases to hand-derived
  gradients [[1,0,0],[0,0,-3]] and a Hessian with only entry [0,0,0]=-4
  (spec 7.3); these binary values are checked exactly. Other comparisons use
  existing central tolerances. No tolerance changes or new AD rules.
- Reconstruction test: mixed-dtype smaller nested leaves, outer-only condition,
  two component axes and current batch (2,1); asserts values, types, static
  fields, shapes and unchanged input shapes. Constructor spies reject calls.
  A shared unconstrained leaf has identity derivative only at valid entries.
- Per-leaf vmap test: a mapped scalar broadcasts with shared batched siblings;
  explicit output shapes and analytical identity/zero Jacobian blocks catch
  skipped broadcasting or failure to mask the unconstrained sibling.
  Draw-column test retains batch shape (2,1), values and valid identity Jacobian.
  Leaf Jacobian blocks verify the mandated barrier itself; no invalid-row
  likelihood derivative is asserted.

## Checks actually run

All uv/gate commands use UV_CACHE_DIR=/private/tmp/survjax-uv-cache.

- Initial full focused module, separate JAX_ENABLE_X64=1/0:
  `uv run --locked --group test pytest tests/unit/test_likelihood_parameters.py
  -q --tb=short`. Both exited 1: 36 passed, 2 failed (46.18s/46.01s).
  The test family's condition hook returned a scalar for rebuilt/per-leaf-vmap
  batches; diagnostics correctly rejected its shape. Corrected the test hook
  to broadcast its condition to current batch_shape. No library change needed.
- Corrected cases in both modes with the same command plus
  `-k 'rebuilt or per_leaf'`: exit 0, 2 passed, 36 deselected in each
  (1.20s/1.17s). The complete corrected suite runs again in the full gate.
- Focused Ruff format/check: exit 0, both files formatted and lint clean.
  mypy src: exit 0, no issues found in 16 source files.
- `scripts/check.sh --through A5`: exit 0, final output
  `check: all gates passed through A5`. Guard: `ok (A6)`; Ruff format:
  `61 files already formatted`; lint: `All checks passed!`; mypy:
  `Success: no issues found in 16 source files`.
  Corrected complete own x64: 572 passed, 7 skipped (90.95s); x32:
  578 passed, 1 skipped (89.58s). This includes all 38 new tests per mode.
  Source doctests: 4 passed per mode (0.93s/0.99s).
  Independent through-A5 x64: 1256 passed, 26 skipped, 769 deselected
  (27.59s); x32: 1261 passed, 21 skipped, 769 deselected (22.98s).
  Full log: `/private/tmp/survjax-a64b1-check.log`.
- `../scripts/spec-since.sh dev`: exit 0, spec unchanged since cf4fef9.
- `git diff --check`: exit 0, no output.

## Limits and uncertainties

No new spec ambiguity or deviation. No public API/export, docs, dependency,
spec or independent-test change. This is helper-level isolation evidence;
A6.4b2 must call it before reference_time/density/survival evaluation and force
invalid contributions to NaN. A6.4c will expose the entry points and run the
synced public isolation and unsupported-mode suites through the A6 gate.
No complete A6 gate, docs build, R fixture, wheel/install or benchmark run.
Implementation plus tests are 300 new lines (41 helper, 259 tests).
