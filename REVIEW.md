# A6.4a rework — Withdraw the superseded family-local guards

Status: implemented on maintainer-selected branch `A6`, based on `2247f88`.
The maintainer approved this next part with "get to it". Specification revision
32, marker `cf4fef9`; independent snapshot `08d97a6`. Stop after this step's
commit; A6.4b1/b2/c have not begun.

## Read first

1. `src/survjax/distributions/exponential.py`: log_prob again uses the direct
   log(rate) formula, without the zero-rate operand guard from `21c2632`.
2. `src/survjax/distributions/weibull.py`: removed the zero-scale/shape operand
   substitutions and value restoration added in that commit; retained all A5
   time-boundary handling. Both files exactly match `3b0c3bf`.
3. Deletion of `tests/unit/test_likelihood_operands.py`: its test-only
   evaluator and zero-only isolation checks served the rejected approach.
4. PLAN.md: A6.4a rework and the proposed central masking in A6.4b1/b2.

## Invariants and test/spec mapping

- Spec 7.1–7.3/12.1 rev 32: isolation belongs in likelihood parameter
  preparation. This step withdraws the family-local mechanism; it does not
  implement or claim isolation for direct methods at invalid parameters.
  The complete replacement test matrix remains planned in A6.4b1/b2/c.
- Spec 4.1/4.5: direct formulas and existing boundary guards are restored
  byte for byte from the pre-A6.4a source. There is no constraint-driven
  repair, new parameter validity domain or change to the public API.
- `tests/unit/test_exponential.py::test_parameter_derivatives` covers the
  analytical interior gradient/Hessian and valid boundary gradients.
  `test_mixed_infinity_jacobians` covers inactive infinity contributions.
- `tests/unit/test_weibull.py::test_mixed_boundary_full_parameter_derivatives`
  covers analytical interior Jacobians/Hessians beside time boundaries, with
  shared/per-row parameters under jit/vmap. The constant-boundary and
  shape-one boundary tests cover the retained A5 guards (spec 4.5/7.2/7.3).
- Independent A5 boundary, tail, closed-form and derivative suites run in the
  full gate, including second-order check_grads and the time-zero grid.
  Focused A6 constraint/parameter-issue suites verify existing declarations,
  diagnostics, transformations and unchanged direct-method behavior.
- No new behavior or test is added for this restoration. Existing tests and
  tolerances are unchanged; only the obsolete dedicated module is removed.
  No spec, independent-test, dependency, docs or protected-file changes.

## Checks actually run

All uv/gate commands use UV_CACHE_DIR=/private/tmp/survjax-uv-cache.

- `../scripts/spec-since.sh dev`: exit 0,
  `dev: spec unchanged since cf4fef9`.
- `git diff --exit-code 3b0c3bf --
  src/survjax/distributions/exponential.py
  src/survjax/distributions/weibull.py`: exit 0, no output.
- Focused suites, with SPEC_THROUGH=A6 and separate JAX_ENABLE_X64=1/0:
  `uv run --locked --group test pytest tests/unit/test_exponential.py
  tests/unit/test_weibull.py tests/spec/test_a6_constraints.py
  tests/spec/test_a6_parameter_issues.py -q --tb=short`.
  x64: exit 0, 176 passed, 2 skipped (13.87s).
  x32: exit 0, 177 passed, 1 skipped (14.34s).
  Existing skips: optional NumPyro absent in both; conversion-warning test
  applies only with x64 disabled.
- `scripts/check.sh --through A5`: exit 0, final output
  `check: all gates passed through A5`. Guard: `ok (A6)`; Ruff format:
  `59 files already formatted`; lint: `All checks passed!`; mypy:
  `Success: no issues found in 15 source files`.
  Own x64: 534 passed, 7 skipped (39.53s); own x32: 540 passed, 1 skipped
  (39.54s). Source doctests: 4 passed in each mode (1.05s/0.93s).
  Independent through-A5 x64: 1256 passed, 26 skipped, 769 deselected
  (25.31s); x32: 1261 passed, 21 skipped, 769 deselected (20.43s).
  Full log: `/private/tmp/survjax-a64a-rework-check.log`.
- `git diff --check`: exit 0, no output. Source/test diff: 194 changed
  lines (5 additions, 189 deletions), within the planned review budget.

## Limits and uncertainties

No new spec ambiguity or deviation. No likelihood core, helper or public entry
point yet. The synced public isolation/unsupported-mode modules are reserved
for A6.4c's through-A6 gate; passing through A5 is not revision-32 likelihood
conformance. No docs build, R fixture, wheel/install or benchmark run.
