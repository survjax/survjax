# A6.4a — Safe built-in operands for zero parameters

Status: approved A6.4a implemented on `A6`, based on `3b0c3bf`.
Spec revision 31, marker be7b253; independent snapshot remains f527f75.
This step adds no likelihood entry point. Stop after this commit for review;
A6.4b/c have not begun.

## Read first

1. `src/survjax/distributions/exponential.py`: log_prob's zero-rate operand.
2. `src/survjax/distributions/weibull.py`: cumulative-hazard and log-hazard
   zero-scale/shape operands, value restoration and boundary handling.
3. `tests/unit/test_likelihood_operands.py`: 36 focused cases.
4. PLAN.md: approved A6.4a scope and remaining implementation parts.

## Invariants and test/spec mapping

- Spec 7.1–7.2: a zero cotangent through log(0) gives 0/0; through exp(inf)
  it can give 0*inf. A final NaN selection does not repair that arithmetic.
  Safe constant operands are substituted before these operations. The
  stored parameters and active valid calculations are unchanged. No floors,
  clipping, stopped real parameters, global dtype changes or custom AD rules.
- Exponential restores log(rate)=-inf at rate=0 after evaluating log on a
  safe operand. Thus its direct log density remains -inf and log survival 0.
- Weibull restores the zero-scale extended formula after finite operand work:
  at positive t, H is inf for k>0, zero for k<0 and NaN for k=0. Log hazard
  is inf for k>1 and NaN otherwise. With k=0 and b>0, H=1 and log h=-inf.
  Boundary restoration remains separate, including the b=0 log-scale limit.
- Spec 4.5: explicit interior-formula and zero-time-limit tests preserve the
  affected direct values, including true -inf and undefined NaN. Expected
  values follow the formulas stated in each test, not a saved code baseline.
- `test_zero_parameter_keeps_other_rows_jacobian_finite` covers rate/shape/
  scale zero between valid rows; active and inactive density branches; forward
  and reverse mode; eager and jit. All valid Jacobian rows are finite, with
  hand-derived diagonals and exactly zero in the invalid parameter column.
- `test_valid_density_derivatives_and_transformations` checks analytical
  valid Exponential/Weibull gradients and Hessians, plus Weibull eager/jit/vmap
  values. A valid interior call runs with jax_debug_nans. Existing boundary,
  tail and derivative tests run through the full gate. No tolerance changes.

## Checks actually run

All uv/gate commands use UV_CACHE_DIR=/private/tmp/survjax-uv-cache.

- New operand module, separately with JAX_ENABLE_X64=1 and 0:
  `uv run --locked --group test pytest tests/unit/test_likelihood_operands.py -q --tb=short`.
  Each exit 0: 36 passed in 4.47s.
- Expanded focus with SPEC_THROUGH=A6, separately in both dtype processes:
  `uv run --locked --group test pytest tests/unit/test_likelihood_operands.py tests/unit/test_exponential.py tests/unit/test_weibull.py tests/spec/test_a6_constraints.py tests/spec/test_a6_parameter_issues.py -q --tb=short`.
  x64: exit 0, 212 passed, 2 skipped (19.51s); x32: exit 0, 213 passed,
  1 skipped (19.94s). Existing skips: optional NumPyro and x64-inapplicable
  conversion-warning test.
- Initial Ruff check reported E731 for an assigned test lambda. Replaced it
  with a named local function; the final full gate checks the corrected file.
- `scripts/check.sh --through A5`: exit 0; guard ok (A6); Ruff format:
  60 files already formatted; lint: All checks passed!; mypy:
  Success: no issues found in 15 source files.
  Own x64: 570 passed, 7 skipped (49.32s); x32: 576 passed, 1 skipped
  (47.99s). Source doctests: 4 passed per mode (1.13s/1.04s).
  Independent through-A5 x64: 1256 passed, 26 skipped, 308 deselected (29.76s);
  x32: 1261 passed, 21 skipped, 308 deselected (24.75s).
  Final output: "check: all gates passed through A5".
- `git diff --check`: exit 0, no output before delivery.

## Limits and uncertainties

No new spec ambiguity or deviation entry. No API/export/dependency/spec/
independent-test/docs edits. No new helper module; Weibull stays at 296 lines.
Isolation evidence covers the zero-parameter cases in the approved plan;
invalid-row derivatives are still unspecified. The test-only composition is
not a public or private library likelihood implementation.
No docs build (public docstrings and docs unchanged), complete A6 gate,
R fixture, install or benchmark run. A6.4b/c still need separate approval.
