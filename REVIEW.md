# A9.1 — Derivative and compiled-composition evidence

Read `tests/unit/test_likelihood_composition.py` first, then
`tests/unit/test_likelihood_custom_derivatives.py`, then the active A9.1
entry in PLAN.md. Approved branch: `A9`; starting commit: `030314b`.
This is a verification-only step. No library code, public API, dependencies,
shared tolerances, spec, independent tests or narrative docs changed.
The two new test modules total 299 lines.

Invariants and spec-to-test mapping:

- `test_traced_construction_diagnostics_and_derivatives` (6.2, 7.3, 12.1):
  Exponential and Weibull with flag/bounds data constructed from traced
  arguments; data and parameter diagnostic `ok` values are returned together
  with the objective, forward/reverse gradients and three Hessian routes.
  Prepared, eager construction and jit construction each match independently
  differentiated exact/right formulas, including Weibull's mixed derivative.
- `test_mapped_likelihood_derivatives_are_analytical` (7.3, 12.1):
  different parameter vectors and datasets have analytical values, gradients
  and Hessians under vmap and jit-of-vmap. Replicates remain independent.
- `test_diagnostics_guard_invalid_replicate_under_jit_vmap` (6.2, 12.1):
  data/parameter diagnostics guard the objective via lax.cond; one invalid
  raw indicator selects an explicit zero fallback. Its valid neighbors keep
  their hand-derived values, gradients and Hessians. The fallback belongs to
  the test's caller workflow, not the library's NaN-likelihood contract.
- `test_derivative_compilation_reused_for_changed_values` (7.2, 12.3):
  changing parameter values, times and event indicators with unchanged
  shapes/dtypes/tree structure reuses one trace. Equal-sized slices retain
  their parent's tree structure/static metadata. Every call is synchronized
  and its values and derivatives checked analytically, including after reuse.
- `test_generic_density_parameter_derivatives` (4.2, 7.3, 12.1):
  a custom family supplies only log survival and a reference, using the
  generic autodiff density. Analytical values, gradients and Hessians agree
  eagerly and under jit; mapped Hessians agree too. Order-two check_grads
  covers forward/reverse modes in both process dtype modes with the existing
  justified unit-scale mean-likelihood tolerances and finite-difference steps.
- `test_substituted_reference_derivatives_are_isolated` (7.2, 12.1):
  the custom exponential reference remains positive finite, but has a
  singular derivative at the tested rate. Right-only and mixed batches
  retain the active formula's gradient and Hessian in all four nested
  forward/reverse combinations, eagerly and through jit.

All numerical expected values come from explicit formulas in the tests.
NumPy inspection runs outside traced functions. Parameters are finite smooth
interior cases; no derivative claim extends to data times or unsupported,
invalid or infinite objectives. Existing tests retain responsibility for
parameter-issue isolation, boundaries, tails and built-in finite differences.
Both dtype processes are selected externally; tests never change JAX config.

Checks completed so far (CPU, Python 3.12.9, JAX 0.11.2, NumPy 2.5.3):

- Focused own tests: `JAX_ENABLE_X64=<1|0>
  UV_CACHE_DIR=/private/tmp/survjax-uv-cache uv run pytest
  tests/unit/test_likelihood_composition.py
  tests/unit/test_likelihood_custom_derivatives.py -q`:
  15 passed in 22.61s (float64), 15 passed in 21.95s (float32), no skips.
- Independent A9 tests: `JAX_ENABLE_X64=<1|0> SPEC_THROUGH=A9
  UV_CACHE_DIR=/private/tmp/survjax-uv-cache uv run pytest
  tests/spec/test_a9_likelihood_derivatives.py
  tests/spec/test_a9_compiled_composition.py -q`:
  40 passed in 37.28s (float64); 27 passed, 13 skipped in 18.37s (float32).
  The 13 skips are the independent suite's existing float64-only finite
  difference checks. No independent tests were edited, skipped or xfailed
  by this step.
- Focused logs: `/private/tmp/survjax-a91-focused-final-x1.log`,
  `/private/tmp/survjax-a91-focused-final-x0.log`;
  independent logs: `/private/tmp/survjax-a91-spec-x1.log`,
  `/private/tmp/survjax-a91-spec-x0.log`.
- Ruff: 74 files already formatted / All checks passed! Mypy:
  Success: no issues found in 17 source files. `git diff --check`: exit 0.

Full gate: `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh
--through A7` exited 0: `check: all gates passed through A7`.
Guard: ok (A9). Ruff: 74 files already formatted / All checks passed!
Mypy: Success: no issues found in 17 source files.

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 789 passed, 7 skipped, 186.51s | 795 passed, 1 skipped, 173.74s |
| Source doctests | 4 passed, 1.01s | 4 passed, 0.95s |
| Independent through A7 | 2114 passed, 28 skipped, 218 deselected, 106.77s | 2118 passed, 24 skipped, 218 deselected, 107.94s |

Full log: [/private/tmp/survjax-a91-gate.log](/private/tmp/survjax-a91-gate.log).
The 15 new own tests pass in both modes with no new skips or exclusions.
The independent gate's 218 deselections belong to later steps; the two A9
modules above were also run separately in full. Only PLAN.md/REVIEW.md result
records changed after the gate. Final Ruff format/lint and whitespace checks
are run after those record updates, before the step commit.

The initial focused run in each dtype had 7 failed / 8 passed
because the new analytical expectations negated NumPy boolean arrays. Casting
the event indicators to float corrected that test-authoring error; no library
change or tolerance adjustment was needed. An initial Ruff check also caught
an unbound loop index in a test lambda; it now binds the index explicitly.

No contract ambiguity or new deviation found. A9.2 installation, A9.3 example,
A9.4 CPU measurements and A9.d1 narrative docs have not started. A8 fixtures
are absent, so this step cannot establish the cumulative A9 or Slice A gate.
No R or docs build is required by this test-only step. Stop after its commit.
