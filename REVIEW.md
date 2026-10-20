# A6.2 — ParameterIssues container and host checkpoint

Status: implemented on `A6.2`, based on `7274dd0`; ready for review.
The maintainer approved the plan, branch creation and minimal docs exception.
Spec revision 30: `.spec-seen` fd150db; synced acceptance snapshot f527f75.
Stop after this step's commit; A6.3 has not begun.

## Read first

1. `src/survjax/diagnostics.py`: ParameterIssues and InvalidParameterError.
2. `tests/unit/test_parameter_issue_container.py`: 17 focused cases.
3. `src/survjax/__init__.py`, `tests/unit/test_import.py`, `docs/api.md`:
   exception export, identity check and approved autodoc entry.
4. PLAN.md: approved A6.2 scope, evidence and A6 docs to-do.

## Invariants and test/spec mapping

- Spec 4.7/6.3: frozen field assignment, ordinary mutable dictionaries;
  supplied boolean masks and aggregates must be consistent. The A6.3 producer
  will compute the union and scalar ok. No constructor checks are added.
- Ordered field/condition key tuples are static metadata; only arrays are
  leaves. Unflatten performs no calculations, including for vmap placeholders.
  `test_round_trip_preserves_order_masks_and_frozen_fields`,
  `test_jit_preserves_multidimensional_masks_and_empty_mappings` and
  `test_vmap_and_jit_keep_order_and_per_example_ok` demonstrate scalar and
  batch shapes, dotted/unsorted keys, empty mappings and stacked ok values.
- Spec 7.2: `test_same_shaped_mask_values_reuse_compilation` changes mask
  values and verifies one trace with updated output.
- Spec 4.7/6.2: scalar field/condition, overlapping dotted-name and concrete
  jit-result checkpoint tests verify return None or InvalidParameterError,
  with each category's counts and first scalar/vector/multidimensional index.
  Expected masks, counts and indices are hand-enumerated, with exact assertions.
- `test_checkpoint_rejects_any_traced_mask` checks jit/vmap, valid/invalid
  inputs and traced/concrete ok: exactly plain ValueError with "host-only".
  All leaves are inspected before NumPy conversion; no callbacks are installed.
- Spec 10.2: import test and independent public-exception test check the export.
  Existing DataIssues calculation, checkpoint and repr code are unchanged.

## Checks actually run

All uv/build commands use UV_CACHE_DIR=/private/tmp/survjax-uv-cache.

- Focused command, separately with JAX_ENABLE_X64=1 and 0, SPEC_THROUGH=A6:
  `uv run --locked --group test pytest tests/unit/test_parameter_issue_container.py tests/unit/test_import.py tests/unit/test_data_issues.py tests/spec/test_a6_parameter_issues.py::test_invalid_parameter_error_is_a_public_value_error -q --tb=short`.
  Each exit 0: 32 passed in 2.83s (31 own, one independent).
- `scripts/check.sh --through A5`: exit 0, guard ok (A6.2); Ruff format:
  57 files already formatted; lint: All checks passed!; mypy:
  Success: no issues found in 14 source files.
  Own suite x64: 518 passed, 7 skipped (44.66s); x32: 524 passed, 1 skipped
  (43.69s). Source doctests: 4 passed per mode (1.24s/1.03s).
  Independent x64: 1256 passed, 26 skipped, 308 deselected (30.39s);
  x32: 1261 passed, 21 skipped, 308 deselected (25.63s).
  Final output: "check: all gates passed through A5".
- `scripts/build_docs.sh`: initial sandboxed run exited 2 because Jupyter
  could not bind a local port (PermissionError). Escalated rerun exited 0;
  both dtype builds executed examples afresh and reported "build succeeded."
- `git diff --check`: exit 0, no output before delivery.

## Limits and uncertainties

No new spec ambiguity found. No independent tests or spec files changed.
The complete parameter-issues acceptance module needs A6.3's producer;
no complete A6 gate, likelihood, derivatives, R fixtures, install or benchmark
checks were run. D-023 remains open for the later likelihood step.
