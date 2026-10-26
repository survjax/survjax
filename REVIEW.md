# SurvivalData size interface — spec 3.2

Read `src/survjax/data.py` (class summary and the three new methods), then
`tests/unit/test_data_size.py`, then the active entry at the top of PLAN.md.
Approved implementation branch: `len_data`.

The read-only shape property returns the stored lower-bound array's shape.
Length counts its leading axis and rejects scalars. Explicit truth testing
always raises TypeError, including for empty data. No size/ndim attributes,
new exports, stored fields, auxiliary metadata or dependencies were added.
Construction, normalization, numerical operations and indexing are unchanged.
The pre-existing data module exceeds the approximate 300-line file guideline;
this localized addition does not introduce a module split or helper layer.
Source/test changes remain below the planned 300-line budget.

Invariants and spec-to-test mapping (all new tests cite section 3.2):

- `test_size_follows_input_dimensions`: both constructors, tuple of Python
  ints, read-only property, scalar length errors, empty length and leading
  axis rather than total elements; also rejects unrequested size/ndim API.
- `test_indexing_and_reconstruction_keep_size_and_tree_structure`: indexing,
  slicing and tree-map slicing determine size; flatten/unflatten retains six
  leaves, parent flags and tree structure despite changing dimensions.
- `test_size_is_static_inside_jit`, `test_scalar_shape_and_length_inside_jit`:
  shape/length are Python values during tracing, including empty and scalar
  cases; jit-returned integers are compared as scalar values.
- `test_vmap_size_describes_each_example`,
  `test_vmap_vector_sees_scalar_examples`: per-example shape and length,
  eagerly under vmap and with jit-of-vmap.
- `test_truth_testing_always_raises`, `test_vmap_truth_testing_raises`:
  bool and Python conditionals reject data; jit and vmap reject truth testing
  while tracing. No derivatives or numerical tolerances apply to shape queries.

Checks actually run:

- Focused own tests: `JAX_ENABLE_X64=<1|0>
  UV_CACHE_DIR=/private/tmp/survjax-uv-cache uv run pytest
  tests/unit/test_data_size.py tests/unit/test_data_indexing.py
  tests/unit/test_data_representation.py tests/unit/test_data_metadata.py
  tests/unit/test_data_tracing.py -q`: 65 passed in 5.31s (float64), 65 passed
  in 4.30s (float32); no skips or failures.
- Ruff formatting of the two changed Python files: one reformatted, one
  unchanged. Focused Ruff lint: All checks passed! `uv run mypy src`:
  Success: no issues found in 17 source files.
- `git diff --check`: exit 0.

Full gate: `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh
--through A7` exited 0: `check: all gates passed through A7`.
Ruff: 72 files already formatted / All checks passed! Mypy: no issues found
in 17 source files.

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 774 passed, 7 skipped, 157.24s | 780 passed, 1 skipped, 150.46s |
| Source doctests | 4 passed, 0.95s | 4 passed, 1.03s |
| Independent through A7 | 2114 passed, 28 skipped, 218 deselected, 103.10s | 2118 passed, 24 skipped, 218 deselected, 102.64s |

Full output: [/private/tmp/survjax-len-data-gate.log](/private/tmp/survjax-len-data-gate.log).
The new own module contributes 22 cases per dtype; the independent shape/len
module contributes 32. No new skips or exclusions; independent tests are
unchanged. Only PLAN.md/REVIEW.md evidence was updated after the full gate.
Final `uv run ruff format --check .` reports 72 files already formatted;
`uv run ruff check .` reports All checks passed! `git diff --check` exits 0
after the evidence update.
No unresolved contract question or new deviation. No R or docs build is
needed for this code step. Narrative docs remain a separate docs to-do in
PLAN.md. This step does not claim Slice A completion. Stop after the commit.
