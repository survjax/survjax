# Review: A3 observation indexing and static metadata

Branch: `A3`. Spec sections 3.2, 6.2, 12.1 and the data portion of 12.2.

## Read first

1. `src/chronox_jax/_data_metadata.py`: input-tracer classification and six
   concrete presence flags, in exact/right/left/interval/left-trunc/right-trunc
   order.
2. `src/chronox_jax/data.py`: constructor entry classification, compile-time
   normalization for concrete inputs, and `__getitem__`.
3. `tests/unit/test_data_indexing.py`: aligned leaves and retained raw issues.
4. `tests/unit/test_data_metadata.py`: concrete/traced behavior and trace counts.

## Invariants

- All six leaves share observation shape. Indexing applies the same index to
  each leaf; reconstruction retains parent auxiliary flags without normalization.
- Concrete metadata reduces over every normalized row, including invalid rows.
  Right censoring is identified by non-exactness and infinite upper bound;
  left censoring has lower zero and is not right censoring; remaining censored
  rows are interval rows. Left/right truncation presence is independent.
- Classification inspects supplied inputs before JAX operations. A tracer at
  any list/tuple nesting depth forces all flags true. Concrete normalization
  used by metadata runs under `jax.ensure_compile_time_eval`, including inside
  transformations. Host conversion occurs only on this concrete path.
- Auxiliary data contains only bool flags. Shapes, values and issue bits are
  absent from auxiliary data; unchanged flags/shapes/dtypes reuse compilation.
- Diagnostics remain a decoded view. Indexed `ok` reduces over indexed rows;
  an empty subset is valid. The six-leaf structure and exports are unchanged.

## Test evidence

| Spec requirement | Implementer tests |
| --- | --- |
| 3.2 aligned indexing, scalar/negative/stride/empty/array/boolean selections, tree-map identity | `test_indexing_preserves_each_leaf` |
| 3.2/12.1 multidimensional indexing and eager/jit/vmap agreement | `test_multidimensional_indexing_and_transforms` |
| 3.2/6.2 stored invalid codes survive indexing and reconstruction; scalar subset validity inside jit | `test_indexed_and_rebuilt_issues_keep_raw_evidence` |
| 3.2 both constructors agree, normalized kinds include invalid rows, subsets inherit flags | `test_normalized_kinds_and_truncation_flags` |
| 3.2 independent kind/truncation flags, explicit defaults, equal treedefs across row counts | `test_independent_presence_flags` |
| 6.2 concrete Python/NumPy/captured JAX inputs inside jit, unrelated tracer arguments | `test_concrete_inputs_inside_jit_keep_eager_flags` |
| 3.2/6.2 nested tracers, single traced argument, arrays created inside jit | `test_observation_tracers_force_all_flags_without_changing_leaves` |
| 3.2/12.1 same flags compile once, changed flags separately, parent-slice and all-true reuse | `test_static_flags_control_compilation_reuse` |

The read-only A3 acceptance suite also covers each supplied observation argument,
issue decoding after transformations, constructor diagnostics and host checkpoints.
This step does not establish completion of the full Slice A gate.

## Checks run

Commands using uv set `UV_CACHE_DIR=/private/tmp/chronox-uv-cache` because the
sandbox denies access to the default cache.

- Focused new tests, `JAX_ENABLE_X64=1 uv run pytest
  tests/unit/test_data_indexing.py tests/unit/test_data_metadata.py -q`:
  **22 passed in 2.68s**.
- `uv run mypy src`: **Success: no issues found in 5 source files**.
- `uv run ruff check .`: **All checks passed!**.
- `scripts/check.sh --through A3`: **check: all gates passed through A3**.
  Guard passed on `A3`; **29 files already formatted**; Ruff passed; mypy
  passed. Own tests: **108 passed, 3 skipped in 5.53s** (x64), **111 passed
  in 4.99s** (x32). Independent tests: **372 passed, 15 skipped in 8.20s**
  (x64), **382 passed, 5 skipped in 7.97s** (x32). Own x64 skips concern
  conversion warnings/unavailable dtype behavior specific to x32.
- After the gate, removed one unused test-data assignment. Re-ran
  `JAX_ENABLE_X64=1 uv run pytest
  tests/unit/test_data_metadata.py::test_static_flags_control_compilation_reuse
  -q`: **1 passed in 0.83s**. `git diff --check`: passed with no output.
- No R, fixture generation, docs build or install smoke test was run for A3.

## Limits and open items

- Indexing follows JAX constraints: boolean masks must be concrete under jit;
  dynamic selections with variable output shapes are not supported by JAX.
- D-010 remains formally open. Its current independent test uses indexed
  data's scalar `ok` and passed; closure belongs to the maintainer.
- Existing `docs/playground.ipynb` edits are excluded from this commit.
