# Review: A1 SurvivalData storage and dtype policy

Read in this order:

1. `PLAN.md` for the approved A1 boundary and the size exception approved in
   conversation on 2026-09-25.
2. `src/chronox_jax/data.py` for normalization, dtype selection and PyTree
   reconstruction; `src/chronox_jax/__init__.py` for the public export.
3. `tests/unit/test_data_representation.py`, `tests/unit/test_data_dtype.py`,
   `tests/conftest.py`, and the updated `tests/unit/test_import.py`.
4. `DEVIATIONS.md` D-001 for the agreed handling of inactive `time2` values.

## Invariants

- The first time/lower array fixes the observation shape; every public field
  and the private issue-bitfield leaf has that shape. The five public fields
  follow the normalized meanings in spec 3.2.
- `exact` is bool, the issue leaf is `int32`, and four time/bound fields have
  one common floating dtype of at least float32. Integer-only input uses the
  configured JAX default float. Raw input dtypes are inspected without
  converting tracers to NumPy.
- In 32-bit mode, implicit float64 input emits one Chronox warning at the
  caller's line and converts to float32. Explicit float64 requests fail.
  Construction does not change JAX's global dtype setting.
- Tree unflattening assigns supplied leaves directly, including arbitrary
  sentinel objects, without rerunning public normalization. The six-leaf
  structure is stable; static metadata is conservatively all true until A3.
- A1 covers valid-input normalization. Full indicator, time, interval and
  truncation validation, plus issue-bit setting, belong to A2. Indexing and
  value-derived static metadata belong to A3.

## Evidence

| Requirement | Tests / check |
| --- | --- |
| Field meanings, shapes, truncation defaults and supplied bounds (spec 3.2) | `test_mixed_kinds_have_meaningful_bounds`, `test_bounds_constructor_keeps_shape_and_truncation`; independent `tests/spec/test_a1_representation.py` |
| Six leaves, dtype and PyTree behavior (spec 3.2) | `test_pytree_has_six_leaves_and_crosses_jit`, `test_unflatten_bypasses_normalization`; independent `test_a1_representation.py` |
| Common float dtype, integer default, promotion, warnings and invalid dtypes (spec 3.3) | `tests/unit/test_data_dtype.py`; independent `tests/spec/test_a1_dtype.py` |
| Public export (spec 10.2) | `test_public_package_imports` |

Final check output, with `UV_CACHE_DIR=/private/tmp/chronox-uv-cache`:

- `scripts/check.sh --through A1`: guard OK; Ruff format `13 files already
  formatted`; Ruff lint `All checks passed!`; mypy `Success: no issues found in
  2 source files`; own tests 64-bit `11 passed, 2 skipped`, 32-bit `13 passed`;
  independent A1 tests 64-bit `41 passed, 8 skipped, 196 deselected`, 32-bit
  `45 passed, 4 skipped, 196 deselected`; `check: all gates passed through A1`.
- `git diff --check`: passed with no output.

No R, fixture or installation check was run; none belongs to A1. The final A1
gate emitted no warnings. The two own-test skips are dtype-specific tests whose
behavior applies only in 32-bit mode; the independent suite has its own
mode-specific skips.

## Uncertainty and next-step boundary

- D-001 remains formally open under the repository rule, although the
  maintainer approved the proposed behavior in conversation. A2 must validate
  active `time2` values and set issue bits; A1 only normalizes valid inputs.
- The A1 acceptance tests exercise both constructors and dtype policy. This
  made the coherent code-plus-test change larger than the preferred roughly
  300 lines. The maintainer explicitly approved one A1 commit on 2026-09-25.
- A1 is not evidence of A2/A3 structural validation, indexing or compile-count
  behavior, nor of later Slice A numerical requirements.
