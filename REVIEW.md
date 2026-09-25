# Review: A1 independent-review fix-up

Read in this order:

1. The independent review at `~/dev/chronox-dev-review/.review/a.md` (the
   requested `a1.md` path was absent), especially F1–F6 and decided Q1.
2. `PLAN.md`, final section, for this fix-up's scope and checks.
3. `src/chronox_jax/data.py`, then the new cases in
   `tests/unit/test_data_dtype.py` and `test_data_representation.py`.
4. `DEVIATIONS.md` D-002 for the maintainer-decided input typing and warning
   rule. D-001 remains the earlier inactive-`time2` question.

## Invariants and review disposition

- **F1 / Q1:** Bare Python `int` and `float` values remain weak for JAX
  promotion. Lists and tuples first take JAX's default array dtype. Inputs
  with a `dtype` attribute carry their dtype. Only dtype-bearing float64 input
  warns in 32-bit mode; `dtype=` is explicit and never warns. With no strong
  floating input, construction uses the configured default float. The library
  neither changes JAX's 64-bit setting nor converts tracers to NumPy.
- **F2:** `_to_shape` rejects any field that would enlarge the first input's
  shape. An exact row from bounds stores `upper == lower`, even when the
  supplied upper is NaN.
- **F3:** `SurvivalData` is still frozen but has identity equality/hash instead
  of dataclass-generated comparisons over arrays. `dataclasses.replace` is
  unsupported because public construction accepts raw observation arguments;
  leaf changes belong to the PyTree path.
- **F4:** One constructor call emits at most one float64 conversion warning,
  even when several typed fields are float64.
- **F5:** The conservative six-position metadata order is named in a comment;
  A3 will compute metadata from concrete values and test compile counts.
- **F6:** Field assignment is explicit. The fix-up is a separate small commit
  on the same A1 branch; no protected paths or dependencies changed.

## Tests and exact checks

| Requirement | Implementer tests |
| --- | --- |
| Weak Python bounds preserve float32 arrays; defaults and scalar/list-only inputs follow JAX (spec 3.3, D-002) | `test_python_scalar_bounds_are_weak_in_both_modes`, `test_omitted_and_explicit_default_bounds_have_same_type_and_tree`, `test_python_float_list_uses_default_float_without_warning`, `test_python_scalar_time_uses_default_float` |
| Typed NumPy float64 stays strong; one warning per call (spec 3.3) | `test_numpy_float64_scalar_truncation_is_strong`, `test_multiple_float64_fields_warn_once_per_call`, existing `test_float64_warning_and_conversion_in_32bit_mode` |
| Shape cannot enlarge; exact bounds normalize upper (spec 3.2) | `test_flag_cannot_enlarge_observation_shape`, `test_exact_bounds_replace_nan_upper_with_lower` |
| Array equality does not try elementwise truth (spec 3.2 representation) | `test_data_equality_uses_identity` |
| Existing A1 public contract | `scripts/check.sh --through A1`, including independent `tests/spec/test_a1_*` in both modes |

Checks run with `UV_CACHE_DIR=/private/tmp/chronox-uv-cache`:

- Ruff format: `13 files already formatted`; Ruff lint: `All checks passed!`.
- mypy: `Success: no issues found in 2 source files`.
- `scripts/run_tests.sh` (through the final gate): x64 enabled `19 passed,
  3 skipped`; x64 disabled `22 passed`.
- `scripts/check.sh --through A1`: guard `ok (a1-survivaldata)`; independent
  spec tests x64 enabled `41 passed, 8 skipped, 196 deselected`; x64 disabled
  `45 passed, 4 skipped, 196 deselected`; final output
  `check: all gates passed through A1`.

No R, fixture, wheel or installation check was run; those are outside this
fix-up. The three own-test skips in x64 mode check behavior specific to x64
being disabled.

## Remaining uncertainty and boundary

- D-002 stays `open` as required. The reviewer records the maintainer's
  decision, and the proposed spec wording must be applied separately on a
  `spec-*` branch. This commit does not edit `spec/` or independent tests.
- D-001 stays `open` despite the maintainer's approval of the current
  normalization. A2 will add active `time2` validation and issue bits.
- Indicator, endpoint and truncation validation remain A2. Indexing,
  value-derived metadata and compile-count tests remain A3.
