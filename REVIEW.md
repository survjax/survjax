# A4.3 review fixes — uncommitted

Implements F-1/F-2 and N-1–N-3 from `../review/slices/A4.3.md`, preserving
the concrete nested-field docstring requested in the preceding review.
The maintainer asked to keep work uncommitted; no commit or staging performed.

## Reading order

1. `src/survjax/distributions/base.py`: the class registration marker, nested
   validation in `initialize`, and short `__getitem__` implementation.
2. New nested-field rejection cases in `tests/unit/test_family_batching.py`,
   its unique tolerance import, and `tests/impl_tolerances.py`.
3. The concrete `distribution()` doctest; then docs wrapping/clamping wording.
4. DEVIATIONS.md D-015 and the active review-fix entry in PLAN.md.

## Invariants and coverage

- F-1 / spec 4.2, 6.2 / D-015: each registered class receives its own private
  marker after successful PyTree registration. Nested values must be instances
  of SurvivalDistribution with a marker in their exact class dictionary;
  inheriting a parent's marker is insufficient. Malformed values raise
  `TypeError` naming the nested field before batch-shape access. Four new cases
  cover an array, None, an unregistered dataclass and an unregistered subclass.
  Existing construction/JAX acceptance cases prove valid nested inputs work.
- F-2 / spec 4.1: construction gives every own/nested dynamic leaf exactly the
  same leading batch rank. Appended full component slices protect the trailing
  axes; JAX handles ellipsis, implicit trailing axes, advanced indices and
  malformed-index errors. Removed custom expansion/counting and its branching.
  Existing nine indexing cases, scalar component-axis rejection, tree slicing,
  traced integer indexing and vmap cases pass. No custom index-error wording
  is retained. As the review noted, a static-only family with no dynamic leaves
  has no leaf operation to index and tree mapping returns it unchanged.
- N-1: renamed the implementer's tolerance module to `impl_tolerances.py`.
  The derivative test imports that unique name. Combined sessions collecting
  both unit and independent suites pass; tests/spec remains untouched.
- N-2 / spec 7.3, 12.1: float32 derivative atol/rtol are now **1e-3**, retaining
  the finite-difference rounding-scale comment. The exact hand-derived
  gradient assertion and second-order forward/reverse `check_grads` pass in
  both dtype modes. Float64 tolerance is unchanged.
- N-3: wrapped docs/index.md's feature line and the declaration error string.
- Clamping: retained JAX's behavior and documented nearest-entry clamping for
  out-of-bounds integer indices in docs/distributions.md.
- Concrete example: complete constant-hazard/scaled-hazard families construct
  a rate-2 baseline with factors `[1, 3]`; broadcasting, log-survival `[-2, -6]`,
  indexing and unchanged baseline shape are executed via doctest in both modes.

## Checks actually run

All uv commands used `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`.

- `scripts/check.sh --through A3`: **all gates passed through A3**.
  Guard **ok (A4.3)**; **35 files already formatted**; Ruff **All checks passed!**;
  mypy **Success: no issues found in 8 source files**.
  Own x64: **179 passed, 6 skipped in 8.85s**.
  Own x32: **184 passed, 1 skipped in 7.80s**.
  Prior independent x64: **372 passed, 15 skipped, 203 deselected in 11.42s**.
  Prior independent x32: **382 passed, 5 skipped, 203 deselected in 11.22s**.
- The same 89 focused A4 acceptance nodes as the original step, listed below:
  `JAX_ENABLE_X64=<mode> SPEC_THROUGH=A4 uv run --locked --offline pytest -q $(cat /private/tmp/survjax-a43-nodes.txt)`.
  x64: **87 passed, 2 skipped in 0.74s**.
  x32: **86 passed, 3 skipped in 0.61s**.
- Combined session, in each separate dtype process:
  `JAX_ENABLE_X64=<mode> SPEC_THROUGH=A3 uv run --locked --offline pytest tests/unit tests/spec -q`.
  x64: **551 passed, 21 skipped, 203 deselected in 16.64s**.
  x32: **566 passed, 6 skipped, 203 deselected in 14.21s**.
  No tolerance import clash. The combined run includes the tightened
  derivative check alongside the independent suites.
- All skips above are existing dtype-specific skips. Focused A4 x64 skips
  require x32 warning behavior; x32 skips require float64/mixed nested dtypes.
  The 203 A4 nodes are deselected in through-A3 runs. No acceptance edits,
  added skips or xfails.
- `scripts/build_docs.sh`: **build succeeded** in x32 and x64; all three
  executable notebooks per mode ran successfully. Used approved escalation
  for local Jupyter kernel ports. Jupyter emitted its TCP encryption warning.
  No visual browser inspection was repeated; the preceding attempt reported
  that the in-app browser was unavailable.
- Executed `distribution.__doc__` with `doctest.DocTestParser` and
  `doctest.DocTestRunner` via `uv run --locked --offline python -c ...`:
  **TestResults(failed=0, attempted=13)** in each separate x64/x32 process.
- `git diff --check`: passed with no output after the final bookkeeping edits.

## Remaining scope

D-015 records the review-directed TypeError policy; it remains open for the
maintainer to resolve. The source review leaves the remaining CLAUDE.md
wording fix to the maintainer; protected role files were not edited.
D-013 remains an A6 handoff. No spec, independent-test or dependency changes.
No mutation run, full A4 gate, wheel/R checks or likelihood debug-nan run.
Direct-method defaults remain A4.4–A4.5. Stop for further review; do not commit.

## Exact focused acceptance nodes

```text
tests/spec/test_a4_registration.py::test_extension_names_are_exported_at_top_level
tests/spec/test_a4_registration.py::test_registered_family_is_a_survival_distribution
tests/spec/test_a4_registration.py::test_positional_parameters_are_rejected
tests/spec/test_a4_registration.py::test_single_positional_parameter_is_rejected
tests/spec/test_a4_registration.py::test_positional_static_is_rejected
tests/spec/test_a4_registration.py::test_missing_parameter_is_rejected
tests/spec/test_a4_registration.py::test_kw_only_recipe_allows_static_before_parameter
tests/spec/test_a4_registration.py::test_parameters_are_the_only_leaves
tests/spec/test_a4_registration.py::test_leaves_hold_the_natural_parameter_values
tests/spec/test_a4_registration.py::test_flatten_unflatten_round_trip_preserves_type_and_parameters
tests/spec/test_a4_registration.py::test_unflatten_bypasses_init
tests/spec/test_a4_registration.py::test_static_field_is_preserved_through_round_trip
tests/spec/test_a4_registration.py::test_static_default_is_used_when_omitted
tests/spec/test_a4_registration.py::test_equal_static_values_give_equal_tree_structures
tests/spec/test_a4_registration.py::test_different_static_values_give_different_tree_structures
tests/spec/test_a4_registration.py::test_no_automatic_subclass_registration
tests/spec/test_a4_registration.py::test_distribution_is_immutable
tests/spec/test_a4_registration.py::test_distribution_passes_through_jit
tests/spec/test_a4_registration.py::test_new_parameter_values_reuse_compilation
tests/spec/test_a4_registration.py::test_different_static_values_compile_separately
tests/spec/test_a4_registration.py::test_static_value_selects_behaviour_under_jit
tests/spec/test_a4_registration.py::test_nested_distribution_parameters_are_leaves_of_the_outer_family
tests/spec/test_a4_registration.py::test_nested_distribution_round_trips_with_its_type
tests/spec/test_a4_registration.py::test_nested_distribution_is_differentiable_through_jit
tests/spec/test_a4_parameters.py::test_scalar_parameters_give_empty_batch_shape
tests/spec/test_a4_parameters.py::test_batch_shape_is_the_parameter_shape
tests/spec/test_a4_parameters.py::test_batch_shape_excludes_component_axes
tests/spec/test_a4_parameters.py::test_multidimensional_batch_shape
tests/spec/test_a4_parameters.py::test_batch_shape_is_a_tuple_of_python_ints
tests/spec/test_a4_parameters.py::test_batch_shape_is_static_under_jit
tests/spec/test_a4_parameters.py::test_component_parameter_with_incompatible_batch_is_rejected
tests/spec/test_a4_parameters.py::test_batch_dimensions_are_broadcast_at_construction
tests/spec/test_a4_parameters.py::test_broadcasting_keeps_parameter_values
tests/spec/test_a4_parameters.py::test_scalar_parameter_is_broadcast_to_batch_shape
tests/spec/test_a4_parameters.py::test_component_axis_stays_trailing_when_batch_is_broadcast
tests/spec/test_a4_parameters.py::test_component_axis_is_not_broadcast_against_batch
tests/spec/test_a4_parameters.py::test_batch_shape_can_come_from_a_component_parameter
tests/spec/test_a4_parameters.py::test_incompatible_batch_shapes_are_rejected
tests/spec/test_a4_parameters.py::test_weibull_style_shape_keyword_is_the_parameter
tests/spec/test_a4_parameters.py::test_distribution_without_shape_field_has_no_shape_attribute
tests/spec/test_a4_parameters.py::test_missing_shape_attribute_raises_attribute_error
tests/spec/test_a4_parameters.py::test_constructor_accepts_negative_parameter_eagerly
tests/spec/test_a4_parameters.py::test_constructor_accepts_nan_and_infinite_parameters_eagerly
tests/spec/test_a4_parameters.py::test_constructor_accepts_invalid_parameters_under_jit
tests/spec/test_a4_parameters.py::test_traced_construction_broadcasts_like_eager
tests/spec/test_a4_parameters.py::test_integer_parameters_use_default_float_dtype
tests/spec/test_a4_parameters.py::test_python_scalar_parameters_use_default_float_dtype
tests/spec/test_a4_parameters.py::test_python_list_parameter_behaves_as_jnp_asarray
tests/spec/test_a4_parameters.py::test_float16_parameter_is_promoted_to_float32
tests/spec/test_a4_parameters.py::test_python_scalar_adopts_float32_array_dtype
tests/spec/test_a4_parameters.py::test_mixed_float_parameters_share_promoted_dtype
tests/spec/test_a4_parameters.py::test_component_parameter_shares_the_common_dtype
tests/spec/test_a4_parameters.py::test_float64_parameter_converts_to_float32_under_32bit_mode
tests/spec/test_a4_parameters.py::test_float64_parameters_warn_once_per_call
tests/spec/test_a4_parameters.py::test_python_scalar_parameters_convert_without_warning
tests/spec/test_a4_parameters.py::test_boolean_parameter_is_rejected
tests/spec/test_a4_parameters.py::test_complex_parameter_is_rejected
tests/spec/test_a4_parameters.py::test_string_parameter_is_rejected
tests/spec/test_a4_parameters.py::test_static_field_is_not_converted
tests/spec/test_a4_parameters.py::test_traced_parameters_keep_their_dtype
tests/spec/test_a4_parameters.py::test_object_parameter_is_rejected
tests/spec/test_a4_parameters.py::test_constructor_takes_no_dtype_keyword
tests/spec/test_a4_parameters.py::test_nested_distribution_keeps_its_own_dtype
tests/spec/test_a4_indexing.py::test_integer_index_gives_scalar_batch
tests/spec/test_a4_indexing.py::test_slice_index_keeps_selected_entries
tests/spec/test_a4_indexing.py::test_integer_array_index_selects_entries
tests/spec/test_a4_indexing.py::test_first_axis_index_of_two_dimensional_batch
tests/spec/test_a4_indexing.py::test_second_axis_index_of_two_dimensional_batch
tests/spec/test_a4_indexing.py::test_full_index_of_two_dimensional_batch
tests/spec/test_a4_indexing.py::test_indexing_keeps_component_axes
tests/spec/test_a4_indexing.py::test_indexing_preserves_static_fields
tests/spec/test_a4_indexing.py::test_tree_map_slicing_updates_batch_shape
tests/spec/test_a4_indexing.py::test_tree_map_slicing_works
tests/spec/test_a4_indexing.py::test_tree_map_slice_matches_index_slice
tests/spec/test_a4_indexing.py::test_batch_shape_inside_vmap_is_the_per_entry_shape
tests/spec/test_a4_indexing.py::test_vmap_over_component_family_keeps_components_inside
tests/spec/test_a4_indexing.py::test_jit_of_indexing_with_traced_index
tests/spec/test_a4_nested.py::test_outer_batch_shape_includes_parameter_batch_with_scalar_base
tests/spec/test_a4_nested.py::test_scalar_base_is_broadcast_to_outer_batch
tests/spec/test_a4_nested.py::test_outer_batch_shape_comes_from_nested_distribution
tests/spec/test_a4_nested.py::test_nested_and_parameter_batches_broadcast_together
tests/spec/test_a4_nested.py::test_nested_component_axis_stays_trailing
tests/spec/test_a4_nested.py::test_incompatible_nested_and_parameter_batches_are_rejected
tests/spec/test_a4_nested.py::test_broadcast_nested_distribution_keeps_its_dtype
tests/spec/test_a4_nested.py::test_nested_static_fields_are_part_of_outer_structure
tests/spec/test_a4_nested.py::test_distribution_helper_takes_no_arguments
tests/spec/test_a4_nested.py::test_indexing_applies_to_nested_leaves
tests/spec/test_a4_nested.py::test_indexing_a_broadcast_scalar_base
tests/spec/test_a4_nested.py::test_tree_map_slicing_applies_to_nested_leaves
```
