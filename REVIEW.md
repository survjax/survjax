# A4.3 review — nested distributions and batch indexing

One implementation step on `A4.3`, based on specification revision 19.

## Reading order

1. `src/survjax/distributions/base.py`: `distribution()`, leaf-derived
   `batch_shape`, batch-only `__getitem__`, and nested construction/PyTree roles.
2. `tests/unit/test_family_batching.py` and `tests/tolerances.py`.
3. Public exports/import smoke test, then `docs/distributions.md` and
   `docs/index.md` for the executable nested/batching example.
4. The active A4.3 entry in `PLAN.md`, then evidence below.

## Invariants and requirement-to-test mapping

- Spec 4.2: nested fields are required, with a no-argument declaration helper.
  `test_nested_field_is_required_and_helper_takes_no_arguments` covers both.
  The only new exported name is `distribution`, from both public namespaces.
- Spec 3.3/4.2: only own parameters undergo dtype normalization. Broadcasting
  nested leaves preserves their original dtypes, trailing components, types
  and static configuration, without reconstructing through public constructors.
  `test_nested_dtype_survives_broadcast_and_index` covers float32 nested leaves
  alongside float64 own leaves in x64; x32 checks its available floating dtype.
- Spec 4.1–4.2/5: all dynamic leaves share leading batch axes after construction,
  including at multiple nesting levels. Broadcasting appends the existing
  component shape after the common batch shape. Inputs remain immutable.
  `test_two_levels_broadcast_components_without_mutating_baseline` covers
  `(2,1)` with `(3,)`, two nesting levels, two component axes and incompatibility.
  Focused acceptance additionally covers scalar/vector baselines and wrappers.
- Spec 4.2: nested families are children, not static objects; JAX recursively
  flattens their parameter leaves and includes nested static values in structure.
  `test_nested_round_trip_preserves_static_fields_and_sentinel_identity` proves
  unflatten bypasses constructors even with sentinel leaves. Acceptance covers
  nested family round trips; the two-level own case covers a nested-only wrapper.
- Spec 4.1–4.2: indexing resolves the batch rank before appending full slices
  for each leaf's component axes. Integers, slices, arrays, tuples, ellipsis,
  None and boolean indices follow JAX rules; components cannot be consumed.
  `test_indexing_preserves_two_component_axes_and_nested_configuration` has
  nine explicit cases, comparing with a hand-built NumPy batch grid.
  `test_index_cannot_consume_component_axes` covers scalar-batch rejection
  and scalar ellipsis. No custom out-of-bounds behavior is introduced.
- Spec 4.1/7.2: `batch_shape` follows current own and nested leaves; it is not
  stored auxiliary data. `test_nested_tree_slicing_and_vmap_use_current_batch_shapes`
  covers tree slicing, vmap per-entry shapes/full components, hand-derived
  log-survival values and a jitted traced integer index.
- Spec 7.2: `test_nested_construction_transforms_and_compilation_reuse` changes
  own and nested dynamic values at fixed shapes/dtypes/structure and observes
  one trace; changing nested static configuration observes a second trace.
  Focused acceptance separately exercises construction under jit/vmap.
- Spec 7.3: `test_gradients_through_nested_broadcast_and_selection` checks the
  hand-derived gradients of `f(r,m)=-2*r*m` through construction/broadcasting/
  indexing, under jit, and second-order forward/reverse `check_grads`.
  Finite-difference tolerances have their scale rationale in `tests/tolerances.py`.

The code/tests change is about 300 lines; the guide adds roughly 30 more.
`base.py` is 280 lines; no private registration module or dependencies added.

## Checks actually run

All uv commands used `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`.

- Final `scripts/check.sh --through A3`: **all gates passed through A3**.
  Guard: **ok (A4.3)**; **35 files already formatted**; Ruff **All checks passed!**;
  mypy **Success: no issues found in 8 source files**.
  Own x64: **175 passed, 6 skipped in 10.10s**.
  Own x32: **180 passed, 1 skipped in 8.93s**.
  Prior independent x64: **372 passed, 15 skipped, 203 deselected in 12.13s**.
  Prior independent x32: **382 passed, 5 skipped, 203 deselected in 11.32s**.
  Own skips are existing dtype-specific cases: x32 conversion/unavailable-dtype
  checks in x64, and float64 promotion in x32. New A4.3 cases run in both modes.
  Final own test runs emitted no warnings.
- Final focused A4 acceptance, 89 nodes listed below, separately in both modes:
  `JAX_ENABLE_X64=<mode> SPEC_THROUGH=A4 uv run --locked --offline pytest -q $(cat /private/tmp/survjax-a43-nodes.txt)`.
  x64: **87 passed, 2 skipped in 0.92s**.
  x32: **86 passed, 3 skipped in 0.65s**.
  x64 skips require x32 conversion-warning behavior; x32 skips require
  float64 promotion or mixed nested float32/float64.
  Nine unselected nodes in these four files depend on later numerical defaults.
  No acceptance files/markers/skips/xfails were changed.
- Final `scripts/build_docs.sh`: **build succeeded** for both x32 and x64,
  executing the distribution guide, playground and survival-data notebooks.
  The first sandboxed attempt failed because Jupyter could not bind a local
  kernel port (`PermissionError: Operation not permitted`). Approved escalation
  allowed the build. Jupyter printed its TCP encryption warning.
  Rendered HTML text in both modes contains `ScaledRecipe` and the paired/
  all-pairs examples. Visual browser inspection was unavailable: the CUA
  in-app browser returned **Browser is not available: iab**.
- `git diff --check`: passed with no output before commit.

Initial focused own runs found two incorrect test inputs (a missing singleton
batch axis and an out-of-bounds NumPy index); both were corrected. An initial
lint run found an unbound loop variable in the broadcasting lambda, now bound
explicitly. An earlier successful gate preceded test simplification, removal
of incidental float64-input warnings and the scalar-boolean indexing fix;
the final gate and focused acceptance results above cover those refinements.

## Remaining scope and uncertainties

No new spec ambiguity was encountered. D-013 remains open for the A6
parameter-diagnostic contract; D-014 retains its maintainer-approved declaration
policy and open status. No spec/deviation changes in this step.
Direct-method defaults/policy remain A4.4–A4.5, built-ins/grids remain A5,
and actual PH/AFT wrappers remain later scope. The full A4 gate, wheel/R
fixture checks and likelihood debug-nan run were not run for this step.
Stop after this commit and await review; do not start A4.4.

## Exact focused acceptance nodes

The temporary node file was generated by reading the four acceptance modules'
ASTs and selecting their top-level test functions, excluding the nine names
specified in PLAN.md. Its exact final contents are:

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
