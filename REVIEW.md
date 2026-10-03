# A4.4 review — direct-method policy and survival/CDF defaults

Implemented on `A4.4` against spec revision 19. Changes remain uncommitted
as requested. No new top-level exports.

## Reading order

1. `src/survjax/distributions/_direct.py`: defaults, safe log-CDF calculation,
   and the registration-time policy wrapper.
2. `src/survjax/distributions/base.py`: explicit default bindings and wrapping
   of log_survival/survival/log_cdf/cumulative_hazard at registration.
3. `tests/unit/test_family_direct.py`, value tolerances in
   `tests/impl_tolerances.py`, and the batching fixture's reference hook.
4. Executable docs/distributions.md, docs index, and the active PLAN.md entry.

## Invariants and tests

- Spec 4.2: defaults are exp(log S), -log S and log(-expm1(log S)). They call
  the preserved raw primitive after the outer wrapper substitutes safe times,
  avoiding recursive dispatch and repeated library policy application.
  `test_mixed_time_policy_eager_jit_vmap` uses hand-derived exponential values
  for all four methods and all time-policy rows, eager/jit/vmap.
- Spec 4.5/7.2: broadcast first, substitute negative/NaN inputs with the
  family's reference under stop_gradient before family calculations, then
  select negative-time constants and NaN outputs. Active parameters are never
  stopped. `test_safe_reference_substitution_and_inactive_gradients` uses
  an unguarded sqrt primitive and parameter-dependent reference; negative and
  NaN times have zero parameter gradients under jit. Valid family parameters
  require a positive finite reference; no domain/reference validator is added.
- Spec 4.5/7.1, D-013: preserve numerical family NaNs without inferring custom
  parameter domains or adding issue masks.
  `test_nan_parameters_propagate_without_poisoning_valid_entries` tests NaN
  parameters beside valid entries at negative and positive times.
- Spec 7.2: for zero CDF, substitute a negative log-S constant before log,
  then return -inf. No floors, clipping or epsilons.
  `test_log_cdf_tiny_mass_against_mpmath` uses an independent 100-digit
  mpmath log-CDF at tiny mass, checks -inf at zero and its parameter gradient.
  The generic lower-tail underflow limit is documented, not repaired.
- Spec 4.2/4.5: defaults and overrides are wrapped. A private function marker
  retains raw methods so a decorated subclass does not stack inherited policy
  wrappers. `test_overrides_and_decorated_subclasses_dispatch_without_primitive`
  checks each derived override at positive/negative/NaN times, raw-marker
  structure, and dispatch with the primitive patched to raise if called.
- Spec 5: `test_paired_and_draw_broadcasting_and_shape_misuse` checks paired
  vectors, explicit (D,1) broadcasting and incompatible-shape TypeError under jit.
- Spec 7.3/12.1: `test_second_order_parameter_derivatives_and_purity` checks
  forward/reverse derivatives through order two on smooth interior cases
  for all four methods and verifies leaf identities remain unchanged.
  Existing compilation and PyTree tests pass after registration integration.
- The batching fixture and both docs recipes now supply required reference
  hooks. Docs execute cumulative hazards [6,12,18], negative-time survival
  [1,1,1], and paired/all-pairs shapes. The concrete nested-field docstring stays.

About 290 changed source/test lines plus the small guide update, excluding
plan/review bookkeeping. The base is 320 lines including its existing concrete
example; the private numerical module keeps this step separate from registration.
No dependencies, global JAX config, protected paths or spec changes.

## Checks actually run

All uv commands used UV_CACHE_DIR=/private/tmp/survjax-uv-cache. Final results
include raw-primitive composition in the defaults.

- `scripts/check.sh --through A3`: **all gates passed through A3**.
  Guard **ok (A4.4)**; **37 files already formatted**; Ruff **All checks passed!**;
  mypy **Success: no issues found in 9 source files**.
  Own x64: **200 passed, 6 skipped in 9.80s**.
  Own x32: **205 passed, 1 skipped in 9.23s**.
  Independent x64: **372 passed, 15 skipped, 203 deselected in 9.65s**.
  Independent x32: **382 passed, 5 skipped, 203 deselected in 9.33s**.
- 112 focused A4 nodes listed below: previous 89 plus 16 generic-default nodes
  and seven primitive/derived policy nodes.
  `JAX_ENABLE_X64=<mode> SPEC_THROUGH=A4 uv run --locked --offline pytest -q $(cat /private/tmp/survjax-a44-nodes.txt)`.
  x64: **110 passed, 2 skipped in 1.37s**.
  x32: **109 passed, 3 skipped in 1.27s**.
- Combined `JAX_ENABLE_X64=<mode> SPEC_THROUGH=A3 uv run --locked --offline pytest tests/unit tests/spec -q`:
  x64: **572 passed, 21 skipped, 203 deselected in 15.87s**.
  x32: **587 passed, 6 skipped, 203 deselected in 15.14s**.
- Skips are existing dtype-specific cases. Focused x64 skips require x32
  warnings; x32 skips require float64 promotion/mixed nested dtypes. The new
  21 direct-method cases run in both modes without warnings. No acceptance
  edits/skips/xfails were introduced; the full 203 A4 nodes remain deselected
  by SPEC_THROUGH=A3 in gate/combined sessions.
- `scripts/build_docs.sh`: **build succeeded** in x32 and x64, executing all
  three notebooks per mode. Approved escalation permits Jupyter kernel ports;
  kernels print their TCP encryption warning. Rendered HTML text in both modes
  contains the new values and precision-limit paragraph. No visual inspection.
- Concrete nested-field docstring run with doctest.DocTestParser/DocTestRunner
  in separate x64/x32 `uv run --locked --offline python -c ...` processes:
  **TestResults(failed=0, attempted=13)** each.
- `git diff --check`: passed with no output after final bookkeeping edits.

Initial focused x32 result: **2 failed, 19 passed**, from an incorrect exact
comparison of JAX float32 and NumPy float64 transcendental results. Corrected
that assertion to use the mandated shared dtype tolerances; no shared tolerance
was loosened. An earlier gate passed before raw-primitive composition; final
checks above were rerun for that change. The first rendered-text probe inserted
spaces between HTML text chunks and failed its literal-value search; joining
chunks without extra spaces verified the same generated content in both modes.

## Remaining scope

No new spec ambiguity. Existing deviations unchanged; D-013 remains an A6
handoff. Density/hazard methods and their policy/defaults stay A4.5; built-ins/
grids stay A5. No full A4 gate, likelihood debug-nan run, R fixtures, wheel smoke
checks, mutation run or performance measurements were run for this step.
Stop for review; leave changes uncommitted and do not start A4.5.

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
tests/spec/test_a4_generic_defaults.py::test_default_survival_is_exp_log_survival
tests/spec/test_a4_generic_defaults.py::test_default_cumulative_hazard_is_minus_log_survival
tests/spec/test_a4_generic_defaults.py::test_default_log_cdf
tests/spec/test_a4_generic_defaults.py::test_default_log_cdf_keeps_precision_for_tiny_cdf
tests/spec/test_a4_generic_defaults.py::test_negative_time_survival_is_one
tests/spec/test_a4_generic_defaults.py::test_negative_time_log_cdf_is_minus_infinity
tests/spec/test_a4_generic_defaults.py::test_negative_time_cumulative_hazard_is_zero
tests/spec/test_a4_generic_defaults.py::test_zero_time_survival_cumulative_hazard_and_log_cdf
tests/spec/test_a4_generic_defaults.py::test_infinite_time_survival_is_zero
tests/spec/test_a4_generic_defaults.py::test_infinite_time_cumulative_hazard_is_infinite
tests/spec/test_a4_generic_defaults.py::test_infinite_time_log_cdf_is_zero
tests/spec/test_a4_generic_defaults.py::test_nan_time_survival_is_nan
tests/spec/test_a4_generic_defaults.py::test_nan_time_cumulative_hazard_is_nan
tests/spec/test_a4_generic_defaults.py::test_nan_time_log_cdf_is_nan
tests/spec/test_a4_generic_defaults.py::test_invalid_parameter_gives_nan_survival_and_log_cdf_defaults
tests/spec/test_a4_generic_defaults.py::test_defaults_agree_under_jit
tests/spec/test_a4_policy_enforcement.py::test_unguarded_log_survival_is_zero_for_negative_time
tests/spec/test_a4_policy_enforcement.py::test_unguarded_survival_is_one_for_negative_time
tests/spec/test_a4_policy_enforcement.py::test_unguarded_cumulative_hazard_is_zero_for_negative_time
tests/spec/test_a4_policy_enforcement.py::test_unguarded_log_cdf_is_minus_infinity_for_negative_time
tests/spec/test_a4_policy_enforcement.py::test_unguarded_infinite_time_log_survival_comes_from_the_family
tests/spec/test_a4_policy_enforcement.py::test_negative_time_log_survival_has_zero_parameter_gradients
tests/spec/test_a4_policy_enforcement.py::test_negative_time_survival_has_zero_parameter_gradients
```
