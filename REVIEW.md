# A4.4 completion review — specification revision 21

Implemented on `A4.4`, continuing maintainer commit `8326b45` after the spec
and acceptance updates merged at `d0b5538`. The maintainer authorized this
completion commit on 2026-10-08. No public API additions, dependencies or
protected-file edits.

## Reading order

1. `src/survjax/distributions/_survival_methods.py`: renamed from `_direct.py`;
   read dtype rejection and safe-time substitution in `apply_policy` first,
   then `_raw_log_survival` and the three defaults.
2. `src/survjax/distributions/base.py`: updated import, default bindings and
   registration calls. Construction and PyTree behavior are unchanged.
3. `tests/unit/test_family_direct.py`: the four new test functions at the end,
   followed by the existing numerical and transformation coverage.
4. `docs/distributions.md`, `DEVIATIONS.md` D-016 through D-018, and the active
   completion status in `PLAN.md`.

## Invariants and requirement evidence

- Spec 4.5 revision 21: times follow JAX conversion/promotion, but boolean,
  complex, string and object dtypes raise TypeError. Inspection uses dtype,
  never tracer values. `test_direct_times_reject_dtype_misuse` covers all four
  methods and six inputs; `test_direct_dtype_rejection_works_on_tracers`
  checks boolean/complex inputs under both jit and vmap.
- Spec 4.5: transient float64 times convert silently under x32, unlike stored
  parameters/data. `test_float64_times_follow_promotion_without_warning`
  checks no warning, output dtype and hand-derived values `-2*(1,2)` in both
  modes. New acceptance cases cover integer times and float64 promotion too.
- Spec 4.5/7.2: negative/NaN times are substituted before family evaluation;
  only the substituted reference is stopped. Zero and positive infinity
  reach the family. `test_zero_time_reaches_family_time_derivative` pins the
  exact derivative `d(-2*t)/dt = -2` at zero, eagerly and under jit; changing
  the negative-time predicate to include zero would fail it. Existing Root
  tests demonstrate safe sqrt inputs and zero inactive parameter gradients.
- Spec 4.5 revision 21: NaN family values at substituted inputs take precedence
  over negative-time constants. No family validity domain is inferred; finite
  invalid parameters need not yield NaN. Existing own NaN-isolation tests and
  the new acceptance cases demonstrate negative time, minus infinity, NaN
  time, unguarded formulas and mixed valid/NaN batches.
- Spec 4.2/7.2: defaults call the raw primitive on already-substituted times.
  The typed `_raw_log_survival` helper centralizes this invariant. Raw method
  markers prevent stacked inherited wrappers. Existing override-dispatch
  tests check that inherited overrides bypass the primitive.
- Spec 4.2: survival is exp(log S), cumulative hazard is -log S, log CDF is
  log(-expm1(log S)). Zero CDF is guarded before log; there are no floors,
  epsilons or clipping. Existing 100-digit mpmath and zero-gradient tests
  verify tiny mass and zero CDF. The generic lower-tail limit is documented.
- Spec 5/7.3: existing paired/draw broadcasting and shape-error tests, mixed
  eager/jit/vmap values, and order-two forward/reverse parameter checks pass.
  Existing registration, nesting, indexing and compilation checks pass too.

## Checks actually run

All uv commands used `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Dtype modes
ran in separate processes.

- `scripts/check.sh --through A3`: exit 0, **all gates passed through A3**.
  Guard **ok (A4.4)**; Ruff **37 files already formatted**, **All checks
  passed!**; mypy **Success: no issues found in 9 source files**.
  Own x64: **234 passed, 6 skipped in 9.52s**.
  Own x32: **239 passed, 1 skipped in 9.44s**.
  Independent x64: **372 passed, 15 skipped, 314 deselected in 9.60s**.
  Independent x32: **382 passed, 5 skipped, 314 deselected in 9.40s**.
- Exact 172 focused A4 nodes listed below: previous 112 plus 60 new in-scope
  time-dtype/NaN-precedence nodes. With `SPEC_THROUGH=A4`,
  `uv run --locked --offline pytest -q <listed node IDs>`:
  x64: **165 passed, 7 skipped in 1.46s**.
  x32: **168 passed, 4 skipped in 1.42s**.
  Skips retain the independent suite's dtype-specific requirements. No
  acceptance tests were edited, skipped or xfailed by this implementation.
- Combined `SPEC_THROUGH=A3 uv run --locked --offline pytest tests/unit tests/spec -q`:
  x64: **606 passed, 21 skipped, 314 deselected in 16.08s**.
  x32: **621 passed, 6 skipped, 314 deselected in 15.25s**.
- `scripts/build_docs.sh`: exit 0; **build succeeded** for x32 and x64,
  executing all three notebooks per mode. Approved escalation allowed local
  Jupyter kernel ports; kernels printed their TCP encryption warning. Both
  generated guides contain the added dtype and NaN-policy text. No visual
  browser inspection was performed.
- Source doctests via `pytest --doctest-modules` on `base.py` and
  `_survival_methods.py`: x64 **1 passed in 0.50s**, x32 **1 passed in 0.53s**.
  The concrete nested-family example remains intact.
- `git diff --check`: exit 0, no output, after the final bookkeeping edits.

One initial lint run rejected an assigned lambda (E731); replaced it with a
plain function before the passing gate. A temporary selector-generation command
also referenced an absent old manifest. The shell then launched unfiltered
pytest instead of the intended focused selection; that attempt failed on
unfinished density/hazard methods (last x32 summary: **134 failed, 795 passed,
12 skipped**). This attempt is not acceptance evidence. Recovered the exact
112 prior selectors from the previous REVIEW.md, added the 60 new selectors,
and reran the intended selection successfully with a fail-fast shell.

## Limits and handoff

D-016 is recorded verbatim as reviewed: generic survival/log-CDF parameter
gradients at positive infinity can be NaN despite finite values. The maintainer
explicitly deferred this fix to A5; it is unchanged here. D-017/D-018 now have
normative spec text and acceptance coverage, but their drafted open statuses
are retained for maintainer closure. No new unresolved implementation choice.

Density/hazard defaults and policy remain A4.5; built-ins/grids remain A5.
No full A4 gate, new mutation run, performance measurements, likelihood
jax_debug_nans run, R fixtures or wheel smoke test was completed for this step.
The two new untracked acceptance files remain untouched and unstaged.
Commit this completion as authorized, then stop before A4.5.

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
tests/spec/test_a4_time_dtypes.py::test_python_bool_time_is_rejected[log_survival]
tests/spec/test_a4_time_dtypes.py::test_python_bool_time_is_rejected[survival]
tests/spec/test_a4_time_dtypes.py::test_python_bool_time_is_rejected[log_cdf]
tests/spec/test_a4_time_dtypes.py::test_python_bool_time_is_rejected[cumulative_hazard]
tests/spec/test_a4_time_dtypes.py::test_numpy_boolean_array_time_is_rejected[log_survival]
tests/spec/test_a4_time_dtypes.py::test_numpy_boolean_array_time_is_rejected[survival]
tests/spec/test_a4_time_dtypes.py::test_numpy_boolean_array_time_is_rejected[log_cdf]
tests/spec/test_a4_time_dtypes.py::test_numpy_boolean_array_time_is_rejected[cumulative_hazard]
tests/spec/test_a4_time_dtypes.py::test_jax_boolean_array_time_is_rejected[log_survival]
tests/spec/test_a4_time_dtypes.py::test_jax_boolean_array_time_is_rejected[survival]
tests/spec/test_a4_time_dtypes.py::test_jax_boolean_array_time_is_rejected[log_cdf]
tests/spec/test_a4_time_dtypes.py::test_jax_boolean_array_time_is_rejected[cumulative_hazard]
tests/spec/test_a4_time_dtypes.py::test_complex_array_time_is_rejected[log_survival]
tests/spec/test_a4_time_dtypes.py::test_complex_array_time_is_rejected[survival]
tests/spec/test_a4_time_dtypes.py::test_complex_array_time_is_rejected[log_cdf]
tests/spec/test_a4_time_dtypes.py::test_complex_array_time_is_rejected[cumulative_hazard]
tests/spec/test_a4_time_dtypes.py::test_python_complex_time_is_rejected[log_survival]
tests/spec/test_a4_time_dtypes.py::test_python_complex_time_is_rejected[survival]
tests/spec/test_a4_time_dtypes.py::test_python_complex_time_is_rejected[log_cdf]
tests/spec/test_a4_time_dtypes.py::test_python_complex_time_is_rejected[cumulative_hazard]
tests/spec/test_a4_time_dtypes.py::test_string_time_is_rejected[log_survival]
tests/spec/test_a4_time_dtypes.py::test_string_time_is_rejected[survival]
tests/spec/test_a4_time_dtypes.py::test_string_time_is_rejected[log_cdf]
tests/spec/test_a4_time_dtypes.py::test_string_time_is_rejected[cumulative_hazard]
tests/spec/test_a4_time_dtypes.py::test_object_array_time_is_rejected[log_survival]
tests/spec/test_a4_time_dtypes.py::test_object_array_time_is_rejected[survival]
tests/spec/test_a4_time_dtypes.py::test_object_array_time_is_rejected[log_cdf]
tests/spec/test_a4_time_dtypes.py::test_object_array_time_is_rejected[cumulative_hazard]
tests/spec/test_a4_time_dtypes.py::test_boolean_time_is_rejected_under_jit
tests/spec/test_a4_time_dtypes.py::test_boolean_time_is_rejected_under_jit_for_every_method[log_survival]
tests/spec/test_a4_time_dtypes.py::test_boolean_time_is_rejected_under_jit_for_every_method[survival]
tests/spec/test_a4_time_dtypes.py::test_boolean_time_is_rejected_under_jit_for_every_method[log_cdf]
tests/spec/test_a4_time_dtypes.py::test_boolean_time_is_rejected_under_jit_for_every_method[cumulative_hazard]
tests/spec/test_a4_time_dtypes.py::test_boolean_time_is_rejected_under_vmap
tests/spec/test_a4_time_dtypes.py::test_float64_time_converts_without_warning_under_32bit_mode[log_survival]
tests/spec/test_a4_time_dtypes.py::test_float64_time_converts_without_warning_under_32bit_mode[survival]
tests/spec/test_a4_time_dtypes.py::test_float64_time_converts_without_warning_under_32bit_mode[log_cdf]
tests/spec/test_a4_time_dtypes.py::test_float64_time_converts_without_warning_under_32bit_mode[cumulative_hazard]
tests/spec/test_a4_time_dtypes.py::test_float64_time_result_is_float32_under_32bit_mode
tests/spec/test_a4_time_dtypes.py::test_float32_time_promotes_to_parameter_dtype_under_64bit_mode
tests/spec/test_a4_time_dtypes.py::test_integer_array_time_log_survival_matches_float_time
tests/spec/test_a4_time_dtypes.py::test_python_int_time_matches_float_time
tests/spec/test_a4_nan_precedence.py::test_nan_parameter_gives_nan_at_negative_time[log_survival]
tests/spec/test_a4_nan_precedence.py::test_nan_parameter_gives_nan_at_negative_time[survival]
tests/spec/test_a4_nan_precedence.py::test_nan_parameter_gives_nan_at_negative_time[log_cdf]
tests/spec/test_a4_nan_precedence.py::test_nan_parameter_gives_nan_at_negative_time[cumulative_hazard]
tests/spec/test_a4_nan_precedence.py::test_nan_parameter_gives_nan_at_minus_infinity[log_survival]
tests/spec/test_a4_nan_precedence.py::test_nan_parameter_gives_nan_at_minus_infinity[survival]
tests/spec/test_a4_nan_precedence.py::test_nan_parameter_gives_nan_at_minus_infinity[log_cdf]
tests/spec/test_a4_nan_precedence.py::test_nan_parameter_gives_nan_at_minus_infinity[cumulative_hazard]
tests/spec/test_a4_nan_precedence.py::test_nan_parameter_gives_nan_at_negative_time_for_unguarded_family[log_survival]
tests/spec/test_a4_nan_precedence.py::test_nan_parameter_gives_nan_at_negative_time_for_unguarded_family[survival]
tests/spec/test_a4_nan_precedence.py::test_nan_parameter_gives_nan_at_negative_time_for_unguarded_family[log_cdf]
tests/spec/test_a4_nan_precedence.py::test_nan_parameter_gives_nan_at_negative_time_for_unguarded_family[cumulative_hazard]
tests/spec/test_a4_nan_precedence.py::test_nan_batch_entry_does_not_change_valid_entry_policy_value[cumulative_hazard]
tests/spec/test_a4_nan_precedence.py::test_nan_batch_entry_does_not_change_valid_entry_policy_value[log_cdf]
tests/spec/test_a4_nan_precedence.py::test_nan_batch_entry_does_not_change_valid_entry_policy_value[log_survival]
tests/spec/test_a4_nan_precedence.py::test_nan_batch_entry_does_not_change_valid_entry_policy_value[survival]
tests/spec/test_a4_nan_precedence.py::test_nan_precedence_applies_elementwise_over_times
tests/spec/test_a4_nan_precedence.py::test_nan_parameter_with_nan_time_gives_nan
```
