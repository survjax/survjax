# Test map

Spec: performance amendment draft (2026-09-30), including the revision 10
diagnostic compatibility decision. Unchanged requirements keep their earlier
revision tags. Tests from the revision 10 decision (sections 6.2, 10.2, 13) are
tagged `rev 10`. Tests from the unaccepted performance amendment (section 7.2)
are tagged `rev perf-draft` and should be retagged with the final revision
number when the maintainer accepts it. Tests are static acceptance tests and have never
been executed against Chronox. Test names below are relative to `tests/spec/`.

## A1: representation and dtype policy (`test_a1_representation.py`, `test_a1_dtype.py`)

| Spec section | Requirement | Tests |
| --- | --- | --- |
| 3.2 | "normalize to five public fields, all with the observation shape" | `test_all_five_fields_have_observation_shape` |
| 3.2 | "`exact` bool" | `test_exact_field_is_boolean` |
| 3.2 | exact rows: `lower` is the exact time, "`upper` equal to `lower`" | `test_exact_row_upper_equals_lower` |
| 3.2 | right censoring: "`+inf` for right censoring" | `test_right_censored_row_has_infinite_upper` |
| 3.2 | left censoring: "`0` for left censoring" | `test_left_censored_row_has_zero_lower` |
| 3.2 | interval: "the set's lower endpoint", "the set's upper endpoint" | `test_interval_row_keeps_both_endpoints` |
| 3.2 | "no placeholder values: every field is meaningful on every row" | `test_mixed_kinds_normalize_row_by_row` |
| 3.2 | "`left_trunc` ... `0` when absent" | `test_absent_left_truncation_normalizes_to_zero` |
| 3.2 | "`right_trunc` ... `+inf` when absent" | `test_absent_right_truncation_normalizes_to_infinity` |
| 3.2 | `left_trunc` = L, `right_trunc` = U | `test_supplied_truncation_is_stored_per_row` |
| 3.2 | "Scalars stay scalar" | `test_scalar_time_gives_scalar_fields`, `test_scalar_time_gives_scalar_leaves` |
| 3.2 | "multidimensional observations are not flattened" | `test_multidimensional_observations_are_not_flattened` |
| 3.2 | "array leaves are the five fields plus one `int32` issue bitfield leaf with the observation shape" | `test_pytree_has_exactly_six_leaves_with_observation_shape`, `test_pytree_leaf_dtypes_are_bool_int32_and_four_floats`, `test_issue_bitfield_is_int32_for_scalar_data` |
| 3.2 | "immutable registered PyTree" | `test_data_passes_through_jit_unchanged`, `test_fields_are_usable_under_vmap`, `test_data_is_immutable` |
| 3.2 | "Reconstruction bypasses public normalization" | `test_unflatten_bypasses_public_normalization`, `test_fields_are_usable_under_vmap` |
| 3.3 | "normalize to a common floating dtype using JAX promotion, at least float32" | `test_time_and_bound_fields_share_one_floating_dtype`, `test_float16_time_is_promoted_to_float32`, `test_bfloat16_time_is_promoted_to_float32`, `test_float32_time_with_int32_truncation_is_float32`, `test_float32_input_stays_float32_under_64bit_mode`, `test_float32_time_with_float64_truncation_promotes_to_float64`, `test_float64_input_is_kept_under_64bit_mode` |
| 3.3 | "integer-only ... input uses the configured default floating dtype" | `test_integer_time_uses_default_float_dtype`, `test_integer_time_and_truncation_use_default_float_dtype`, `test_integer_bounds_use_default_float_dtype`, `test_python_int_list_time_uses_default_float_dtype` |
| 3.3 | "scalar-only input uses the configured default floating dtype" | `test_python_float_scalar_time_uses_default_float_dtype`, `test_python_int_scalar_time_uses_default_float_dtype`, `test_python_scalar_time_and_truncation_use_default_float_dtype`, `test_python_scalar_bounds_use_default_float_dtype` |
| 3.3 | "including JAX's weak typing: Python scalars adopt the dtype of the array inputs" | `test_python_float_truncation_adopts_float32_time_dtype`, `test_python_int_truncation_adopts_float32_time_dtype`, `test_python_infinite_upper_adopts_float32_lower_dtype`, `test_python_float_truncation_with_integer_time_uses_default_float_dtype` |
| 3.3 | "Python lists behave as `jnp.asarray` of the list" | `test_python_float_list_time_uses_default_float_dtype`, `test_python_int_list_time_uses_default_float_dtype`, `test_python_list_bounds_use_default_float_dtype`, `test_python_list_truncation_promotes_as_jnp_asarray` |
| 3.3 | "Omitting a truncation argument and passing its default value give the same dtype" (default is `None`, Q-015 resolved) | `test_passing_none_truncation_matches_omitting_it` |
| 3.3 | weak typing applied to Python `0`/`+inf` truncation (the values stored when truncation is absent) | `test_passing_zero_left_truncation_matches_omitting_it_for_float32_time`, `test_passing_infinite_right_truncation_matches_omitting_it_for_float32_time`, `test_passing_default_truncation_matches_omitting_it_for_integer_time`, `test_passing_default_truncation_matches_omitting_it_in_from_bounds` |
| 3.3 | "Array input with dtype float64 (NumPy, pandas, JAX) when JAX's 64-bit mode is disabled is converted to float32 ... It is not an error" | `test_float64_input_converts_to_float32_under_32bit_mode`, `test_float64_bounds_convert_to_float32_under_32bit_mode`, `test_pandas_float64_series_converts_to_float32_with_one_warning` (Q-016 resolved) |
| 3.3 | "with one warning per call site" | `test_float64_input_warns_once_per_call`, `test_repeated_calls_from_one_call_site_warn_once`, `test_two_call_sites_warn_once_each`, `test_pandas_float64_series_converts_to_float32_with_one_warning` |
| 3.3 | "Python scalars and lists carry no chosen precision and convert without a warning" | `test_python_float_scalar_converts_without_warning`, `test_python_float_list_converts_without_warning`, `test_python_list_bounds_convert_without_warning`, `test_python_scalar_truncation_with_float32_array_does_not_warn` |
| 3.3 | "An explicit request for an unavailable dtype ... fails clearly" | `test_explicit_float64_request_fails_under_32bit_mode`, `test_explicit_float64_request_fails_in_from_bounds_under_32bit_mode` |
| 3.1, 3.3 | `dtype=` keyword honoured when available | `test_explicit_float32_request_gives_float32_fields`, `test_explicit_float64_request_gives_float64_fields_under_64bit_mode` |
| 3.3 | "Boolean times, complex values, strings and object or mixed-None arrays are rejected" | `test_boolean_time_is_rejected`, `test_boolean_time2_is_rejected`, `test_boolean_lower_bound_is_rejected`, `test_boolean_truncation_is_rejected`, `test_complex_time_is_rejected`, `test_complex_truncation_is_rejected`, `test_string_time_is_rejected`, `test_object_array_with_none_time_is_rejected` |
| 3.3 | "Chronox never changes JAX's global 64-bit setting" | `test_construction_does_not_change_global_64bit_setting`, `test_failed_float64_request_does_not_enable_64bit_mode` |

### A1: deliberately not tested

- **Warning text and category** for float64 conversion (CLAUDE.md: not
  asserted; Q-004 records how "call site" is read).
- **No warning for float32 arrays, or float64 under 64-bit mode:** implied
  but not stated by 3.3 (only Python scalars and lists are stated to convert
  without a warning). No-warning tests ignore `DeprecationWarning` and
  `PendingDeprecationWarning`, which third-party libraries may emit.
- **JAX float64 arrays under 32-bit mode:** named in 3.3, but a JAX array
  cannot hold float64 while 64-bit mode is disabled, and tests must not toggle
  the mode. NumPy and pandas cover the requirement.
- **pandas when it is not installed:** the pandas test skips
  (`pytest.importorskip`); the spec does not make pandas a dependency.
- **Explicit `dtype=float16` or integer dtype requests:** the spec does not say
  whether "at least float32" overrides an explicit request or rejects it.
- **The order of PyTree leaves and the name of the bitfield leaf:** not fixed by
  3.2. Tests identify the bitfield as the only integer leaf.
- **Parameter dtype policy, likelihood dtype, integer counts in KM/metrics**
  (3.3 bullets 4–5): these belong to distributions, likelihoods and metrics,
  outside A1–A3.
- **"float64 is recommended for reference ... work"; "Precision lost before
  values reach Chronox cannot be recovered":** guidance, not testable behaviour.

## A2: constructor and data requirements

### `test_a2_boundaries.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 3.3 | zero exact times are invalid; no epsilon is added. | `test_exact_time_zero_sets_issue` |
| 3.3 | negative exact times are invalid. | `test_negative_exact_time_sets_issue` |
| 3.3 | infinite exact times are invalid. | `test_infinite_exact_time_sets_issue` |
| 3.1 | missing required values are invalid; no implicit dropping. | `test_nan_exact_time_sets_issue` |
| 3.3 | exact times must be strictly positive; a tiny positive time is valid and unaltered. | `test_tiny_positive_exact_time_is_kept_unchanged` |
| 3.3 | censoring endpoints are nonnegative. | `test_negative_right_censoring_time_sets_issue` |
| 3.3 | censoring endpoints are nonnegative. | `test_negative_left_censoring_time_sets_issue` |
| 3.3 | censoring endpoints are nonnegative. | `test_negative_interval_lower_endpoint_sets_issue` |
| 3.3 | in the flag constructor, right censoring at +inf is invalid. | `test_right_censoring_at_infinity_sets_issue` |
| 3.3 | in the flag constructor, left censoring at +inf is invalid. | `test_left_censoring_at_infinity_sets_issue` |
| 3.1 | missing required values are invalid. | `test_nan_right_censoring_time_sets_issue` |
| 2/7.1 | left censoring at 0 is the empty set (0, 0], which is invalid. | `test_left_censoring_at_zero_sets_issue` |
| 3.3 | \`upper\` may be +inf. | `test_interval_with_infinite_upper_endpoint_is_accepted` |
| 3.3 | lower truncation must be nonnegative. | `test_negative_left_truncation_sets_issue` |
| 3.3 | lower truncation must be finite. | `test_infinite_left_truncation_sets_issue` |
| 3.2 | NaN truncation bounds are invalid. | `test_nan_left_truncation_sets_issue` |
| 3.2 | NaN truncation bounds are invalid. | `test_nan_right_truncation_sets_issue` |
| 3.3 | L<U is required. | `test_equal_truncation_bounds_set_issue` |
| 3.3 | L<U is required. | `test_reversed_truncation_bounds_set_issue` |
| 3.3 | upper truncation may be +inf. | `test_infinite_right_truncation_is_accepted` |
| 3.3 | lower truncation must be finite and nonnegative, so 0 is valid. | `test_explicit_zero_left_truncation_is_accepted` |
| 3.3 | truncation membership is L<t<=U, so t=L is outside. | `test_exact_time_equal_to_left_truncation_sets_issue` |
| 3.3 | truncation membership is L<t<=U. | `test_exact_time_below_left_truncation_sets_issue` |
| 3.3 | truncation membership is L<t<=U, so t=U is inside. | `test_exact_time_equal_to_right_truncation_is_accepted` |
| 3.3 | truncation membership is L<t<=U. | `test_exact_time_above_right_truncation_sets_issue` |
| 3.3 | truncation membership is L<t<=U. | `test_exact_time_strictly_inside_window_is_accepted` |
| 2 | right-censored before entry (c<L) is invalid (censored_before_entry). | `test_right_censoring_before_entry_sets_issue` |
| 2/3.3 | right-censored at entry (c=L>0) is valid (uninformative). | `test_right_censoring_at_entry_is_accepted` |
| 3.3 | \`(c, +inf]\` with c=L=0 is valid (uninformative). | `test_right_censoring_at_zero_without_truncation_is_accepted` |
| 3.3 | \`(c, +inf]\` with c=L is valid including finite right truncation. | `test_right_censoring_at_entry_with_finite_right_truncation_is_accepted` |
| 3.1 | an interval-flag row with \`time2=+inf\` and \`time<L\` is invalid (censored_before_entry). | `test_interval_with_infinite_time2_starting_before_entry_sets_issue` |
| 3.1 | an interval-flag row with \`time2=+inf\` and \`time=L\` is valid (uninformative). | `test_interval_with_infinite_time2_starting_at_entry_is_accepted` |
| 3.1 | \`time2=+inf\` with \`time=L\` is valid for any valid truncation window (here finite U). | `test_interval_with_infinite_time2_at_entry_with_finite_right_truncation_is_accepted` |
| 3.3 | \`from_bounds(0, inf)\` is uninformative (valid) when L=0. | `test_bounds_zero_to_infinity_without_truncation_is_accepted` |
| 3.3 | \`from_bounds(0, inf)\` is invalid when L>0. | `test_bounds_zero_to_infinity_with_left_truncation_sets_issue` |
| 2 | c<L is invalid for any censored row with upper +inf, however it was constructed. | `test_bounds_right_censoring_before_entry_sets_issue` |
| 2/3.3 | \`from_bounds(L, inf)\` is right censoring at entry, which is valid. | `test_bounds_right_censoring_at_entry_is_accepted` |
| 2 | partial overlap with the truncation window is allowed (L < c < U). | `test_right_censoring_after_entry_is_accepted` |
| 2 | empty intersection, right-censored at c>=U (here c=U), is invalid. | `test_right_censoring_at_upper_truncation_sets_issue` |
| 2 | empty intersection, right-censored at c>=U (here c>U), is invalid. | `test_right_censoring_beyond_upper_truncation_sets_issue` |
| 2 | empty intersection, (0, c] with c=L misses (L, U], is invalid. | `test_left_censoring_at_left_truncation_sets_issue` |
| 2 | empty intersection, (0, c] with c<L, is invalid. | `test_left_censoring_before_left_truncation_sets_issue` |
| 2 | empty intersection, an interval ending at or before L, is invalid. | `test_interval_ending_at_left_truncation_sets_issue` |
| 2 | empty intersection, (a, b] with a>=U, is invalid. | `test_interval_starting_at_right_truncation_sets_issue` |
| 2 | partial overlap with the truncation window is allowed (L < c < U). | `test_left_censoring_inside_window_is_accepted` |
| 2 | partial overlap with the truncation window is allowed. | `test_interval_straddling_left_truncation_is_accepted` |
| 2 | partial overlap with the truncation window is allowed. | `test_interval_straddling_right_truncation_is_accepted` |
| 2 | left-censored at c>=U covers the whole window and is valid (uninformative); here c=U. | `test_left_censoring_at_upper_truncation_is_accepted` |
| 2 | left-censored at c>=U covers the whole window and is valid (uninformative); here c>U. | `test_left_censoring_beyond_upper_truncation_is_accepted` |
| 2 | an interval containing (L, U] is valid (uninformative). | `test_interval_containing_window_is_accepted` |
| 2 | an interval containing (L, U], here equal to it, is valid (uninformative). | `test_interval_equal_to_window_is_accepted` |
| 3.1/6.2 | no implicit dropping; all rows remain and the invalid row is flagged. | `test_one_invalid_row_is_retained_with_its_issue` |

### `test_a2_bounds_constructor.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 3.1 | censored rows mean lower < T <= upper. | `test_finite_bounds_row_is_interval_censored` |
| 3.1 | \`upper=+inf\` expresses right censoring. | `test_infinite_upper_expresses_right_censoring` |
| 3.1 | \`lower=0\` expresses left censoring. | `test_zero_lower_expresses_left_censoring` |
| 3.1 | exact rows are marked by \`exact\`; their time is \`lower\`; \`upper\` may equal \`lower\`. | `test_exact_mask_row_uses_lower_as_time` |
| 3.1 | a NaN upper on an exact row normalizes to \`lower\`. | `test_exact_row_with_nan_upper_normalizes_upper_to_lower` |
| 3.1 | exactness is never inferred; without \`exact\` every row is censored. | `test_omitted_exact_mask_means_no_exact_rows` |
| 3.1/3.2 | \`left_trunc\` and \`right_trunc\` keywords give L and U. | `test_bounds_truncation_is_stored` |
| 3.2 | absent truncation normalizes to L=0 and U=+inf. | `test_bounds_truncation_defaults` |
| 3.2 | \`lower\` determines the observation shape; other fields broadcast to it. | `test_scalar_upper_broadcasts_to_lower_shape` |
| 3.1 | equal endpoints on a non-exact row are invalid. | `test_equal_endpoints_on_non_exact_row_set_issue` |
| 3.1 | exactness is not inferred from equal endpoints. | `test_equal_endpoints_without_exact_mask_set_issue` |
| 7.1 | reversed endpoints are invalid (censored rows mean lower < T <= upper). | `test_reversed_bounds_set_issue` |
| 3.1 | an exact row's \`upper\` must equal \`lower\` or be NaN. | `test_exact_row_with_different_upper_sets_issue` |
| 3.3 | exact times must be strictly positive (bounds constructor). | `test_exact_row_at_zero_sets_issue` |
| 3.3 | censoring endpoints are nonnegative. | `test_negative_lower_sets_issue` |
| 3.3 | \`lower\` is finite. | `test_infinite_lower_sets_issue` |
| 3.1 | missing required values are invalid. | `test_nan_lower_sets_issue` |
| 3.1/6.2 | a non-exact \`from_bounds\` row with NaN \`upper\` is invalid (\`missing_time2\`). | `test_nan_upper_on_censored_row_sets_issue` |
| 6.2 | a supplied \`from_bounds\` upper that is negative is invalid (\`invalid_time2\`). | `test_negative_upper_sets_issue` |
| 6.2 | a supplied \`from_bounds\` upper of \`-inf\` is invalid (\`invalid_time2\`). | `test_minus_infinite_upper_sets_issue` |
| 3.2 | fields must broadcast to the shape of \`lower\` without enlarging it. | `test_column_upper_cannot_enlarge_vector_lower` |
| 3.2 | fields must broadcast to the shape of \`lower\` without enlarging it. | `test_column_exact_mask_cannot_enlarge_vector_lower` |
| 12.1 | both constructors normalize identically (exact rows). | `test_exact_rows_normalize_identically` |
| 12.1 | both constructors normalize identically (right censoring). | `test_right_censored_rows_normalize_identically` |
| 12.1 | both constructors normalize identically (left censoring). | `test_left_censored_rows_normalize_identically` |
| 12.1 | both constructors normalize identically (interval censoring). | `test_interval_rows_normalize_identically` |
| 12.1 | both constructors normalize identically (mixed kinds with truncation). | `test_mixed_truncated_rows_normalize_identically` |
| 12.1 | for any valid untruncated rows, both constructors give identical fields. | `test_constructors_normalize_identically_property` |

### `test_a2_flag_constructor.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 3.1 | \`SurvivalData(time, event)\` uses true for exact events and false for right censoring. | `test_event_true_is_exact_and_false_is_right_censored` |
| 3.1 | the second positional argument is the event indicator, not a censoring flag. | `test_second_positional_argument_is_event_not_censoring_flag` |
| 3.1 | indicators accept numbers exactly equal to 0 or 1. | `test_integer_event_codes_zero_and_one_are_accepted` |
| 3.1 | indicators accept numbers exactly equal to 0 or 1, including floats. | `test_float_event_values_zero_and_one_are_accepted` |
| 3.1 | omitting \`event\` and all censoring flags means every row is exact. | `test_omitting_event_and_flags_makes_every_row_exact` |
| 3.1 | explicit right flags select the row's kind. | `test_right_censored_flag_selects_right_censoring` |
| 3.1 | explicit left flags select the row's kind; \`time\` is the censoring time. | `test_left_censored_flag_selects_left_censoring` |
| 3.1 | \`time\` is an interval's lower endpoint, \`time2\` its upper endpoint. | `test_interval_flag_uses_time_and_time2_as_endpoints` |
| 3.1 | with all flags false a row is exact. | `test_all_flags_false_row_is_exact` |
| 3.1 | indicators accept numbers exactly equal to 0 or 1 (censoring flags). | `test_integer_flags_zero_and_one_are_accepted` |
| 3.1 | \`time2\` is read only on interval rows; values on other rows are ignored. | `test_time2_values_on_non_interval_rows_are_ignored` |
| 3.1 | an interval-flag row with \`time2=+inf\` normalizes to right censoring at \`c=time\`. | `test_interval_with_infinite_time2_normalizes_as_right_censoring` |
| 3.2 | kinds are derived views; interval (0, c] normalizes like left censoring at c. | `test_interval_with_zero_time_normalizes_as_left_censoring` |
| 3.1 | event=false and right_censored=true both mean right censoring. | `test_event_form_and_right_flag_form_normalize_identically` |
| 3.1 | \`event\` together with any non-None censoring flag is rejected, even if all false. | `test_event_with_all_false_right_flag_sets_issue` |
| 3.1 | \`event\` together with any non-None censoring flag is rejected, even if all false. | `test_event_with_all_false_left_flag_sets_issue` |
| 3.1 | \`event\` together with any non-None censoring flag is rejected, even if all false. | `test_event_with_all_false_interval_flag_sets_issue` |
| 3.1 | at most one flag may be true per row. | `test_right_and_left_flags_on_one_row_set_issue` |
| 3.1 | at most one flag may be true per row. | `test_left_and_interval_flags_on_one_row_set_issue` |
| 3.1 | kind is never inferred from equal endpoints; interval rows require time < time2. | `test_interval_with_equal_endpoints_sets_issue_not_made_exact` |
| 3.1 | interval rows require time < time2. | `test_interval_with_reversed_endpoints_sets_issue` |
| 3.1 | \`time2\` is required for interval rows; missing required values are invalid. | `test_interval_without_time2_sets_issue` |
| 3.1 | missing required values are invalid; there is no implicit dropping or imputation. | `test_interval_with_nan_time2_sets_issue` |
| 3.1 | other indicator codes are rejected. | `test_event_code_two_sets_issue` |
| 3.1 | R-style 1/2 event coding is not inferred; the 2s are invalid codes. | `test_r_style_one_two_event_coding_is_not_inferred` |
| 3.1 | indicators accept only numbers exactly equal to 0 or 1. | `test_negative_event_code_sets_issue` |
| 3.1 | indicators accept only numbers exactly equal to 0 or 1. | `test_fractional_event_value_sets_issue` |
| 3.1 | missing values and NaN indicators are rejected. | `test_nan_event_sets_issue` |
| 3.1 | infinite indicators are rejected. | `test_infinite_event_sets_issue` |
| 3.1 | string indicators are rejected. | `test_string_event_sets_issue` |
| 3.1 | missing indicator values are rejected. | `test_object_event_with_none_sets_issue` |
| 3.1 | censoring flags accept only booleans or numbers exactly 0 or 1. | `test_censoring_flag_code_two_sets_issue` |
| 3.2 | other fields broadcast to the shape of \`time\`. | `test_scalar_event_broadcasts_to_time_shape` |
| 3.2 | other fields broadcast to the observation shape. | `test_scalar_truncation_broadcasts_to_time_shape` |
| 3.2 | fields broadcast to the observation shape of \`time\` (2, 3). | `test_trailing_axis_truncation_broadcasts_over_multidimensional_time` |
| 3.2 | \`time\` (N,) with flags (N,1) is invalid for N>1. | `test_column_event_cannot_enlarge_vector_time` |
| 3.2 | \`time\` (N,) with flags (N,1) is invalid for N>1. | `test_column_censoring_flag_cannot_enlarge_vector_time` |
| 3.2 | every other field must broadcast to the observation shape without enlarging it. | `test_column_truncation_cannot_enlarge_vector_time` |
| 3.2 | the first time argument determines the observation shape; fields cannot enlarge it. | `test_vector_event_cannot_enlarge_scalar_time` |
| 3.2 | \`time2\` must broadcast to the shape of \`time\` without enlarging it. | `test_column_time2_cannot_enlarge_vector_time` |

## A3: constructor and data requirements

### `test_a3_bounds_issues.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 6.2 | the bounds form returns data and exposes diagnostics through a property. | `test_bounds_constructor_returns_data_with_diagnostics_property` |
| 6.2 | the bounds form is the only public bounds observation constructor. | `test_bounds_constructor_has_no_issue_returning_convenience` |
| 6.2 | identical fields and issue bits also for accepted rows that carry the uninformative bit. | `test_bounds_data_and_issues_matches_from_bounds_on_uninformative_input` |
| 6.2 | the bounds constructor records invalid rows without raising. | `test_bounds_data_and_issues_does_not_raise_on_invalid_rows` |
| 6.2 | valid bounds data set no rule mask, no invalid row, and \`ok\` is true. | `test_bounds_data_and_issues_valid_data_has_no_issues` |
| 6.2 | bounds construction and diagnostics agree eagerly and under jit. | `test_bounds_data_and_issues_under_jit_matches_eager` |
| 6.2 | \`from_bounds\` \`lower\` uses \`invalid_time\`; censored lower bounds must be nonnegative. | `test_bounds_negative_lower_sets_invalid_time` |
| 6.2 | \`from_bounds\` \`lower\` uses \`invalid_time\`; censored lower bounds must be finite. | `test_bounds_infinite_lower_sets_invalid_time` |
| 6.2 | \`from_bounds\` \`lower\` uses \`invalid_time\`, including when it is NaN. | `test_bounds_nan_lower_sets_invalid_time` |
| 6.2 | exact times must be finite and positive; an exact \`lower\` of 0 sets \`invalid_time\`. | `test_bounds_exact_lower_at_zero_sets_invalid_time` |
| 6.2 | domains are evaluated before order, so a malformed lower is not a reversed interval (inf >= inf). | `test_bounds_infinite_lower_is_not_also_invalid_interval` |
| 6.2 | a non-exact \`from_bounds\` row with NaN \`upper\` sets \`missing_time2\`. | `test_bounds_nan_upper_on_censored_row_sets_missing_time2` |
| 6.2 | an exact \`from_bounds\` row may have NaN \`upper\` and does not set \`missing_time2\`. | `test_bounds_nan_upper_on_exact_row_sets_no_issue` |
| 6.2 | presence is evaluated before order, so a missing upper is not a reversed interval. | `test_bounds_nan_upper_is_not_also_invalid_interval` |
| 6.2 | a supplied \`from_bounds\` upper that is negative sets \`invalid_time2\`. | `test_bounds_negative_upper_sets_invalid_time2` |
| 6.2 | a supplied \`from_bounds\` upper of \`-inf\` sets \`invalid_time2\`. | `test_bounds_minus_infinite_upper_sets_invalid_time2` |
| 6.2 | an exact \`from_bounds\` row whose upper is neither its lower nor NaN sets \`invalid_time2\`. | `test_bounds_exact_row_with_different_upper_sets_invalid_time2` |
| 6.2 | a domain error in \`upper\` is neither missing nor a reversed interval (here 0 > -1). | `test_bounds_negative_upper_is_not_also_invalid_interval_or_missing` |
| 6.2 | a non-exact \`from_bounds\` row with \`lower >= upper\` (here equal) sets \`invalid_interval\`. | `test_bounds_equal_endpoints_on_censored_row_set_invalid_interval` |
| 6.2 | a non-exact \`from_bounds\` row with \`lower >= upper\` (here reversed) sets \`invalid_interval\`. | `test_bounds_reversed_endpoints_set_invalid_interval` |
| 6.2 | a \`+inf\` non-exact bounds upper is valid. | `test_bounds_infinite_upper_is_valid` |
| 2/6.2 | a censored row with \`upper=+inf\` and \`lower<L\` sets \`censored_before_entry\`, however constructed. | `test_bounds_right_censoring_before_entry_sets_censored_before_entry` |
| 3.3 | \`from_bounds(0, inf)\` is invalid when L>0 (\`lower<L\`, so \`censored_before_entry\`). | `test_bounds_zero_to_infinity_with_left_truncation_sets_censored_before_entry` |
| 3.3 | \`from_bounds(0, inf)\` is uninformative when L=0. | `test_bounds_zero_to_infinity_without_truncation_is_uninformative_and_valid` |
| 2/6.2 | \`(L, +inf]\` from bounds is valid and \`uninformative\`, not \`censored_before_entry\`. | `test_bounds_right_censoring_at_entry_is_uninformative_and_valid` |
| 6.2 | argument misuse (nonnumeric fields) raises in both constructors. | `test_bounds_data_and_issues_raises_on_nonnumeric_lower` |
| 6.2 | argument misuse (incompatible shapes) raises in both constructors. | `test_bounds_data_and_issues_raises_on_incompatible_shapes` |
| 6.2 | argument misuse raises independent of tracing. | `test_bounds_data_and_issues_raises_on_incompatible_shapes_under_jit` |

### `test_a3_data_issues.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 6.2 | the flag form returns data and exposes diagnostics through a property. | `test_flag_constructor_returns_data_with_diagnostics_property` |
| 6.2 | the flag form is the only public flag observation constructor. | `test_flag_constructor_has_no_issue_returning_convenience` |
| 6.2 | invalid rows give issues, without an eager exception. | `test_flag_data_and_issues_does_not_raise_on_invalid_rows` |
| 6.2 | nonnumeric time fields cannot form an object and raise instead of setting a row mask. | `test_flag_data_and_issues_raises_on_nonnumeric_time` |
| 6.2/3.2 | incompatible shapes cannot form an object and raise instead of setting a row mask. | `test_flag_data_and_issues_raises_on_incompatible_shapes` |
| 6.2 | per-rule masks have the data shape. | `test_issue_masks_have_observation_shape` |
| 6.2 | per-rule masks have the data shape, (2, 3) here. | `test_issue_masks_keep_multidimensional_observation_shape` |
| 6.2 | \`ok\` is the scalar validity reduction. | `test_ok_is_a_scalar` |
| 6.2 | valid data sets no rule mask, no invalid row, and \`ok\` is true. | `test_valid_data_has_no_issues` |
| 6.2/3.1 | other indicator codes set \`invalid_indicator\`. | `test_event_code_two_sets_invalid_indicator` |
| 3.1 | R-style 1/2 event coding is not inferred; each 2 is an invalid indicator. | `test_r_style_coding_flags_each_two_as_invalid_indicator` |
| 6.2/3.1 | a NaN indicator sets \`invalid_indicator\`. | `test_nan_event_sets_invalid_indicator` |
| 6.2/3.1 | a censoring flag other than exactly 0 or 1 sets \`invalid_indicator\`. | `test_fractional_flag_sets_invalid_indicator` |
| 6.2/3.1 | more than one true flag on a row sets \`conflicting_indicators\`. | `test_two_true_flags_set_conflicting_indicators` |
| 6.2/3.1 | an interval row without an upper endpoint sets \`missing_time2\`. | `test_nan_time2_on_interval_row_sets_missing_time2` |
| 6.2 | an active interval row with omitted \`time2\` sets \`missing_time2\`. | `test_omitted_time2_on_interval_row_sets_missing_time2` |
| 6.2 | \`missing_time2\` is for active interval rows only. | `test_nan_time2_on_inactive_rows_does_not_set_missing_time2` |
| 3.1 | \`time2\` values on non-interval rows (finite, negative, -inf, NaN) are ignored and set no issue. | `test_time2_values_on_non_interval_rows_set_no_issue` |
| 6.2 | presence is evaluated before interval order, so a missing upper is not a reversed interval. | `test_nan_time2_is_not_also_invalid_interval` |
| 6.2 | an active interval \`time2\` that is negative sets \`invalid_time2\`. | `test_negative_time2_sets_invalid_time2` |
| 6.2 | an active interval \`time2\` of \`-inf\` sets \`invalid_time2\`. | `test_minus_infinite_time2_sets_invalid_time2` |
| 6.2 | endpoint domains are evaluated before interval order; one malformed upper is not a reversed interval. | `test_negative_time2_is_not_also_invalid_interval` |
| 6.2 | \`-inf\` is a domain error in \`time2\`, neither missing nor a reversed interval. | `test_minus_infinite_time2_is_not_also_invalid_interval_or_missing` |
| 3.3 | right censoring at +inf is invalid (\`invalid_time\`). | `test_right_censoring_at_infinity_sets_invalid_time` |
| 3.3 | left censoring at +inf is invalid (\`invalid_time\`). | `test_left_censoring_at_infinity_sets_invalid_time` |
| 3.3/6.2 | a zero exact time violates 3.3 and sets \`invalid_time\`. | `test_zero_exact_time_sets_invalid_time` |
| 3.3/6.2 | a negative exact time violates 3.3 and sets \`invalid_time\`. | `test_negative_exact_time_sets_invalid_time` |
| 3.3/6.2 | an infinite exact time violates 3.3 and sets \`invalid_time\`. | `test_infinite_exact_time_sets_invalid_time` |
| 3.3/6.2 | a negative censoring time violates 3.3 and sets \`invalid_time\`. | `test_negative_censoring_time_sets_invalid_time` |
| 6.2 | \`time\` uses \`invalid_time\` on any row, including an interval row's lower endpoint. | `test_negative_interval_lower_endpoint_sets_invalid_time` |
| 6.2/3.3 | censored lower bounds must be finite; an interval row's \`time=+inf\` sets \`invalid_time\`. | `test_infinite_interval_lower_endpoint_sets_invalid_time` |
| 6.2 | domains are evaluated before order, so a malformed lower is not a reversed interval (inf >= inf). | `test_infinite_interval_lower_endpoint_is_not_also_invalid_interval` |
| 6.2 | a malformed lower endpoint is not also labelled a reversed interval (here -1 > -2). | `test_negative_interval_lower_endpoint_is_not_also_invalid_interval` |
| 6.2 | a NaN \`time\` sets \`invalid_time\` (no separate missing-value mask for the first endpoint). | `test_nan_exact_time_sets_invalid_time` |
| 6.2 | a NaN \`time\` sets \`invalid_time\` on a censored row too. | `test_nan_right_censoring_time_sets_invalid_time` |
| 6.2 | a NaN \`time\` sets \`invalid_time\`, including an interval row's lower endpoint. | `test_nan_interval_lower_endpoint_sets_invalid_time` |
| 6.2 | presence is evaluated before order, so a NaN lower is not a reversed interval. | `test_nan_interval_lower_endpoint_is_not_also_invalid_interval` |
| 3.1/6.2 | an interval row with \`lower >= upper\` (here equal) sets \`invalid_interval\`. | `test_equal_interval_endpoints_set_invalid_interval` |
| 3.1/6.2 | an interval row with \`lower >= upper\` (here reversed) sets \`invalid_interval\`. | `test_reversed_interval_endpoints_set_invalid_interval` |
| 6.2 | a \`+inf\` interval \`time2\` is valid. | `test_infinite_time2_is_a_valid_interval_upper` |
| 3.3 | lower truncation must be nonnegative (\`invalid_truncation\`). | `test_negative_left_truncation_sets_invalid_truncation` |
| 3.3 | lower truncation must be finite (\`invalid_truncation\`). | `test_infinite_left_truncation_sets_invalid_truncation` |
| 3.2 | NaN truncation bounds are invalid (\`invalid_truncation\`). | `test_nan_left_truncation_sets_invalid_truncation` |
| 3.2 | NaN truncation bounds are invalid (\`invalid_truncation\`). | `test_nan_right_truncation_sets_invalid_truncation` |
| 3.3 | L<U is required (\`invalid_truncation\`). | `test_equal_truncation_bounds_set_invalid_truncation` |
| 3.3 | L<U is required (\`invalid_truncation\`). | `test_reversed_truncation_bounds_set_invalid_truncation` |
| 3.3 | membership is L<t<=U; t=L sets \`exact_outside_truncation\`. | `test_exact_time_at_left_truncation_sets_exact_outside_truncation` |
| 3.3 | membership is L<t<=U; t<L sets \`exact_outside_truncation\`. | `test_exact_time_below_left_truncation_sets_exact_outside_truncation` |
| 3.3 | membership is L<t<=U; t>U sets \`exact_outside_truncation\`. | `test_exact_time_above_right_truncation_sets_exact_outside_truncation` |
| 3.3 | membership is L<t<=U; t=U is inside. | `test_exact_time_at_right_truncation_is_valid` |
| 2/6.2 | right-censored at c<L is invalid (\`censored_before_entry\`). | `test_right_censoring_before_entry_sets_censored_before_entry` |
| 3.1/6.2 | an interval-flag row with \`time2=+inf\` and \`time<L\` sets \`censored_before_entry\`. | `test_interval_to_infinity_before_entry_sets_censored_before_entry` |
| 6.2 | \`censored_before_entry\` applies to \`lower<L\`, not \`lower=L\`. | `test_right_censoring_at_entry_does_not_set_censored_before_entry` |
| 2 | right-censored at c>=U is an empty intersection (\`empty_observation\`, Q-010). | `test_right_censoring_at_upper_truncation_sets_empty_observation` |
| 2 | (0, c] with c<=L is an empty intersection (\`empty_observation\`, Q-010). | `test_left_censoring_at_left_truncation_sets_empty_observation` |
| 2 | an interval ending at or before L is an empty intersection (\`empty_observation\`, Q-010). | `test_interval_ending_at_left_truncation_sets_empty_observation` |
| 2 | (a, b] with a>=U is an empty intersection (\`empty_observation\`, Q-010). | `test_interval_starting_at_right_truncation_sets_empty_observation` |
| 2/6.2 | right-censored at c=L>0 is valid and flagged \`uninformative\`. | `test_right_censoring_at_entry_is_uninformative_and_valid` |
| 3.3 | \`(c, +inf]\` with c=L=0 is valid and flagged \`uninformative\`. | `test_right_censoring_at_zero_without_truncation_is_uninformative_and_valid` |
| 3.3 | \`(c, +inf]\` with c=L is uninformative including finite right truncation. | `test_right_censoring_at_entry_with_finite_right_truncation_is_uninformative_and_valid` |
| 3.1 | an interval-flag row with \`time2=+inf\` and \`time=L\` is valid and \`uninformative\`. | `test_interval_to_infinity_at_entry_is_uninformative_and_valid` |
| 3.1 | \`time2=+inf\` with \`time=L\` is \`uninformative\` for any valid truncation window (here finite U). | `test_interval_to_infinity_at_entry_with_finite_right_truncation_is_uninformative_and_valid` |
| 2/6.2 | left-censored at c>=U is valid and flagged \`uninformative\`, not in \`invalid_rows\`. | `test_left_censoring_at_upper_truncation_is_uninformative_and_valid` |
| 2/6.2 | left-censored at c>U is valid and flagged \`uninformative\`. | `test_left_censoring_beyond_upper_truncation_is_uninformative_and_valid` |
| 2/6.2 | an interval containing (L, U] is valid and flagged \`uninformative\`. | `test_interval_containing_window_is_uninformative_and_valid` |
| 2 | only censored sets covering the whole window are uninformative. | `test_partially_overlapping_rows_are_not_uninformative` |
| 2/6.2 | uninformative rows are valid, so the eager constructor does not raise. | `test_eager_constructor_accepts_uninformative_rows` |
| 3.2/12.1 | issue bitfield decoding: each rule's mask marks exactly its own row. | `test_distinct_rules_on_distinct_rows_decode_independently` |
| 3.2/12.1 | the bitfield preserves several pieces of raw-input evidence on one row. | `test_two_rules_on_one_row_are_both_decoded` |
| 6.2 | the union of the ten rule masks is \`invalid_rows\`. | `test_invalid_rows_is_the_union_of_the_ten_rule_masks` |
| 6.2 | \`uninformative\` is reported separately and is not part of \`invalid_rows\`. | `test_uninformative_is_not_part_of_invalid_rows` |
| 6.2 | \`ok\` is the scalar validity reduction. | `test_ok_is_false_when_any_row_is_invalid` |
| 6.2 | \`ok\` reduces \`invalid_rows\`, which excludes \`uninformative\` (Q-011). | `test_ok_is_true_with_only_uninformative_rows` |
| 6.2 | invalid rows raise \`chronox_jax.InvalidDataError\`, a subclass of \`ValueError\`. | `test_invalid_data_error_is_a_public_value_error` |
| 10.2/13 | "`TracedInputError` is retired and is not exported." | `test_traced_input_error_is_retired` |
| 6.2 | \`DataIssues.raise_if_invalid()\` is an explicit host-only checkpoint. | `test_raise_if_invalid_raises_for_invalid_rows` |
| 6.2 | an invalid host checkpoint reports counts and first offending indices. | `test_host_checkpoint_reports_count_and_first_offending_index` |
| 6.2 | the checkpoint does not raise for valid data. | `test_raise_if_invalid_passes_for_valid_data` |
| 6.2 | uninformative rows are valid, so the checkpoint does not raise. | `test_raise_if_invalid_passes_for_uninformative_rows` |
| 6.2 | the flag constructor agrees eagerly and under jit. | `test_flag_data_and_issues_under_jit_matches_eager` |
| 6.2 | the flag constructor agrees under vmap for each dataset. | `test_flag_data_and_issues_under_vmap_matches_eager_per_dataset` |
| 6.3 | DataIssues is a registered PyTree, so it can be returned from jit. | `test_data_issues_is_a_pytree_returned_from_jit` |

### `test_a3_indexing.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 3.2 | \`data[idx]\` is supported; an integer index gives scalar fields. | `test_integer_index_gives_scalar_observation` |
| 3.2 | \`data[idx]\` follows array indexing, including negative indices. | `test_negative_integer_index` |
| 3.2 | slicing over observation axes is supported. | `test_slice_index` |
| 3.2 | slicing over observation axes is supported (step 2). | `test_strided_slice_index` |
| 3.2 | indexing makes minibatches (integer-array index, with repetition). | `test_integer_array_index_for_minibatch` |
| 3.2 | indexing makes train/validation/test splits (boolean mask). | `test_boolean_mask_index_for_split` |
| 3.2 | \`data[idx]\` over multidimensional observation axes. | `test_indexing_multidimensional_observations` |
| 3.2 | \`jax.tree.map\` slicing over observation axes is supported and agrees with \`data[idx]\`. | `test_tree_map_slicing_matches_indexing` |
| 3.2 | every leaf has the observation shape, so indexing slices each leaf, the bitfield included. | `test_indexing_carries_every_leaf_including_issue_bitfield` |
| 3.2 | slicing works on data passed through jit (minibatches inside compiled code). | `test_slicing_inside_jit` |
| 3.2 | for any integer-array index, \`data[idx].field == data.field[idx]\`. | `test_indexing_commutes_with_fields_property` |

### `test_a3_issue_decoding.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 3.2/6.2 | fresh flag data decode stored invalid-indicator evidence. | `test_fresh_flag_data_decodes_invalid_indicator` |
| 3.2/6.2 | fresh bounds data decode a stored interval issue. | `test_fresh_bounds_data_decodes_invalid_interval` |
| 6.2 | a valid construction can carry the \`uninformative\` bit. | `test_valid_data_carries_the_uninformative_bit` |
| 6.2 | for \`data[idx]\`, each per-row mask equals the parent's mask indexed by \`idx\` (slice). | `test_issues_of_a_slice_index_the_parent_masks` |
| 6.2 | the same for an integer-array index (minibatch). | `test_issues_of_an_integer_array_index_the_parent_masks` |
| 6.2 | the same for a boolean-mask index (split). | `test_issues_of_a_boolean_mask_index_the_parent_masks` |
| 6.2 | a scalar observation decodes to scalar masks equal to the parent's row. | `test_issues_of_an_integer_index_give_scalar_masks` |
| 3.2/6.2 | \`jax.tree.map\` slicing carries the bitfield, so decoding matches the parent's rows. | `test_issues_of_tree_map_slice_index_the_parent_masks` |
| 6.2 | \`ok\` reduces over the indexed rows only; a slice of valid rows is ok though its parent is not. | `test_ok_of_a_slice_reduces_over_the_slice_only` |
| 6.2/13 | "The decoder is the `data.issues` property only; there is no callable `data.issues()` compatibility alias." | `test_issues_has_no_callable_alias` |
| 6.2 | \`data.issues\` is pure and works under jit. | `test_issues_under_jit_matches_eager` |
| 3.2/6.2 | transformed data retain raw invalid-indicator evidence. | `test_issues_of_data_returned_from_jit_preserve_raw_indicator` |
| 3.2/6.2 | data rebuilt from its leaves decode to the same issues. | `test_issues_of_data_rebuilt_from_leaves_matches_original` |

### `test_a3_static_metadata.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 3.2 | flags record which kinds are present, not the values. | `test_same_kinds_with_different_values_share_tree_structure` |
| 3.2 | auxiliary data records which derived kinds are present (right). | `test_adding_a_right_censored_row_changes_tree_structure` |
| 3.2 | auxiliary data records which derived kinds are present (left). | `test_adding_a_left_censored_row_changes_tree_structure` |
| 3.2 | auxiliary data records which derived kinds are present (interval). | `test_adding_an_interval_row_changes_tree_structure` |
| 3.2 | right, left and interval are separate derived kinds. | `test_each_censored_kind_gives_a_distinct_tree_structure` |
| 3.2 | auxiliary data records whether any left truncation is present (any \`left_trunc > 0\`). | `test_left_truncation_changes_tree_structure` |
| 3.2 | auxiliary data records whether any right truncation is present (any \`right_trunc < +inf\`). | `test_right_truncation_changes_tree_structure` |
| 3.2 | left and right truncation are separate flags. | `test_left_and_right_truncation_give_different_tree_structures` |
| 3.2 | the auxiliary data holds only the flags, so equal flags give equal tree structures. | `test_equal_flags_with_different_row_counts_share_tree_structure` |
| 3.2 | flags come from normalized fields; explicit L=0 and U=+inf normalize like absent truncation. | `test_explicit_zero_and_infinite_truncation_give_the_same_flags_as_absent` |
| 3.2/12.1 | both constructors normalize identically, including the static metadata. | `test_both_constructors_give_the_same_tree_structure` |
| 3.2 | with no traced observation input, flags are computed from the normalized fields, by either route. | `test_eager_flag_constructor_computes_flags_from_values` |
| 3.2 | the bounds constructor with non-traced inputs computes flags from values. | `test_eager_bounds_constructor_computes_flags_from_values` |
| 3.2 | flags are computed from the normalized fields of all rows, including rows with issue bits. | `test_flags_include_rows_with_issue_bits` |
| 3.2 | with no traced observation input, flags are computed from values even inside a transformed function. | `test_ordinary_construction_inside_jit_computes_flags_from_values` |
| 3.2 | constructors classify inputs as in 6.2; non-traced inputs inside jit give value-based flags. | `test_construction_from_concrete_inputs_inside_jit_computes_flags_from_values` |
| 3.2 | traced arguments that are not observation inputs do not force all flags true. | `test_ordinary_construction_inside_jit_with_other_traced_args_computes_flags_from_values` |
| 3.2 | slices inherit flags unchanged, even when the slice lacks a parent kind. | `test_slice_inherits_parent_flags` |
| 3.2 | a scalar observation taken from a parent keeps the parent's flags. | `test_integer_index_inherits_parent_flags` |
| 3.2 | any one traced observation input (here only \`left_trunc\`) makes all flags true. | `test_single_traced_observation_input_sets_all_flags` |
| 3.2 | the bounds constructor with a traced observation input gives all flags true. | `test_traced_bounds_construction_sets_all_flags` |
| 3.2 | when any observation input is traced (ordinary constructor), all flags are true, whatever the values. | `test_traced_construction_sets_all_flags` |
| 3.2 | eager data containing every kind and both truncations has all flags true, as traced data does. | `test_traced_construction_of_full_data_matches_eager_structure` |
| 3.2 | datasets with the same flags, shapes and dtypes share one compilation. | `test_same_flags_compile_once` |
| 3.2 | datasets with different flags have different tree structures and compile separately. | `test_different_flags_compile_separately` |
| 3.2 | traced-built data (all flags true) and eager data with every kind share one compilation. | `test_traced_and_eager_all_flag_data_share_one_compilation` |
| 3.2 | equal-shape slices of one parent keep its flags, so they share one compilation. | `test_slice_reuses_parent_flags_for_compilation` |

### `test_a3_tracing.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 3.2/6.2 | concrete NumPy inputs inside jit construct with value-based flags. | `test_concrete_numpy_inputs_inside_jit_keep_value_based_metadata` |
| 6.2 | concrete invalid observations inside jit return data and issue bits. | `test_concrete_invalid_flag_inputs_inside_jit_return_issues` |
| 6.2 | concrete invalid bounds inside jit return data and issue bits. | `test_concrete_invalid_bounds_inputs_inside_jit_return_issues` |
| 6.2 | tracing unrelated values does not reject invalid concrete observations. | `test_other_traced_argument_does_not_change_concrete_input_validation` |
| 6.2 | traced flag-form time has the same normalized fields and issues. | `test_traced_time_matches_eager` |
| 6.2 | traced event codes retain invalid-indicator evidence. | `test_traced_event_matches_eager` |
| 6.2 | traced right-censoring flags are accepted. | `test_traced_right_flag_matches_eager` |
| 6.2 | traced left-censoring flags are accepted. | `test_traced_left_flag_matches_eager` |
| 6.2 | traced interval-censoring flags are accepted. | `test_traced_interval_flag_matches_eager` |
| 6.2 | traced time2 is accepted and checked on interval rows. | `test_traced_time2_matches_eager` |
| 6.2 | traced left truncation is accepted and checked. | `test_traced_left_truncation_matches_eager` |
| 6.2 | traced right truncation is accepted and checked. | `test_traced_right_truncation_matches_eager` |
| 6.2 | a tracer nested in a Python list is an accepted observation input. | `test_nested_list_with_tracer_is_accepted` |
| 6.2 | a tracer in a Python tuple is an accepted observation input. | `test_tuple_with_tracer_is_accepted` |
| 6.2 | a JAX array created inside jit is accepted as traced input. | `test_jnp_array_created_inside_jit_is_accepted` |
| 6.2 | traced bounds lower is accepted and validated. | `test_traced_bounds_lower_matches_eager` |
| 6.2 | traced bounds upper is accepted and validated. | `test_traced_bounds_upper_matches_eager` |
| 6.2 | traced bounds exact indicator is accepted and validated. | `test_traced_bounds_exact_matches_eager` |
| 6.2 | traced bounds left truncation is accepted. | `test_traced_bounds_left_truncation_matches_eager` |
| 6.2 | traced bounds right truncation is accepted. | `test_traced_bounds_right_truncation_matches_eager` |
| 6.2 | bounds inputs may contain a tracer inside nested tuples. | `test_bounds_nested_tuple_with_tracer_is_accepted` |
| 6.2/12.1 | bounds construction agrees under vmap for valid and invalid rows. | `test_bounds_vmap_agrees_with_eager_for_valid_and_invalid_rows` |
| 6.2 | "`ok` is `~jnp.any(invalid_rows)`"; a scalar over every row of data returned from vmap, and over the indexed rows only for `data[idx]`. | `test_ok_of_vmap_returned_data_reduces_over_all_rows` |
| 6.2 | event combined with a censoring flag raises TypeError under jit. | `test_flag_misuse_still_raises_when_inputs_are_traced` |
| 6.2 | incompatible bounds shapes raise TypeError under jit. | `test_bounds_shape_misuse_still_raises_when_inputs_are_traced` |
| 6.2 | a nonnumeric field raises TypeError even when another input is traced. | `test_nonnumeric_field_misuse_raises_with_other_traced_input` |
| 3.3/6.2 | a traced flag call cannot request unavailable float64. | `test_flag_unavailable_explicit_dtype_raises_under_tracing` |
| 3.3/6.2 | a traced bounds call cannot request unavailable float64. | `test_bounds_unavailable_explicit_dtype_raises_under_tracing` |
| 6.2 | data.issues.ok is a scalar bool usable in compiled control flow. | `test_compiled_ok_controls_branch_selection` |
| 6.2/13 | raise_if_invalid on traced diagnostics "raises a plain `ValueError` with a host-only message, not `InvalidDataError`". | `test_traced_diagnostics_cannot_raise_at_host_checkpoint` |
| 6.2/13 | the same, when the traced rows are invalid. | `test_traced_invalid_diagnostics_misuse_is_not_invalid_data_error` |
| 6.2 | concrete diagnostics returned from jit raise only at the host checkpoint. | `test_compiled_invalid_diagnostics_raise_at_explicit_host_checkpoint` |
| 7.2 (draft) | compiled construction and diagnostic access run "without ... host callbacks". | `test_compiled_construction_and_diagnostics_install_no_host_callback` |
| 7.2 (draft) | ... "without library-induced host transfers" once warmed. | `test_warm_compiled_construction_and_diagnostics_make_no_host_transfer` |
| 7.2 (draft) | "Changing dynamic ... observation values must reuse compilation" (construction inside jit). | `test_compiled_construction_reuses_compilation_for_new_observation_values` |

## A2–A3: deliberately not tested

- Exact punctuation and wording in `InvalidDataError` messages: section 6.2
  requires counts and first offending indices, tested as decimal integers
  (Q-017), while formatting remains open.
- The host-only message text of the traced-checkpoint `ValueError` (6.2):
  wording is not fixed.
- Host synchronization (7.2) beyond the jaxpr callback and transfer-guard
  checks: JAX offers no public, reliable way to observe it from a test.
- Performance measurements, benchmarks, peak memory, accelerator execution
  and documentation builds (10.3, 10.5, 12.3): these are recorded evidence and
  review items, not acceptance assertions, and none has a pass/fail threshold.
- Compilation reuse for likelihood and gradient evaluation (7.2, 12.3):
  likelihood tests begin at A6.
- Exact normalized field values on malformed observation inputs: sections 3.2
  and 6.2 require consistent eager/traced normalization and preserved issue
  evidence, but do not assign every malformed field a value.
- Likelihood propagation of stored invalid-input bits and the all-censored
  exponential estimability example: section 12.1 requires these for Slice A,
  but likelihood tests begin at A6 under the maintainer's step schedule.
- The simulation study guide (11.4): non-normative since the performance
  amendment. The 12.1 "Compiled composition" row (construction, diagnostics
  and likelihood inside `jit`, with parameter derivatives) needs likelihoods,
  so it belongs to A6/A9.
- Branch pruning by static metadata (3.2, 7.2): likelihood behavior belongs
  to a later step/slice; A3 covers tree structures and compile counts.
