# Test map

Spec revision 7. Revision 4 plus the dtype-policy amendment of commit 91f5f62
(A1 rows citing 3.3 dtype quote the amended text); revision 5 changed right
censoring at entry and endpoint issue attribution; revision 6 added
`from_bounds_with_issues`, the traced-input classification, argument misuse in
all four constructors, and value-based static metadata inside transformations;
revision 7 named the exception types, added `data.issues()`, ignored `time2` on
non-interval rows, separated the truncation flags, ordered misuse before the
traced-input check and removed outside-interval censoring. Only tests for
requirements that changed carry `rev 5`, `rev 6` or `rev 7` tags. Steps follow the mapping in API_ASSUMPTIONS Q-001. Test names
are relative to `tests/spec/`. None of these tests has ever been executed.

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

## A2: constructors and boundary rules (`test_a2_flag_constructor.py`, `test_a2_bounds_constructor.py`, `test_a2_boundaries.py`)

Invalid rows are tested through the eager constructor, which "raises
descriptive structural errors" (6.2). The DataIssues field for each rule is
tested in A3.

| Spec section | Requirement | Tests |
| --- | --- | --- |
| 3.1 | "`SurvivalData(time, event)` uses true/1 for exact events and false/0 for right censoring" | `test_event_true_is_exact_and_false_is_right_censored`, `test_integer_event_codes_zero_and_one_are_accepted`, `test_float_event_values_zero_and_one_are_accepted` |
| 3.1 | "The second positional argument is the event indicator, not a censoring flag" | `test_second_positional_argument_is_event_not_censoring_flag`, `test_event_form_and_right_flag_form_normalize_identically` |
| 3.1 | "Omitting `event` and all censoring flags means every row is exact" | `test_omitting_event_and_flags_makes_every_row_exact` |
| 3.1 | "explicit right/left/interval flags select each row's kind; with all flags false a row is exact" | `test_right_censored_flag_selects_right_censoring`, `test_left_censored_flag_selects_left_censoring`, `test_interval_flag_uses_time_and_time2_as_endpoints`, `test_all_flags_false_row_is_exact`, `test_integer_flags_zero_and_one_are_accepted` |
| 3.1 | "At most one flag may be true per row" | `test_right_and_left_flags_on_one_row_are_rejected`, `test_left_and_interval_flags_on_one_row_are_rejected` |
| 3.1 | "Supplying `event` together with any non-None censoring flag is rejected, even if the flag contains only false values" | `test_event_with_all_false_right_flag_is_rejected`, `test_event_with_all_false_left_flag_is_rejected`, `test_event_with_all_false_interval_flag_is_rejected` |
| 3.1 | "`time` is ... an interval's lower endpoint. `time2` is an interval's upper endpoint" | `test_interval_flag_uses_time_and_time2_as_endpoints` |
| 3.1 | "`time2` is read only on interval rows; its values on other rows are ignored and set no issue" (rev 7) | `test_time2_values_on_non_interval_rows_are_ignored`, `test_time2_values_on_non_interval_rows_set_no_issue`, `test_nan_time2_on_inactive_rows_does_not_set_missing_time2` |
| 3.1 | "Kind is never inferred from equal or nearly equal endpoints; interval rows require `time < time2`" | `test_interval_with_equal_endpoints_is_rejected_not_made_exact`, `test_interval_with_reversed_endpoints_is_rejected` |
| 3.1 | "Indicators accept booleans or numbers exactly equal to 0 or 1 ... Missing values, NaN, infinity, other codes and strings are rejected" | `test_event_code_two_is_rejected`, `test_negative_event_code_is_rejected`, `test_fractional_event_value_is_rejected`, `test_nan_event_is_rejected`, `test_infinite_event_is_rejected`, `test_string_event_is_rejected`, `test_object_event_with_none_is_rejected`, `test_censoring_flag_code_two_is_rejected` |
| 3.1 | "R-style 1/2 event coding is not inferred" | `test_r_style_one_two_event_coding_is_not_inferred` |
| 3.1 | "Missing required values are invalid. There is no implicit dropping or imputation" | `test_interval_without_time2_is_rejected`, `test_interval_with_nan_time2_is_rejected`, `test_nan_exact_time_is_rejected`, `test_nan_right_censoring_time_is_rejected`, `test_nan_lower_is_rejected`, `test_nan_upper_on_censored_row_is_rejected`, `test_one_invalid_row_rejects_the_whole_eager_construction` |
| 3.1 | bounds: "Censored rows mean `lower < T <= upper`. `upper=+inf` expresses right censoring and `lower=0` left censoring" | `test_finite_bounds_row_is_interval_censored`, `test_infinite_upper_expresses_right_censoring`, `test_zero_lower_expresses_left_censoring`, `test_reversed_bounds_are_rejected` |
| 3.1 | bounds: "Exact rows are marked by the explicit `exact` mask; their time is `lower`, and their `upper` must equal `lower` or be NaN. A NaN upper on an exact row normalizes to `lower`" | `test_exact_mask_row_uses_lower_as_time`, `test_exact_row_with_nan_upper_normalizes_upper_to_lower`, `test_exact_row_with_different_upper_is_rejected` |
| 3.1 | bounds: "without inferring exactness ... Equal endpoints on a non-exact row are invalid" | `test_omitted_exact_mask_means_no_exact_rows`, `test_equal_endpoints_on_non_exact_row_are_rejected`, `test_equal_endpoints_without_exact_mask_are_rejected` |
| 3.1 | bounds: `left_trunc`, `right_trunc` keywords | `test_bounds_truncation_is_stored`, `test_bounds_truncation_defaults` |
| 3.2 | "The first time argument determines the observation shape. Every other field must broadcast to that shape without enlarging it" | `test_scalar_event_broadcasts_to_time_shape`, `test_scalar_truncation_broadcasts_to_time_shape`, `test_trailing_axis_truncation_broadcasts_over_multidimensional_time`, `test_scalar_upper_broadcasts_to_lower_shape`, `test_column_event_cannot_enlarge_vector_time`, `test_column_censoring_flag_cannot_enlarge_vector_time`, `test_column_truncation_cannot_enlarge_vector_time`, `test_vector_event_cannot_enlarge_scalar_time`, `test_column_time2_cannot_enlarge_vector_time`, `test_column_upper_cannot_enlarge_vector_lower`, `test_column_exact_mask_cannot_enlarge_vector_lower` |
| 3.2 | "Right/left/interval 'kinds' are derived views (`upper==inf`, `lower==0`)" | `test_interval_with_zero_time_normalizes_as_left_censoring` |
| 3.1 | "An interval-flag row with `time2=+inf` normalizes to right censoring at `c=time`. It is valid and `uninformative` when `time=L`, and invalid with `censored_before_entry` when `time<L`, for any valid truncation window" | `test_interval_with_infinite_time2_normalizes_as_right_censoring`, `test_interval_with_infinite_time2_starting_at_entry_is_accepted`, `test_interval_with_infinite_time2_at_entry_with_finite_right_truncation_is_accepted`, `test_interval_with_infinite_time2_starting_before_entry_is_rejected` |
| 12.1 | "both constructors normalize identically" | `test_exact_rows_normalize_identically`, `test_right_censored_rows_normalize_identically`, `test_left_censored_rows_normalize_identically`, `test_interval_rows_normalize_identically`, `test_mixed_truncated_rows_normalize_identically`, `test_constructors_normalize_identically_property` (hypothesis, `derandomize=True`) |
| 3.3 | "Exact times must be finite and strictly positive. Zero, negative and infinite exact times are invalid; no epsilon is added" | `test_exact_time_zero_is_rejected`, `test_negative_exact_time_is_rejected`, `test_infinite_exact_time_is_rejected`, `test_tiny_positive_exact_time_is_kept_unchanged`, `test_exact_row_at_zero_is_rejected` |
| 3.3 | "Truncation membership is `L<t<=U`" | `test_exact_time_equal_to_left_truncation_is_rejected`, `test_exact_time_below_left_truncation_is_rejected`, `test_exact_time_equal_to_right_truncation_is_accepted`, `test_exact_time_above_right_truncation_is_rejected`, `test_exact_time_strictly_inside_window_is_accepted` |
| 3.3 | "Censoring endpoints are nonnegative. `lower` is finite; `upper` may be `+inf`" | `test_negative_right_censoring_time_is_rejected`, `test_negative_left_censoring_time_is_rejected`, `test_negative_interval_lower_endpoint_is_rejected`, `test_negative_lower_is_rejected`, `test_infinite_lower_is_rejected`, `test_interval_with_infinite_upper_endpoint_is_accepted` |
| 3.3 | "right censoring at `+inf` and left censoring at `+inf` are both invalid" | `test_right_censoring_at_infinity_is_rejected`, `test_left_censoring_at_infinity_is_rejected` |
| 3.3, 3.2 | "Lower truncation must be finite and nonnegative; upper truncation may be `+inf`; `L<U` is required"; "NaN truncation bounds are invalid" | `test_negative_left_truncation_is_rejected`, `test_infinite_left_truncation_is_rejected`, `test_nan_left_truncation_is_rejected`, `test_nan_right_truncation_is_rejected`, `test_equal_truncation_bounds_are_rejected`, `test_reversed_truncation_bounds_are_rejected`, `test_infinite_right_truncation_is_accepted`, `test_explicit_zero_left_truncation_is_accepted` |
| 2 | "Right-censored before entry (`c<L`): invalid data ... applies to any censored row with upper bound `+inf`, however it was constructed" | `test_right_censoring_before_entry_is_rejected`, `test_bounds_right_censoring_before_entry_is_rejected`, `test_right_censoring_after_entry_is_accepted` |
| 2, 3.3 | "Right-censored at entry (`c=L`): valid and `uninformative`"; "including `c=L=0` and finite right truncation" | `test_right_censoring_at_entry_is_accepted`, `test_right_censoring_at_zero_without_truncation_is_accepted`, `test_right_censoring_at_entry_with_finite_right_truncation_is_accepted`, `test_bounds_right_censoring_at_entry_is_accepted` |
| 3.3 | "`from_bounds(0, inf)` is uninformative when `L=0` and invalid when `L>0`" | `test_bounds_zero_to_infinity_without_truncation_is_accepted`, `test_bounds_zero_to_infinity_with_left_truncation_is_rejected` |
| 6.2 | bounds: "`missing_time2`: ... a non-exact `from_bounds` row has NaN `upper`"; "`invalid_time2`: ... a supplied `from_bounds` upper is negative or `-inf`" (eager rejection; attribution in A3 `test_a3_bounds_issues.py`) | `test_nan_upper_on_censored_row_is_rejected`, `test_negative_upper_is_rejected`, `test_minus_infinite_upper_is_rejected`, `test_exact_row_with_different_upper_is_rejected` |
| 2 | "Empty intersection (for example, right-censored at `c>=U`, or an interval ending at or before `L`): invalid data" | `test_right_censoring_at_upper_truncation_is_rejected`, `test_right_censoring_beyond_upper_truncation_is_rejected`, `test_left_censoring_at_left_truncation_is_rejected`, `test_left_censoring_before_left_truncation_is_rejected`, `test_interval_ending_at_left_truncation_is_rejected`, `test_interval_starting_at_right_truncation_is_rejected`, `test_left_censoring_at_zero_is_rejected` |
| 2 | "Partial overlap with the truncation window is allowed" | `test_right_censoring_after_entry_is_accepted`, `test_left_censoring_inside_window_is_accepted`, `test_interval_straddling_left_truncation_is_accepted`, `test_interval_straddling_right_truncation_is_accepted` |
| 2 | "Other censored sets covering the whole window (left-censored at `c>=U`, an interval containing `(L,U]`): valid" | `test_left_censoring_at_upper_truncation_is_accepted`, `test_left_censoring_beyond_upper_truncation_is_accepted`, `test_interval_containing_window_is_accepted`, `test_interval_equal_to_window_is_accepted` |

### A2: deliberately not tested

- **Error message content** ("reporting counts and the first offending
  indices", 6.2): the format is open; only the raise is asserted.
- **The `exact` mask of `from_bounds` given as 0/1 numbers or with invalid
  codes:** the indicator rules in 3.1 sit under the flag constructor; whether
  they apply to `exact` is not stated.
- **`time` of shape (1,) with flags (1,1):** the spec's example is qualified
  "for N>1", so the N=1 case is left open.
- **Truncation of rows from `from_bounds` against uninformative/empty rules:**
  the rules are properties of the normalized fields, so they are tested once,
  through the flag constructor. The bounds constructor's equivalence is covered
  by the identical-normalization tests.

## A3: DataIssues, eager versus traced construction, indexing, static metadata (`test_a3_data_issues.py`, `test_a3_bounds_issues.py`, `test_a3_issue_decoding.py`, `test_a3_tracing.py`, `test_a3_indexing.py`, `test_a3_static_metadata.py`)

| Spec section | Requirement | Tests |
| --- | --- | --- |
| 6.2 | "`SurvivalData.with_issues(...) -> (data, DataIssues)` # pure; works under jit" | `test_with_issues_returns_data_and_issues_pair`, `test_with_issues_data_matches_ordinary_constructor_on_valid_input`, `test_with_issues_does_not_raise_on_invalid_rows`, `test_with_issues_under_jit_matches_eager`, `test_with_issues_under_vmap_matches_eager_per_dataset` |
| 6.2 | "per-rule masks with the data shape, an invalid-row union and a scalar validity reduction" | `test_issue_masks_have_observation_shape`, `test_issue_masks_keep_multidimensional_observation_shape`, `test_ok_is_a_scalar`, `test_valid_data_has_no_issues` |
| 6.2, 3.1 | `invalid_indicator` (indicator codes, NaN, R-style 1/2) | `test_event_code_two_sets_invalid_indicator`, `test_r_style_coding_flags_each_two_as_invalid_indicator`, `test_nan_event_sets_invalid_indicator`, `test_fractional_flag_sets_invalid_indicator` |
| 6.2, 3.1 | `conflicting_indicators` ("At most one flag may be true per row") | `test_two_true_flags_set_conflicting_indicators` |
| 6.2 | `missing_time2` ("an active interval row has omitted or NaN `time2`") | `test_nan_time2_on_interval_row_sets_missing_time2`, `test_omitted_time2_on_interval_row_sets_missing_time2`, `test_nan_time2_on_inactive_rows_does_not_set_missing_time2` |
| 6.2 | `invalid_time2` ("an active interval `time2` ... is negative or `-inf`") | `test_negative_time2_sets_invalid_time2`, `test_minus_infinite_time2_sets_invalid_time2` |
| 6.2 | "Evaluate presence and endpoint domains before interval order, so a malformed lower or upper is not also labelled a reversed interval" | `test_nan_time2_is_not_also_invalid_interval`, `test_negative_time2_is_not_also_invalid_interval`, `test_minus_infinite_time2_is_not_also_invalid_interval_or_missing`, `test_infinite_interval_lower_endpoint_is_not_also_invalid_interval`, `test_negative_interval_lower_endpoint_is_not_also_invalid_interval`, `test_nan_interval_lower_endpoint_is_not_also_invalid_interval` |
| 6.2 | "Inputs with nonnumeric time/bound fields or incompatible shapes cannot form an object and raise instead of setting a row mask" | `test_with_issues_raises_on_nonnumeric_time`, `test_with_issues_raises_on_incompatible_shapes` |
| 6.2, 3.3 | `invalid_time` ("right censoring at `+inf` and left censoring at `+inf` are both invalid (`invalid_time`)"; "The flag constructor's `time` uses `invalid_time` when its active value violates section 3.3 on any row, including an interval row's lower endpoint"; "A NaN `time` sets `invalid_time`") | `test_right_censoring_at_infinity_sets_invalid_time`, `test_left_censoring_at_infinity_sets_invalid_time`, `test_zero_exact_time_sets_invalid_time`, `test_negative_exact_time_sets_invalid_time`, `test_infinite_exact_time_sets_invalid_time`, `test_negative_censoring_time_sets_invalid_time`, `test_negative_interval_lower_endpoint_sets_invalid_time`, `test_infinite_interval_lower_endpoint_sets_invalid_time`, `test_nan_exact_time_sets_invalid_time`, `test_nan_right_censoring_time_sets_invalid_time`, `test_nan_interval_lower_endpoint_sets_invalid_time` |
| 6.2, 3.1 | `invalid_interval` ("an explicitly interval-censored row ... has `lower >= upper`"; "A `+inf` interval `time2` ... is valid") | `test_equal_interval_endpoints_set_invalid_interval`, `test_reversed_interval_endpoints_set_invalid_interval`, `test_infinite_time2_is_a_valid_interval_upper` |
| 6.2, 3.3, 3.2 | `invalid_truncation` | `test_negative_left_truncation_sets_invalid_truncation`, `test_infinite_left_truncation_sets_invalid_truncation`, `test_nan_left_truncation_sets_invalid_truncation`, `test_nan_right_truncation_sets_invalid_truncation`, `test_equal_truncation_bounds_set_invalid_truncation`, `test_reversed_truncation_bounds_set_invalid_truncation` |
| 6.2, 3.3 | `exact_outside_truncation` ("Truncation membership is `L<t<=U`") | `test_exact_time_at_left_truncation_sets_exact_outside_truncation`, `test_exact_time_below_left_truncation_sets_exact_outside_truncation`, `test_exact_time_above_right_truncation_sets_exact_outside_truncation`, `test_exact_time_at_right_truncation_is_valid` |
| 6.2, 2 | `censored_before_entry` ("applies to censored rows with `upper=+inf` and `lower<L`, not `lower=L`") | `test_right_censoring_before_entry_sets_censored_before_entry`, `test_interval_to_infinity_before_entry_sets_censored_before_entry`, `test_right_censoring_at_entry_does_not_set_censored_before_entry` |
| 6.2, 2 | `empty_observation` ("Empty intersection ... invalid data", Q-010) | `test_right_censoring_at_upper_truncation_sets_empty_observation`, `test_left_censoring_at_left_truncation_sets_empty_observation`, `test_interval_ending_at_left_truncation_sets_empty_observation`, `test_interval_starting_at_right_truncation_sets_empty_observation` |
| 6.2, 2, 3.3 | "Right-censored at entry (`c=L`): valid and `uninformative`" (incl. `c=L=0` and finite `U`); 3.1 "interval-flag row with `time2=+inf` ... valid and `uninformative` when `time=L` ... for any valid truncation window" | `test_right_censoring_at_entry_is_uninformative_and_valid`, `test_right_censoring_at_zero_without_truncation_is_uninformative_and_valid`, `test_right_censoring_at_entry_with_finite_right_truncation_is_uninformative_and_valid`, `test_interval_to_infinity_at_entry_is_uninformative_and_valid`, `test_interval_to_infinity_at_entry_with_finite_right_truncation_is_uninformative_and_valid` |
| 6.2, 2 | "`uninformative` is reported separately and is **not** part of `invalid_rows`" | `test_left_censoring_at_upper_truncation_is_uninformative_and_valid`, `test_left_censoring_beyond_upper_truncation_is_uninformative_and_valid`, `test_interval_containing_window_is_uninformative_and_valid`, `test_partially_overlapping_rows_are_not_uninformative`, `test_uninformative_is_not_part_of_invalid_rows`, `test_eager_constructor_accepts_uninformative_rows` |
| 6.2 | "Their union is `invalid_rows`; `ok` is the scalar reduction" | `test_invalid_rows_is_the_union_of_the_ten_rule_masks`, `test_ok_is_false_when_any_row_is_invalid`, `test_ok_is_true_with_only_uninformative_rows` |
| 3.2, 12.1 | "one `int32` issue bitfield ... `DataIssues` decodes the bitfield"; "issue bitfield decoding" | `test_distinct_rules_on_distinct_rows_decode_independently`, `test_two_rules_on_one_row_are_both_decoded` |
| 6.2, 3.2 | "`data.issues()` decodes the issue bitfield of any `SurvivalData` ... It is pure and works under `jit`. For data from a pure constructor it returns the same masks as that constructor's `DataIssues`" (rev 7) | `test_issues_returns_a_data_issues_container`, `test_issues_matches_with_issues_on_fresh_data`, `test_issues_matches_from_bounds_with_issues_on_fresh_data`, `test_issues_under_jit_matches_eager`, `test_issues_of_data_returned_from_jit_matches_construction`, `test_issues_of_data_rebuilt_from_leaves_matches_original` |
| 6.2 | "For `data[idx]`, each per-row mask equals the parent's mask indexed by `idx`, and `ok` reduces over the indexed rows only" | `test_issues_of_a_slice_index_the_parent_masks`, `test_issues_of_an_integer_array_index_the_parent_masks`, `test_issues_of_a_boolean_mask_index_the_parent_masks`, `test_issues_of_an_integer_index_give_scalar_masks`, `test_issues_of_tree_map_slice_index_the_parent_masks`, `test_ok_of_a_slice_reduces_over_the_slice_only` |
| 6.2, 3.3 | "Exception types. Invalid observation rows raise `chronox_jax.InvalidDataError`, a subclass of `ValueError` ... Argument misuse raises `TypeError`. An explicit dtype request that cannot be honoured raises `ValueError`, not `InvalidDataError`" (rev 7) | `test_invalid_data_error_is_a_public_value_error`; every rejection test asserts its kind: `pytest.raises(INVALID_DATA)` (59), `pytest.raises(MISUSE)` (29), `raises_dtype_request()` (3) |
| 6.2 | "raises `chronox_jax.TracedInputError` at trace time, and its message names the pure counterpart"; "A traced observation input to an ordinary constructor raises `chronox_jax.TracedInputError`, a subclass of `TypeError`"; misuse raises "`TypeError`, not `TracedInputError`" (rev 7) | `test_traced_input_error_is_a_public_type_error`; every trace-time rejection test uses `pytest.raises(TRACED_INPUT, match=<counterpart>)`; the two misuse-first tests use `raises_misuse()` |
| 6.2 | "Data from an ordinary constructor can carry only the `uninformative` bit" | `test_ordinary_constructor_data_carries_only_the_uninformative_bit` |
| 6.2 | "`DataIssues.raise_if_invalid()` # explicit host-only checkpoint" | `test_raise_if_invalid_raises_for_invalid_rows`, `test_raise_if_invalid_passes_for_valid_data`, `test_raise_if_invalid_passes_for_uninformative_rows` |
| 6.2, 12.1 | "If any observation input is traced, the ordinary constructor raises at trace time and names its pure counterpart"; "rejection when any observation input, or a list element of one, is traced" | `test_ordinary_constructor_under_jit_raises_pointing_to_with_issues`, `test_ordinary_constructor_under_vmap_raises_pointing_to_with_issues`, `test_bounds_constructor_under_jit_raises_pointing_to_from_bounds_with_issues`, `test_traced_time_is_rejected_naming_with_issues`, `test_traced_event_is_rejected_naming_with_issues`, `test_traced_right_censored_flag_is_rejected_naming_with_issues`, `test_traced_left_censored_flag_is_rejected_naming_with_issues`, `test_traced_interval_censored_flag_is_rejected_naming_with_issues`, `test_traced_time2_is_rejected_naming_with_issues`, `test_traced_left_trunc_is_rejected_naming_with_issues`, `test_traced_right_trunc_is_rejected_naming_with_issues`, `test_traced_lower_is_rejected_naming_from_bounds_with_issues`, `test_traced_upper_is_rejected_naming_from_bounds_with_issues`, `test_traced_exact_mask_is_rejected_naming_from_bounds_with_issues`, `test_traced_bounds_left_trunc_is_rejected_naming_from_bounds_with_issues`, `test_traced_bounds_right_trunc_is_rejected_naming_from_bounds_with_issues` |
| 6.2 | "An input is traced if it is a JAX tracer, or if it is a Python list or tuple containing a tracer at any nesting depth"; "Arrays created with `jax.numpy` inside the function are traced, even when their values are constant" | `test_list_containing_a_tracer_is_rejected`, `test_tuple_containing_a_tracer_is_rejected`, `test_nested_list_containing_a_tracer_is_rejected`, `test_jnp_array_created_inside_jit_is_rejected` |
| 6.2, 12.1 | "Inside a transformed function, the following are non-traced: NumPy arrays; Python scalars; lists and tuples of those; concrete JAX arrays created outside the function"; "When none is, normalization and validation run eagerly, also during tracing"; "ordinary construction from non-traced inputs inside `jit`, including raising on invalid rows" | `test_ordinary_constructor_accepts_numpy_inputs_inside_jit`, `test_ordinary_constructor_inside_jit_matches_outside`, `test_ordinary_constructor_accepts_python_scalars_inside_jit`, `test_ordinary_constructor_accepts_python_lists_inside_jit`, `test_ordinary_constructor_accepts_concrete_jax_array_from_outside_jit`, `test_bounds_constructor_accepts_numpy_inputs_inside_jit`, `test_ordinary_constructor_raises_on_invalid_numpy_rows_inside_jit`, `test_bounds_constructor_raises_on_invalid_numpy_rows_inside_jit`, `test_ordinary_constructor_raises_on_invalid_rows_inside_jit_with_other_traced_args` |
| 6.2, 12.1 | "Argument misuse raises in all four constructors, independent of values and of tracing"; "argument misuse raising in the pure constructors" | `test_with_issues_raises_on_event_with_censoring_flag`, `test_with_issues_raises_on_event_with_censoring_flag_under_jit`, `test_with_issues_raises_on_nonnumeric_time`, `test_with_issues_raises_on_incompatible_shapes`, `test_from_bounds_with_issues_raises_on_nonnumeric_lower`, `test_from_bounds_with_issues_raises_on_incompatible_shapes`, `test_from_bounds_with_issues_raises_on_incompatible_shapes_under_jit` |
| 6.2 | "Argument misuse is checked before the traced-input check, so misuse raises the misuse error in every constructor, also when observation inputs are traced" (rev 7) | `test_ordinary_constructor_raises_misuse_error_before_traced_input_error`, `test_bounds_constructor_raises_misuse_error_before_traced_input_error` |
| 6.2 | "`SurvivalData.from_bounds_with_issues(...) -> (data, DataIssues)` # pure; works under jit"; "For inputs accepted by the ordinary constructor, both yield identical fields and issue bits" | `test_from_bounds_with_issues_returns_data_and_issues_pair`, `test_from_bounds_with_issues_matches_from_bounds_on_valid_input`, `test_from_bounds_with_issues_matches_from_bounds_on_uninformative_input`, `test_from_bounds_with_issues_does_not_raise_on_invalid_rows`, `test_from_bounds_with_issues_valid_data_has_no_issues`, `test_from_bounds_with_issues_under_jit_matches_eager` |
| 6.2, 3.3 | bounds: "The `from_bounds` `lower` uses the same mask [`invalid_time`], including when it is NaN" | `test_bounds_negative_lower_sets_invalid_time`, `test_bounds_infinite_lower_sets_invalid_time`, `test_bounds_nan_lower_sets_invalid_time`, `test_bounds_exact_lower_at_zero_sets_invalid_time`, `test_bounds_infinite_lower_is_not_also_invalid_interval` |
| 6.2 | bounds: `missing_time2` ("a non-exact `from_bounds` row has NaN `upper`. An exact `from_bounds` row may have NaN `upper` and does not set this issue") | `test_bounds_nan_upper_on_censored_row_sets_missing_time2`, `test_bounds_nan_upper_on_exact_row_sets_no_issue`, `test_bounds_nan_upper_is_not_also_invalid_interval` |
| 6.2 | bounds: `invalid_time2` ("a supplied `from_bounds` upper is negative or `-inf`; also, an exact `from_bounds` row supplies an upper other than its lower or NaN") | `test_bounds_negative_upper_sets_invalid_time2`, `test_bounds_minus_infinite_upper_sets_invalid_time2`, `test_bounds_exact_row_with_different_upper_sets_invalid_time2`, `test_bounds_negative_upper_is_not_also_invalid_interval_or_missing` |
| 6.2 | bounds: `invalid_interval` ("a non-exact `from_bounds` row has `lower >= upper` ... A `+inf` ... non-exact bounds upper is valid") | `test_bounds_equal_endpoints_on_censored_row_set_invalid_interval`, `test_bounds_reversed_endpoints_set_invalid_interval`, `test_bounds_infinite_upper_is_valid` |
| 2, 3.3, 6.2 | bounds: `censored_before_entry` and `uninformative` ("`from_bounds(0, inf)` is uninformative when `L=0` and invalid when `L>0`"; `c=L` valid) | `test_bounds_right_censoring_before_entry_sets_censored_before_entry`, `test_bounds_zero_to_infinity_with_left_truncation_sets_censored_before_entry`, `test_bounds_zero_to_infinity_without_truncation_is_uninformative_and_valid`, `test_bounds_right_censoring_at_entry_is_uninformative_and_valid` |
| 6.3 | "Every diagnostic container (`DataIssues`, ...) ... is a registered PyTree ... accessed by field name" | `test_data_issues_is_a_pytree_returned_from_jit`, `test_with_issues_under_jit_matches_eager` |
| 3.2 | "`data[idx]` and `jax.tree.map` slicing over observation axes are supported and tested" | `test_integer_index_gives_scalar_observation`, `test_negative_integer_index`, `test_slice_index`, `test_strided_slice_index`, `test_integer_array_index_for_minibatch`, `test_boolean_mask_index_for_split`, `test_indexing_multidimensional_observations`, `test_tree_map_slicing_matches_indexing`, `test_slicing_inside_jit`, `test_indexing_commutes_with_fields_property` (hypothesis, `derandomize=True`) |
| 3.2 | "every leaf has the observation shape" (bitfield carried by indexing) | `test_indexing_carries_every_leaf_including_issue_bitfield` |
| 3.2 | "auxiliary data carries 'may contain' flags: which derived kinds are present, whether any left truncation is present (any `left_trunc > 0`), and, as a separate flag, whether any right truncation is present (any `right_trunc < +inf`)"; "The auxiliary data holds only these flags; datasets with equal flags have equal tree structures" (rev 7); "When no observation input is traced, the flags are computed from the normalized fields of all rows" | `test_left_and_right_truncation_give_different_tree_structures`, `test_equal_flags_with_different_row_counts_share_tree_structure`, `test_explicit_zero_and_infinite_truncation_give_the_same_flags_as_absent`, `test_eager_from_bounds_with_issues_computes_flags_from_values`, `test_same_kinds_with_different_values_share_tree_structure`, `test_adding_a_right_censored_row_changes_tree_structure`, `test_adding_a_left_censored_row_changes_tree_structure`, `test_adding_an_interval_row_changes_tree_structure`, `test_each_censored_kind_gives_a_distinct_tree_structure`, `test_left_truncation_changes_tree_structure`, `test_right_truncation_changes_tree_structure`, `test_both_constructors_give_the_same_tree_structure`, `test_eager_with_issues_computes_flags_from_values` |
| 3.2 | "This includes rows with issue bits, and applies also when construction happens inside a transformed function"; "metadata for non-traced ... inputs" | `test_flags_include_rows_with_issue_bits`, `test_ordinary_construction_inside_jit_computes_flags_from_values`, `test_pure_construction_from_non_traced_inputs_inside_jit_computes_flags_from_values`, `test_ordinary_construction_inside_jit_with_other_traced_args_computes_flags_from_values` |
| 3.2 | "When any observation input is traced, which is possible only with a pure constructor, all flags are true" | `test_traced_construction_sets_all_flags`, `test_traced_construction_of_full_data_matches_eager_structure`, `test_single_traced_observation_input_sets_all_flags`, `test_traced_bounds_construction_sets_all_flags` |
| 3.2 | "Slices inherit them unchanged" | `test_slice_inherits_parent_flags`, `test_integer_index_inherits_parent_flags` |
| 3.2, 12.1 | "datasets with different flags have different tree structures and compile separately"; "compile-count tests for static metadata" | `test_same_flags_compile_once`, `test_different_flags_compile_separately`, `test_traced_and_eager_all_flag_data_share_one_compilation`, `test_slice_reuses_parent_flags_for_compilation` |

### A3: tests sensitive to JAX behaviour

- **`test_jnp_array_created_inside_jit_is_rejected`:** spec 6.2 defines
  "traced" as "is a JAX tracer" and gives constant `jax.numpy` arrays built
  inside a transformed function as an example. That holds because current JAX
  stages every such operation as a tracer. A JAX release that returned a
  concrete array here would fail this test without any change in Chronox, so
  check the JAX version first if it fails. Kept by maintainer decision
  (2026-09-29); the spec wording stays as it is.

### A3: deliberately not tested

- **Clear masks on a broken row, beyond the rev 5 orderings:** "independent
  problems ... may still set multiple issue masks on one row", so apart from
  the malformed-endpoint and `c=L` cases other masks are not asserted clear.
- **Normalized field values on rows with invalid inputs:** unspecified; "A
  returned data object alone is not a validity certificate". Tests only require
  eager and jit to agree. Rows whose inputs are individually valid but break a
  truncation or coverage rule are read as normalizing by the 3.2 table ("no
  placeholder values"); `test_flags_include_rows_with_issue_bits` relies on
  this for a right-censored row at `c=U` (maintainer decision, 2026-09-29: no
  spec change).
- **Error message content** of the eager constructor and `raise_if_invalid`
  ("counts and the first offending indices"): format open.
- **Pruning branches by metadata and identical pruned results** (3.2, 7.2):
  likelihood behaviour, Slice B.
- **`LikelihoodIssues`, status `IntEnum`, other result containers** (6.2,
  6.3): outside A1–A3.
- **Tracing via transformations other than `jit` and `vmap`** (`grad`,
  `scan`, ...): the classification is by input type, so `jit` and `vmap`
  stand for all of them.
