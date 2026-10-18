# Test map

Spec: revision 27 (see `.spec-seen`). Unchanged requirements keep their
earlier revision tags. Tests from the revision 10 decision (sections 6.2, 10.2,
13) are tagged `rev 10`. Tests from the performance amendment (section 7.2),
accepted in revision 11, are tagged `rev 11`. A4 tests derive from revision 11 and
are tagged `rev 11`, except `test_a4_policy_enforcement.py` (rev 12, base
enforcement of section 4.5) and `test_a4_hazard_at_zero_survival.py` (rev 13,
`log_hazard` where `S(t)=0`), and the parameter dtype tests on bool, complex,
string and object inputs, `dtype=` and nested dtypes (rev 15), and the `batch_shape` and shape-misuse tests
(rev 16), the `.shape` attribute tests (rev 17), `test_a4_nested.py` (rev 18), and the keyword-only construction tests
(rev 19), and `test_a4_time_dtypes.py`, `test_a4_nan_precedence.py` and the
`t<0` zero-gradient tests in `test_a4_policy_enforcement.py` (rev 20), and
the `log_hazard=-inf` at `t<0`, string and object `t`, and traced-`TypeError`
tests (rev 21). A5 tests were written against revision 21 and are tagged
`rev 21`, except those whose requirements revision 22 settled (built-ins as
subclasses, `from_log`, `reference_time()` shape and dtype, NaN at `t=+inf`,
boundary gradients, the grid argument, the `bc` vector), tagged `rev 22`, as
are the `log_prob(+inf)` NaN tests in `test_a4_nan_precedence.py`. Grid tests
that revision 24 extended to every `_grid` twin, and the new grid tests, are
tagged `rev 24`. Revision 23 (documentation workflow, 10.5) changed no tested
requirement. Revision 25 (`log_cdf` relative precision in both tails, 4.1,
4.2, 7.2, 12.1) adds `test_a4_log_cdf_upper_tail.py` and
`test_a5_log_cdf_tails.py`, tagged `rev 25` like `test_default_log_cdf`; the
`test_a5_closed_forms.py` interior tests keep `rev 21` but now compare
`log_cdf` with zero absolute tolerance (12.1, rev 25). Revision 26 (base
supplies the survival limits at `t=0` and `t=+inf`; `log_survival_at_infinity()`;
2, 4.2, 4.5, 12.1) adds `test_a4_base_boundaries.py` and one A5 grid test,
retags the `t=0`/`t=+inf` tests whose requirement text changed as `rev 26`,
moves the `t=0` density-limit tests from the generic `log_prob` default (no
longer guaranteed there) to a `log_prob` override, and drops `t=0` from tests
that evaluated the generic `log_prob` default there. Revision 27 (4.5: for
valid parameters the substituted input is a finite `t>0`; 12.2: Slice A
includes the base's use of `log_survival_at_infinity()`) retags the
`boundary_nan_hazard` boundary tests in `test_a4_base_boundaries.py` as
`rev 27` and confirms the improper-family tests belong to A4. The mpmath references
need `mpmath` in the test environment. Tests are static acceptance tests and have never
been executed against survjax. Test names below are relative to `tests/spec/`.

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
| 3.3 | "survjax never changes JAX's global 64-bit setting" | `test_construction_does_not_change_global_64bit_setting`, `test_failed_float64_request_does_not_enable_64bit_mode` |

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
  values reach survjax cannot be recovered":** guidance, not testable behaviour.

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
| 6.2 | invalid rows raise \`survjax.InvalidDataError\`, a subclass of \`ValueError\`. | `test_invalid_data_error_is_a_public_value_error` |
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
| 7.2 | compiled construction and diagnostic access run "without ... host callbacks". | `test_compiled_construction_and_diagnostics_install_no_host_callback` |
| 7.2 | ... "without library-induced host transfers" once warmed. | `test_warm_compiled_construction_and_diagnostics_make_no_host_transfer` |
| 7.2 | "Changing dynamic ... observation values must reuse compilation" (construction inside jit). | `test_compiled_construction_reuses_compilation_for_new_observation_values` |

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

## A4: distribution base (spec 4.1, 4.2, 4.5; rev 11)

All A4 tests use custom families defined in `spec_families.py` through the
public extension interface; built-in families start at A5.

### `test_a4_registration.py`: registration, PyTree, static and nested fields

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.2 | `survjax.SurvivalDistribution`, `register_family`, `param`, `static` and `distribution` exist. | `test_extension_names_are_exported_at_top_level` |
| 4.2 | the decorator keeps the class a subclass of `SurvivalDistribution`. | `test_registered_family_is_a_survival_distribution` |
| 4.2 | registered families are constructed with keyword arguments only; positional raises `TypeError`. | `test_positional_parameters_are_rejected` |
| 4.2 | a single positional parameter also raises `TypeError`. | `test_single_positional_parameter_is_rejected` |
| 4.2 | keyword-only applies to static fields too. | `test_positional_static_is_rejected` |
| 4.2 | "Parameter fields have no default and are required". | `test_missing_parameter_is_rejected` |
| 4.2 | with `kw_only=True`, a defaulted static field may precede a required parameter. | `test_kw_only_recipe_allows_static_before_parameter` |
| 4.1/4.2 | parameter arrays are dynamic leaves; a static field is not a leaf. | `test_parameters_are_the_only_leaves` |
| 4.1 | natural parameters are stored as the leaves (values unchanged, not transformed). | `test_leaves_hold_the_natural_parameter_values` |
| 4.2 | register_family generates PyTree flatten/unflatten. | `test_flatten_unflatten_round_trip_preserves_type_and_parameters` |
| 4.2 | reconstruction bypasses `__init__` (arbitrary leaves are neither broadcast nor converted). | `test_unflatten_bypasses_init` |
| 4.2 | `static(...)` holds hashable configuration carried by the PyTree. | `test_static_field_is_preserved_through_round_trip` |
| 4.2 | `survjax.static(default=...)` supplies the field default. | `test_static_default_is_used_when_omitted` |
| 4.2 | static configuration is not data; equal configuration and shapes share a structure. | `test_equal_static_values_give_equal_tree_structures` |
| 4.2 | static configuration is part of the tree structure. | `test_different_static_values_give_different_tree_structures` |
| 4.2 | "There is no automatic subclass registration": an undecorated subclass is a PyTree leaf. | `test_no_automatic_subclass_registration` |
| 4.1/4.2 | distributions are immutable frozen dataclasses. | `test_distribution_is_immutable` |
| 4.1 | methods are pure (no mutation or caches): leaves are unchanged after calls. | `test_methods_do_not_mutate_parameters` |
| 4.1 | a distribution is a PyTree that can be passed into and returned from jit. | `test_distribution_passes_through_jit` |
| 4.1 and 7.2 | parameters are dynamic leaves; changed values with equal shapes, dtypes and statics do not retrace. | `test_new_parameter_values_reuse_compilation` |
| 4.2 and 7.2 | static configuration is compile-time; a different value retraces. | `test_different_static_values_compile_separately` |
| 4.2 | a static field is concrete during tracing, so Python control flow on it works under jit. | `test_static_value_selects_behaviour_under_jit` |
| 4.2 | `survjax.distribution()` declares a nested distribution; its parameters are dynamic leaves too. | `test_nested_distribution_parameters_are_leaves_of_the_outer_family` |
| 4.2 | flatten/unflatten reconstructs the nested distribution with its own type. | `test_nested_distribution_round_trips_with_its_type` |
| 4.1/4.2 | nested parameters are dynamic leaves, differentiable under jit. | `test_nested_distribution_is_differentiable_through_jit` |

### `test_a4_parameters.py`: batch shape, broadcasting, values, traced construction, dtype

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.2 | `batch_shape` from parameter fields; scalars give `()`. | `test_scalar_parameters_give_empty_batch_shape` |
| 4.2 | `batch_shape` from parameter fields. | `test_batch_shape_is_the_parameter_shape` |
| 4.1/4.2 | `batch_shape` excludes component axes (here `w` has `component_ndim=1`). | `test_batch_shape_excludes_component_axes` |
| 4.1 | batch dimensions may be multidimensional, e.g. `(D, N)`. | `test_multidimensional_batch_shape` |
| 4.1 | `batch_shape` is "a property returning a tuple of Python ints". | `test_batch_shape_is_a_tuple_of_python_ints` |
| 4.1 | `batch_shape` is Python ints, so it can size arrays inside `jit`. | `test_batch_shape_is_static_under_jit` |
| 4.1 | leading (batch) axes of a component parameter that do not broadcast raise `TypeError`. | `test_component_parameter_with_incompatible_batch_is_rejected` |
| 4.1 | "every parameter leaf shares the leading batch shape" after construction. | `test_batch_dimensions_are_broadcast_at_construction` |
| 4.1 | broadcasting replicates values; it does not transform them. | `test_broadcasting_keeps_parameter_values` |
| 4.1 | a scalar parameter leaf gains the batch shape of the others. | `test_scalar_parameter_is_broadcast_to_batch_shape` |
| 4.1 | "component axes such as a coefficient vector stay trailing". | `test_component_axis_stays_trailing_when_batch_is_broadcast` |
| 4.1/4.2 | a component axis whose length equals the batch length is still a component axis. | `test_component_axis_is_not_broadcast_against_batch` |
| 4.1/4.2 | leading axes of a component parameter are batch axes and broadcast the others. | `test_batch_shape_can_come_from_a_component_parameter` |
| 4.1 | incompatible parameter batch shapes at construction are static misuse (`TypeError`). | `test_incompatible_batch_shapes_are_rejected` |
| 4.1 | `.shape` is never the array shape; a `shape` keyword is a parameter leaf. | `test_weibull_style_shape_keyword_is_the_parameter` |
| 4.1 | "The base class defines no `shape` attribute". | `test_distribution_without_shape_field_has_no_shape_attribute` |
| 4.1 | accessing `.shape` on a family without that field raises `AttributeError`, so `getattr` defaults work. | `test_missing_shape_attribute_raises_attribute_error` |
| 4.1 | "Constructors never check parameter values"; the value is stored unchanged. | `test_constructor_accepts_negative_parameter_eagerly` |
| 4.1 | constructors never check parameter values (NaN, inf are stored). | `test_constructor_accepts_nan_and_infinite_parameters_eagerly` |
| 4.1 | eager and jit construction behave the same for invalid parameter values. | `test_constructor_accepts_invalid_parameters_under_jit` |
| 4.1 | construction with traced parameters inside a model must work. | `test_construction_with_traced_parameters_inside_jit` |
| 4.1 | batch broadcasting at construction also applies to traced parameters. | `test_traced_construction_broadcasts_like_eager` |
| 4.1 | construction from batched traced parameters inside `vmap` works. | `test_construction_inside_vmap` |
| 4.1 | traced construction inside an objective is differentiable. | `test_gradient_through_construction` |
| 3.3 | parameters follow the data dtype policy; integer-only input uses the default float dtype. | `test_integer_parameters_use_default_float_dtype` |
| 3.3 | scalar-only input uses the configured default floating dtype. | `test_python_scalar_parameters_use_default_float_dtype` |
| 3.3 | Python lists behave as `jnp.asarray` of the list (floats: default float dtype). | `test_python_list_parameter_behaves_as_jnp_asarray` |
| 3.3 | the result is at least float32. | `test_float16_parameter_is_promoted_to_float32` |
| 3.3 | weak typing: a Python scalar parameter adopts the dtype of an array parameter. | `test_python_scalar_adopts_float32_array_dtype` |
| 3.3 | parameters normalize to a common floating dtype using JAX promotion. | `test_mixed_float_parameters_share_promoted_dtype` |
| 3.3 | the dtype policy keeps structured (component) dimensions and applies to them too. | `test_component_parameter_shares_the_common_dtype` |
| 3.3 | float64 array input with 64-bit mode disabled converts to float32; not an error. | `test_float64_parameter_converts_to_float32_under_32bit_mode` |
| 3.3 | one warning per call site, even when several parameters arrive as float64 (Q-004). | `test_float64_parameters_warn_once_per_call` |
| 3.3 | Python scalars and lists carry no chosen precision and convert without a warning. | `test_python_scalar_parameters_convert_without_warning` |
| 3.3 | boolean values are argument misuse (`TypeError`) for parameters as for times. | `test_boolean_parameter_is_rejected` |
| 3.3 | complex values are argument misuse (`TypeError`). | `test_complex_parameter_is_rejected` |
| 3.3 | strings are argument misuse (`TypeError`). | `test_string_parameter_is_rejected` |
| 4.2 | dtype normalization applies to parameters; static configuration stays as given. | `test_static_field_is_not_converted` |
| 3.3 | traced parameters follow the same policy (float32 traced input stays float32). | `test_traced_parameters_keep_their_dtype` |
| 3.3 | object inputs raise `TypeError`. | `test_object_parameter_is_rejected` |
| 3.3 | "Constructors take no `dtype` keyword; callers cast inputs themselves". | `test_constructor_takes_no_dtype_keyword` |
| 3.3 | "Nested distribution fields keep their own dtypes"; the outer family's leaves are normalized on their own. | `test_nested_distribution_keeps_its_own_dtype` |

### `test_a4_indexing.py`: batch indexing, tree-map slicing, vmap

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.1 | `dist[idx]` indexes batch dimensions. | `test_integer_index_gives_scalar_batch` |
| 4.1 | `dist[idx]` with a slice selects batch entries. | `test_slice_index_keeps_selected_entries` |
| 4.1 | `dist[idx]` with an integer array (a minibatch) selects batch entries. | `test_integer_array_index_selects_entries` |
| 4.1 | indexing a `(2, 3)` batch on its first axis gives batch `(3,)` for every leaf. | `test_first_axis_index_of_two_dimensional_batch` |
| 4.1 | `dist[:, j]` indexes the second batch axis. | `test_second_axis_index_of_two_dimensional_batch` |
| 4.1 | indexing every batch axis gives a scalar distribution. | `test_full_index_of_two_dimensional_batch` |
| 4.1/4.2 | batch indexing never touches trailing component axes. | `test_indexing_keeps_component_axes` |
| 4.2 | indexing changes leaves, not static configuration. | `test_indexing_preserves_static_fields` |
| 4.1 | `dist[idx].log_prob(t)` equals `dist.log_prob(t)[idx]`. | `test_indexed_distribution_matches_indexed_values` |
| 4.1 | `batch_shape` follows the current leaves after `jax.tree.map` slicing (it is not stored). | `test_tree_map_slicing_updates_batch_shape` |
| 4.1 | "`jax.tree.map` slicing ... over a distribution work[s]". | `test_tree_map_slicing_works` |
| 4.1 | tree-map slicing and `dist[idx]` agree on leaves and tree structure. | `test_tree_map_slice_matches_index_slice` |
| 4.1 | `vmap` over a distribution works and agrees with the batched method. | `test_vmap_over_distribution_matches_batched_call` |
| 4.1/5 | `vmap` over a distribution and a time vector pairs entries, like the batched call. | `test_vmap_over_distribution_and_time_pairs_entries` |
| 4.1 | `batch_shape` is "computed from the current parameter leaves ...; under `vmap` it is the per-entry shape". | `test_batch_shape_inside_vmap_is_the_per_entry_shape` |
| 4.1/4.2 | under `vmap` each entry sees its full component vector. | `test_vmap_over_component_family_keeps_components_inside` |
| 4.1 | batch indexing composes with jit (dynamic integer index). | `test_jit_of_indexing_with_traced_index` |

### `test_a4_log_prob_default.py`: autodiff `log_prob` default, shapes, derivatives

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.2 | default `log_prob` is `log(-d/dt log S) + log S`; here `log r - r t`. | `test_default_log_prob_of_constant_hazard` |
| 4.2 | default `log_prob` for a Weibull-like family with `k<1`. | `test_default_log_prob_of_power_hazard_decreasing_hazard` |
| 4.2 | default `log_prob` for a Weibull-like family with `k=1` (`log(1/b) - t/b`). | `test_default_log_prob_of_power_hazard_constant_hazard` |
| 4.2 | default `log_prob` for a Weibull-like family with `k>1`. | `test_default_log_prob_of_power_hazard_increasing_hazard` |
| 4.2 | the default is accurate "wherever the hazard is representable in linear space". | `test_default_log_prob_over_wide_time_range` |
| 4.2 | the default works for families with component axes. | `test_default_log_prob_with_component_parameter` |
| 4.2 | the default works through a nested distribution field. | `test_default_log_prob_of_nested_family` |
| 4.2 | `log_prob` is an optional override; a family's own method replaces the default. | `test_log_prob_override_is_used` |
| 4.5 | direct methods "carry no issue masks": the result is one array, not a tuple. | `test_log_prob_returns_a_single_array` |
| 3.3 | calculations use the common promoted dtype (default float here). | `test_log_prob_dtype_matches_parameters` |
| 5 | scalar time and scalar distribution give `()`. | `test_scalar_time_scalar_distribution_gives_scalar` |
| 5 | time scalar, distribution `(N,)` gives `(N,)`. | `test_scalar_time_batched_distribution_gives_batch_shape` |
| 5 | time `(N,)`, distribution `(N,)` pair elementwise. | `test_paired_time_and_distribution` |
| 5 | trailing-axis broadcasting: time `(K, 1)` against batch `(N,)` gives `(K, N)`. | `test_time_column_against_batch_gives_outer_shape` |
| 5 | time `(K,)` against a scalar distribution gives `(K,)`. | `test_time_vector_against_scalar_distribution` |
| 5 | "Component axes never appear in outputs". | `test_component_axes_never_appear_in_outputs` |
| 4.1/5 | a batch `(D,)` against time `(N,)` with different lengths, neither 1, raises `TypeError`. | `test_mismatched_lengths_do_not_broadcast` |
| 4.1 | time `(4, 1)` against batch `(2, 3)` does not broadcast: `TypeError`. | `test_time_column_not_broadcasting_against_two_dimensional_batch_is_rejected` |
| 4.1 | the shape rule applies to "a direct method", including `log_survival` and `survival`. | `test_mismatched_time_is_rejected_by_every_direct_method` |
| 4.1 | a nested family's direct method also raises `TypeError` on a non-broadcasting time. | `test_mismatched_time_is_rejected_for_nested_family` |
| 4.1/6.2 | static shape misuse raises also when traced. | `test_mismatched_time_is_rejected_under_jit` |
| 12.1 | eager and `jit` agree. | `test_default_log_prob_agrees_under_jit` |
| 12.1 | eager and `vmap` (over a time axis) agree. | `test_default_log_prob_agrees_under_vmap_over_time` |
| 7.3 | parameter gradient of the default; `d/dr (log r - r t) = 1/r - t`. | `test_gradient_of_default_log_prob_constant_hazard` |
| 7.3 | parameter gradients of the default for a Weibull-like family. | `test_gradient_of_default_log_prob_power_hazard` |
| 7.3 | first parameter derivatives in forward and reverse mode agree. | `test_forward_and_reverse_gradients_agree` |
| 7.3 | second parameter derivatives of the default (one extra autodiff order); `d2/dr2 = -1/r**2`. | `test_hessian_of_default_log_prob_constant_hazard` |
| 7.3 | `d2/dk2 log f = -1/k**2 - z**k u**2` (from `d/dk = 1/k + u - z**k u`). | `test_hessian_of_default_log_prob_power_hazard_in_shape` |
| 7.3/12.1 | `check_grads(order=2, modes=("fwd","rev"))` on an interior case. | `test_check_grads_order_two_constant_hazard` |
| 7.3/12.1 | `check_grads(order=2, modes=("fwd","rev"))` for the Weibull-like default. | `test_check_grads_order_two_power_hazard` |
| 7.3 | parameter gradients compose with `jit` and `vmap`; `d/dr = 1/r - t`. | `test_gradient_of_default_log_prob_under_jit_and_vmap` |

### `test_a4_value_policy.py`: direct-method value policy for `log_prob`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.5 | `t<0` gives `log_prob=-inf`. | `test_negative_time_log_prob_is_minus_infinity` |
| 4.5 | methods are total over `[-inf, +inf]`; `t=-inf` is `t<0`. | `test_minus_infinity_time_log_prob_is_minus_infinity` |
| 4.5 | `t<0` gives `-inf` even when the density is infinite at `0` (Weibull-like `k<1`). | `test_negative_time_log_prob_for_infinite_density_at_zero` |
| 4.5 | rev 26: at `t=0` a `log_prob` override supplies the density limit; Weibull-like `k<1` gives `+inf`. | `test_zero_time_log_prob_override_infinite_density_for_shape_below_one` |
| 4.5 | rev 26: Weibull-like `k=1` override at `t=0` gives `log(1/b)` (no epsilon, no NaN). | `test_zero_time_log_prob_override_density_for_shape_one` |
| 4.5 | rev 26: Weibull-like `k>1` override at `t=0` gives `-inf`. | `test_zero_time_log_prob_override_zero_density_for_shape_above_one` |
| 4.5 | rev 26: "an override of `log_prob` ... is called at `t=0`" (sentinel override). | `test_zero_time_log_prob_override_is_called` |
| 4.5 | rev 26: a time vector starting at `t=0` with Weibull-like `k=1` (override) is finite: `log(1/b) - t/b`. | `test_grid_from_zero_with_shape_one_is_finite` |
| 4.5 | rev 26: `t=+inf` gives `log_prob=-inf` (base-supplied). | `test_infinite_time_log_prob_constant_hazard` |
| 4.5 | `t=+inf` gives `-inf` for a decreasing hazard (Weibull-like `k<1`). | `test_infinite_time_log_prob_with_vanishing_hazard` |
| 4.5 | `t=+inf` gives `-inf`, not NaN, when the hazard diverges (Weibull-like `k>1`). | `test_infinite_time_log_prob_with_diverging_hazard` |
| 4.5 | NaN `t` gives NaN. | `test_nan_time_log_prob_is_nan` |
| 4.5 | NaN `t` gives NaN (Weibull-like family). | `test_nan_time_log_prob_is_nan_for_power_hazard` |
| 4.5 | every case is handled per element of one array, without raising (rev 26: `t=0` left out). | `test_policy_applies_elementwise_in_one_call` |
| 4.5/12.1 | policy values are the same eagerly and under `jit` (rev 26: `t=0` left out). | `test_policy_values_agree_under_jit` |
| 4.5/12.1 | policy values are the same eagerly and under `vmap` over time (rev 26: `t=0` left out). | `test_boundary_values_agree_under_vmap` |
| 4.5 | invalid parameters give NaN, without raising. | `test_invalid_parameter_gives_nan_log_prob` |
| 4.5 | invalid (NaN) parameters give NaN at an interior time. | `test_nan_parameter_gives_nan_log_prob` |
| 4.5 | an invalid batch entry gives NaN only in its own position. | `test_invalid_parameter_does_not_affect_other_batch_entries` |

### `test_a4_generic_defaults.py`: other pointwise direct-method defaults (Slice A by rev 14)

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 8.1 | `survival(t)` is `P(T>t)`, i.e. `exp(log_survival(t))`. | `test_default_survival_is_exp_log_survival` |
| 4.2 | default `cumulative_hazard = -log S`. | `test_default_cumulative_hazard_is_minus_log_survival` |
| 4.2/7.2 | rev 25: default `log_cdf = log1mexp(log S)`, with zero absolute tolerance (`H=15`, `log F=-3e-7`). | `test_default_log_cdf` |
| 4.2 | the default `log_cdf` is accurate until `\|log S\|` underflows. | `test_default_log_cdf_keeps_precision_for_tiny_cdf` |
| 4.2 | default `log_hazard = log(-d/dt log S)`; here `log r` at every time. | `test_default_log_hazard_constant_hazard` |
| 8.1 | `hazard(t) = f(t)/S(t)`; here `r`. | `test_default_hazard_constant_hazard` |
| 4.2 | default `log_hazard` for a Weibull-like family: `log(k/b) + (k-1) log(t/b)`. | `test_default_log_hazard_power_hazard` |
| 4.2 | the derivative form stays accurate where `H` is large (no `log_prob - log_survival` cancellation). | `test_default_log_hazard_with_large_cumulative_hazard` |
| 4.5 | `t<0` gives `S=1`. | `test_negative_time_survival_is_one` |
| 4.5 | `t<0` gives `log_cdf=-inf`. | `test_negative_time_log_cdf_is_minus_infinity` |
| 4.5 | `t<0` gives `H=0`. | `test_negative_time_cumulative_hazard_is_zero` |
| 4.5 | `t<0` gives `h=0`. | `test_negative_time_hazard_is_zero` |
| 4.5 | rev 21: `t<0` gives `log_hazard=-inf`. | `test_negative_time_log_hazard_is_minus_infinity` |
| 4.5 | rev 26: at `t=0` the base supplies `S(0)=1`, `H(0)=0`, `log F(0)=-inf`. | `test_zero_time_survival_cumulative_hazard_and_log_cdf` |
| 4.5 | rev 26: at `t=+inf`, `S` follows from `log_survival_at_infinity()` (default `-inf`, so `S=0`). | `test_infinite_time_survival_is_zero` |
| 4.5 | rev 26: `t=+inf` gives `H = -log S = +inf` with the proper default. | `test_infinite_time_cumulative_hazard_is_infinite` |
| 4.5 | rev 26: `t=+inf` gives `log F = log1mexp(-inf) = 0` with the proper default. | `test_infinite_time_log_cdf_is_zero` |
| 4.5 | NaN `t` gives NaN (`survival`). | `test_nan_time_survival_is_nan` |
| 4.5 | NaN `t` gives NaN (`cumulative_hazard`). | `test_nan_time_cumulative_hazard_is_nan` |
| 4.5 | NaN `t` gives NaN (`log_cdf`). | `test_nan_time_log_cdf_is_nan` |
| 4.5 | NaN `t` gives NaN (`log_hazard`). | `test_nan_time_log_hazard_is_nan` |
| 4.5 | NaN `t` gives NaN (`hazard`). | `test_nan_time_hazard_is_nan` |
| 4.5 | invalid (NaN) parameters give NaN. | `test_invalid_parameter_gives_nan_survival_and_log_cdf_defaults` |
| 12.1 | eager and `jit` agree for the generic defaults, including boundary inputs. | `test_defaults_agree_under_jit` |

### `test_a4_policy_enforcement.py`: base applies the value policy (rev 12, Q-019 resolved)

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.2/4.5 | `log_survival` is called only on `[0, +inf]`; `t<0` gives `log S = 0`. | `test_unguarded_log_survival_is_zero_for_negative_time` |
| 4.5 | `t<0` gives `log_prob=-inf` for a family whose formula is undefined there. | `test_unguarded_default_log_prob_is_minus_infinity_for_negative_time` |
| 4.5 | `t<0` gives `S=1`. | `test_unguarded_survival_is_one_for_negative_time` |
| 4.5 | `t<0` gives `H=0`. | `test_unguarded_cumulative_hazard_is_zero_for_negative_time` |
| 4.5 | `t<0` gives `log_cdf=-inf`. | `test_unguarded_log_cdf_is_minus_infinity_for_negative_time` |
| 4.5 | `t<0` gives `h=0`. | `test_unguarded_hazard_is_zero_for_negative_time` |
| 4.5 | rev 21: `t<0` gives `log_hazard=-inf` for a family whose formula is undefined there. | `test_unguarded_log_hazard_is_minus_infinity_for_negative_time` |
| 4.5 | NaN `t` gives NaN for `log_survival` and `log_prob`. | `test_unguarded_nan_time_gives_nan` |
| 4.5 | rev 26: at `t=0`, `log S=0`, `S=1`, `log_cdf=-inf`, `H=0` come from the base (`k` in `0.5, 1, 2`). | `test_unguarded_zero_time_survival_values_come_from_the_base` |
| 4.5 | rev 26: at `t=+inf` the base gives `log_prob=-inf` (also where a default formula would give `+inf - inf`). | `test_unguarded_infinite_time_log_prob_is_minus_infinity` |
| 4.5 | rev 26: at `t=+inf`, `log_survival` equals `log_survival_at_infinity()` (default `-inf`). | `test_unguarded_infinite_time_log_survival_comes_from_the_base` |
| 4.2 | rev 26: for finite `t>0` the family's own formula is used. | `test_unguarded_interior_values_are_the_family_formula` |
| 4.5/5 | rev 26: the policy applies per element of a broadcast `(K, N)` `log_cdf` result (`t<0`, `t=0`, interior). | `test_unguarded_policy_applies_elementwise_with_batches` |
| 4.5/12.1 | rev 26: policy values are the same eagerly and under `jit` (`t=0` only for `log_survival`). | `test_unguarded_policy_agrees_under_jit` |
| 4.5 | the base applies the policy to "methods the family overrides" (`t<0`: `-inf`). | `test_unguarded_log_prob_override_is_minus_infinity_for_negative_time` |
| 4.5 | NaN `t` gives NaN for an overridden method. | `test_unguarded_log_prob_override_nan_time_gives_nan` |
| 4.2 | rev 26: the override is used for finite `t>0` (same value as the hand-derived density). | `test_unguarded_log_prob_override_supplies_interior_values` |
| 4.5 | policy values have no NaN parameter gradients; `log S = 0` is constant, so gradients are 0. | `test_negative_time_log_survival_has_zero_parameter_gradients` |
| 4.5 | `S=1` for `t<0` has zero parameter gradients. | `test_negative_time_survival_has_zero_parameter_gradients` |
| 4.5 | rev 20: "With valid parameters the policy values keep zero parameter gradients" (`log_prob=-inf`). | `test_negative_time_log_prob_has_zero_parameter_gradients` |
| 4.5 | rev 20: `H=0` for `t<0` keeps zero parameter gradients. | `test_negative_time_cumulative_hazard_has_zero_parameter_gradients` |
| 4.5 | rev 20: `log_cdf=-inf` for `t<0` keeps zero parameter gradients. | `test_negative_time_log_cdf_has_zero_parameter_gradients` |
| 4.5 | rev 20: `h=0` for `t<0` keeps zero parameter gradients. | `test_negative_time_hazard_has_zero_parameter_gradients` |
| 4.5 | rev 21: `log_hazard=-inf` for `t<0` keeps zero parameter gradients. | `test_negative_time_log_hazard_has_zero_parameter_gradients` |
| 4.5 | an overridden method is called only on a safe input under `stop_gradient`; rev 20: its policy value keeps zero parameter gradients. | `test_negative_time_override_has_zero_parameter_gradients` |

### `test_a4_time_dtypes.py`: dtype of the direct-method `t` (rev 20)

Spec 4.5: "The `t` argument follows JAX conversion and promotion. Boolean,
complex, string and object `t` raise `TypeError` ..., also during tracing"
(rev 21; bool and complex since rev 20); "a float64 array `t` with 64-bit mode
disabled is converted to float32 without a warning" (rev 20). Parametrized tests run
for all seven Slice A direct methods (`log_survival`, `survival`, `log_cdf`,
`cumulative_hazard`, `log_prob`, `log_hazard`, `hazard`).

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.5 | boolean `t` raises `TypeError` (Python `bool`). | `test_python_bool_time_is_rejected[*]` |
| 4.5 | boolean `t` raises `TypeError` (NumPy boolean array). | `test_numpy_boolean_array_time_is_rejected[*]` |
| 4.5 | boolean `t` raises `TypeError` (JAX boolean array). | `test_jax_boolean_array_time_is_rejected[*]` |
| 4.5 | complex `t` raises `TypeError` (NumPy complex array). | `test_complex_array_time_is_rejected[*]` |
| 4.5 | complex `t` raises `TypeError` (Python `complex`). | `test_python_complex_time_is_rejected[*]` |
| 4.5 | rev 21: string `t` raises `TypeError`. | `test_string_time_is_rejected[*]` |
| 4.5 | rev 21: object `t` raises `TypeError`. | `test_object_array_time_is_rejected[*]` |
| 4.5 | rev 21: boolean `t` raises `TypeError` "also during tracing" (`jit`). | `test_boolean_time_is_rejected_under_jit` |
| 4.5 | rev 21: complex `t` raises `TypeError` "also during tracing" (`jit`). | `test_complex_time_is_rejected_under_jit` |
| 4.5 | rev 21: the traced check covers every direct method. | `test_boolean_time_is_rejected_under_jit_for_every_method[*]` |
| 4.5 | rev 21: boolean `t` raises `TypeError` when traced by `vmap`. | `test_boolean_time_is_rejected_under_vmap` |
| 4.5 | float64 array `t` under 32-bit mode converts without a warning (skipped under 64-bit mode). | `test_float64_time_converts_without_warning_under_32bit_mode[*]` |
| 4.5 | float64 `t` is converted to float32; result float32 with the hand value (32-bit mode only). | `test_float64_time_result_is_float32_under_32bit_mode` |
| 4.5 | `t` follows JAX promotion; "the output precision is set by the parameters" (float32 `t`, float64 parameters; 64-bit mode only). | `test_float32_time_promotes_to_parameter_dtype_under_64bit_mode` |
| 4.5 | integer `t` gives the equal float time's value (`log_survival`, hand-derived). | `test_integer_array_time_log_survival_matches_float_time` |
| 4.5 | integer `t` gives the equal float time's value for the autodiff `log_prob` default (rev 26: `t=1,2,3`, avoiding `t=0`). | `test_integer_array_time_log_prob_matches_float_time` |
| 4.5 | integer `t` gives the equal float time's value for the `hazard` default. | `test_integer_array_time_hazard_matches_float_time` |
| 4.5 | Python `int` `t` gives the equal float time's value. | `test_python_int_time_matches_float_time` |
| 4.5 | integer `t<0` gets the `t<0` policy value. | `test_negative_integer_time_gets_policy_value` |

### `test_a4_nan_precedence.py`: NaN family value at `t<0` and NaN `t` (rev 20)

Spec 4.5: "For `t<0` and NaN `t`, if the family's value at the substituted
input is NaN, the result is NaN; otherwise the policy value applies."

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.5 | `rate=NaN` at `t=-1` gives NaN, for each of the seven direct methods. | `test_nan_parameter_gives_nan_at_negative_time[*]` |
| 4.5 | the same at `t=-inf`. | `test_nan_parameter_gives_nan_at_minus_infinity[*]` |
| 4.5 | the same for an unguarded family (`naive_power`, `scale=NaN`). | `test_nan_parameter_gives_nan_at_negative_time_for_unguarded_family[*]` |
| 4.5 | batch `rate=(NaN, 2)` at `t=-1`: NaN entry NaN; valid entry keeps `log S=0`, `S=1`, `log F=-inf`, `H=0`, `log_prob=-inf`, `h=0`, and (rev 21) `log_hazard=-inf`. | `test_nan_batch_entry_does_not_change_valid_entry_policy_value[*]` |
| 4.5/5 | elementwise over a broadcast `(N, K)` result of times and batch. | `test_nan_precedence_applies_elementwise_over_times` |
| 4.5/12.1 | the same values eagerly and under `jit`. | `test_nan_precedence_agrees_under_jit` |
| 4.5 | the rule covers NaN `t`: NaN `t` with a NaN parameter is NaN. | `test_nan_parameter_with_nan_time_gives_nan` |
| 4.5 | rev 26: the base-supplied `log_prob(+inf)` is NaN when the family's value at the substituted input is NaN (guarded and unguarded families). | `test_nan_parameter_log_prob_at_infinity_is_nan`, `test_nan_parameter_log_prob_at_infinity_is_nan_for_unguarded_family` |
| 4.5 | rev 26: elementwise: a valid `k=2` entry keeps `log_prob(+inf)=-inf` next to a NaN entry. | `test_nan_batch_entry_log_prob_at_infinity_keeps_valid_entry_minus_infinity` |
| 4.5 | "With valid parameters the policy values keep zero parameter gradients". | the six `test_negative_time_*_has_zero_parameter_gradients` tests above, plus the rev 12 `log_survival` and `survival` ones |

### `test_a4_hazard_at_zero_survival.py`: `log_hazard` and `hazard` where `S(t)=0` and at `t=+inf` (rev 13, 26)

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.2/4.5 | rev 26: at `t=+inf` the generic `log_hazard` default gives NaN. | `test_default_log_hazard_is_nan_at_infinity` |
| 4.5 | rev 26: at `t=+inf`, `hazard` from the generic default is NaN. | `test_default_hazard_is_nan_at_infinity` |
| 4.5 | rev 26: the same for a Weibull-like family (`k=2`, derivative form `+inf`) and `k=0.5` (`-inf`). | `test_default_log_hazard_is_nan_at_infinity_for_unguarded_family` |
| 4.2/4.5 | rev 26: elementwise: `log r` at finite `t`, NaN at `+inf`, in one call. | `test_default_log_hazard_is_nan_only_where_survival_is_zero` |
| 4.2 | `S(t)>0` is the mathematical condition, read from `log S`, not from an underflowed `exp(log S)`. | `test_default_log_hazard_with_tiny_but_positive_survival_is_not_nan` |
| 4.5 | rev 26: at `t=+inf` the `log_hazard` override is called and supplies the limit (`log r`). | `test_log_hazard_override_supplies_value_at_infinity` |
| 4.5 | rev 26: `hazard` at `t=+inf` comes from the `log_hazard` override (`r`). | `test_hazard_uses_log_hazard_override_at_infinity` |
| 4.5 | the override is wrapped by the base: `t<0` gives `h=0`, NaN `t` gives NaN. | `test_log_hazard_override_still_follows_policy_for_negative_and_nan_time` |
| 4.5 | rev 21: the `t<0` row applies to a `log_hazard` override: `-inf`, not `log r`. | `test_log_hazard_override_is_minus_infinity_for_negative_time` |
| 4.2 | inside the support the default is `log(-d/dt log S)`; for uniform on `(0, c]`, `-log(c - t)`. | `test_default_log_hazard_inside_bounded_support` |
| 4.5 | where `S(t)=0` beyond a finite support end (`t=c`, `t>c`, `t=+inf`), the default gives NaN. | `test_default_log_hazard_is_nan_at_and_beyond_support_end` |
| 4.5 | `hazard` beyond a finite support end is NaN from the generic default. | `test_default_hazard_is_nan_beyond_support_end` |
| 4.5/12.1 | the same values eagerly and under `jit`. | `test_zero_survival_rule_agrees_under_jit` |

### `test_a4_log_cdf_upper_tail.py`: generic `log_cdf` default in the upper tail (rev 25)

ConstHazard (`log S = -rate t`), `rate = 0.5`, `t = 2H`. `[h]` marks a
parametrization over `H`: upper tail `20, 33, 55` and `70` (float32) or `600`
(float64); derivatives at the split points `0.5, 0.69, 0.70, 1.2` and at
`20, 33`.

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.2/12.1 | "never rounds a representable `log F` to zero": negative, nonzero, against mpmath `log1p(-exp(-H))`, zero absolute tolerance. | `test_default_log_cdf_in_upper_tail_is_negative_and_accurate[h]` |
| 12.1 | the same under `jit` and `vmap`. | `test_default_log_cdf_upper_tail_under_jit`, `test_default_log_cdf_upper_tail_under_vmap_over_time` |
| 7.2/12.1 | values on both sides of the `log1mexp` split `H = log 2`. | `test_default_log_cdf_on_both_sides_of_log1mexp_split` |
| 7.3/12.1 | first and second rate derivatives (`grad`, `jacfwd`, rev-over-rev, fwd-over-rev) at the split and upper-tail points: `t/expm1(H)`. | `test_default_log_cdf_rate_derivatives[h]` |
| 12.1 | `t = (0, interior, upper tail, +inf)` under `jax_debug_nans`. | `test_default_log_cdf_mixed_times_under_debug_nans` |

### `test_a4_nested.py`: nested distribution fields (rev 18)

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.2 | the outer `batch_shape` is the broadcast of parameter and nested batch shapes. | `test_outer_batch_shape_includes_parameter_batch_with_scalar_base` |
| 4.2 | "nested distributions are broadcast to it" at construction. | `test_scalar_base_is_broadcast_to_outer_batch` |
| 4.2 | a nested batch shape counts like a parameter's; outer parameters broadcast to it. | `test_outer_batch_shape_comes_from_nested_distribution` |
| 4.2 | base `(2, 1)` and outer parameter `(3,)` give batch `(2, 3)` for every leaf. | `test_nested_and_parameter_batches_broadcast_together` |
| 4.2 | broadcasting a nested distribution keeps its component axes. | `test_nested_component_axis_stays_trailing` |
| 4.1/4.2 | nested and parameter batches that do not broadcast raise `TypeError`. | `test_incompatible_nested_and_parameter_batches_are_rejected` |
| 3.3/4.2 | broadcasting keeps the nested distribution's dtype (rev 15: nested dtypes are kept). | `test_broadcast_nested_distribution_keeps_its_dtype` |
| 4.2 | "nested static fields are part of its structure". | `test_nested_static_fields_are_part_of_outer_structure` |
| 4.2 | "`survjax.distribution()` takes no arguments in v0". | `test_distribution_helper_takes_no_arguments` |
| 4.2 | `dist[idx]` indexes nested leaves too. | `test_indexing_applies_to_nested_leaves` |
| 4.2 | a base that was scalar before broadcasting is indexed like any batch leaf. | `test_indexing_a_broadcast_scalar_base` |
| 4.2 | `jax.tree.map` slicing reaches nested leaves. | `test_tree_map_slicing_applies_to_nested_leaves` |
| 4.2 | `vmap` over a family built from a scalar base works and matches the batched call. | `test_vmap_over_outer_distribution_with_broadcast_base` |
| 4.2 | values with a `(2, 3)` broadcast of base and outer parameter. | `test_broadcast_nested_values_in_log_prob` |

### `test_a4_base_boundaries.py`: base-supplied survival limits, `log_survival_at_infinity()` (rev 26, 27)

Families: `boundary_nan_hazard` (`log S = -r t (t/t)`, NaN at `0` and `+inf`),
`softplus_log_logistic` (the spec 4.5 test case), `mixture_cure` (improper,
`S(+inf) = c`, overriding `log_survival_at_infinity()`). `[m]` runs over
`log_survival`, `survival`, `log_cdf`, `cumulative_hazard` unless noted.

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.2 | default `log_survival_at_infinity()`: "`-inf` per batch entry ...: an array of shape `batch_shape` in the parameter dtype" (scalar, `(3,)`, component family). | `test_default_log_survival_at_infinity_is_minus_infinity_for_scalar_family`, `test_default_log_survival_at_infinity_has_one_entry_per_batch_entry`, `test_default_log_survival_at_infinity_excludes_component_axes` |
| 4.2/4.5 | `log_survival` is "Called for finite `t>0`; the base supplies the values at `t=0` and `t=+inf`"; rev 27: "For valid parameters the substituted input is a finite `t>0`", so a family NaN at both still gives the policy values (and `log_prob(+inf)=-inf`). | `test_zero_time_value_comes_from_the_base_when_family_is_nan_there[m]`, `test_infinite_time_value_comes_from_the_base_when_family_is_nan_there[m, log_prob]` |
| 4.2 | the family formula is used for finite `t>0` (`log_survival`, default `log_prob`). | `test_nan_at_boundary_family_interior_values_are_its_formula` |
| 4.5 | elementwise over `(-1, 0, 2, +inf, NaN)`. | `test_nan_at_boundary_family_values_elementwise_in_one_call` |
| 4.5/12.1 | the same under `jit` and `vmap`. | `test_nan_at_boundary_family_values_agree_under_jit[m]`, `test_nan_at_boundary_family_values_agree_under_vmap[m]` |
| 4.5 | base-supplied values have gradient "`0` for `S(0)`, `log F(0)`, `H(0)` and, with the proper default, for `S(+inf)` and `log F(+inf)`" (also `log S(0)`). | `test_nan_at_boundary_family_zero_time_has_zero_rate_gradient[m]`, `test_nan_at_boundary_family_infinite_time_has_zero_rate_gradient[*]` |
| 4.5/12.1 | test case: the plain softplus log-logistic "gives `S(0)=1` with zero gradient and `S(+inf)=0`"; also `log S(0)`, `log F(0)`, `H(0)` and `log F(+inf)` gradients. | `test_softplus_log_logistic_survival_at_zero_is_one`, `test_softplus_log_logistic_survival_at_zero_has_zero_gradients`, `test_softplus_log_logistic_zero_time_values_have_zero_gradients[m]`, `test_softplus_log_logistic_survival_at_infinity_is_zero`, `test_softplus_log_logistic_infinite_time_values_have_zero_gradients[*]` |
| 4.2/7.3 | interior log-logistic values and `beta` gradient (hand-derived). | `test_softplus_log_logistic_interior_values`, `test_softplus_log_logistic_interior_gradient_is_finite` |
| 4.2 | "Improper families return `log S(+inf)`, for example the log cure fraction". | `test_improper_family_log_survival_at_infinity_override_is_used` |
| 2/4.5 | improper `t=+inf`: `log S = log_survival_at_infinity()`, `S`, `log_cdf = log1mexp(log S)`, `H = -log S`; `log_prob=-inf`; generic `log_hazard`/`hazard` NaN. | `test_improper_family_values_at_infinity[m]`, `test_improper_family_log_prob_at_infinity_is_minus_infinity`, `test_improper_family_generic_log_hazard_at_infinity_is_nan` |
| 2/4.5 | `S(0)=1` "for every family" (improper too); interior values follow the formula. | `test_improper_family_values_at_zero[m]`, `test_improper_family_interior_values` |
| 4.2/4.5/12.1 | improper `+inf` values are differentiated "through `log_survival_at_infinity()`" (`d/dc` hand-derived, `d/dr = 0`). | `test_improper_family_gradients_at_infinity[m]` |
| 4.2/4.5 | per batch entry, and under `jit` and `vmap`. | `test_improper_family_values_at_infinity_per_batch_entry`, `test_improper_family_values_at_infinity_agree_under_jit_and_vmap[m]` |
| 4.5 | "if ... at `t=+inf` `log_survival_at_infinity()` is NaN, the result is NaN" (`cure = -1`, family NaN everywhere). | `test_nan_log_survival_at_infinity_gives_nan_at_infinity[m, log_prob]` |
| 4.5 | the same where the family is finite at some substituted inputs (`cure = -0.5`: finite for `t < 2.197`), so only the `log_survival_at_infinity()` clause gives NaN. | `test_nan_log_survival_at_infinity_wins_over_finite_family_value[m, log_prob]` |
| 4.5 | a `log_hazard` override is called at `t=0` and supplies the hazard limit; `log_prob(+inf)=-inf` from the base also over a `log_prob` override. | `test_log_hazard_override_is_called_at_zero`, `test_log_prob_override_at_infinity_gives_base_value` |
| 4.5 | base-supplied values at `t=0` and `t=+inf` are NaN when the family is NaN at the substituted input (`rate=NaN`); a valid batch entry keeps the base values. | `test_nan_parameter_gives_nan_at_zero_time[m]`, `test_nan_parameter_gives_nan_at_infinite_time[m]`, `test_nan_batch_entry_keeps_valid_entry_boundary_values[m]` |

### A4: deliberately not tested

- **Finite invalid parameters at `t<0`, and finite invalid parameters on
  `[0, +inf]`** (4.5, rev 21): the substituted input is not specified, so the
  result "may be NaN or the policy value"; on `[0, +inf]` the invalid-parameters
  row gives "whatever the family's formula gives". Neither is a testable
  requirement beyond the NaN cases already covered (Q-026, resolved).
- **Absence of a float64 `t` warning deduplicated by the implementation:**
  `test_float64_time_converts_without_warning_under_32bit_mode` resets
  Python's warning registry but cannot detect a warning that an earlier call
  in the same process suppressed through its own once-only flag.

- **`reference_time()`** (4.2): required of every family, but A4 has no
  behaviour that consumes it; it is exercised by the inactive-input rules of
  7.2 (A6) and by summaries (Slice D). The spec does not say what happens when
  a family omits a required member, so no test defines a family without one.
- **Built-in families and `from_log` constructors** (4.1, 4.5 Weibull rows on
  `Weibull` itself, gradient of `S(0)` in `k`, survival grids): covered at A5
  (section below).
- **`log_interval_prob`, `log_conditional_interval_prob`,
  `inverse_cumulative_hazard`, `support()`, `parameter_issues()`, summaries,
  `sample`, `check_family`** (4.2 optional members): Slices B and D under
  spec 12.2 (rev 14). `parameter_issues()` may be needed for likelihood
  diagnostics at A6 (7.1); its return type is unspecified, to be raised then.
- **Gradients with respect to time, and of family-supplied boundary values
  for custom families:** 7.3 puts time gradients outside the guarantee; rev 26
  gives custom density and hazard limits "no such guarantee". Base-supplied
  values (`log S`, `S`, `log F`, `H` at `0` and `+inf`) are tested for custom
  families in `test_a4_base_boundaries.py`; family-supplied limits on the
  built-ins at A5.
- **The generic `log_prob`, `log_hazard` and `hazard` defaults at `t=0`**
  (4.5, rev 26): "that result carries no value or gradient guarantee". The
  earlier `t=0` density tests now use a `log_prob` override; the
  exponential-like `log r` at `t=0` from the generic default
  (`test_zero_time_constant_hazard_density`) was removed.
- **`log_survival` not being called at `t=0`/`t=+inf`, observed directly:**
  call counting is not observable through the public API; the NaN-at-boundary
  family shows that its boundary values are not used.
- **Overrides of `log_interval_prob` and `log_conditional_interval_prob`
  receiving `0` and `+inf`** (4.2, rev 26), **`H(+inf)` in the solver** (8.2)
  and **`check_family` limit checks** (4.2): Slices B and D.
- **Likelihood limits from the base (7.1, rev 26):** A6 and later; no
  likelihood tests exist yet.
- **`HazardFamily` keeping the proper default** (4.3): Slice D.
- **Forward-mode specifically** (4.2: "by forward-mode autodiff"): the method
  of differentiation is not observable from values; tests check the result,
  first and second parameter derivatives in both modes.
- **Float32 `check_grads(order=2)`:** second-order finite differences in
  float32 are too noisy at `check_grads` default tolerances; float32 runs the
  hand-derived first and second derivative tests instead.
- **Derivative tolerances:** hand-derived gradient and Hessian checks use the
  12.1 ordinary tolerances; the expected values are O(1) smooth interior
  quantities, for which autodiff error is far below those scales.

## A5: Exponential and Weibull closed forms, boundary values, `_grid` twins (spec 4.1, 4.5, 5; rev 21–24)

Hand derivations are in each module docstring. `[*]` marks a test
parametrized over the seven Slice A direct methods (`log_survival`,
`survival`, `log_cdf`, `cumulative_hazard`, `log_prob`, `log_hazard`,
`hazard`) and, in `test_a5_boundaries.py`, over four cases: Exponential
`r=0.5` and Weibull `b=2` with `k` in `(0.5, 1, 2)`. In
`test_a5_survival_grids.py`, `[*]` marks a test parametrized over the seven
`_grid` twins of those methods (rev 24).

### `test_a5_construction.py`: built-in classes, keywords, leaves, batch, `reference_time`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 10.2 | distributions are re-exported at top level. | `test_exponential_is_exported_at_top_level`, `test_weibull_is_exported_at_top_level` |
| 4.1 | rev 22: "Built-in families are registered subclasses of `SurvivalDistribution`". | `test_exponential_is_a_survival_distribution`, `test_weibull_is_a_survival_distribution` |
| 4.2 | "Registered families, built-in and custom, are constructed with keyword arguments only". | `test_exponential_rejects_positional_rate`, `test_weibull_rejects_positional_parameters` |
| 4.2 | "Parameter fields have no default and are required". | `test_weibull_requires_both_parameters`, `test_exponential_requires_rate` |
| 4.1 | "Natural parameters are stored as the leaves, under the names the constructor uses" (`rate`; `shape`, `scale`). | `test_exponential_stores_rate_as_its_only_leaf`, `test_weibull_stores_shape_and_scale_as_its_only_leaves` |
| 4.1 | "The Weibull keyword stays `shape`"; `batch_shape` is the array shape. | `test_weibull_shape_attribute_is_the_parameter_not_the_batch_shape` |
| 4.1 | no `shape` field: `.shape` raises `AttributeError`. | `test_exponential_has_no_shape_attribute` |
| 4.1 | `batch_shape` from the parameter leaves. | `test_scalar_exponential_has_empty_batch_shape` |
| 4.1 | "Batch dimensions are broadcast at construction". | `test_weibull_parameters_are_broadcast_at_construction` |
| 4.1/5 | incompatible parameter batches raise `TypeError`. | `test_weibull_incompatible_parameter_batches_are_rejected` |
| 5 | `Weibull(shape=k[:, None], scale=...)` gives batch `(D,N)`. | `test_weibull_posterior_style_construction_gives_draw_by_individual_batch` |
| 4.1 | "Constructors never check parameter values", eagerly and under `jit`. | `test_built_in_constructors_accept_invalid_values_eagerly`, `test_built_in_constructors_accept_invalid_values_under_jit` |
| 4.1 | construction with traced parameters (`jit`, `vmap`). | `test_weibull_construction_with_traced_parameters_matches_eager`, `test_exponential_construction_inside_vmap` |
| 4.1 | `dist[idx]`, `jax.tree.map` slicing, `vmap` over a built-in. | `test_weibull_indexing_selects_batch_entries`, `test_exponential_tree_map_slicing`, `test_vmap_over_weibull_matches_batched_call` |
| 3.3 | parameter dtype policy on built-ins: integer input, weak typing, promotion (64-bit mode only). | `test_integer_rate_uses_default_float_dtype`, `test_weibull_python_scalar_adopts_float32_array_dtype`, `test_weibull_mixed_float_parameters_share_promoted_dtype` |
| 4.2 | rev 22: `reference_time()` is an array of shape `batch_shape` in the parameter dtype, positive and finite for valid parameters. | `test_exponential_reference_time_is_positive_finite_per_batch_entry`, `test_weibull_reference_time_is_positive_finite_per_batch_entry` |
| 4.1/12.1 | `reference_time()` eager and `jit` agree. | `test_weibull_reference_time_under_jit_matches_eager` |

### `test_a5_from_log.py`: log-parameter constructors

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.1 | `from_log` "return[s] the same class". | `test_exponential_from_log_returns_exponential`, `test_weibull_from_log_returns_weibull` |
| 4.1/4.6 | rev 22: `k=exp(log_shape)`, `b=exp(log_scale)`, `r=exp(log_rate)`, "stored as the natural-parameter leaves". | `test_exponential_from_log_stores_natural_rate`, `test_weibull_from_log_stores_natural_shape_and_scale` |
| 4.1 | batch dimensions broadcast as at construction. | `test_weibull_from_log_broadcasts_batch_dimensions` |
| 4.1 | rev 22: values and gradients "equal those of the ordinary constructor wherever `exp` is finite". | `test_exponential_from_log_values_match_natural_constructor`, `test_weibull_from_log_values_match_natural_constructor` |
| 4.1 | traced construction through `from_log`. | `test_from_log_with_traced_inputs_under_jit` |
| 4.1/7.3 | parameter gradients with respect to the log parameters. | `test_gradient_through_exponential_from_log`, `test_gradient_through_weibull_from_log` |

### `test_a5_closed_forms.py`: values and tails

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.1/8.1 | Exponential `S=exp(-r*t)` and derived methods, paired batch. | `test_exponential_interior_values[*]` |
| 4.1/8.1 | Weibull `S=exp(-(t/b)**k)` for `k<1`, `k=1`, `k>1`. | `test_weibull_interior_values_decreasing_hazard[*]`, `test_weibull_interior_values_constant_hazard[*]`, `test_weibull_interior_values_increasing_hazard[*]` |
| 4.1/5 | batch `(N,)` against times `(N,)` pairs entries. | `test_weibull_paired_batch_values[*]` |
| 4.1 | identity `Weibull(shape=1, scale=b) = Exponential(rate=1/b)`, `t=0` included. | `test_weibull_with_shape_one_equals_exponential_with_rate_one_over_scale[*]` |
| 4.2 | `log_survival` keeps relative precision when `\|log S\|` is small (`H=2e-30`, `1e-20`). | `test_exponential_small_cumulative_hazard_keeps_relative_precision`, `test_weibull_small_cumulative_hazard_keeps_relative_precision` |
| 4.2 | `log_cdf` accurate in the lower tail. | `test_exponential_log_cdf_in_lower_tail`, `test_weibull_log_cdf_in_lower_tail` |
| 4.1 | Weibull `k<1` near `t=0`: large finite `log h`, `log f`. | `test_weibull_log_hazard_near_zero_with_decreasing_hazard`, `test_weibull_log_prob_near_zero_with_decreasing_hazard` |
| 4.1/12.1 | upper tail in log space where `S`, `f` underflow. | `test_exponential_log_survival_in_upper_tail`, `test_exponential_log_prob_in_upper_tail`, `test_weibull_log_survival_in_upper_tail`, `test_weibull_log_prob_in_upper_tail`, `test_weibull_cumulative_hazard_in_upper_tail` |
| 4.2 | "Built-ins override [`log_hazard`] analytically"; the generic form cancels when `H` is large. | `test_exponential_log_hazard_in_upper_tail_is_analytic`, `test_exponential_hazard_in_upper_tail_is_analytic`, `test_weibull_log_hazard_in_upper_tail_is_analytic`, `test_weibull_hazard_in_upper_tail_is_analytic` |
| 8.1 | `survival` underflows to `0`, not NaN. | `test_survival_underflows_to_zero_not_nan_in_upper_tail` |

### `test_a5_log_cdf_tails.py`: built-in `log_cdf` in both tails (rev 25)

Exponential `r = 0.5` at `t = 2H`; Weibull `k = 2.5`, `b = 3` at
`t = 3 H**(1/k)`, with `H` recomputed in mpmath from the rounded `t`. `[h]` as
for `test_a4_log_cdf_upper_tail.py`; `[name]` over Exponential and Weibull
(and ConstHazard for the grid). Upper-tail tolerance: `helpers.tail_rtol`,
zero absolute tolerance, scale justified there (`S` has condition number `H`).

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.1/12.1 | upper tail "negative there and never rounds to zero", against mpmath, at `H` about 20, 33, 55 and deeper. | `test_exponential_log_cdf_in_upper_tail_is_negative_and_accurate[h]`, `test_weibull_log_cdf_in_upper_tail_is_negative_and_accurate[h]` |
| 12.1 | the same "under `jit` and `vmap`, and through `log_cdf_grid`" (grid also for the generic default). | `test_upper_tail_log_cdf_under_jit[name]`, `test_upper_tail_log_cdf_under_vmap_over_time[name]`, `test_upper_tail_log_cdf_grid[name]` |
| 4.1/12.1 | lower tail "even where `H` or `F` underflows": `log F = log H`. | `test_exponential_log_cdf_where_cumulative_hazard_underflows`, `test_weibull_log_cdf_where_cumulative_hazard_underflows` |
| 7.2/12.1 | values on both sides of the split `H = log 2`. | `test_log_cdf_on_both_sides_of_log1mexp_split[name]` |
| 7.3/12.1 | first and second parameter derivatives at the split and upper-tail points (Exponential `t/expm1(H)`; Weibull in `k` and `b`). | `test_exponential_log_cdf_rate_derivatives[h]`, `test_weibull_log_cdf_shape_derivatives[h]`, `test_weibull_log_cdf_scale_derivatives[h]` |
| 12.1 | mixed `0`, interior, upper-tail, `+inf` times under `jax_debug_nans` (Weibull `k` in `0.5, 1, 2.5`). | `test_exponential_log_cdf_mixed_times_under_debug_nans`, `test_weibull_log_cdf_mixed_times_under_debug_nans[k]` |

### `test_a5_boundaries.py`: value policy and boundary values on built-ins

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.5 | `t<0` row (`-1e-30`, `-1`, `-inf`) for all seven methods. | `test_negative_time_gives_policy_value[*]` |
| 4.5 | `t<0` policy values keep zero parameter gradients. | `test_negative_time_policy_values_have_zero_weibull_gradients[*]`, `test_negative_time_policy_values_have_zero_exponential_gradient[*]` |
| 4.5 | rev 26: `t=0` row: `log S`, `S`, `log F`, `H` from the base; Weibull `k<1` infinite density, `k=1` `log(1/b)`, `k>1` `-inf`; Exponential `log r` from the overrides. | `test_zero_time_boundary_value[*]` |
| 4.1 | `xlogy`-style term: `k=1`, `t=0` gives `log(1/b)`, not NaN. | `test_weibull_shape_one_density_at_zero_is_not_nan` |
| 4.5 | rev 26: `t=+inf` row (`log_survival_at_infinity()` is `-inf` for built-ins); `log_hazard`/`hazard` are the override's limit. | `test_infinite_time_boundary_value[*]` |
| 4.5 | NaN `t` gives NaN. | `test_nan_time_gives_nan[*]` |
| 4.5 | rev 26: NaN parameters give NaN at every `t`, base-supplied values included (NaN at the substituted input). | `test_nan_rate_gives_nan_for_every_time[*]`, `test_nan_weibull_scale_gives_nan_for_every_time[*]` |
| 4.5 | an invalid batch entry affects only its own position. | `test_nan_batch_entry_does_not_affect_other_entries` |
| 4.5 | elementwise over one array, without raising. | `test_weibull_policy_applies_elementwise_in_one_call` |
| 4.5/12.1 | boundary values agree under `jit` and `vmap`. | `test_boundary_values_agree_under_jit[*]`, `test_boundary_values_agree_under_vmap_over_time[*]` |
| 4.5 | test case "the gradient of `S(0)` with respect to `k`" (`k` in `0.5, 1, 2`), also in `b`. | `test_gradient_of_weibull_survival_at_zero_in_shape[*]`, `test_gradient_of_weibull_survival_at_zero_in_scale[*]` |
| 4.5 | "avoid NaN parameter gradients where the value is finite": `log S(0)`, `H(0)`; `k=1` `log f(0)`, `log h(0)` in `b`; Exponential in `r`. | `test_gradient_of_weibull_log_survival_and_cumulative_hazard_at_zero[*]`, `test_gradient_in_scale_of_weibull_shape_one_density_at_zero[*]`, `test_gradient_of_exponential_density_at_zero[*]` |
| 4.5/7.3 | the `S(0)` gradient under `jit` and `vmap`. | `test_gradient_of_weibull_survival_at_zero_under_jit_and_vmap` |
| 4.5 | rev 26: at `t=+inf`, the base-supplied `S` and `log F` have zero parameter gradients with the proper default (Weibull `k` in `0.5, 1, 2`; Exponential). | `test_gradient_of_weibull_finite_values_at_infinity[*]`, `test_gradient_of_exponential_finite_values_at_infinity[*]` |
| 4.5 | rev 22: gradients at `+inf` equal the derivative of the limit: Exponential `log h`, `h`; Weibull `k=1` `log h` in `b`. | `test_gradient_of_exponential_hazard_at_infinity`, `test_gradient_of_weibull_shape_one_log_hazard_at_infinity` |
| 4.5/7.3 | rev 22: the `+inf` gradients under `jit` and `vmap`. | `test_gradient_at_infinity_under_jit_and_vmap[*]` |

### `test_a5_survival_grids.py`: `_grid` twins and shape rules

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 5 | scalar time against `(N,)`; paired `(N,)`; `(K,1,1)` against `(D,N)`. | `test_scalar_time_against_batch_gives_batch_shape`, `test_time_vector_against_equal_batch_pairs_entries`, `test_explicit_time_first_axis_gives_shared_grid_shape` |
| 4.1/5 | a time that does not broadcast raises `TypeError`. | `test_time_that_does_not_broadcast_against_batch_is_rejected` |
| 4.2/5/8.1 | rev 24: "Every pointwise direct method of `t` ... has a grid twin named with a `_grid` suffix" (all seven exercised by every `[*]` test). | `test_grid_on_scalar_distribution_has_grid_shape[*]` and the other `[*]` tests |
| 5 | rev 24: returns "`(K, *batch_shape)`": `(K,)`, `(K,N)`, `(K,D,N)`. | `test_grid_on_scalar_distribution_has_grid_shape[*]`, `test_grid_on_batch_is_time_first[*]`, `test_grid_on_two_dimensional_batch_gives_k_d_n[*]` |
| 5 | rev 24: "for batch `(D,N)` this is `m(t[:, None, None])`, shape `(K,D,N)`". | `test_grid_equals_direct_method_on_grid_with_new_axes[*]` |
| 5 | rev 24: with `K=N` a grid is shared, never paired (the motivating `log_prob` case included). | `test_grid_with_length_equal_to_batch_equals_direct_method_on_column[*]`, `test_grid_with_length_equal_to_batch_is_not_paired`, `test_log_prob_grid_with_length_equal_to_batch_is_not_paired` |
| 5 | rev 24: "`t` need not be sorted or distinct; output row `i` belongs to `t[i]`". | `test_grid_accepts_unsorted_repeated_times[*]` |
| 5 | "Component axes never appear in outputs"; the base provides grids for every registered family (custom family). | `test_grid_has_no_component_axes`, `test_custom_family_with_only_log_survival_has_every_grid[*]` |
| 5/8.1 | grid values for each grid method and family. | `test_weibull_survival_grid_values`, `test_weibull_cumulative_hazard_grid_values`, `test_weibull_hazard_grid_values`, `test_weibull_log_survival_grid_values`, `test_weibull_log_prob_grid_values`, `test_weibull_log_cdf_grid_values`, `test_weibull_log_hazard_grid_values`, `test_exponential_grids_values`, `test_exponential_log_grids_values` |
| 4.5/12.1 | "survival grid from `t=0` with Weibull `k=1`", with the section 11.1 `bc` fixed vector (rev 22); hazard and density from `t=0` finite (rev 24: also via `log_prob_grid`). | `test_survival_grid_from_zero_with_weibull_shape_one`, `test_hazard_grid_from_zero_with_weibull_shape_one_is_finite`, `test_log_prob_on_grid_from_zero_with_weibull_shape_one_is_finite`, `test_log_prob_grid_from_zero_with_weibull_shape_one_is_finite` |
| 4.5/5 | grids follow the value policy, including the `S=0` analytic limit; rev 24: "value rules" for the log-scale twins. | `test_survival_grid_applies_value_policy`, `test_hazard_grid_at_infinity_uses_analytic_limit`, `test_log_survival_grid_applies_value_policy`, `test_log_prob_grid_applies_value_policy`, `test_log_cdf_grid_applies_value_policy`, `test_log_hazard_grid_applies_value_policy` |
| 4.5 | float64 NumPy grid under 32-bit mode converts without warning (32-bit mode only). | `test_float64_numpy_grid_converts_without_warning_under_32bit_mode` |
| 12.1/7.3 | grids under `jit`, `vmap` over the distribution, and parameter gradients. | `test_grid_agrees_under_jit[*]`, `test_vmap_over_distribution_of_survival_grid`, `test_gradient_through_survival_grid` |
| 5 | the time vector is "the first positional argument, named `t`". | `test_grid_time_can_be_passed_as_keyword_t[*]` |
| 5 | a scalar or 2-D `t` (including `(K,1)`) raises `TypeError`, also under `jit`. | `test_scalar_grid_is_rejected[*]`, `test_two_dimensional_grid_is_rejected[*]`, `test_column_grid_is_rejected`, `test_two_dimensional_grid_is_rejected_under_jit` |
| 5/4.5 | grid `t` follows the 4.5 dtype rules: boolean raises `TypeError`; integer converts. | `test_boolean_grid_is_rejected[*]`, `test_integer_grid_matches_float_grid` |
| 4.5/5 | rev 26: `survival_grid` through `t=0` and `t=+inf` for the plain softplus log-logistic gives base values `1` and `0`, with a finite (zero) `beta` gradient. | `test_survival_grid_of_naive_log_logistic_uses_base_boundary_values` |

### `test_a5_derivatives.py`: transformations and parameter derivatives

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 12.1 | eager and `jit` agree for every direct method. | `test_weibull_methods_agree_under_jit[*]`, `test_exponential_methods_agree_under_jit[*]` |
| 4.1/12.1 | `vmap` over paired distribution and time agrees. | `test_weibull_methods_agree_under_vmap_over_distribution_and_time[*]` |
| 7.3 | hand-derived first derivatives of `log f` and `log S`. | `test_exponential_log_prob_gradient`, `test_exponential_log_survival_gradient`, `test_weibull_log_prob_gradient`, `test_weibull_log_survival_gradient` |
| 7.3 | forward and reverse modes agree; gradients under `jit` and `vmap`. | `test_weibull_forward_and_reverse_gradients_agree`, `test_weibull_gradient_under_jit_and_vmap` |
| 7.3 | hand-derived second derivatives. | `test_exponential_log_prob_hessian`, `test_weibull_log_prob_hessian_in_shape`, `test_weibull_log_prob_hessian_in_scale` |
| 7.3/12.1 | `check_grads(order=2, modes=("fwd","rev"))` for `log_prob`, `log_survival`, `log_cdf`, `log_hazard` (64-bit mode only), and at Weibull `k=1`. | `test_check_grads_order_two_exponential[*]`, `test_check_grads_order_two_weibull[*]`, `test_check_grads_order_two_weibull_at_shape_one` |

### A5: deliberately not tested

- **`from_log` with overflowing log parameters** (4.1, rev 22): the result
  falls under the invalid-parameters row, which fixes no value (Q-028).
- **Gradient in `k` of Weibull `log f(0)` at `k=1`** (4.5, rev 22): "outside
  the guarantee" (Q-029).
- **NaN-free boundary gradients for custom families** (4.5, rev 22): they
  hold "when their `log_survival` does", which is the author's duty, not the
  base's.
- **Finite invalid parameters** (for example `rate<0`, `shape=0`): the family
  formula decides (4.5 invalid-parameters row); there is no specified value
  to assert beyond NaN parameters.
- **Value of `reference_time()`**: only "positive finite" is specified; its
  value is implementation work (13).
- **Exact equality of the `k=1` Weibull and the exponential:** compared at
  12.1 tolerances, not bitwise.
- **Float64 parameter warning on built-ins** (3.3): the base rule is tested
  at A4 on custom families; built-ins add nothing family-specific.
- **Log-normal and log-logistic, `inverse_cumulative_hazard`, summaries,
  `check_family`:** Slice D.
- **"Families do not implement or override" the `_grid` twins** (4.2/5,
  rev 24): this is a rule for family authors; the spec gives no observable
  outcome for a family that defines a `_grid` method, so nothing is asserted.
  The twins on a custom family that defines only `log_survival` are tested.
- **Grid twins of summaries, an `outer=True` keyword, a reshape helper**
  (rev 24): explicitly out of scope.
- **`vmap` and gradients for every `_grid` twin:** each twin is specified as
  its direct method on reshaped `t`, whose transformations `test_a5_derivatives.py`
  covers; grids add `jit` for all seven, and `vmap` and gradients for
  `survival_grid`.
- **The `log_cdf` switch point and substituted operands** (4.1/7.2, rev 25):
  "an implementation choice"; only values and derivatives on both sides of
  `H = log 2` and NaN-freedom under `jax_debug_nans` are tested.
- **Generic-default lower tail where `|log S|` underflows** (4.2, rev 25): the
  default is accurate only "until `|log S|` underflows"; nothing to assert
  beyond the existing `t=1e-20` case.
- **`log1mexp` itself:** a private helper (7.2); tested only through
  `log_cdf`.
- **Parameter gradients under `jax_debug_nans`:** 12.1 lists values on mixed
  vectors; gradients are tested on finite interior and upper-tail points.
- **Float32 `check_grads(order=2)`:** as at A4, float32 runs the
  hand-derived first and second derivatives instead.
- **Tail scales:** tail tests use the 12.1 tolerances; where the expected
  value is far below 1 (`H=1e-20`, `2e-30`) or far above it (`1e14`), the
  12.1 relative tolerance alone (`helpers.assert_rel_close`), since the
  absolute tolerance would be vacuous or the absolute error scales with the
  magnitude.
