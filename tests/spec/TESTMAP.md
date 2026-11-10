# Test map

Spec: revision 56 (see `.spec-seen`). Unchanged requirements keep their
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
`rev 27` and confirms the improper-family tests belong to A4. Revision 28
(decision 0042: parameter constraints, `ParameterIssues`,
`parameter_conditions()`, the likelihood's `invalid_parameters`; 4.1, 4.2,
4.4, 4.5, 4.7, 6.3, 7.1, 10.2, 12.1, 12.2) adds three A6 files tagged
`rev 28` (step placement: Q-035); it leaves A1–A5 requirements unchanged
(direct-method values are explicitly unaffected), so no earlier test was
changed or retagged. Revision 29 (decision 0043: the base method
`dist.log_likelihood`, identical to the function; `log_likelihood_with_info`
and `LikelihoodIssues` removed; causes of NaN rows read from `data.issues` and
`dist.parameter_issues()` by the 7.1 recipe; 4.1, 4.2, 6.1–6.3, 7.1, 11.4,
12.1, 12.2) adds `test_a6_likelihood_method.py` and rewrites the A6
likelihood tests that read `LikelihoodIssues` masks: they now assert the NaN
rows against `data.issues.invalid_rows` and
`parameter_issues().invalid_entries` (helper `nan_causes`, the recipe), and
the `_with_info` variants of the tests become method variants. Tests whose
assertion or quoted text changed are tagged `rev 29`; the rest keep `rev 28`.
A1–A5 and the `ParameterIssues` tests are unaffected. Revision 30 (4.7:
constraints compare with JAX's ordinary operators in the parameter dtype;
subnormals may be classified either way) changes only
`test_a6_constraints.py`: `test_unit_interval_includes_zero_and_one` now
probes below 0 at `-finfo.tiny` instead of the negative subnormal
`nextafter(0, -inf)`, and new `rev 30` tests pin the cases ordinary
comparison still fixes (`±finfo.tiny`, `-0.0`). Revision 31 (decision
0044: unsupported Slice A modes return NaN rows eagerly and under
transformations, preserving supported rows, without adding data or parameter
issues; the triage recipe checks mode support; 7.1, 12.2) rewrites
`test_a6_unsupported_modes.py` (rev 31; Q-037 resolved) and adds the mode
check to `helpers.nan_causes` (new `helpers.supported_mode`,
`helpers.unsupported_nan`); callers' data are all supported, so their
assertions are unchanged. Revision 32 (decision 0045: the likelihood stops
gradients on parameter entries of rows with issues; isolation to second
order, both modes, every violation class and family; 7.1, 7.2, 7.3, 12.1)
adds `test_a6_likelihood_gradient_isolation.py` and retags the three
gradient tests of `test_a6_likelihood_parameter_issues.py` whose quoted
mechanism text was removed. Revision 33 (decision 0046: right takes
precedence over left for a non-exact `(0, +inf]` row, also on rows with
issues; registration rejects unmarked fields and inherited initializers;
nested fields require registered instances; isolation stops at the
parameter leaves; triage after edited fields; Slice A numerical scope;
3.2, 4.2, 7.1, 12.2) adds three `test_a3_static_metadata.py` tests, four
`test_a4_registration.py` tests and four `test_a4_nested.py` tests, all
tagged `rev 33`. It states no behaviour change, so no existing test was
changed or retagged; no existing test relied on the claims rev 33 withdrew
(isolation upstream of the parameter leaves, exact triage of edited data).
Revision 34 (decision 0047: a valid row whose
log contribution overflows gives `-inf`; with shared parameters an infinite
row leaves the other rows' reverse-mode derivatives outside the guarantee,
forward mode and per-row parameters unaffected; the fixture's survival at
each row's own time, schema defined by the spec tester; 7.1, 7.3, 11.1, 12.1)
adds five `rev 34` overflow tests to `test_a7_likelihood_tails.py`, retags
the A8 survival and grid tests `rev 34`, and resolves Q-039. No existing test
asserted reverse-mode derivatives next to an infinite row. A9 and A10 (the
maintainer's 2026-10-23 replan adding Harrell's C to Slice A; Q-047) were
written against revision 34. Revision 35 (decision 0048: `dist.log_likelihood`
is the only public likelihood entry point; no top-level `survjax.log_likelihood`;
6.1, 7.1, 10.2, 12.1, 12.2) moves every likelihood call in A6–A9 to the
method (helper `ll` now calls it; `ll_method` removed) without retagging
tests whose requirement did not change. It replaces the top-level export and
module tests with `test_no_top_level_log_likelihood`, resolves Q-038,
replaces the function-versus-method equality tests with raw-form versus
`SurvivalData`-form agreement in values, gradients and Hessians, and deletes
tests that became exact duplicates once both sides were the method. Tests
whose assertion or quoted text changed are tagged `rev 35`. Revision 36
(decision 0049: `data.shape`, `len(data)`, undefined truth value; 3.2, 12.1,
12.2) adds `test_a3_shape_and_len.py`, tagged `rev 36` and marked A3 (Q-048).
Revision 37 (decision 0050: first-public-release milestone with
`ProportionalHazards`, its private PH closure and staged RMST, section 8.7;
1, 4.2, 4.4, 4.5, 8.1, 8.4, 8.7, 10.4, 11.2, 11.3, 12.1, 12.2, 13) changes no
requirement that an existing test asserts: no test used PH wrappers or
summaries, and the A10 sections 9.1–9.2 are unchanged. Its new requirements
are not yet tested (see "Not covered" below and Q-049). Revisions 38 and 39 (8.7, 12.1: staged RMST boundary
guarantees narrowed, NaN-entry tangents deferred, Weibull `tau=0`
guarantee first-order only, parameter-derivative tolerances in
log-parameter units) change only staged RMST, which no test covers yet. Revision 40
(decision 0052; 4.4, 7.3, 12.1, 12.2: extreme-scaling PH parameter
derivatives outside the initial-release guarantee) changes only
`ProportionalHazards`, which no test covers yet; the built-in-family
derivative tests of A5–A9 are unaffected. Revision 41 (11.2 neural demo
float32 and censoring) changes no tested requirement. A11 (`ProportionalHazards`
without RMST) and A12 (staged RMST and PH–RMST) follow the maintainer's
2026-11-06 schedule (Q-049) and are written against revision 41, tagged
`rev 41`. Revisions 42–46 (decisions 0053–0057) resolve Q-050–Q-054: keyword-only
wrappers with field `baseline` (rev 42, one new test, two retags), wrapper
`log_hazard` overrides carrying baseline limits (rev 43, four new tests),
RMST derivative tolerance factors (rev 44: the A12 derivative tests and
helpers are retagged; no assertion changed), shape Hessians raise or are
correct (rev 45, retags), and absent APIs plus `tau` shape and dtype rules
(rev 46, four new tests; the absence test now expects `AttributeError`). Revision 47 (use-case 01 censoring limits) and the keyword-only PH example fixes change no test. Revisions 48–50 (8.5, 8.7, 8.8, 12.1, 12.2: the independently gated
post-release generic RMST, mean and quantile stage, its `SummaryOptions` /
`SummaryInfo` schema and superseded verification rows) were first covered
by a step A13 batch. Revision 51 (decision 0058) replaces that fixed stage
with per-method lifting in any order (8.8), so that batch is now relabelled
into post-release feature tests (`test_summary_<feature>_*.py`, marked
`feature(...)`, tagged `rev 51`; see "Post-release features" below). The A12
rejection and absence tests the A13 batch had removed are restored under
`step("A12")` with `until(<feature>)`, so they run until the lifting feature is
enabled (`SPEC_FEATURES`). A12 value and first-derivative regressions are
unchanged. Slice B observation support (12.2) is staged, at the unchanged
revision 51, as the maintainer's features `observation.censoring` and
`observation.truncation` (2026-11-09): the first-release NaN assertions for
those modes in `test_a6_unsupported_modes.py` and `test_a11_ph_likelihood.py`
keep `step(...)` and gain `until(<feature>)` per case, and
`helpers.supported_mode` (the 7.1 recipe's support check) expands with the
enabled features (Q-064). Revision 52 (8.3, 8.5) defines quadrature
convergence as one adjacent pair of summed combined estimates, `num_levels`
as rounds per entry and `error_estimate` as the last difference: the RMST and
mean policy count, work-exhaustion and error-estimate tests are retagged
`rev 52` and tightened, eight tests are new, and Q-061 is narrowed. Revision
53 (8.3) states that a converged quadrature entry returns `Q_l` of the
converging pair and a failure status returns NaN: the RMST and mean NaN-value
tests for `non_finite_arithmetic` and `work_exhausted` and
`test_error_estimate_is_in_time_units` are retagged `rev 53` with unchanged
assertions, and Q-061 is narrowed to the bracket-limit status. Revision 54
(8.1–8.5, 12.1, 12.2) settles Q-057, Q-058, Q-060 to Q-063 and Q-065. Changed
tests (tagged `rev 54`): the three `test_parameter_violations_are_not_reported_converged`
become `test_parameter_violations_report_invalid_input`;
`test_generic_invalid_cutoff_is_not_reported_converged` becomes
`test_invalid_cutoff_reports_invalid_input` (every family, `-inf` added);
`test_invalid_probability_is_not_reported_converged` becomes
`test_invalid_probability_reports_invalid_input`;
`test_bracket_limit_failure_is_reported` becomes
`test_bracket_limit_reports_failed_bracket`;
`test_improper_family_mean_is_never_finite` is replaced by
`test_improper_family_mean_is_plus_infinity` and
`test_improper_family_mean_is_known_infinite`;
`test_improper_mean_has_a_non_finite_tangent` becomes
`test_improper_mean_has_a_nan_tangent`;
`test_upper_endpoint_quantile_has_a_non_finite_tangent` becomes
`test_endpoint_quantile_has_nan_tangents` (`p` 0 and 1, rate and `p`
directions); `test_inverse_cumulative_hazard_at_zero_is_zero` and
`test_improper_family_inverse_cumulative_hazard_beyond_the_cure_level_is_infinite`
(`e=+inf` added) are retagged. New tests cover `converged` endpoint statuses,
the generic known-infinite mean (no quadrature, per entry), `mean_with_info`
overrides (log-logistic `beta` 0.5, 1, 2; a sentinel), `work_exhausted` inside
a bracket, the `inverse_cumulative_hazard` contract (invalid `e`, `e=+inf`,
`options`, no `_with_info` twin) and NaN tangents (endpoints, invalid inputs,
parameter issues). The generic `10*rtol` factor is now spec (12.1); only
comments changed. `log_interval_prob` joins Slice B (12.2); module docstring
only. Design d104627 (4.2 aligned with 8.1; within rev 54) applies the 4.5
dtype rule to the `e` of `inverse_cumulative_hazard` (Q-063 settled): three
new policy tests, tagged `rev 54 (design d104627)`. Revision 56 (8.1, 8.4,
8.5; Q-067 settled, Q-057's last open case settled) fixes the
`mean_with_info` override contract and makes an improper family's `+inf`
quantiles `known_infinite` with NaN tangents. Changed: the improper `p = 1`
status test becomes
`test_improper_family_upper_endpoint_quantile_reports_known_infinite`
(Q-068); the four custom-override mean tests are retagged `rev 56`; the custom
`known_mean_log_logistic` supplies its own NaN tangents. New tests: interior
and per-entry `known_infinite` quantiles, NaN tangents of known-infinite
quantiles, analytical built-in means with NaN error estimate and zero counts,
and `mean()` carrying an override's tangents. Revision 57 (8.4, 8.5, 12.2;
Q-068 and Q-070 settled): isolation takes precedence over NaN tangents, which
are guaranteed in forward mode only; an improper `quantile(1)` is
`known_infinite`; `check_family` stays in Slice D. No test asserted a
reverse-mode NaN of a bad entry (every NaN-tangent test uses `jax.jvp`), so
none changed for that. The custom `known_mean_log_logistic` override now
gives a constant NaN tangent instead of `dx * nan`, so it does not itself
leak `0 * nan` into a shared `alpha` in reverse mode (forward NaN tangents
unchanged). `test_improper_family_upper_endpoint_quantile_reports_known_infinite`
is retagged `rev 57` (same assertion). New isolation tests (forward and
reverse, shared parameters, hand-derived): beside a known-infinite mean
(custom override; improper generic family; built-in `LogLogistic` with
`beta = [3, 0.8]`), known-infinite quantiles (`p = 0.9` and `p = 1`), endpoint
quantiles (default and bounded support), `inverse_cumulative_hazard` at `e`
0 and `+inf`, and invalid-input entries (invalid `p`, invalid `e`, parameter
issues). Revision 55 (8.8, 12.2) brings
the log-normal and log-logistic families forward; they are covered by the new
features `family.lognormal` and `family.loglogistic` (`test_family_*.py`,
tagged `rev 55`; analytical-summary tests also list the summary feature).
No earlier test changed. The A10 concordance `unsupported` rows are a
permanent v0 rule of 9.1 ("Supported observations are exact and
right-censored rows"), not a Slice A restriction, and are not staged. The mpmath references
need `mpmath`, and the A10 property tests `hypothesis`, in the test environment. Tests are static acceptance tests and have never
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

### `test_a3_shape_and_len.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 3.2 | "`data.shape` is the observation shape as a tuple of Python ints" (`(N,)`). | `test_shape_of_vector_data_is_a_tuple_of_python_ints` |
| 3.2 | the same for scalar data, `()`. | `test_shape_of_scalar_data_is_empty_tuple` |
| 3.2 | "Empty data `(0,)` are allowed". | `test_shape_of_empty_data` |
| 3.2/12.1 | two-dimensional data. | `test_shape_of_two_dimensional_data` |
| 3.2 | the bounds form has the same `shape`. | `test_shape_from_bounds_constructor` |
| 3.2 | `shape` is the observation shape that every field has. | `test_shape_equals_field_shape` |
| 3.2/12.1 | `shape` after indexing (slice, integer, integer array, 2-D, empty slice). | `test_shape_after_slicing`, `test_shape_after_integer_index_is_scalar`, `test_shape_after_integer_array_index`, `test_shape_after_indexing_two_dimensional_data`, `test_shape_after_empty_slice` |
| 3.2 | "read from the stored leaves rather than kept in the auxiliary data". | `test_shape_is_read_from_stored_leaves`, `test_shape_is_not_in_the_tree_structure` |
| 3.2/12.1 | "static under `jit`" (static ints). | `test_shape_is_static_under_jit` |
| 3.2/12.1 | "per-example under `vmap`". | `test_shape_is_per_example_under_vmap`, `test_shape_is_per_example_under_vmap_of_two_dimensional_data` |
| 3.2 | "There is no `size` or `ndim`". | `test_no_size_attribute`, `test_no_ndim_attribute` |
| 3.2 | "`len(data)` is `data.shape[0]`". | `test_len_of_vector_data_is_number_of_observations` |
| 3.2/12.1 | "on scalar data it raises `TypeError`". | `test_len_of_scalar_data_raises_type_error`, `test_len_of_integer_indexed_data_raises_type_error` |
| 3.2 | empty data "have length 0". | `test_len_of_empty_data_is_zero`, `test_len_of_empty_slice_is_zero` |
| 3.2 | "`len` counts only the leading axis; the total is `math.prod(data.shape)`". | `test_len_of_two_dimensional_data_counts_leading_axis` |
| 3.2/12.1 | `len` after indexing. | `test_len_after_slicing` |
| 3.2/12.1 | `len` under `jit` is a static int. | `test_len_under_jit_is_static` |
| 3.2/12.1 | `len` under `vmap` follows the per-example shape. | `test_len_under_vmap_of_vector_data_raises_type_error`, `test_len_under_vmap_of_two_dimensional_data_is_per_example` |
| 3.2/12.1 | "`bool(data)` (and so `if data:`) raises `TypeError` instead of testing emptiness". | `test_bool_of_nonempty_data_raises_type_error`, `test_bool_of_empty_data_raises_type_error`, `test_bool_of_scalar_data_raises_type_error`, `test_if_data_raises_type_error` |

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
| 3.2 | "Right takes precedence": a non-exact row with `upper=+inf` is right-censored even when `lower=0` (`from_bounds(0, inf)`); valid left rows have `upper<+inf` (rev 33). | `test_zero_to_inf_row_is_classified_as_right_censored`, `test_zero_to_inf_row_is_not_classified_as_left_censored` |
| 3.2 | "This classification is not a validity check; rows with issue bits are still classified" (rev 33). | `test_zero_to_inf_row_with_issue_is_still_classified_as_right_censored` |
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

- `data.shape` and `len(data)` on traced construction inside `jit` (rev 36):
  covered for data passed into `jit`; construction inside `jit` gives the same
  leaves, so no separate test was added.
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
| 4.2 | "Every dataclass field must use one of these helpers; registration raises a `TypeError` naming an unmarked field" (rev 33). | `test_unmarked_field_is_rejected_at_registration`, `test_unmarked_field_error_names_the_field`, `test_unmarked_field_without_default_is_rejected_at_registration` |
| 4.2 | "a subclass that only inherits a registered initializer raises `TypeError` at registration" (rev 33). | `test_subclass_inheriting_registered_initializer_is_rejected` |
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
| 4.2 | a `distribution()` field must hold an instance of a registered class; an unregistered subclass of a registered family, an array or another object raises a `TypeError` naming the field, before nested batch shapes are read (rev 33). | `test_nested_field_rejects_unregistered_subclass_instance`, `test_nested_field_error_names_the_field`, `test_nested_field_rejects_non_distribution_before_reading_batch_shape`, `test_nested_field_rejects_array_value` |
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
  spec 12.2 (rev 14). `parameter_issues()` and `parameter_conditions()` were
  specified by rev 28 (4.7) and are tested at A6 (section below).
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

## A6: parameter constraints and diagnostics (spec 4.1, 4.2, 4.5, 4.7, 6.3, 7.1, 10.2; rev 28, 29)

Step placement by maintainer decision (Q-035). The rest of A6 is in the
next section.

### `test_a6_constraints.py`: the `survjax.constraints` vocabulary

Each constraint is checked eagerly, under `jit` and under `vmap` (helper
`_check`), with a `bool` result; the suite runs in both dtypes.

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 10.2 | the `constraints` module is re-exported as `survjax.constraints`. | `test_constraints_module_is_exported_at_top_level` |
| 4.7 | the nine vocabulary names exist. | `test_vocabulary_names_exist` |
| 4.7 | callable; `event_dim` 0 (absent means 0) for scalar constraints, 1 for `ordered` and `simplex`. | `test_scalar_constraints_have_event_dim_zero`, `test_vector_constraints_have_event_dim_one` |
| 4.7 | `real`: finite; NaN and `±inf` outside. | `test_real_accepts_finite_values`, `test_real_rejects_nan_and_infinities` |
| 4.7 | `positive`: finite, `>0`; boundary `0` outside. | `test_positive_accepts_finite_positive_values`, `test_positive_rejects_zero`, `test_positive_rejects_negative_values`, `test_positive_rejects_nan_and_infinities` |
| 4.7 (rev 30) | ordinary comparison in the parameter dtype: `-finfo.tiny` (normal) outside `positive` and `nonnegative`, `finfo.tiny` inside `nonnegative`; `-0.0` outside `positive`, inside `nonnegative`. | `test_positive_rejects_smallest_negative_normal`, `test_positive_rejects_negative_zero`, `test_nonnegative_accepts_smallest_positive_normal`, `test_nonnegative_rejects_smallest_negative_normal`, `test_nonnegative_accepts_negative_zero` |
| 4.7 | `nonnegative`: finite, `>=0`; boundary `0` inside. | `test_nonnegative_accepts_zero_and_positive_values`, `test_nonnegative_rejects_negative_values`, `test_nonnegative_rejects_nan_and_infinities` |
| 4.7 | `greater_than(a)`: finite, `>a`; `a` outside, its successor inside; `int` bounds. | `test_greater_than_is_strict_at_its_bound`, `test_greater_than_with_negative_bound`, `test_greater_than_accepts_python_int_bound`, `test_greater_than_rejects_nan_and_infinities` |
| 4.7 | `interval(a, b)`: open at both ends. | `test_interval_is_open_at_both_ends`, `test_interval_rejects_nan_and_infinities` |
| 4.7 | `closed_interval(a, b)`: closed at both ends. | `test_closed_interval_includes_both_ends`, `test_closed_interval_rejects_nan_and_infinities` |
| 4.7 | `unit_interval`: `0<=x<=1` (below 0 probed at `-finfo.tiny`, rev 30). | `test_unit_interval_includes_zero_and_one`, `test_unit_interval_rejects_nan_and_infinities` |
| 4.7/12.1 | bounds are finite Python numbers with `a<b`; anything else raises `TypeError`. | `test_greater_than_rejects_non_finite_bound`, `test_greater_than_rejects_array_bound`, `test_greater_than_rejects_string_bound`, `test_interval_constraints_reject_invalid_bounds`, `test_interval_constraints_reject_array_bounds` |
| 4.7/12.1 | `ordered`: finite, strictly increasing along the last axis; fails ties; reduces one axis. | `test_ordered_accepts_strictly_increasing_vectors`, `test_ordered_rejects_ties`, `test_ordered_rejects_decreasing_steps`, `test_ordered_rejects_infinite_entries`, `test_ordered_rejects_nan_entries`, `test_ordered_reduces_only_the_last_axis` |
| 4.7/12.1 | `simplex`: passes `softmax` output up to length 1000 in both dtypes; fails length 10 summing to `1+1e-3`; zero entries allowed; negative, NaN, infinite entries outside; reduces one axis. | `test_simplex_accepts_softmax_output`, `test_simplex_rejects_sum_off_by_1e_minus_3`, `test_simplex_accepts_zero_entries`, `test_simplex_rejects_negative_entries_even_when_sum_is_one`, `test_simplex_rejects_nan_and_infinite_entries`, `test_simplex_reduces_only_the_last_axis` |

### `test_a6_parameter_issues.py`: declarations, `ParameterIssues`, host checkpoint, built-ins

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.2 | `survjax.param(constraint=...)`. | `test_param_accepts_constraint_keyword` |
| 4.7 | constraints live in field metadata, not in the PyTree. | `test_constraints_are_not_pytree_leaves`, `test_constrained_and_unconstrained_twins_have_equal_leaves` |
| 4.2/4.7 | `parameter_constraints`: own constrained fields, declaration order, undeclared and nested excluded, read-only. | `test_parameter_constraints_lists_own_constrained_fields_in_declaration_order`, `test_parameter_constraints_excludes_nested_distributions`, `test_parameter_constraints_is_empty_without_declarations`, `test_parameter_constraints_is_read_only` |
| 4.7/12.1 | decoration-time `TypeError` for a non-callable constraint and for `event_dim > component_ndim`; `event_dim == component_ndim` accepted. | `test_register_family_rejects_non_callable_constraint`, `test_register_family_rejects_vector_constraint_on_scalar_field`, `test_register_family_rejects_event_dim_above_component_ndim`, `test_register_family_accepts_event_dim_equal_to_component_ndim` |
| 4.7 | a plain callable is a constraint with `event_dim` 0. | `test_plain_function_is_a_constraint_with_event_dim_zero` |
| 4.7/12.1 | a NumPyro constraint object is accepted (skipped without NumPyro). | `test_numpyro_constraint_object_is_accepted` |
| 4.2 | the `parameter_conditions()` default is an empty mapping. | `test_default_parameter_conditions_is_empty` |
| 4.7 | a family with nothing declared reports no issues; `fields`/`conditions` are `dict`s. | `test_family_with_nothing_declared_reports_no_issues`, `test_issue_fields_are_dicts` |
| 4.7 | field masks: polarity, undeclared fields absent, `any` over remaining component axes, `event_dim` consuming the component axis, `batch_shape` masks, scalar batch. | `test_field_mask_marks_violations_true`, `test_undeclared_fields_are_absent_from_fields`, `test_remaining_component_axis_is_reduced_by_any`, `test_vector_constraint_consumes_the_component_axis`, `test_masks_have_batch_shape_for_two_dimensional_batch`, `test_scalar_distribution_has_scalar_masks` |
| 4.7 | condition masks; `invalid_entries` union; scalar `ok`. | `test_condition_mask_marks_violations_true`, `test_invalid_entries_is_union_of_fields_and_conditions`, `test_ok_is_scalar_negated_any` |
| 4.7 | nested keys dotted, own first; nested masks broadcast to the outer batch; nested violations join `invalid_entries`; keys fixed by the class. | `test_nested_field_keys_are_dotted_and_follow_own_fields`, `test_nested_condition_keys_are_dotted_and_follow_own_conditions`, `test_nested_masks_are_broadcast_to_outer_batch_shape`, `test_nested_violation_makes_outer_issues_not_ok`, `test_issue_keys_are_fixed_by_the_class` |
| 4.7 | `dist[idx].parameter_issues()` equals the parent's masks indexed by `idx` (slice, int, int array). | `test_indexed_issues_equal_indexed_parent_masks` |
| 4.7/6.3 | eager/`jit`/`vmap` agreement, traced construction, registered PyTree. | `test_parameter_issues_under_jit_matches_eager`, `test_parameter_issues_with_traced_construction_matches_eager`, `test_parameter_issues_under_vmap_matches_eager`, `test_parameter_issues_is_a_pytree_round_trip` |
| 4.7/12.1 | malformed `parameter_conditions()` (non-mapping, empty/dotted/non-string key, non-bool or wrongly shaped value) raises `TypeError`, eagerly and under tracing; other mappings accepted. | `test_malformed_parameter_conditions_raise_type_error`, `test_malformed_parameter_conditions_raise_type_error_under_jit`, `test_other_mapping_types_are_accepted_from_parameter_conditions` |
| 4.7/10.2 | `InvalidParameterError` is exported and subclasses `ValueError`. | `test_invalid_parameter_error_is_a_public_value_error` |
| 4.7 | `raise_if_invalid()`: passes without violations (and with nothing declared); raises for field and condition violations; message names fields, conditions and dotted nested keys with counts and first indices; works on issues returned from `jit`; plain `ValueError` on traced diagnostics. | `test_raise_if_invalid_passes_without_violations`, `test_raise_if_invalid_passes_with_nothing_declared`, `test_raise_if_invalid_raises_invalid_parameter_error_for_field_violation`, `test_raise_if_invalid_raises_invalid_parameter_error_for_condition_violation`, `test_raise_if_invalid_message_names_field_with_count_and_first_index`, `test_raise_if_invalid_message_names_condition_and_nested_field`, `test_raise_if_invalid_works_on_issues_returned_from_jit`, `test_raise_if_invalid_on_traced_diagnostics_raises_plain_value_error` |
| 4.1 | built-in declarations: Exponential `rate`, Weibull `shape`/`scale` are `positive`; `from_log` natural parameters are checked. | `test_exponential_declares_rate_positive`, `test_weibull_declares_shape_and_scale_positive`, `test_built_in_constraints_behave_as_positive`, `test_from_log_overflow_is_reported_as_parameter_issue`, `test_weibull_from_log_overflow_is_reported_as_parameter_issue` |
| 4.5 | declared violations and conditions do not change direct-method values. | `test_declared_violation_does_not_change_direct_method_values`, `test_declared_violation_keeps_finite_formula_value`, `test_condition_violation_does_not_change_direct_method_values` |

### `test_a6_likelihood_parameter_issues.py`: NaN likelihood rows for parameter issues (rev 29)

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 7.1 | rows marked by `parameter_issues().invalid_entries` are NaN even where the formula is finite (Exponential `rate=-1`, `0`; Weibull `scale=inf`, `shape=-1`; `from_log` overflow; a named condition; a nested violation; a custom declared family). | `test_declared_violation_is_nan_where_formula_is_finite`, `test_zero_rate_right_censored_row_is_nan`, `test_infinite_weibull_scale_right_censored_row_is_nan`, `test_negative_weibull_shape_right_censored_row_is_nan`, `test_from_log_overflow_row_is_nan`, `test_condition_violation_is_nan_on_exact_row`, `test_nested_violation_is_nan`, `test_declared_twin_of_finite_value_is_nan` |
| 7.1/5 | with valid data the NaN rows are `invalid_entries` broadcast to the contribution shape (paired, scalar distribution, `(D,1)` draws); valid parameters give no NaN. | `test_nan_rows_are_parameter_issues_invalid_entries`, `test_scalar_invalid_distribution_marks_every_row`, `test_invalid_entries_broadcast_over_draws`, `test_valid_parameters_give_no_nan_rows` |
| 7.1 | triage recipe: a row with invalid data and parameters is attributed to both; undeclared NaN is the remainder (`other`). | `test_invalid_parameters_and_invalid_data_overlap`, `test_undeclared_nan_is_the_recipe_remainder`, `test_undeclared_negative_rate_log_is_the_recipe_remainder` |
| 7.1/12.1 | named-condition and constraint violations in one batch give exactly their NaN rows; the raw form forces NaN too (rev 35). | `test_named_condition_and_constraint_violations_give_nan_rows`, `test_raw_form_forces_nan_on_parameter_violations` |
| 7.1/12.1 | rows without issues keep their values and parameter gradients (per-row parameters; Jacobian in both modes, eager and `jit`; sum over valid rows). Rev 32 widening: `test_a6_likelihood_gradient_isolation.py`. | `test_rows_without_issues_keep_their_values`, `test_rows_without_issues_keep_their_gradients`, `test_rows_without_issues_keep_their_gradients_under_jit`, `test_sum_over_valid_rows_has_alone_gradient` |
| 7.1/5 | `jit` and `vmap` agreement of values and `invalid_entries`. | `test_parameter_violations_under_jit_match_eager`, `test_parameter_violations_under_vmap_match_batched_call` |

### A6 (section 4.7 part): deliberately not tested

- **Values on rows with parameter issues beyond NaN, and their gradients**
  (7.1, 7.3): "Gradients on rows with issues are unspecified". The gradient
  of a valid row with respect to the invalid row's parameter is asserted
  zero, because evaluated alone the valid row does not depend on it.
- **Finite values from undeclared invalid parameters in the likelihood**:
  4.5 gives "possibly finite" values and 7.2 leaves `reference_time()`
  unspecified for invalid parameters, so neither the value nor the absence
  of an issue is asserted.
- **A shared (scalar) leaf that is itself invalid**: every row then has an
  issue, so no row is left to isolate. A shared valid leaf beside a per-row
  invalid one is tested (rev 32: leaves are broadcast to `batch_shape`).
- **Undeclared invalidity** (7.1, rev 32): "is not masked and has no
  isolation guarantee".
- **The 7.2 `stop_gradient` exception as such** (rev 32): observable only as
  the isolation, which is tested.
- **Exact `raise_if_invalid()` message text**: only the presence of names,
  counts and first indices.
- **`simplex` exactly at the `4*n*eps` tolerance edge, `ordered` of length
  one**: summation order is an implementation choice, and the other is a
  corner case the table does not single out.
- **Subnormal values and bounds, and values within rounding of a converted
  bound** (4.7, rev 30): "may be classified either way" where the backend
  flushes subnormals, so no expectation is asserted (for example
  `nonnegative(nextafter(0, -inf))`). Every boundary neighbour in the tests
  is a normal number and every bound is exact in both dtypes.
- **Wrapper declarations** (4.4: linear predictor `real`, `baseline.` prefix):
  the PH and AFT wrappers are Slice D; nesting is covered by a custom family.
- **The 11.4 recipe** (a custom family with constraints and one condition):
  documentation, A9 and later.
- **`parameter_issues()` is not overridden by families**: a rule for family
  authors with no observable outcome.

## A6: exact and right-censored likelihood, method, unsupported modes (spec 4.2, 5, 6.1–6.3, 7.1–7.3, 12.2; rev 28, 29, 31, 32, 35)

Hand-derived exponential and Weibull contributions (`Z=1` without
truncation). Tails, boundaries, mpmath and `jax_debug_nans` are A7; R
fixtures A8; `check_grads` and Hessians A9 (except the raw-form versus
data-form derivative agreement, 12.1). Shared helpers `ll` (the method,
the only entry point since rev 35) and `nan_causes`
(the 7.1 triage recipe) are in `helpers.py`.

### `test_a6_likelihood_values.py`: values

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 2/7.1 | exact row `log f(t)`, right-censored row `log S(c)`; exponential. | `test_exponential_exact_contribution`, `test_exponential_right_censored_contribution`, `test_exponential_mixed_rows_with_per_row_rates`, `test_exponential_shared_rate_over_rows`, `test_exponential_from_log_matches_ordinary_constructor` |
| 4.1/7.1 | Weibull, `k>1` and `k<1`, exact and right; `k=1` equals exponential; mixed per-row parameters. | `test_weibull_exact_contribution_shape_above_one`, `test_weibull_right_censored_contribution_shape_above_one`, `test_weibull_exact_contribution_shape_below_one`, `test_weibull_right_censored_contribution_shape_below_one`, `test_weibull_shape_one_equals_exponential`, `test_weibull_mixed_rows_with_per_row_parameters` |
| 4.2/7.1 | "The likelihood never dispatches on family type": custom family through generic defaults, equal to the built-in; `log_prob` override used for exact rows. | `test_custom_family_uses_generic_defaults`, `test_custom_family_matches_built_in`, `test_log_prob_override_is_used_for_exact_rows` |
| 2/3.3/7.1 | right-censored at `c=L=0` (flag and bounds forms) is uninformative and exactly zero, with zero gradient. | `test_right_censored_at_zero_contributes_exactly_zero`, `test_bounds_form_zero_to_infinity_contributes_exactly_zero`, `test_uninformative_row_has_zero_gradient` |
| 7.1 | true zero density or mass is `-inf`, not NaN, and the recipe finds no cause. | `test_zero_density_is_minus_infinity_not_nan`, `test_zero_survival_is_minus_infinity_not_nan` |
| 12.1 | all-censored exponential data remain valid. | `test_all_censored_exponential_data_is_valid` |
| 3.3 | likelihood dtype is the promoted data/parameter dtype. | `test_likelihood_dtype_is_default_float_in_32bit_mode`, `test_likelihood_promotes_float32_parameters_with_float64_data`, `test_likelihood_promotes_float32_data_with_float64_parameters` |

### `test_a6_likelihood_shapes.py`: section 5 shapes

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 5 | scalar/scalar `()`; `(N,)` with scalar; paired `(N,)`; `(D,1)` and `(D,N)` give `(D,N)`; the documented Weibull regression pattern. | `test_scalar_data_scalar_distribution_gives_scalar`, `test_vector_data_scalar_distribution`, `test_vector_data_paired_distribution`, `test_draw_column_distribution_gives_draws_by_individuals`, `test_full_draw_matrix_distribution`, `test_weibull_draw_column_regression_pattern` |
| 3.2/5 | multidimensional data keep every axis; trailing broadcasting of `(B,N)` data. | `test_multidimensional_data_keeps_its_axes`, `test_batched_datasets_against_trailing_distribution` |
| 5 | bare `(D,)` against `(N,)`, `D != N`: `TypeError`. | `test_mismatched_bare_batch_raises_type_error` |

### `test_a6_likelihood_invalid_data.py`: stored input issues

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 3.3/7.1 | invalid exact times (`0`, negative, `inf`, NaN) and right-censoring times (negative, `inf`, NaN): NaN rows equal `data.issues.invalid_rows`; the valid row keeps its value. | `test_invalid_exact_time_is_nan`, `test_invalid_right_censoring_time_is_nan` |
| 7.1 | invalid indicator (code 2, NaN) is NaN although normalized fields look valid. | `test_invalid_event_code_is_nan_although_normalized_fields_look_valid`, `test_nan_event_indicator_is_nan` |
| 3.1/7.1 | bounds form: mismatched exact upper is NaN; valid bounds rows equal the flag form. | `test_bounds_exact_row_with_mismatched_upper_is_nan`, `test_bounds_valid_exact_and_right_rows_match_flag_form` |
| 7.1/12.1 | propagation after indexing, reconstruction, data returned from `jit`, traced construction under `jit`, `vmap` over data. | `test_indexed_data_keeps_invalid_rows`, `test_rebuilt_data_keeps_invalid_rows`, `test_data_returned_from_jit_keeps_invalid_rows`, `test_traced_construction_and_likelihood_under_jit`, `test_vmap_over_data_keeps_invalid_rows` |
| 6.1/7.1 | raw form (function and method): event code 2 and invalid times give NaN eagerly and under `jit`, on the rows `SurvivalData(time, event).issues` marks. | `test_raw_form_invalid_event_code_returns_nan_eagerly`, `test_raw_form_invalid_time_returns_nan_eagerly`, `test_raw_form_method_invalid_event_code_returns_nan_eagerly`, `test_raw_form_invalid_event_code_under_jit` |
| 7.1 | uninformative bits alone do not invalidate; the recipe attributes invalid-data rows to data only; broadcast over draws. | `test_uninformative_bits_alone_do_not_invalidate`, `test_invalid_data_rows_triage_to_data_only`, `test_invalid_data_broadcasts_over_draws` |

### `test_a6_likelihood_entry_points.py`: input forms, transformations, compilation (through `dist.log_likelihood`, rev 35)

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 7.1/10.2/12.1 | "There is no top-level `survjax.log_likelihood` function" (rev 35). | `test_no_top_level_log_likelihood` |
| 6.2/6.3 | the likelihood "has no `_with_info` variant"; `LikelihoodIssues` removed (rev 29). | `test_likelihood_has_no_with_info_variant`, `test_likelihood_issues_container_is_removed` |
| 6.1/7.1 | data form positional and by keyword; raw form equals data form; integer event codes. | `test_data_form_positional`, `test_data_form_by_keyword`, `test_raw_form_equals_data_form`, `test_raw_form_accepts_integer_event_codes` |
| 6.1/6.2 | `TypeError` for a raw array as `data`, mixed forms, a missing `time` or `event`, no data, constructor keywords in the raw form, raw shape mismatch. | `test_raw_array_passed_as_data_raises_type_error`, `test_mixing_forms_raises_type_error`, `test_data_with_only_event_keyword_raises_type_error`, `test_raw_form_without_event_raises_type_error`, `test_raw_form_without_time_raises_type_error`, `test_no_data_at_all_raises_type_error`, `test_raw_form_rejects_constructor_keywords`, `test_raw_form_shape_mismatch_raises_type_error` |
| 6.1/12.1 | eager/`jit`/`vmap` agreement, traced construction and raw form inside `jit`; summed-likelihood gradient. | `test_likelihood_under_jit_matches_eager`, `test_traced_construction_inside_jit_matches_eager`, `test_raw_form_inside_jit_matches_eager`, `test_vmap_over_data_matches_batched_call`, `test_vmap_over_parameter_draws_matches_batched_call`, `test_gradient_of_summed_likelihood_is_hand_derived` |
| 7.2 | compilation reuse for new values (prepared data and traced construction); no host transfers on the compiled path (likelihood, `data.issues.ok`, `parameter_issues()`, gradient). | `test_new_values_reuse_compilation`, `test_traced_construction_reuses_compilation`, `test_compiled_path_makes_no_host_transfers` |

### `test_a6_likelihood_method.py`: `dist.log_likelihood` (rev 29, 35)

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 4.2/7.1 | a base method on every registered family: built-ins, a custom family that does not define it, a nested family. | `test_built_in_families_have_the_method`, `test_custom_family_has_the_method_without_defining_it`, `test_nested_family_has_the_method` |
| 7.1 | Weibull per-row parameters, hand-derived. | `test_weibull_per_row_parameters_are_hand_derived` |
| 7.1 | positional `time`/`event` rejected (keyword-only). | `test_time_and_event_are_keyword_only` |
| 6.1/12.1 | "`dist.log_likelihood(time=..., event=...)` equal to `dist.log_likelihood(SurvivalData(time, event))` in values and first and second parameter derivatives" (rev 35): Weibull summed and per-row values exactly, gradient and Hessian. | `test_raw_form_value_equals_data_form_value`, `test_raw_form_per_row_values_equal_data_form`, `test_raw_form_gradient_equals_data_form_gradient`, `test_raw_form_hessian_equals_data_form_hessian` |

The input forms, static misuse, `jit`/`vmap` and the hand-derived gradient
are in `test_a6_likelihood_entry_points.py` (rev 35 removed their duplicates
here).

Invalid-data and parameter-issue NaN rows through the method are in
`test_a6_likelihood_invalid_data.py` and
`test_a6_likelihood_parameter_issues.py`; the static shape mismatch is in
`test_a6_likelihood_shapes.py`.

### `test_a6_unsupported_modes.py`: NaN rows for unsupported modes (rev 31)

Cases: left, interval, bounds-left, bounds-interval, left-truncated exact and
right, right-truncated exact; row 0 is a supported exact row.

Slice B staging: tests marked **(staged)** below carry, per case,
`until("observation.censoring")` (left, interval, bounds-left,
bounds-interval) or `until("observation.truncation")` (the three truncated
cases), so no NaN or mode-support assertion runs once its mode is enabled.
The other tests assert only facts that stay true after Slice B (the rows are
valid data; the exact row keeps its value and derivatives) and run
unconditionally.

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 3.2/7.1 | the rows are valid data and add no data issue. | `test_unsupported_row_is_valid_data` |
| 7.1 | the NaN row adds no parameter issue **(staged)**. | `test_unsupported_row_adds_no_parameter_issue` |
| 7.1/12.2 | NaN on the row without an exception **(staged)**; the supported row keeps its value. | `test_unsupported_row_is_nan`, `test_supported_row_keeps_its_value` |
| 7.1/12.2 | the same under `jit` (prepared data), traced construction and `vmap` over draws **(staged)**. | `test_unsupported_row_is_nan_under_jit`, `test_unsupported_mode_from_traced_construction_is_nan`, `test_unsupported_row_is_nan_under_vmap` |
| 7.1/12.2 | the Slice A triage recipe attributes the row to mode support (`unsupported_nan`), not `other` **(staged)**. | `test_triage_recipe_attributes_row_to_mode_support` |
| 7.1/7.3 | the supported row keeps its first and second rate derivatives, per-row (Jacobian both modes, Hessian, `jit`) and with a shared rate. | `test_supported_row_keeps_its_gradient_per_row`, `test_supported_row_keeps_its_hessian_per_row`, `test_supported_row_keeps_its_gradient_with_shared_rate`, `test_supported_row_keeps_its_gradient_under_jit` |

### `test_a6_likelihood_gradient_isolation.py`: isolation of rows without parameter issues (rev 32)

Each case: three rows, row 1 violates a declaration. "Alone" is the family
built from the per-row leaves at rows 0 and 2 on those data rows, as a
function of the same flat parameter vector.

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 7.1/12.1 | values of rows without issues equal alone; row 1 NaN. Violation classes zero, negative, NaN, `+inf`, `-inf` (and non-finite `real`), and named conditions; Exponential, Weibull (per-row scale with shared shape, per-row shape with shared scale), custom `constrained_rate`, `capped_rate` condition, `component_constrained` (batch leaf and component leaf), nested `scaled_capped` (base leaf, outer leaf, base and outer conditions). | `test_rows_without_issues_keep_values[*]` |
| 7.1/7.3/12.1 | first and second derivatives equal alone, finite, in every leaf: `jacrev`, `jacfwd`, fwd-over-rev, rev-over-rev, fwd-over-fwd, rev-over-fwd. | `test_rows_without_issues_keep_derivatives[*]` |
| 7.1/7.3 | the same under `jit`. | `test_rows_without_issues_keep_derivatives_under_jit[*]` |
| 7.1/7.3 | hand-derived anchors: Exponential per-row Hessian; Weibull shared-shape gradient and Hessian with a negative per-row scale; nested shared `log_ratio` with a NaN base rate. | `test_exponential_isolated_hessian_is_hand_derived`, `test_weibull_shared_shape_derivatives_are_hand_derived`, `test_nested_shared_log_ratio_derivatives_are_hand_derived` |

### A6 (likelihood part): deliberately not tested

- **Unsupported-row derivatives** (7.1, rev 31): "unspecified".
- **Families do not override `log_likelihood`** (4.2): a rule for family
  authors; whether registration rejects an override is not specified.
- **`Z=0` NaN rows** (the recipe's `other`): `Z=0` needs truncation, Slice B;
  the bounded-support zero-numerator versus zero-denominator case of 12.1 is
  Slice B for the same reason. Now covered by `observation.truncation`
  (`test_observation_truncation_diagnostics.py`).
- **Invalid rows whose normalized kind may be left or interval** (conflicting
  flags, equal non-exact bounds): rev 31 makes them NaN either way, but the
  triage of such rows (both invalid data and possibly unsupported) is left
  to Slice B with the modes themselves. Now covered by
  `observation.censoring` (`test_observation_censoring_diagnostics.py`).
- **Inactive-input substitution** (7.2: own window for exact rows,
  `reference_time()` for censored rows, `stop_gradient`): not observable on
  valid exact/right data beyond finite values and gradients; A7
  (`jax_debug_nans`) and A9 (`check_grads`) exercise it.
- **Static-metadata pruning and "different metadata may compile
  separately"**: pruning is optional and Slice B; separate compilation is
  permitted, not required.
- **No host callbacks or synchronisation** (7.2) beyond the transfer guard:
  callbacks are not observable through the public API.
- **Tails, narrow cases, boundaries, mpmath and R fixtures**: A7 and A8.
- **Second derivatives, Hessians, `check_grads`**: A9 (7.3), except the
  rev 31/32 isolation of supported rows and of rows without parameter
  issues, which 12.1 lists with the parameter diagnostics.
- **"Multiplying invalid rows by zero does not make them valid"**: guidance
  for callers, not library behaviour.
- **Reach of isolation upstream of the parameter leaves** (7.1, rev 33):
  overflow in a caller's `exp(X @ beta)` or in `from_log` "can still give
  NaN gradients" for shared upstream parameters. This is a permission, not a
  requirement, so neither NaN nor finite gradients are asserted there.
- **NaN loss with a finite gradient** (7.1, rev 33): "can have"; gradients on
  rows with issues stay unspecified, so a finite summed gradient is not
  required.
- **Triage after edited fields** (7.1, rev 33): `other` "may also include"
  rows the likelihood finds invalid in current fields. Permissive; the
  recipe's exactness is tested only for data as constructed (unchanged).

## A7: likelihood numerics, tails, boundaries, mpmath, `jax_debug_nans` (spec 2, 4.5, 7.1–7.3, 12.1, 12.2; rev 33)

Slice A scope per 12.2 (rev 33): exact and right-censored rows without
truncation, and the direct methods. The direct-method tails, boundaries and
`log_cdf` mpmath checks are already in A4/A5 (`test_a4_log_cdf_upper_tail.py`,
`test_a5_closed_forms.py`, `test_a5_log_cdf_tails.py`,
`test_a5_boundaries.py`, `test_a4_base_boundaries.py`); A7 adds the
likelihood. References: `helpers.mp_exponential_contribution` and
`helpers.mp_weibull_contribution` (mpmath, 50 digits, at dtype-rounded
inputs), compared with zero absolute tolerance and a relative tolerance that
is the 12.1 `rtol` widened only by the documented term-cancellation bound
(`helpers._reference`).

### `test_a7_likelihood_mpmath.py`: contributions across scales

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 2/7.1/12.1 | exponential exact `log r - r t` and right `-r c`, rates `1e-6` to `1e3`, against mpmath. | `test_exponential_exact_contribution_matches_mpmath[*]`, `test_exponential_right_censored_contribution_matches_mpmath[*]` |
| 2/4.1/7.1/12.1 | Weibull exact and right, `k` 0.3 to 8, `b` `1e-3` to `1e4`, the `bc`-like vector, against mpmath. | `test_weibull_exact_contribution_matches_mpmath[*]`, `test_weibull_right_censored_contribution_matches_mpmath[*]` |
| 5/7.1/12.1 | per-row parameters and alternating kinds in one batch. | `test_weibull_mixed_batch_matches_mpmath_row_by_row` |
| 7.3/12.1 | the same under `jit` and `vmap` over rows. | `test_exponential_mixed_batch_matches_mpmath_under_jit_and_vmap` |

### `test_a7_likelihood_tails.py`: tiny masses, both tails, no floors

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 7.1/12.1 | "tiny nonzero masses": right-censored `-H` with `H=1e-20`, `1e-30` keeps relative precision and never rounds to zero. | `test_exponential_right_censored_tiny_hazard_keeps_relative_precision`, `test_weibull_right_censored_tiny_hazard_keeps_relative_precision` |
| 7.3/12.1 | tiny derivatives compared with zero `atol`: `-c`; `-H log z`, `kH/b`. | `test_exponential_right_censored_tiny_hazard_rate_gradient`, `test_weibull_right_censored_tiny_hazard_parameter_gradients` |
| 7.1/12.1 | upper tail where `S` or `f` underflows: finite log values, not `-inf`, below `log(tiny)` (no floor); `H=1e32`. | `test_exponential_right_censored_where_survival_underflows`, `test_exponential_exact_where_density_underflows`, `test_weibull_right_censored_where_survival_underflows`, `test_weibull_exact_where_density_underflows`, `test_weibull_exact_with_shape_below_one_where_density_underflows`, `test_weibull_exact_with_huge_cumulative_hazard_is_finite`, `test_weibull_right_censored_with_huge_cumulative_hazard_is_finite` |
| 7.3/12.1 | hand-derived parameter gradients in the upper tail. | `test_exponential_upper_tail_rate_gradients`, `test_weibull_upper_tail_exact_parameter_gradients` |
| 4.1/7.1/12.1 | lower tail of exact rows (`t=1e-30`, `1e-12`; tiny rate). | `test_weibull_exact_near_zero_with_decreasing_hazard_is_large_and_finite`, `test_weibull_exact_near_zero_with_increasing_hazard_is_finite`, `test_exponential_exact_near_zero_is_log_rate`, `test_exponential_exact_with_tiny_rate` |
| 7.1/7.2 | tail rows leave an interior row's value unchanged. | `test_tail_rows_do_not_change_interior_rows` |
| 7.1/12.1 | rev 34: a valid row whose log contribution overflows (Weibull `k=50`, `b=1`, `t=1e10`, `H=1e500`) gives `-inf`, not NaN, exact and right; no NaN cause; other rows unchanged. | `test_right_censored_row_whose_contribution_overflows_is_minus_infinity`, `test_exact_row_whose_contribution_overflows_is_minus_infinity`, `test_overflowing_row_has_no_nan_cause_and_leaves_other_rows_unchanged` |
| 7.3 | rev 34: next to an infinite row, "Forward-mode derivatives and per-row parameters are unaffected". | `test_forward_mode_derivatives_of_other_rows_survive_an_overflowing_row`, `test_reverse_mode_derivatives_with_per_row_parameters_survive_an_overflowing_row` |

### `test_a7_likelihood_boundaries.py`: boundaries

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 3.3/7.1/12.1 | exact and right rows at the smallest normal time `tiny`, against mpmath; finite exact-row gradients there. | `test_exponential_exact_at_smallest_normal_time`, `test_weibull_exact_at_smallest_normal_time[k]`, `test_exponential_right_censored_at_smallest_normal_time`, `test_weibull_right_censored_at_smallest_normal_time_with_shape_below_one`, `test_weibull_exact_at_smallest_normal_time_has_finite_gradients[k]` |
| 2/4.5/7.1/7.2 | right-censored at `0`: exactly zero and zero gradients for Weibull `k` in `0.5, 1, 2.5` (infinite `f(0)` at `k<1`: inactive density piece must not leak); next to exact rows sharing a rate. | `test_weibull_right_censored_at_zero_is_exactly_zero[k]`, `test_weibull_right_censored_at_zero_has_zero_gradients[k]`, `test_exponential_right_censored_at_zero_has_zero_gradient_in_mixed_batch` |
| 2/7.1 | improper family: "`P(c<T<=+inf)=S(c)`", so a right row is `log S(c)`, tending to `log cure`; exact row through the generic default; `c=0` exactly zero; no NaN cause. | `test_improper_family_right_censored_contribution_is_log_survival`, `test_improper_family_right_censored_far_tail_tends_to_log_cure`, `test_improper_family_exact_contribution`, `test_improper_family_right_censored_at_zero_is_exactly_zero`, `test_improper_family_mixed_rows_have_no_nan_cause` |

### `test_a7_debug_nans.py`: eager `jax_debug_nans` run

Seven-row valid batch: exact interior, tiny, upper tail; right at `0`, tiny
`H`, interior, upper tail. Values and gradients of the summed contributions.

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 7.2/12.1/12.2 | "the likelihood suite run eagerly under `jax_debug_nans` on valid mixed batches": Exponential (shared and per-row rates, hand values), Weibull `k` in `0.3, 0.5, 1, 2.5, 8` and per-row, `from_log`, a custom family on generic defaults, an improper family, bounds-form data with a `(0, +inf]` row. | `test_exponential_mixed_batch_under_debug_nans`, `test_exponential_per_row_rates_under_debug_nans`, `test_weibull_mixed_batch_under_debug_nans[k]`, `test_weibull_per_row_parameters_under_debug_nans`, `test_weibull_from_log_under_debug_nans`, `test_custom_family_with_generic_defaults_under_debug_nans`, `test_improper_family_under_debug_nans`, `test_bounds_form_batch_under_debug_nans` |

### A7: deliberately not tested

- **Narrow intervals, interval cancellation, `Z=0`, tiny `Z`, bounded-support
  zero numerator versus zero denominator** (12.1): Slice B evidence (12.2,
  rev 33).
- **Values below the normal range** (subnormal `t/b`, `H` or contributions):
  the spec's relative-precision clauses are stated for normal numbers, and
  backends may flush subnormals.
- **Reverse-mode derivatives with shared parameters next to an infinite
  row, and the infinite row's own derivatives** (7.3, rev 34): outside the
  guarantee.
- **Generic `log_prob` default in the far upper tail** (custom families):
  no tail guarantee is specified for it; the improper family's exact rows
  stay where `f` is representable.
- **`check_grads` and Hessians**: A9.
- **Bitwise equality across batch layouts**: not specified; compared at
  12.1 tolerances.

## A8: committed synthetic `bc`-like R fixtures (spec 10.4, 11.1, 12.1, 12.2; rev 33)

Fixture `fixtures/bc_synthetic_weibull.json`, schema in `fixtures/README.md`
(Q-039, resolved by rev 34: "The spec tester defines the file name and
schema"). No fixture is committed yet, so every A8 test currently fails at
load time by design (a reference test must never pass without its values).
`[name]` runs over the three 11.1 vectors (`fixed_a`, `fixed_b`,
`r_optimum`); `[source]` over flexsurv and survreg. Tolerances: 12.1 in each
dtype, which in float64 are the 11.1 targets (`atol=1e-10`, `rtol=1e-8`).

### `test_a8_bc_fixtures.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 11.1 | fixture integrity: 686 rows, groups 229/229/228; exact and right rows with positive finite times; vectors equal to 11.1; flexsurv and survreg agree; 141×3 grid from 0 to 7. | `test_fixture_has_bc_size_and_group_sizes`, `test_fixture_rows_are_exact_or_right_censored_with_positive_finite_times`, `test_fixture_parameter_vectors_are_those_of_section_11_1[name]`, `test_fixture_sources_agree_with_each_other[name]`, `test_fixture_grid_has_shape_141_by_3[name]` |
| 11.1/12.1/12.2 | "per-row log likelihoods ... from flexsurv and survreg at the three parameter vectors", float64 `atol=1e-10, rtol=1e-8`. | `test_per_row_log_likelihood_matches_r[name-source]` |
| 6.1/7.1/11.1 | the raw `time=`/`event=` form gives the same per-row values. | `test_raw_form_log_likelihood_matches_r[name]` |
| 4.1/11.1 | the 11.1 objective's construction `k=exp(theta[0])`, `b_i=exp(X_i beta)` through `Weibull.from_log` under `jit`. | `test_design_matrix_construction_with_from_log_matches_r[name]` |
| 7.1/11.1 | the summed per-row values match R's sum. | `test_summed_log_likelihood_matches_r[name]` |
| 8.1/11.1/12.1 | rev 34: "survival values `S_i(t_i)` at each row's own time"; `log_survival` on censored rows and `log_prob` on exact rows equal the R contributions. | `test_per_row_survival_matches_r[name-source]`, `test_per_row_log_survival_matches_right_censored_contributions[name]`, `test_per_row_log_prob_matches_exact_contributions[name]` |
| 5/11.1 | rev 34: group curves on the demo grid via `survival_grid`, shape `(141, 3)`, `S(0)=1`. | `test_group_survival_curves_match_r[name]` |

### A8: deliberately not tested

- **The 11.1 audit totals** (`-840.82...`, `-815.98...`, `-811.94...`):
  they are for the real `bc` rows, which are local-only (10.4) and never in
  regular tests; the classical demo (Slice C) checks them.
- **Fixtures for left, interval and truncated rows** (12.1 "all observation
  kinds and truncation"), survfit, concordance, timeROC, riskRegression:
  Slices B and E.
- **The fit, Hessian and independent optimum check** (11.1 checks 3–5):
  Slice C, never in CI.
- **How the fixture was simulated** (seed, censoring pattern "similar" to
  `bc`): recorded in `meta` and the generator, not testable as a property
  beyond sizes, groups and kinds.
- **Generating fixtures:** the spec tester never runs R or derives
  reference values from survjax; the maintainer commits them (10.4).

## A9: derivative and transformation suite, compiled composition, install checks (spec 7.2, 7.3, 10.1–10.3, 12.1, 12.2; rev 34)

Tests carry the revision of the requirement they quote (most `rev 28`; the
12.1 compiled-composition row `rev 33`; the 12.2 wheel gate `rev 34`).
`check_grads` runs in 64-bit mode only; float32 runs the hand-derived first
and second derivatives (as at A4/A5). Hand-derived Weibull gradients and
Hessians are in the docstring of `test_a9_likelihood_derivatives.py`.

### `test_a9_likelihood_derivatives.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 7.3 | hand-derived Hessians of the summed likelihood: exponential shared and per-row (diagonal) rates; Weibull `(k,b)` gradient and full Hessian at `k<1`, `k=1`, `k>1`; per-row shapes (diagonal); `from_log` by the chain rule. | `test_exponential_shared_rate_hessian_is_hand_derived`, `test_exponential_per_row_rate_hessian_is_diagonal_and_hand_derived`, `test_weibull_shared_parameter_gradient_is_hand_derived`, `test_weibull_shared_parameter_hessian_is_hand_derived[k]`, `test_weibull_per_row_shape_hessian_is_diagonal_and_hand_derived`, `test_weibull_from_log_hessian_follows_the_chain_rule` |
| 2/7.2/7.3 | right-censored at `0`: zero Hessian, including `k<1` (inactive density piece does not leak). | `test_right_censored_at_zero_has_zero_hessian[k]` |
| 7.3 | "forward and reverse modes, including Hessians": four `jac*` compositions agree; per-row Jacobian in both modes. | `test_forward_and_reverse_hessians_agree`, `test_per_row_jacobian_in_both_modes` |
| 7.3/12.1 | composed with `jit` and `vmap` (draws, datasets); `jit` of `vmap` and the raw form give the hand-derived Hessian. | `test_hessian_under_jit_matches_eager`, `test_hessian_under_vmap_over_parameter_draws`, `test_gradient_under_vmap_over_datasets`, `test_hessian_under_jit_of_vmap_is_hand_derived`, `test_raw_form_hessian_equals_data_form` |
| 7.3/12.1 | `check_grads(order=2, modes=("fwd","rev"))`: exponential shared and per-row; Weibull shared (`k` 0.6, 1, 1.4, 3), per-row, `(D,1)` draws, `from_log`; under `jit`, `vmap`, and with construction inside `jit`. | `test_check_grads_exponential_shared_rate`, `test_check_grads_exponential_per_row_contributions`, `test_check_grads_weibull_shared_parameters[k]`, `test_check_grads_weibull_per_row_contributions`, `test_check_grads_weibull_draw_by_individual_contributions`, `test_check_grads_weibull_from_log`, `test_check_grads_under_jit`, `test_check_grads_under_vmap_over_draws`, `test_check_grads_with_traced_construction` |
| 4.2/7.3 | "The custom-family recipe is verified where its primitives are twice differentiable"; autodiff `log_prob` default "adds one derivative order". | `test_check_grads_custom_family_on_generic_defaults`, `test_custom_family_hessian_matches_built_in` |
| 7.2/12.1 | "reference-input gradient isolation": a `reference_time()` with infinite parameter derivative leaves right-censored and mixed gradients and Hessians hand-derived and finite (`stop_gradient` on substituted values). | `test_reference_input_does_not_reach_right_censored_gradients`, `test_reference_input_does_not_reach_mixed_batch_gradients` |

### `test_a9_compiled_composition.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 12.1 | "ordinary construction, diagnostic access and likelihood evaluation compose inside `jit`; first and second parameter derivatives agree with direct evaluation". | `test_construction_diagnostics_and_derivatives_inside_jit_match_direct_evaluation` |
| 6.2/12.1 | `data.issues.ok` "usable in compiled control flow" (`lax.cond` selecting loss and gradient). | `test_diagnostics_select_the_loss_in_compiled_control_flow` |
| 7.3/12.1 | second-order derivatives in a compiled loop (test-written Newton, no external solver) reach the closed-form exponential MLE; `vmap` over replicates with one invalid replicate isolated. | `test_newton_iterations_inside_jit_reach_the_closed_form_exponential_mle`, `test_vmapped_compiled_fits_over_replicate_datasets` |
| 7.2/12.3 | compilation reuse for the Hessian with new values, and for gradients over same-shaped minibatches of one parent. | `test_compiled_hessian_reuses_compilation_for_new_values`, `test_minibatch_gradients_reuse_compilation` |

### `test_a9_package.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 10.2 | top-level re-exports of the Slice A data object, distributions and error classes (no likelihood function since rev 35; its absence is `test_no_top_level_log_likelihood`). | `test_slice_a_names_are_exported_at_top_level[name]` |
| 10.2/10.3 | `py.typed`; metadata `Name`, `Requires-Python >=3.12`, version a PEP 440 version no earlier than the initial `0.0.1.dev0`, Apache license (Q-041). | `test_package_has_type_marker`, `test_distribution_name_is_survjax`, `test_distribution_requires_python_312`, `test_distribution_version_is_not_before_the_initial_version`, `test_distribution_license_metadata_is_apache` |
| 10.3/12.1 | "core code imports and runs without NumPyro, Equinox, Optax, Optimistix ... or plotting libraries" (fresh interpreter, imports blocked). | `test_core_imports_and_runs_without_demo_dependencies` |

### `test_a9_wheel_smoke.py`

Wheel path from `SURVJAX_WHEEL` (Q-040); missing means fail.

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 10.2 | pure `py3-none-any` wheel with `survjax/__init__.py` and `survjax/py.typed`; metadata name and Python; no demo dependency in `Requires-Dist`. | `test_wheel_is_pure_python`, `test_wheel_contains_package_and_type_marker`, `test_wheel_metadata_names_survjax_and_python_312`, `test_wheel_requires_no_demo_dependencies` |
| 10.3/12.2 | "Smoke-test wheel installation and imports outside the source tree": unpacked wheel imported from an empty working directory evaluates a likelihood. | `test_installed_wheel_imports_and_evaluates_outside_the_source_tree` |

Already covered earlier and not repeated: `TracedInputError` not exported
(A3), `survjax.constraints` (A6), error classes as `ValueError` subclasses
(A3, A6), compile counts for static metadata (A3), compilation reuse and no
host transfers on the compiled likelihood path (A6), `jax_debug_nans` (A7).

### A9: deliberately not tested

- **Wrappers** (`check_grads` "for families and wrappers"): Slice D.
- **The tiny synthetic example, executable documentation, Jupyter playground
  and rendered-page review** (12.2): documentation work, executed by the docs
  build (10.5); the compiled-composition tests check the capabilities such an
  example uses.
- **Recorded CPU performance baseline** (12.3): a recorded benchmark, not a
  pass/fail test ("no timing gate in ordinary CI").
- **mypy on core source and on an installed out-of-tree custom-family
  example** (10.3): a type-checking run, not a pytest; the out-of-tree family
  itself runs throughout (`spec_families.py`).
- **Source archive** (10.3): "check wheel and source archive before release",
  not Slice A.
- **Pre-commit, Ruff and CI configuration, `uv.lock`, `.python-version`,
  LICENSE file contents** (10.1–10.3): repository files, not observable from
  the installed package except the metadata tested.
- **Derivatives with respect to data times, at support-change boundaries, at
  invalid or infinite likelihoods, and above second order** (7.3): outside
  the guarantee.
- **Custom JAX derivative rules** (7.3): "must be documented and tested" by
  the implementation; not observable as such through the public API beyond
  the derivative values tested here.
- **Linux CI job and GPU verification** (10.3): infrastructure, not tests.

## A10: Harrell's concordance (spec 6.1–6.3, 9.1, 9.2, 10.2, 12.1; rev 34)

Step A10 is the maintainer's replan of 2026-10-23 (Q-047); assertions come
only from spec rev 34, all tagged `rev 34`. Hand-enumerated pairs are in
each test's docstring; `helpers.harrell_reference` transcribes the 9.2
rules for the property tests and the fixture-integrity check. Info field
names, count representation and mask shape: Q-043 to Q-045.

### `test_a10_concordance_api.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 9.2/10.2 | both entry points at top level and in `survjax.metrics`; `ConcordanceInfo` in its module (Q-042). | `test_concordance_entry_points_are_exported_at_top_level`, `test_metrics_module_holds_the_entry_points`, `test_concordance_info_is_discoverable_in_its_module` |
| 6.3/9.2 | `_with_info` returns exactly `(value, info)`; same value; the four counts and three masks (reference case `C=0.75`). | `test_with_info_returns_exactly_value_and_info`, `test_with_info_value_equals_plain_value`, `test_info_holds_hand_enumerated_counts[name]`, `test_info_masks_are_false_for_valid_comparable_data[name]` |
| 6.3 | the info is a PyTree returned from `jit` and `vmap`; flatten/unflatten. | `test_info_is_a_pytree_returned_from_jit`, `test_info_is_returned_from_vmap`, `test_info_leaves_rebuild_an_equal_info` |
| 6.1 | data form positional and by keyword; raw form (also `_with_info`, integer codes); `TypeError` for raw array as `data`, mixing, missing `time`/`event`, no data, constructor keywords, an `eval_times` keyword (not in the 9.2 signature). | `test_data_form_positional`, `test_data_form_by_keyword`, `test_raw_form_equals_data_form`, `test_raw_form_accepts_integer_event_codes`, `test_raw_form_with_info`, `test_raw_array_passed_as_data_raises_type_error`, `test_mixing_forms_raises_type_error`, `test_raw_form_without_event_raises_type_error`, `test_raw_form_without_time_raises_type_error`, `test_no_data_at_all_raises_type_error`, `test_raw_form_rejects_constructor_keywords`, `test_time_grid_keyword_is_not_accepted` |
| 6.1 | eager raw-form event code 2 returns NaN, not an exception; raw form under `jit`. | `test_raw_form_invalid_event_code_returns_nan_eagerly`, `test_raw_form_under_jit_matches_eager` |
| 9.1 | "ratios are computed in the data's floating dtype", including float32 data with float64 scores (64-bit mode). | `test_result_dtype_is_data_float_dtype`, `test_result_dtype_follows_float32_data_with_float64_scores` |

### `test_a10_concordance_values.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 9.2 | concordant (`C=1`), reversed (`0`), all tied scores (`0.5`); mixed case; seven-row case with every tie kind; `comparable = concordant + discordant + tied`. | `test_perfectly_concordant_scores_give_one`, `test_reversed_scores_give_zero`, `test_all_tied_scores_give_one_half`, `test_mixed_case_with_censoring_and_ties`, `test_counts_partition_comparable_pairs` |
| 9.1/9.2 | pair rules: earlier exact comparable, earlier censored not; equal-time event/event and censor/censor excluded; equal-time event/censor comparable with the event earlier (concordant, discordant, row-order independent, tied half credit); each unordered pair once. | `test_earlier_exact_later_censored_is_comparable`, `test_earlier_censored_is_not_comparable`, `test_equal_time_event_pair_is_excluded`, `test_equal_time_censor_pair_is_excluded`, `test_equal_time_event_censor_pair_with_larger_event_score_is_concordant`, `test_equal_time_event_censor_pair_with_smaller_event_score_is_discordant`, `test_equal_time_event_censor_pair_does_not_depend_on_row_order`, `test_equal_time_event_censor_pair_with_tied_scores_gets_half_credit`, `test_each_unordered_pair_is_counted_once` |
| 9.2 | "No comparable pairs gives NaN": all censored, singleton, events at one time; mask set only then. | `test_all_censored_gives_nan`, `test_single_observation_gives_nan`, `test_events_all_at_one_time_give_nan`, `test_no_comparable_pair_mask_is_false_when_pairs_exist` |
| 9.2 | row permutation, negated scores (`1-C`), increasing transformation; uninformative `(0,+inf]` row adds nothing; bounds form equals flag form. | `test_row_permutation_leaves_value_and_counts_unchanged`, `test_negated_scores_swap_concordant_and_discordant`, `test_increasing_transformation_of_scores_leaves_value_unchanged`, `test_uninformative_row_adds_no_pair_and_no_issue`, `test_bounds_form_data_give_the_flag_form_value` |
| 6.1/12.1 | `jit` (values and counts) and traced construction inside `jit`. | `test_value_under_jit_matches_eager`, `test_traced_construction_inside_jit` |

### `test_a10_concordance_diagnostics.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 9.2/6.1 | "Scores are finite": NaN, `+inf`, `-inf` give NaN plus `invalid`, eagerly and under `jit`; only the affected draw. | `test_non_finite_score_gives_nan_and_invalid[bad]`, `test_non_finite_score_does_not_raise_under_jit`, `test_non_finite_score_affects_only_its_draw` |
| 6.2/9.1 | stored data issues (event code 2; exact time negative, zero, `inf`, NaN) give NaN plus `invalid`; traced construction; after indexing; only the affected dataset. | `test_invalid_event_code_gives_nan_and_invalid`, `test_invalid_exact_time_gives_nan_and_invalid[bad]`, `test_invalid_data_under_jit_with_traced_construction`, `test_invalid_indexed_data_keeps_its_issue`, `test_invalid_data_in_one_dataset_leaves_the_other_dataset` |
| 9.1 | left, interval (flag and bounds), left- and right-truncated rows are valid data and give NaN plus `unsupported`, eagerly, under `jit`, from traced construction; supported rows under traced metadata stay supported; only the affected dataset; explicit `0`/`+inf` truncation is supported. | `test_unsupported_row_is_valid_data[case]`, `test_unsupported_row_gives_nan_and_unsupported[case]`, `test_unsupported_row_gives_nan_under_jit[case]`, `test_unsupported_row_from_traced_construction_gives_nan`, `test_traced_construction_of_supported_rows_is_not_unsupported`, `test_unsupported_row_in_one_dataset_leaves_the_other_dataset`, `test_explicit_zero_left_and_infinite_right_truncation_are_supported` |

### `test_a10_concordance_shapes.py`

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 9.2 | the five rows of the shape table, data and raw forms; info counts per draw. | `test_vector_data_and_vector_scores_give_a_scalar`, `test_score_draws_give_one_value_per_draw`, `test_batched_datasets_with_matching_scores`, `test_dataset_column_against_score_draws`, `test_two_leading_score_axes`, `test_raw_form_follows_the_shape_table`, `test_with_info_counts_follow_the_result_shape`, `test_dataset_q_hand_values[name]` |
| 9.1/6.2 | last axis must match and is "never broadcast" (`(N±1,)`, `(1,)` scores, `(N,1)` scores, `(1,)` data); incompatible leading axes; under `jit`: `TypeError`. | `test_mismatched_observation_length_raises_type_error`, `test_single_score_is_not_broadcast_across_the_cohort`, `test_column_of_scores_is_not_broadcast_across_the_cohort`, `test_single_observation_data_is_not_broadcast_across_scores`, `test_incompatible_leading_axes_raise_type_error`, `test_mismatched_observation_length_raises_under_jit` |
| 12.1/7.2 | `vmap` over draws and datasets, `jit(vmap)` with masks; compilation reuse for new values. | `test_vmap_over_score_draws_matches_batched_call`, `test_vmap_over_datasets_matches_batched_call`, `test_jit_of_vmap_matches_eager`, `test_new_scores_reuse_compilation` |

### `test_a10_concordance_properties.py` (hypothesis, `derandomize=True`)

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 9.2 | counts, value and NaN/mask equal direct enumeration of the pair rules on tie-heavy cohorts. | `test_counts_and_value_equal_direct_enumeration` |
| 9.2 | permutation invariance; `C(-s) = 1 - C(s)` with swapped counts. | `test_row_permutation_invariance`, `test_negated_scores_give_one_minus_c` |
| 9.1 | `(B,N)` equals `B` separate calls (never pooled). | `test_batched_datasets_equal_separate_calls` |

### `test_a10_concordance_fixtures.py`

Fixture `fixtures/concordance_synthetic.json`, schema in
`fixtures/README.md` (Q-046). Not committed yet, so these tests fail at
load time by design.

| Spec section | Requirement covered | Test |
| --- | --- | --- |
| 9.2/10.4 | fixture integrity: recorded R call (`reverse=TRUE`, `timewt="n"`); every tie pattern present; supported rows; R counts equal the 9.2 rules. | `test_fixture_records_the_r_call`, `test_fixture_covers_the_required_tie_patterns`, `test_fixture_rows_are_exact_or_right_censored_with_positive_finite_times`, `test_fixture_counts_agree_with_the_spec_pair_rules` |
| 9.2/12.1 | "Cross-check against `survival::concordance(..., reverse = TRUE, timewt = "n")`": values, counts, `(D,N)` in one call, raw form. | `test_concordance_matches_r`, `test_counts_match_r`, `test_score_draws_match_r_in_one_call`, `test_raw_form_matches_r` |

### A10: deliberately not tested

- **Count overflow beyond `int32`** (9.1 hi/lo accumulator): needs more
  than `2**31` comparable pairs (about 65,000 rows, `O(N**2)` work); the
  accumulator is private, so its overflow handling is implementation unit
  testing (the draft outline says so too). Only the exposed counts are read
  (Q-044).
- **"do not allocate an N-by-N pair matrix", O(N) workspace, documented
  complexity** (9.2): not observable through results; peak memory is not
  measured by tests (12.3).
- **No score gradient** (9.1: "No useful gradient is promised"): nothing
  to assert.
- **Counts of NaN results** caused by invalid or unsupported rows, and
  whether `no_comparable_pairs` is set together with `invalid` or
  `unsupported`: unspecified.
- **Boolean or integer scores, scores of a different floating dtype in
  32-bit mode**: the spec says only "finite, caller-supplied".
- **Random subsets** (9.2): caller practice; "No automatic subsampling or new
  approximation API is required".
- **Documentation, playground, held-out synthetic example** (draft A10
  outline): documentation steps, not tests (Q-047).
- **Kaplan–Meier, Brier, IBS, AUC, distribution-versus-array prediction
  inputs, `G=0`, delayed entry**: Slice E.

## A11: `ProportionalHazards` without RMST (spec 3.3, 4.1, 4.2, 4.4, 4.5, 4.7, 5, 6.1, 7.1, 7.3, 12.1, 12.2; rev 41)

Construction goes through `helpers.ph` (keyword-only fields `baseline`, `log_hazard_ratio`; rev 42). Hand references
use `H = exp(eta) H0`; derivative references are hand-derived (algebra in
the module docstrings). PH identity tests compare with
`Weibull(shape=k, scale=b0*exp(-eta/k))` and `Exponential(rate=r*exp(eta))`
written as functions of `(k, b0, eta)` / `(r, eta)`. Every derivative case
is ordinary range (7.3 exception).

### `test_a11_ph_construction.py`

| Spec | Requirement | Tests |
| --- | --- | --- |
| 10.2 | wrappers re-exported at top level | `test_proportional_hazards_is_exported_at_top_level` |
| 4.2/4.4 | the wrapper is a registered distribution | `test_proportional_hazards_is_a_survival_distribution` |
| 4.1/4.2 | natural parameters stored under constructor names; nested leaves are outer leaves; PyTree round trip | `test_baseline_and_linear_predictor_are_stored_under_constructor_names`, `test_leaves_are_the_baseline_parameters_and_the_linear_predictor`, `test_flatten_unflatten_round_trip`, `test_tree_map_keeps_wrapper_and_baseline_types` |
| 4.4/5 | batch is the broadcast of baseline batch and predictor shape, both broadcast at construction; `(S, 1)` x `(S, 2)`; "baseline `()`, linear predictor `(D,N)`" | `test_scalar_baseline_with_matrix_predictor_has_the_predictor_batch`, `test_vector_baseline_with_scalar_predictor_has_the_baseline_batch`, `test_draw_baseline_broadcasts_against_draw_by_arm_predictor` |
| 4.4 | keyword-only: a positional baseline raises `TypeError` (rev 42) | `test_positional_baseline_is_rejected` |
| 4.1 | incompatible batches raise `TypeError`; no `.shape` attribute | `test_incompatible_baseline_and_predictor_batches_are_rejected`, `test_wrapper_has_no_shape_attribute` |
| 4.1 | constructors never check values; traced construction under `jit`/`vmap` | `test_constructor_does_not_check_parameter_values`, `test_construction_with_traced_parameters_under_jit`, `test_construction_under_vmap_over_the_predictor` |
| 3.3 | dtype policy for the wrapper's own leaf: integer/scalar default float, bool and complex `TypeError`, no `dtype` keyword, float64 under 32-bit converts with one warning; nested baseline keeps its dtype, calculations promote | `test_integer_predictor_uses_the_default_float_dtype`, `test_python_scalar_predictor_uses_the_default_float_dtype`, `test_boolean_predictor_is_rejected`, `test_complex_predictor_is_rejected`, `test_constructor_takes_no_dtype_keyword`, `test_float64_predictor_in_32_bit_mode_converts_with_one_warning`, `test_nested_baseline_keeps_its_own_dtype` |
| 4.2 | a non-distribution `distribution()` field raises `TypeError` naming it | `test_non_distribution_baseline_is_rejected_naming_the_field` |
| 4.1/4.2 | indexing, tree-map slicing, `vmap`, `jit` with nested leaves | `test_indexing_selects_baseline_and_predictor_entries`, `test_tree_map_slicing_applies_to_nested_leaves`, `test_vmap_over_the_wrapper`, `test_wrapper_passes_through_jit_as_an_argument` |

### `test_a11_ph_parameter_issues.py`

| Spec | Requirement | Tests |
| --- | --- | --- |
| 4.4/4.7 | "The linear predictor declares `real`"; `parameter_constraints` has own fields only | `test_wrapper_declares_only_the_linear_predictor`, `test_linear_predictor_constraint_is_real` |
| 4.4/4.7 | baseline issues under the `baseline.` prefix, own keys first; masks, union, `ok` | `test_exponential_baseline_keys_are_prefixed`, `test_weibull_baseline_keys_are_prefixed`, `test_valid_wrapper_reports_no_issue` |
| 4.7 | nested masks broadcast to the outer batch, incl. `(S, 1)` x `(S, 2)` | `test_shared_baseline_mask_is_broadcast_to_the_outer_batch`, `test_draw_baseline_masks_broadcast_against_draw_by_arm_predictor` |
| 4.7 | nested conditions with dotted keys (custom baseline) | `test_custom_baseline_condition_appears_with_the_prefix` |
| 4.7/6.3 | `jit`/`vmap` agreement, indexing equality | `test_parameter_issues_agree_under_jit`, `test_parameter_issues_agree_under_vmap`, `test_indexed_wrapper_issues_equal_indexed_masks` |
| 4.7 | `raise_if_invalid()` names violated fields; returns when valid | `test_raise_if_invalid_names_the_prefixed_baseline_field`, `test_raise_if_invalid_names_the_linear_predictor`, `test_raise_if_invalid_returns_for_valid_wrapper` |
| 4.5/4.7 | declarations do not change direct-method values | `test_declared_violation_does_not_change_finite_direct_values` |

### `test_a11_ph_direct_methods.py`

| Spec | Requirement | Tests |
| --- | --- | --- |
| 4.4/8.1 | all seven direct methods, PH-Weibull, PH-Exponential and a custom baseline with generic defaults, against hand values | `test_ph_weibull_method_is_hand_derived`, `test_ph_exponential_method_is_hand_derived`, `test_ph_custom_baseline_method_is_hand_derived`, `test_zero_predictor_leaves_the_baseline_law` |
| 4.4 | "`eta>0` means higher hazard under PH" | `test_positive_predictor_means_higher_hazard` |
| 4.5 | `t<0` rows, NaN `t`, base-supplied `t=0` and `t=+inf` values, `log f(0)` limits (k=1, k>1, k<1), zero gradients of base-supplied values in baseline parameters and `eta` | `test_negative_time_policy_values`, `test_nan_time_gives_nan`, `test_base_supplied_values_at_zero`, `test_base_supplied_values_at_infinity`, `test_density_limit_at_zero_for_weibull_baseline`, `test_base_supplied_values_have_zero_parameter_gradients` |
| 4.4/4.5 | `log_hazard` row `log h0(t) + eta` carries the baseline's limits at `t=0` and `t=+inf` (Weibull `k<1`, `k=1`, `k>1`; Exponential), equal to the equivalent Weibull (rev 43) | `test_ph_weibull_log_hazard_limits_carry_through`, `test_ph_weibull_hazard_limits_carry_through`, `test_ph_weibull_hazard_limits_equal_equivalent_weibull`, `test_ph_exponential_hazard_at_the_boundaries` |
| 7.3/12.1 | values stay accurate at the 7.3 extreme-scaling examples (Weibull `eta=±720`; Exponential `rate=1e-200`, `eta=500`), float64, against mpmath | `test_extreme_scaling_weibull_values_stay_accurate`, `test_extreme_scaling_exponential_values_stay_accurate` |
| 5 | scalar, paired and explicit time-first shapes against `(S, 2)`; non-broadcasting time `TypeError` | `test_scalar_time_against_draw_by_arm_batch`, `test_paired_time_vector_against_arms`, `test_explicit_time_first_axis`, `test_time_that_does_not_broadcast_is_rejected` |
| 5 | every `_grid` twin equals `m(t[:, None, None])` with shape `(K, S, 2)`, unsorted/repeated `t`, `K=N`, non-1-D `t` `TypeError`, grid from `t=0`; the `log_cdf_grid` reference check is relative only where the reference `S` is a normal number in the active dtype (4.2, D-044) | `test_grid_twin_equals_time_first_direct_method`, `test_grid_with_as_many_times_as_entries_is_not_paired`, `test_grid_rejects_time_that_is_not_one_dimensional`, `test_grid_from_zero_includes_base_supplied_values` |
| 12.1 | eager/`jit`/`vmap` agreement for every method; result dtype | `test_methods_agree_under_jit`, `test_methods_agree_under_vmap`, `test_result_dtype_is_the_parameter_dtype` |

### `test_a11_ph_identities.py`

| Spec | Requirement | Tests |
| --- | --- | --- |
| 4.4/12.1 | every direct method equals the equivalent Weibull/Exponential in values, incl. grids with `(S, 1)` x `(S, 2)` | `test_ph_weibull_values_equal_equivalent_weibull`, `test_ph_exponential_values_equal_equivalent_exponential`, `test_ph_weibull_grids_equal_equivalent_weibull_with_draw_by_arm_broadcasting`, `test_ph_exponential_grid_equals_equivalent_with_draw_by_arm_broadcasting` |
| 4.4 | the likelihood identity, incl. `(S, 1)` x `(S, 2)` | `test_ph_weibull_likelihood_equals_equivalent_weibull`, `test_ph_exponential_likelihood_equals_equivalent_exponential`, `test_likelihood_identity_with_draw_by_arm_broadcasting` |
| 4.4/7.3 | first derivatives in `eta` and the baseline parameters, forward and reverse, direct methods and likelihood, with broadcasting | `test_ph_weibull_first_derivatives_equal_equivalent_weibull`, `test_ph_exponential_first_derivatives_equal_equivalent_exponential`, `test_likelihood_first_derivatives_equal_equivalent_weibull`, `test_likelihood_first_derivatives_equal_equivalent_exponential`, `test_first_derivatives_with_draw_by_arm_broadcasting` |

### `test_a11_ph_derivatives.py`

| Spec | Requirement | Tests |
| --- | --- | --- |
| 7.3/12.1 | ordinary-range gradients and full Hessians incl. mixed baseline/`eta` terms, all four mode compositions, `log_prob` and `log_survival` | `test_ph_weibull_gradient_is_hand_derived`, `test_ph_weibull_hessian_with_mixed_terms_is_hand_derived`, `test_ph_exponential_gradient_is_hand_derived`, `test_ph_exponential_hessian_with_mixed_terms_is_hand_derived` |
| 7.3 | likelihood gradients and Hessians (shared and per-row `eta`) | `test_ph_weibull_likelihood_hessian_is_hand_derived`, `test_ph_weibull_likelihood_gradient_is_hand_derived`, `test_ph_exponential_likelihood_hessian_is_hand_derived`, `test_per_row_predictor_hessian_is_diagonal` |
| 7.3 | composed with `jit` and `vmap`; `(S, 1)` x `(S, 2)` gradient | `test_hessian_under_jit_matches_hand_value`, `test_hessian_under_vmap_over_parameter_draws`, `test_gradient_under_jit_of_vmap_over_predictor`, `test_gradient_with_draw_by_arm_broadcasting_is_hand_derived` |
| 12.1 | `check_grads(order=2, modes=("fwd","rev"))` for wrappers (float64) | `test_check_grads_order_two_ph_weibull`, `test_check_grads_order_two_ph_exponential`, `test_check_grads_order_two_ph_weibull_likelihood`, `test_check_grads_order_two_ph_exponential_likelihood` |

### `test_a11_ph_likelihood.py`

| Spec | Requirement | Tests |
| --- | --- | --- |
| 4.4/7.1 | exact and right-censored contributions keep the PH formulas (Weibull, Exponential, custom baseline); per-row `eta` | `test_ph_weibull_mixed_rows_are_hand_derived`, `test_ph_exponential_mixed_rows_are_hand_derived`, `test_per_row_predictor_pairs_with_rows`, `test_custom_baseline_likelihood_is_hand_derived` |
| 2/7.1 | right-censored at `c=L=0` is exactly zero; overflowing valid contribution is `-inf` | `test_right_censored_at_zero_contributes_exactly_zero`, `test_overflowing_contribution_is_minus_infinity` |
| 5 | `(N,)`, `(D,1)` x `(N,)`, `(S, 2)` x `(2,)`; mismatch `TypeError` | `test_scalar_wrapper_against_observations`, `test_draw_column_against_observations`, `test_draw_by_arm_batch_against_paired_arms`, `test_mismatched_batch_and_data_lengths_are_rejected` |
| 6.1/12.1 | raw form equals data form in values, gradients, Hessians; `jit`, `vmap`, traced construction | `test_raw_form_equals_data_form_in_values_and_derivatives`, `test_likelihood_agrees_under_jit`, `test_likelihood_agrees_under_vmap_over_predictor_draws`, `test_traced_construction_and_likelihood_inside_jit` |
| 7.1 | NaN rows for invalid data, non-finite `eta`, invalid baseline, nested custom condition; triage attributes them | `test_invalid_data_rows_are_nan_and_others_kept`, `test_non_finite_predictor_row_is_nan`, `test_invalid_baseline_row_is_nan`, `test_custom_baseline_condition_row_is_nan` |
| 7.1/12.2 | unsupported modes (left, interval, left/right truncation) NaN eagerly and under `jit`, supported rows keep values, no issue added (staged per case: `until("observation.censoring")` / `until("observation.truncation")`); supported-row derivatives next to a left-censored row (unconditional) | `test_unsupported_mode_row_is_nan_and_supported_row_kept`, `test_unsupported_row_leaves_supported_row_derivatives` |

### `test_a11_ph_isolation.py`

| Spec | Requirement | Tests |
| --- | --- | --- |
| 7.1/7.3/12.1 | rows without issues keep values and first/second derivatives (forward and reverse, `jit`) for non-finite `eta`, zero/negative/NaN/`±inf` baseline parameters (shared and per-row), a nested custom condition | `test_case_reports_the_issue_on_the_bad_row`, `test_rows_without_issues_keep_values`, `test_rows_without_issues_keep_derivatives`, `test_rows_without_issues_keep_derivatives_under_jit` |
| 7.1 | hand anchors: shared `eta` with a NaN baseline row; shared baseline with an infinite `eta` row | `test_shared_predictor_derivatives_are_hand_derived`, `test_shared_baseline_derivatives_with_infinite_predictor_row` |

### A11: deliberately not tested

- **Derivatives in the 7.3 extreme-scaling regime** (the three quoted
  examples and float32 stress cases): outside the initial-release guarantee;
  only their values are asserted. The quoted correct gradients are not
  required.
- **Float32 analogues of the extreme-value checks**: no PH-specific
  tail-precision statement exists beyond 7.3's float64 examples; ordinary
  float32 values are covered.
- **`log_hazard`/`hazard` boundaries for a custom baseline**: rev 43 says it
  carries through the generic default's boundary behaviour, which itself has
  no value guarantee at `t=0`.
- **Delegated summaries in the identity** (`rmst`): A12. `mean`, `median`,
  `quantile`, `inverse_cumulative_hazard`, `sample`: not in the first
  release (8.7), see A12.
- **AFT wrapper, PH over log-normal/log-logistic baselines**: Slice D.
- **Derivatives of a PH wrapper over a custom baseline**: 7.3 guarantees
  wrappers over built-in baselines only; custom baselines are covered for
  values, issues and isolation.
- **Exact warning text, repr**: unspecified.

## A12: staged RMST and PH–RMST (spec 4.4, 4.5, 5, 8.1, 8.4, 8.7, 10.2, 12.1; rev 41)

References: `rmst_reference.py`, mpmath quadrature (40 digits) of the
physical law `S(t)` and of `dS/dtheta`, split at cumulative-hazard levels and
cut at `H = 2000`; higher orders by mpmath `diff` of the closed forms (60
digits). Inputs are rounded to the active dtype before the reference is
evaluated. Values: 12.1 `rtol`, zero absolute tolerance. Derivatives: the
12.1 RMST scales with factor 12.1 `rtol`, `eta` at `|rmst|`, higher orders
in log units (rev 44).

### `test_a12_rmst_reference.py` (no survjax calls)

| Spec | Requirement | Tests |
| --- | --- | --- |
| 8.1/8.7/12.1 | the reference integration itself: equals the closed form, its derivatives and a refined quadrature over the full grid; no mass missed on `k=0.2`, `tau/b=1e5`; `k=1` and PH-law identities | `test_quadrature_value_equals_closed_form`, `test_quadrature_derivatives_equal_closed_form_derivatives`, `test_refined_breakpoints_do_not_change_the_reference`, `test_very_large_interval_with_heavy_tail_keeps_its_mass`, `test_weibull_shape_one_reference_equals_exponential_identity`, `test_ph_law_reference_equals_effective_scale_closed_form`, `test_exponential_reference_derivative_equals_numerical_derivative` |

### `test_a12_rmst_values.py`

| Spec | Requirement | Tests |
| --- | --- | --- |
| 8.7 | Exponential `-expm1(-r*tau)/r` | `test_exponential_rmst_matches_reference`, `test_exponential_rmst_is_vectorized_over_rates` |
| 8.7/12.1 | Weibull over `k` in [0.2, 50], `tau/b` in [1e-2, 1e5], both dtypes; underflow (RMST about `tau`) and overflow (about the mean) tails | `test_weibull_rmst_matches_quadrature`, `test_weibull_rmst_where_the_power_underflows_is_about_tau`, `test_weibull_rmst_where_the_power_overflows_is_about_the_mean`, `test_weibull_rmst_is_vectorized_over_shapes` |
| 12.1 | Exponential special case (`k=1`); scale changes | `test_weibull_shape_one_equals_exponential`, `test_weibull_scale_change` |
| 4.4/8.7 | PH-Weibull against the physical law and the equivalent Weibull; PH-Exponential hand value; `(S, 1)` x `(S, 2)`; both power tails through PH | `test_ph_weibull_rmst_matches_physical_law_quadrature`, `test_ph_weibull_rmst_equals_equivalent_weibull`, `test_ph_exponential_rmst_is_hand_derived`, `test_ph_rmst_with_draw_by_arm_broadcasting`, `test_ph_weibull_rmst_tail_cases` |
| 3.3 | result dtype | `test_rmst_result_dtype_is_the_parameter_dtype` |

### `test_a12_rmst_derivatives.py`

| Spec | Requirement | Tests |
| --- | --- | --- |
| 8.7/12.1 | first derivatives in `k`, `b`, `tau` over the full grid, forward and reverse, both dtypes; finite and correct in the power tails | `test_weibull_first_derivatives_match_quadrature`, `test_weibull_first_derivatives_are_finite_in_the_power_tails` |
| 8.7 | `d rmst/d tau = S(tau)` | `test_cutoff_derivative_is_the_survival_at_the_cutoff` |
| 8.7 | Exponential, PH-Weibull (`k`, `b0`, `eta`, `tau`, physical law), PH-Exponential first derivatives; PH identity in derivatives | `test_exponential_first_derivatives_match_reference`, `test_ph_weibull_first_derivatives_match_physical_law_quadrature`, `test_ph_exponential_first_derivatives_are_hand_derived`, `test_ph_weibull_first_derivatives_equal_equivalent_weibull` |
| 4.5/8.7 | `tau=0`: value 0, parameter derivatives 0, `tau` derivative 1 | `test_rmst_at_zero_is_zero`, `test_first_derivatives_at_zero_cutoff` |
| 8.7/12.1 | Exponential all orders: Hessians (four compositions), third derivatives, `check_grads(order=2)`, PH-Exponential | `test_exponential_hessian_in_rate_and_cutoff`, `test_ph_exponential_hessian_in_rate_predictor_and_cutoff`, `test_exponential_third_derivatives`, `test_check_grads_order_two_exponential_rmst`, `test_check_grads_order_two_ph_exponential_rmst` |
| 8.7/12.1 | Weibull and PH-Weibull derivatives not involving `k` on interior `tau>0`: Hessians in `b`/`b0`, `eta`, `tau`; third `tau` derivative; `check_grads` without the shape direction | `test_weibull_hessian_in_scale_and_cutoff`, `test_ph_weibull_hessian_in_scale_predictor_and_cutoff`, `test_weibull_third_cutoff_derivative`, `test_check_grads_order_two_weibull_rmst_without_shape`, `test_check_grads_order_two_ph_weibull_rmst_without_shape` |
| 8.7/12.1 | shape Hessians (`k`, `(k, b, tau)`, PH `(k, eta)`) raise `NotImplementedError` or are correct, never wrong (rev 45); `until("summary.rmst")` | `test_weibull_shape_hessian_raises_or_is_correct`, `test_weibull_full_hessian_raises_or_is_correct`, `test_ph_weibull_shape_hessian_raises_or_is_correct` |

### `test_a12_rmst_policy.py`

| Spec | Requirement | Tests |
| --- | --- | --- |
| 4.5/8.7 | NaN for negative, NaN, `±inf` `tau`, eager and `jit`; valid entries kept | `test_invalid_cutoff_gives_nan`, `test_invalid_cutoff_entries_leave_valid_entries` |
| 8.7 | NaN wherever `parameter_issues()` reports a violation (baseline, `eta`) | `test_parameter_issue_entries_are_nan`, `test_parameter_issue_nan_under_jit` |
| 8.7 | valid entries keep values and first derivatives beside NaN entries, incl. shared parameters, forward/reverse, `jit`; Exponential Hessian isolation; hand-checked shared-scale gradient | `test_valid_entries_keep_values_beside_nan_entries`, `test_valid_entries_keep_first_derivatives_beside_nan_entries`, `test_valid_entries_keep_first_derivatives_under_jit`, `test_exponential_shared_rate_hessian_is_isolated`, `test_shared_scale_gradient_beside_invalid_cutoff_is_hand_checked` |
| 8.7 | `rmst(5.0)` `(S, 2)`; `rmst(tau_grid[:, None, None])` `(m, S, 2)`; trailing pairing; native grid; no `rmst_grid`; non-broadcasting `tau` `TypeError` (rev 46) | `test_scalar_cutoff_against_draws_by_arms`, `test_explicit_cutoff_grid_against_draws_by_arms`, `test_cutoff_vector_pairs_with_the_trailing_arm_axis`, `test_cutoff_grid_for_native_weibull_batch`, `test_there_is_no_rmst_grid`, `test_cutoff_that_does_not_broadcast_is_rejected` |
| 4.5/8.7 | `tau` dtype rules: boolean, complex, string, object `TypeError` (eager and traced); float64 under 32-bit mode converts without warning (rev 46) | `test_cutoff_with_a_non_numeric_dtype_is_rejected`, `test_boolean_cutoff_is_rejected_during_tracing`, `test_float64_cutoff_converts_without_warning_in_32_bit_mode` |
| 12.1 | eager/`jit`/`vmap` agreement | `test_rmst_agrees_under_jit`, `test_rmst_agrees_under_vmap_over_distribution_and_cutoff`, `test_rmst_gradient_agrees_under_vmap_and_jit` |
| 8.7/12.1 | `rmst` on a custom family and on PH of a custom baseline raises `NotImplementedError` (call and trace), by type not values; no `options` argument (rev 41); `until("summary.rmst")` | `test_rmst_on_a_custom_family_raises`, `test_rmst_check_depends_on_the_type_not_values`, `test_rmst_on_a_ph_wrapper_of_a_custom_baseline_raises`, `test_rmst_takes_no_options_argument` |
| 8.7/12.1 | `mean` raises everywhere (`until("summary.mean")`); `median` and `quantile` raise everywhere (`until("summary.quantile")`), call and trace (rev 41) | `test_mean_raises_on_every_family`, `test_median_raises_on_every_family`, `test_quantile_raises_on_every_family` |
| 8.7 | absent (`AttributeError`, rev 46): `rmst_with_info` (`until("summary.rmst")`), `inverse_cumulative_hazard` (`until("summary.quantile")`), `sample` (`until("sampling")`) | `test_absent_first_release_methods` |
| 8.7 | `SummaryOptions`, `SummaryInfo` not exported from any public module (rev 46); `until("summary.containers")` | `test_summary_configuration_types_are_not_provided` |

### A12: deliberately not tested

- **Weibull/PH-Weibull higher derivatives at `tau=0`** and **the tangent of a
  NaN entry itself**: outside the staged guarantee (8.7).
- **A required raise from some shape Hessian**: not required (12.1, rev 45).
- **Second-order isolation for Weibull families**: shape Hessians are
  restricted; Exponential second-order isolation is tested
  (the `summary.rmst` feature tests Weibull Hessian isolation).
- **`tau` outside `[1e-2, 1e5] * b`, `k` outside [0.2, 50]**: no accuracy
  requirement.
- **Section 8.4 quadrature tangent, `SummaryInfo` statuses, generic RMST,
  `mean`/`median`/`quantile` values, sampling**: post-release features
  (below) and Slice D.
- **Executable documentation of the derivative limitation**: a
  documentation gate, not a test.

## Post-release features (spec 4.2, 4.4, 4.5, 8.1-8.5, 8.7, 8.8, 12.1; rev 51-56)

Section 8.8 (rev 51) enables each section 8.1 method, and the two containers,
by its own step in any order, under the full contract of 8.1–8.6; 8.7's tail
accuracy, value policy and shape rules stay in force and apply in the same way
to `p`. Tests carry `@pytest.mark.feature(...)` with the names of the
"Post-release features" table and run only when every named feature is in
`SPEC_FEATURES`. A test that also needs `options` or `SummaryInfo` lists
`summary.containers` too ("+ containers" below). The `sampling` feature has no
tests yet (maintainer: not needed yet).

References: 8.1/8.2 closed forms (`summary_reference`, mpmath `diff` at 60
digits for derivatives; hand algebra in docstrings), mpmath quadrature of the
physical law (`rmst_reference`, `summary_reference.log_logistic_rmst`), hand
values for uniform and mixture-cure families. Built-in closed forms and
Weibull RMST use the 12.1 scales; generic quadrature/solver paths use
`GENERIC_FACTOR = 10` times them (12.1 rev 54: "compared at `10*rtol`
relative"; derivative units: Q-058). Shared cases live in
`summary_cases.py` (`entry_tangents` for per-entry NaN-tangent checks).
Families: `spec_families.uniform_support`, `nan_beyond`,
`far_reference_log_logistic`, `const_hazard`, `power_hazard`,
`softplus_log_logistic`, `mixture_cure`, `constrained_rate`, `capped_rate`,
and (rev 54) `known_mean_log_logistic`, `mean_with_info_sentinel`.

### `summary.containers`: `test_summary_containers_schema.py`

| Spec | Requirement | Tests |
| --- | --- | --- |
| 8.5 | `survjax.SummaryOptions` (and `SummaryInfo`, Q-056) at the top level | `test_summary_containers_are_exported_at_top_level` |
| 8.5 | `SummaryOptions(rtol=None, atol=None, max_levels=8, max_solver_steps=64, max_bracket_steps=64)`; stored values; frozen | `test_summary_options_defaults`, `test_summary_options_stores_explicit_values`, `test_summary_options_is_frozen` |
| 8.5 | "Explicit tolerances are dynamic PyTree leaves; the three work limits are static metadata"; one trace across tolerance values, retrace on a limit | `test_explicit_tolerances_are_the_pytree_leaves`, `test_tolerance_values_do_not_change_the_tree_structure`, `test_work_limits_are_static_metadata`, `test_options_cross_jit_with_tolerances_traced_and_limits_static` |
| 8.5 | `SummaryInfo.Status` `IntEnum` codes 0-5 | `test_status_is_an_int_enum_with_the_specified_codes` |
| 8.5 | "a frozen PyTree with all five fields as array leaves"; "Construction stores these fields" | `test_summary_info_construction_stores_its_fields`, `test_summary_info_has_five_array_leaves`, `test_summary_info_round_trips_through_tree_map`, `test_summary_info_passes_through_jit`, `test_summary_info_is_frozen` |

### `summary.rmst`: `test_summary_rmst_*.py`

| Spec | Requirement | Tests |
| --- | --- | --- |
| 4.2/4.4/8.8 | every family (built-ins, PH closures, custom, PH without closure) provides `rmst`; no `rmst_grid` | `test_every_family_provides_rmst`, `test_there_is_still_no_rmst_grid` (surface) |
| 8.1/8.5/8.7 (+ containers) | `rmst_with_info` returns `(values, SummaryInfo)` from the same backend; `converged`, integer codes, field shapes, counts; `options=None` defaults; keyword-only `options`; built-ins accept options | `test_rmst_with_info_returns_values_and_summary_info`, `test_valid_interior_rmst_reports_converged`, `test_rmst_info_fields_are_shaped_like_the_values`, `test_rmst_options_none_selects_the_defaults`, `test_rmst_options_are_keyword_only`, `test_built_in_rmst_accepts_explicit_options` (surface) |
| 8.3/8.8 | generic fallbacks: constant and power hazards, batched; PH without closure; bounded support; improper finite RMST; heavy tail; scale changes; dtype | `test_generic_rmst_of_a_constant_hazard`, `test_generic_rmst_of_a_power_hazard_matches_quadrature`, `test_generic_rmst_is_vectorized_over_a_batch`, `test_ph_rmst_on_a_baseline_without_closure_matches_the_physical_law`, `test_bounded_support_rmst_is_hand_derived`, `test_improper_family_has_finite_rmst`, `test_heavy_tail_generic_rmst`, `test_generic_rmst_scales_with_the_time_scale`, `test_generic_rmst_is_in_the_parameter_dtype` (values) |
| 4.5/8.7 | 8.7 `tau` policy on generic paths: NaN, `rmst(0)=0`, dtype `TypeError` (eager, traced), float64 conversion; `tau_grid[:, None, None]`; non-broadcasting `tau`; parameter violations NaN | `test_generic_invalid_cutoff_gives_nan`, `test_generic_rmst_at_zero_is_zero`, `test_generic_cutoff_with_a_non_numeric_dtype_is_rejected`, `test_generic_boolean_cutoff_is_rejected_during_tracing`, `test_generic_float64_cutoff_converts_without_warning_in_32_bit_mode`, `test_generic_cutoff_grid_with_explicit_axes`, `test_generic_cutoff_that_does_not_broadcast_is_rejected`, `test_parameter_violations_give_nan_rmst`, `test_non_finite_integrand_gives_nan_rmst`, `test_rmst_agrees_under_jit_and_vmap` (policy) |
| 8.5 rev 54 (+ containers) | "Invalid arguments (`tau<0` or non-finite ...) and entries with parameter issues have status `invalid_input` and NaN values"; "`rmst(0)=0` ... `converged`"; per entry | `test_invalid_cutoff_reports_invalid_input`, `test_parameter_violations_report_invalid_input`, `test_zero_cutoff_reports_converged`, `test_cutoff_statuses_are_per_entry` (policy) |
| 8.3/8.5 (+ containers) | violation status under `jit`; info shapes on grids; `non_finite_arithmetic` per entry; `work_exhausted`; counts; error estimate in time units; `jit`/`vmap` of info; options into `jit` | `test_cutoff_grid_with_info_against_draws_by_arms`, `test_parameter_violation_status_under_jit`, `test_non_finite_integrand_is_a_failure_status`, `test_quadrature_work_exhaustion_is_reported`, `test_generic_rmst_counts`, `test_error_estimate_is_in_time_units`, `test_rmst_with_info_agrees_under_jit_and_vmap`, `test_options_cross_jit_into_a_generic_rmst` (policy) |
| 8.3/8.5 rev 52 (+ containers) | "one adjacent pair ... No second successful pair is required; one evaluated level alone cannot establish convergence" (converged `num_levels` in `[2, max_levels]`, loose `atol` converges at 2); "`num_levels` counts the rounds ... not the sum of counts over split intervals" (exhaustion at `max_levels=3` across the `s` split gives 3); "`error_estimate` is the last absolute difference ... including on work exhaustion" (finite, positive), "NaN if fewer than two estimates exist" (`max_levels=1`) "or if a non-finite integrand"; rev 53: "A converged entry returns `Q_l` of the converging pair" (error estimate within `atol + rtol*abs(value)`), a failure status "returns NaN" | `test_one_adjacent_pair_suffices_for_convergence`, `test_work_exhaustion_counts_rounds_not_split_intervals`, `test_single_level_reports_one_round_and_a_nan_error_estimate`, `test_non_finite_integrand_has_a_nan_error_estimate`, `test_generic_rmst_counts`, `test_non_finite_integrand_is_a_failure_status`, `test_error_estimate_is_in_time_units`, `test_quadrature_work_exhaustion_is_reported`, `test_non_finite_integrand_gives_nan_rmst` (policy); `test_non_converged_entry_has_a_nan_tangent` (derivatives, premise only) |
| 8.4/8.8 | Weibull and PH-Weibull shape Hessians correct (interior, heavy, both power tails), four compositions; `check_grads` with the shape direction | `test_weibull_rmst_shape_second_derivative_is_correct`, `test_weibull_rmst_full_hessian_is_correct`, `test_ph_weibull_rmst_hessian_with_shape_is_correct`, `test_check_grads_order_two_weibull_rmst_with_shape`, `test_check_grads_order_two_ph_weibull_rmst_with_shape` (derivatives) |
| 8.4 | `tau=0` first derivatives (parameters 0, `tau` 1) on every kind of family (rev 51, family-neutral) | `test_rmst_first_derivatives_at_zero_cutoff` (derivatives) |
| 8.4 | generic tangents: first derivatives, Hessian, PH without closure, `check_grads`, `jit`/`vmap` | `test_generic_rmst_first_derivatives_match_quadrature`, `test_generic_rmst_hessian`, `test_ph_rmst_without_closure_derivatives_match_the_physical_law`, `test_check_grads_order_two_generic_rmst`, `test_generic_rmst_gradient_agrees_under_jit_and_vmap` (derivatives) |
| 8.4/8.7 | isolation with shared parameters (invalid `tau`, non-finite failure, Weibull Hessian); NaN tangent of a failed entry; NaN tangent of a non-converged entry (+ containers) | `test_generic_rmst_isolates_invalid_cutoffs_with_shared_rate`, `test_valid_entries_isolated_from_a_non_finite_failure`, `test_weibull_rmst_hessian_isolation_with_shared_parameters`, `test_non_finite_failure_entry_has_a_nan_tangent`, `test_non_converged_entry_has_a_nan_tangent` (derivatives) |
| 8.4 rev 54 | "invalid-input entries" get NaN tangents: invalid `tau` (`<0`, NaN, `±inf`) on every family, with `tau=0` and interior entries finite; parameter-issue entries (built-in and generic) | `test_invalid_cutoff_entries_have_nan_tangents`, `test_parameter_violation_entry_has_a_nan_rmst_tangent` (derivatives) |

### `summary.mean`: `test_summary_mean_*.py`

| Spec | Requirement | Tests |
| --- | --- | --- |
| 4.2/4.4/8.8 | every family provides `mean` | `test_every_family_provides_mean` (surface) |
| 8.1/8.5/8.7 (+ containers) | `mean_with_info` from the same backend; `converged`; field shapes; `options=None`; keyword-only; built-ins accept options | `test_mean_with_info_returns_values_and_summary_info`, `test_valid_mean_reports_converged`, `test_mean_info_fields_are_shaped_like_the_values`, `test_mean_options_none_selects_the_defaults`, `test_mean_options_are_keyword_only`, `test_built_in_mean_accepts_explicit_options` (surface) |
| 8.1 | Exponential `1/r`, Weibull `b*Γ(1+1/k)`, batched, scale changes, dtype | `test_exponential_mean_is_the_reciprocal_rate`, `test_weibull_mean_closed_form`, `test_built_in_mean_is_vectorized_over_the_batch`, `test_built_in_mean_scales_with_the_time_scale`, `test_mean_is_in_the_parameter_dtype` (values) |
| 8.3/4.4 | generic fallbacks (constant, power, batched, scale changes); PH identities (closure, generic, PH-Exponential, `(S, 1)` x `(S, 2)`); bounded support; heavy tails | `test_generic_mean_of_a_constant_hazard`, `test_generic_mean_of_a_power_hazard`, `test_generic_mean_is_vectorized_over_a_batch`, `test_generic_mean_scales_with_the_time_scale`, `test_ph_weibull_mean_equals_the_equivalent_weibull`, `test_ph_exponential_mean_equals_the_equivalent_exponential`, `test_ph_mean_with_draw_by_arm_broadcasting`, `test_bounded_support_mean_is_half_the_end`, `test_heavy_tail_generic_mean` (values) |
| 8.1 rev 54 | generic mean "returns `+inf` ... wherever `log_survival_at_infinity() > -inf`" (improper family) | `test_improper_family_mean_is_plus_infinity` (values) |
| 8.1/8.5 rev 54 (+ containers) | "with status `known_infinite`, without quadrature" (zero counts, NaN error estimate), decided per entry; `beta` near 1 correct or unresolved, never `+inf` | `test_improper_family_mean_is_known_infinite`, `test_improper_family_mean_uses_no_quadrature`, `test_known_infinite_mean_is_decided_per_entry`, `test_heavy_tail_near_one_mean_is_correct_or_reported_unresolved` (values) |
| 8.1 rev 54, 56 (+ containers) | "A family reports it by overriding `mean_with_info`; `mean` returns the value part"; rev 56 override signature `mean_with_info(self, *, options=None)`, "may ignore `options`": custom log-logistic override (`beta` 0.5 and 1 `+inf`/`known_infinite`; `beta` 2 `alpha*π/2`/`converged`; per entry); sentinel override returned by `mean()` with and without `options` | `test_log_logistic_mean_override_reports_known_infinite`, `test_log_logistic_mean_override_closed_form_for_beta_two`, `test_log_logistic_mean_override_is_per_entry`, `test_mean_returns_the_value_part_of_a_mean_with_info_override` (values) |
| 8.1 rev 56 (+ containers) | "An analytical answer reports status `converged` (or `known_infinite`), a NaN `error_estimate` and zero work counts": Exponential, Weibull and PH-closure means | `test_built_in_analytical_mean_reports_no_work` (values) |
| 8.3/8.7 | `batch_shape`; parameter violations NaN; non-finite integrand NaN; `jit`/`vmap` | `test_mean_has_the_batch_shape`, `test_parameter_violations_give_nan_mean`, `test_non_finite_integrand_gives_nan_mean`, `test_mean_agrees_under_jit_and_vmap` (policy) |
| 8.3/8.5 (+ containers) | info shapes; violations `invalid_input` (rev 54); `non_finite_arithmetic`; `work_exhausted`; counts; `jit`/`vmap` of info | `test_mean_info_has_the_batch_shape`, `test_parameter_violations_report_invalid_input`, `test_non_finite_integrand_in_the_mean_is_a_failure_status`, `test_quadrature_work_exhaustion_of_the_mean_is_reported`, `test_generic_mean_counts`, `test_mean_with_info_agrees_under_jit_and_vmap` (policy) |
| 8.3/8.5 rev 52 (+ containers) | the same quadrature stopping rules on the mean: one pair suffices; `[0, s]` and `[s, +inf)` share rounds; last-difference error estimate, NaN with one level or a non-finite integrand; rev 53: a failure status "returns NaN" | `test_one_adjacent_pair_suffices_for_mean_convergence`, `test_mean_work_exhaustion_counts_rounds_not_split_intervals`, `test_single_level_mean_reports_one_round_and_a_nan_error_estimate`, `test_non_finite_integrand_in_the_mean_has_a_nan_error_estimate`, `test_generic_mean_counts`, `test_quadrature_work_exhaustion_of_the_mean_is_reported`, `test_non_finite_integrand_in_the_mean_is_a_failure_status`, `test_non_finite_integrand_gives_nan_mean` (policy) |
| 8.4 | built-in first derivatives and Hessian; PH chain rule; generic first derivatives and Hessian; `check_grads`; `jit`/`vmap`; isolation; non-finite tangent of the improper mean | `test_exponential_mean_derivative`, `test_weibull_mean_derivatives`, `test_weibull_mean_hessian`, `test_ph_weibull_mean_gradient_equals_the_equivalent_weibull`, `test_generic_mean_first_derivatives`, `test_generic_mean_hessian`, `test_check_grads_order_two_mean`, `test_generic_mean_gradient_agrees_under_jit_and_vmap`, `test_mean_isolates_failing_entries_with_shared_scale` (derivatives) |
| 8.4 rev 54 | NaN tangents of "known-infinite means" (improper family, rate and cure directions) and of "invalid-input entries" (parameter issues, built-in and generic) | `test_improper_mean_has_a_nan_tangent`, `test_parameter_violation_entry_has_a_nan_mean_tangent` (derivatives) |
| 8.1/8.4 rev 56 (+ containers) | "The override is responsible for the section 8.4 NaN tangents of a known infinite mean"; `mean()` returns the value part, so it carries the override's NaN (`beta<=1`) and finite (`beta=2`) tangents | `test_mean_carries_the_tangents_of_a_mean_with_info_override` (derivatives) |
| 8.4 rev 57 | "An entry with a NaN tangent never gives NaN derivatives to other entries, in forward or reverse mode, including through shared parameters": hand-derived derivatives of finite means beside a known-infinite override entry (shared `alpha`, + containers), a known-infinite improper entry (shared rate, generic), and a non-finite-`eta` entry (shared baseline rate, PH closure and generic) | `test_finite_mean_keeps_its_derivative_beside_an_override_known_infinite_entry`, `test_finite_mean_keeps_its_derivative_beside_a_known_infinite_improper_entry`, `test_valid_means_keep_hand_derived_derivatives_beside_an_invalid_input_entry` (derivatives) |

### `summary.quantile`: `test_summary_quantile_*.py`

| Spec | Requirement | Tests |
| --- | --- | --- |
| 4.2/4.4/8.1/8.8 | every family provides `quantile`, `median` (= `quantile(0.5)`, no longer raising) and public `inverse_cumulative_hazard` (no longer absent); `quantile(p) = H⁻¹(-log1p(-p))` | `test_every_family_provides_quantile`, `test_median_is_the_half_quantile`, `test_every_family_provides_inverse_cumulative_hazard`, `test_quantile_is_the_inverse_cumulative_hazard_of_minus_log1p_minus_p` (surface) |
| 8.1/8.5/8.7 (+ containers) | `quantile_with_info` and `median_with_info` from the same backend; `converged`; field shapes; `options=None`; keyword-only for all four; built-ins accept options | `test_quantile_with_info_returns_values_and_summary_info`, `test_valid_interior_quantile_reports_converged`, `test_quantile_info_fields_are_shaped_like_the_values`, `test_quantile_options_none_selects_the_defaults`, `test_quantile_options_are_keyword_only`, `test_built_in_quantile_accepts_explicit_options`, `test_median_with_info_is_the_half_quantile_with_info` (surface) |
| 8.1/8.2 | closed forms: Exponential and Weibull quantile, median and inverse cumulative hazard; scale changes; dtype | `test_exponential_quantile_closed_form`, `test_weibull_quantile_closed_form`, `test_built_in_quantile_scales_with_the_time_scale`, `test_exponential_median_is_log_two_over_rate`, `test_weibull_median_closed_form`, `test_exponential_inverse_cumulative_hazard_closed_form`, `test_weibull_inverse_cumulative_hazard_closed_form`, `test_quantile_results_are_in_the_parameter_dtype` (values) |
| 8.1 | endpoints from `support()` (default and bounded); rev 54: "`e=0` gives the lower and `e=+inf` the upper end of `support()`" (default, bounded; improper `+inf`); `H(H⁻¹(e)) = e` on every family | `test_quantile_endpoints_follow_the_default_support`, `test_bounded_support_quantile_endpoints_come_from_support`, `test_inverse_cumulative_hazard_at_zero_is_zero`, `test_inverse_cumulative_hazard_at_infinity_is_the_upper_support_end`, `test_bounded_support_inverse_cumulative_hazard_endpoints`, `test_improper_family_inverse_cumulative_hazard_beyond_the_cure_level_is_infinite`, `test_cumulative_hazard_inverts_the_inverse_cumulative_hazard` (values) |
| 8.2/4.4 | generic solver (constant and power hazards, median, batched, scale changes); PH identities (closure and generic, `H0⁻¹(e*exp(-eta))`, PH-Exponential, `(S, 1)` x `(S, 2)`); bounded support; improper below and beyond the cure level (`+inf`) | `test_generic_quantile_of_a_constant_hazard`, `test_generic_quantile_of_a_power_hazard`, `test_generic_inverse_cumulative_hazard_of_a_power_hazard`, `test_generic_median_of_a_power_hazard`, `test_generic_quantile_is_vectorized_over_a_batch`, `test_generic_quantile_scales_with_the_time_scale`, `test_ph_weibull_quantile_equals_the_equivalent_weibull`, `test_ph_inverse_cumulative_hazard_is_the_baseline_inverse_at_scaled_hazard`, `test_ph_exponential_quantile_equals_the_equivalent_exponential`, `test_ph_quantile_with_draw_by_arm_broadcasting`, `test_bounded_support_quantile_is_hand_derived`, `test_bounded_support_inverse_cumulative_hazard_is_hand_derived`, `test_improper_family_quantile_below_the_cure_level`, `test_improper_family_quantile_beyond_the_cure_level_is_infinite`, `test_improper_family_inverse_cumulative_hazard_below_the_cure_level`, `test_improper_family_inverse_cumulative_hazard_beyond_the_cure_level_is_infinite` (values) |
| 4.5/8.7/8.8 | `p` policy: NaN outside `[0,1]` (eager, `jit`), valid entries kept; dtype `TypeError` (eager, traced); float64 conversion; shapes (scalar, `p_grid[:, None, None]`, non-broadcasting `TypeError`); `e_grid` axes; violations NaN for quantile and median; default-limit convergence far from `reference_time()`; `jit`/`vmap` incl. median and `H⁻¹`; `vmap` over `p` | `test_invalid_probability_gives_nan`, `test_invalid_probability_entries_leave_valid_entries`, `test_probability_with_a_non_numeric_dtype_is_rejected`, `test_boolean_probability_is_rejected_during_tracing`, `test_float64_probability_converts_without_warning_in_32_bit_mode`, `test_scalar_probability_against_draws_by_arms`, `test_probability_grid_with_explicit_axes`, `test_probability_that_does_not_broadcast_is_rejected`, `test_inverse_cumulative_hazard_grid_with_explicit_axes`, `test_parameter_violations_give_nan_quantile`, `test_far_reference_quantile_converges_with_default_limits`, `test_quantile_agrees_under_jit_and_vmap`, `test_quantile_agrees_under_vmap_over_probabilities` (policy) |
| 8.1 rev 54 | `inverse_cumulative_hazard(e, *, options=None)`: "negative or NaN `e` gives NaN" (eager, `jit`, per entry); "no `_with_info` twin" (`AttributeError`) | `test_invalid_hazard_level_gives_nan`, `test_invalid_hazard_levels_leave_valid_entries`, `test_inverse_cumulative_hazard_has_no_with_info_twin` (policy) |
| 4.5/8.1 rev 54 (d104627) | "the section 4.5 dtype rule applies to `e` as to `t`": boolean, complex, string, object `e` raise `TypeError` (eager, traced); float64 `e` converts without a warning in 32-bit mode | `test_hazard_level_with_a_non_numeric_dtype_is_rejected`, `test_boolean_hazard_level_is_rejected_during_tracing`, `test_float64_hazard_level_converts_without_warning_in_32_bit_mode` (policy) |
| 8.5 rev 56, 57 (+ containers) | "On an improper family, a quantile that is `+inf` because `p >= 1 - S(+inf)` has status `known_infinite`": interior `p` (0.75, 0.9) beside a converged `p = 0.5`; per batch entry (cure `[0.3, 0]`); rev 57 "including `quantile(1)` (this rule takes precedence over the endpoint rule)" (Q-068 settled) | `test_improper_family_quantile_beyond_the_cure_level_reports_known_infinite`, `test_known_infinite_quantile_is_decided_per_entry_of_the_batch`, `test_improper_family_upper_endpoint_quantile_reports_known_infinite` (policy) |
| 8.1/8.5 rev 54 (+ containers) | `inverse_cumulative_hazard` "takes `options` for its solver limits": `None` defaults, keyword-only, one bracket step gives NaN ("failures are NaN") while the default converges | `test_inverse_cumulative_hazard_options_none_selects_the_defaults`, `test_inverse_cumulative_hazard_options_are_keyword_only`, `test_inverse_cumulative_hazard_options_set_the_solver_limits` (policy) |
| 8.2/8.5 rev 54 (+ containers) | invalid `p` (`>1`, `<0`, NaN, `+inf`) and parameter issues `invalid_input`; endpoint quantiles `converged` (default support, bounded `end`); "`failed_bracket` when the bracket step limit ... is reached", "`work_exhausted` when `max_solver_steps` is reached inside a bracket", with the counts at the limit | `test_invalid_probability_reports_invalid_input`, `test_parameter_violations_report_invalid_input`, `test_endpoint_quantiles_report_converged`, `test_bounded_support_endpoint_quantiles_report_converged`, `test_bracket_limit_reports_failed_bracket`, `test_solver_step_limit_inside_a_bracket_reports_work_exhausted` (policy) |
| 8.2/8.5 (+ containers) | `_with_info` dtype and broadcast `TypeError`; info grid shapes; default-limit `converged`; counts; `jit`/`vmap` of info | `test_probability_with_a_non_numeric_dtype_is_rejected_with_info`, `test_probability_grid_info_has_the_broadcast_shape`, `test_probability_that_does_not_broadcast_is_rejected_with_info`, `test_far_reference_quantile_reports_converged_with_default_limits`, `test_generic_quantile_counts`, `test_quantile_with_info_agrees_under_jit_and_vmap` (policy) |
| 8.4 | built-in quantile and `H⁻¹` first derivatives (`dq/dp = 1/f(q)`, `dt/de = 1/h(t)`), Hessian; median = quantile derivative; PH chain rule; generic first derivatives (quantile, `H⁻¹`) and Hessian; `check_grads`; `jit`/`vmap`; isolation; non-finite `quantile(1)` tangent | `test_exponential_quantile_derivatives`, `test_weibull_quantile_derivatives`, `test_weibull_quantile_hessian`, `test_weibull_inverse_cumulative_hazard_derivatives`, `test_median_derivatives_equal_the_half_quantile_derivatives`, `test_ph_weibull_quantile_gradient_equals_the_equivalent_weibull`, `test_generic_quantile_first_derivatives`, `test_generic_inverse_cumulative_hazard_derivatives`, `test_generic_quantile_hessian`, `test_check_grads_order_two_quantile`, `test_generic_quantile_gradient_agrees_under_jit_and_vmap`, `test_generic_quantile_isolates_invalid_probabilities` (derivatives) |
| 8.4 rev 54 | NaN tangents "even where a one-sided or constant derivative exists": endpoint quantiles `p` 0 and 1 (rate and `p`; built-in and generic), bounded `quantile(1) = end`, `H⁻¹` at `e` 0 and `+inf` (rate and `e`), invalid `p` and `e` entries, parameter-issue entries; interior entries finite | `test_endpoint_quantile_has_nan_tangents`, `test_bounded_support_upper_endpoint_quantile_has_a_nan_tangent`, `test_inverse_cumulative_hazard_endpoints_have_nan_tangents`, `test_invalid_probability_entries_have_nan_tangents`, `test_invalid_hazard_level_entries_have_nan_tangents`, `test_parameter_violation_entry_has_a_nan_quantile_tangent` (derivatives) |
| 8.4 rev 56 | "known-infinite ... quantiles" get NaN tangents: improper `+inf` quantile in cure, rate and `p`; finite below the cure level | `test_known_infinite_quantile_has_nan_tangents` (derivatives) |
| 8.4 rev 57 | isolation "in forward or reverse mode, including through shared parameters", hand-derived: interior quantile beside known-infinite `p = 0.9` and `quantile(1)` (shared cure and rate); beside endpoint quantiles `p` 0 and 1 (shared rate, built-in and generic; shared bounded `end`); interior `H⁻¹` beside `e` 0 and `+inf` (shared rate); beside invalid `p`, invalid `e` (built-in and generic), a Weibull shape violation (shared scale) and a non-finite `eta` (shared baseline rate, PH closure and generic) | `test_finite_quantile_keeps_its_derivatives_beside_known_infinite_quantiles`, `test_interior_quantile_keeps_its_rate_derivative_beside_endpoint_quantiles`, `test_interior_quantile_keeps_its_derivative_beside_bounded_support_endpoints`, `test_interior_inverse_cumulative_hazard_keeps_its_rate_derivative_beside_endpoints`, `test_valid_quantiles_keep_their_rate_derivative_beside_invalid_probabilities`, `test_valid_levels_keep_their_rate_derivative_beside_invalid_hazard_levels`, `test_valid_quantiles_keep_their_scale_derivative_beside_a_parameter_violation`, `test_valid_quantiles_keep_their_rate_derivative_beside_an_invalid_log_hazard_ratio` (derivatives) |
| 8.4 rev 57 | "Its own NaN tangent is guaranteed in forward mode; its own reverse-mode derivative is unspecified": every NaN-tangent test above uses `jax.jvp`; no reverse-mode derivative of a NaN-tangent entry is asserted (mean, quantile, RMST, both brought-forward families) | (all `*_nan_tangent*` tests) |

### `observation.censoring`: `test_observation_censoring_*.py` (Slice B, rev 51)

Left- and interval-censored rows without truncation. References: hand
algebra in the module docstrings (Exponential and Weibull closed forms of
`log F` and `log(S(a) - S(b))`), and `observation_reference` (50-digit mpmath
of the spec 2 table, values and `mpmath.diff` derivatives). Row builder
`helpers.flag_data`.

| Spec | Requirement | Tests |
| --- | --- | --- |
| 2/7.1 | left `log F(c)`, interval `log(S(a)-S(b))`, Exponential and Weibull (`k<1`, `=1`, `>1`), hand-derived; hand formulas equal mpmath | `test_exponential_left_censored_row_is_hand_derived`, `test_exponential_interval_censored_row_is_hand_derived`, `test_weibull_left_censored_row_is_hand_derived`, `test_weibull_interval_censored_row_is_hand_derived`, `test_hand_values_agree_with_mpmath` (values) |
| 1/2/3.1/12.1 | mixed exact/right/left/interval batches; bounds form equals flag form; `time2` ignored on left rows; finite negative values | `test_weibull_mixed_batch_with_exact_and_right_rows`, `test_exponential_mixed_batch_with_exact_and_right_rows`, `test_bounds_form_equals_flag_form`, `test_time2_is_ignored_on_left_rows`, `test_censored_rows_are_finite_and_negative` (values) |
| 3.2/5 | `(D,1)` draws, per-row pairing, 2-D observations | `test_draw_column_against_censored_observations`, `test_per_row_parameters_pair_with_censored_rows`, `test_two_dimensional_censored_data` (values) |
| 2/4.2 | generic defaults (`power_hazard`) equal Weibull; improper `mixture_cure` left/interval | `test_custom_family_on_generic_defaults_matches_weibull`, `test_improper_family_left_and_interval_rows` (values) |
| 7.1/12.1 | zero model mass gives `-inf` (bounded support, untruncated half of the zero-numerator/zero-denominator pair); overflowing interval contribution `-inf`, not NaN | `test_interval_beyond_bounded_support_is_minus_infinity`, `test_overflowing_interval_contribution_is_minus_infinity` (values) |
| 3.1/3.3/6.2/7.1 | invalid left/interval rows (left at `+inf`, negative, NaN; reversed, equal, NaN and negative `time2`; conflicting flags; bounds equal, NaN and negative upper) are NaN exactly on `invalid_rows`, triaged to data, rule masks where 6.2 fixes them; eager, `jit`, traced construction, indexing | `test_invalid_censored_row_is_nan_and_attributed_to_data`, `test_invalid_censored_row_under_jit`, `test_invalid_censored_row_from_traced_construction`, `test_indexing_keeps_issues_of_censored_rows` (diagnostics) |
| 6.2/7.1 | valid mixed batch: no NaN, empty triage masks (Q-064); valid censored rows add no issue and are not `uninformative`; parameter violations NaN on censored rows, triaged to parameters | `test_valid_mixed_batch_has_no_nan_and_no_triage_mask`, `test_valid_censored_rows_add_no_issue`, `test_parameter_issue_on_a_censored_row_is_nan` (diagnostics) |
| 3.2/6.2/7.2/12.1 | eager/`jit`/`vmap` agreement: prepared data, traced flag and bounds construction, metadata-pruning invariance, `vmap` over draws and datasets, `jit(vmap)`, indexing and `tree.map` slicing, compilation reuse (prepared and traced), compiled composition with gradient | `test_prepared_data_under_jit_matches_eager`, `test_traced_flag_construction_inside_jit_matches_eager`, `test_traced_bounds_construction_inside_jit_matches_eager`, `test_concrete_metadata_without_left_rows_does_not_change_values`, `test_vmap_over_parameter_draws`, `test_vmap_over_datasets`, `test_jit_of_vmap_over_draws_matches_eager_batch`, `test_indexed_censored_data_keeps_row_values`, `test_new_censored_values_reuse_compilation`, `test_traced_censored_construction_reuses_compilation`, `test_censored_rows_compose_inside_one_compiled_objective` (transformations) |
| 7.3/12.1 | first and second derivatives: Exponential hand-derived (per-row Jacobians both modes, second derivatives three ways, shared-rate mixed batch); Weibull gradients and Hessians against mpmath; four fwd/rev Hessian compositions; `jit`, `vmap` over draws and datasets; `check_grads(order=2)` (float64) on Exponential, Weibull shared/per-row/`from_log`, `jit`/`vmap`, custom family; custom Hessian | `test_exponential_per_row_jacobian_is_hand_derived`, `test_exponential_left_and_interval_second_derivatives_are_hand_derived`, `test_exponential_shared_rate_mixed_batch_derivatives_are_hand_derived`, `test_weibull_left_and_interval_gradients_match_mpmath`, `test_weibull_mixed_batch_hessian_matches_mpmath`, `test_weibull_mixed_batch_gradient_matches_mpmath`, `test_forward_and_reverse_hessians_agree_with_mpmath`, `test_hessian_under_jit_matches_eager`, `test_hessian_under_vmap_over_parameter_draws`, `test_gradient_under_vmap_over_datasets`, `test_check_grads_*`, `test_custom_family_hessian_matches_mpmath` (derivatives) |
| 7.1/7.2/7.3 | isolation of rows without parameter issues on censored rows (zero, negative, NaN, `+inf` scale), both modes, to second order; reference-input isolation for censored rows | `test_rows_without_parameter_issues_keep_derivatives`, `test_reference_input_does_not_reach_censored_row_gradients` (derivatives) |
| 12.1 | cross-kind identities (hypothesis): interval `(c,+inf]` = right; `from_bounds(0,c)` = left; `logaddexp` of adjacent sets equals the union (left+interval, interval+interval, interval+right); left+right = 0; full coverage exactly zero | `test_interval_to_infinity_equals_right_censoring`, `test_bounds_from_zero_equals_left_censoring`, `test_left_plus_adjacent_interval_is_the_larger_left`, `test_adjacent_intervals_add_to_their_union`, `test_interval_plus_right_tail_is_the_earlier_right`, `test_left_plus_right_at_the_same_time_covers_everything`, `test_full_coverage_is_exactly_zero` (properties) |
| 4.1/7.2/12.1/12.2 | tiny masses (left at `1e-30`, Weibull `H` underflow, tiny intervals); upper tails (left `log F ~ -S` at `H` = 20, 33, 55, 70/600; intervals with `H(a)` = 900–1000); narrow intervals (`2**-20`, `2**-40`) and Weibull increments; eager `jax_debug_nans` on valid mixed censored batches (Exponential, Weibull, per-row, custom, bounds form) | `test_exponential_left_row_at_tiny_time_keeps_relative_precision`, `test_weibull_left_row_where_the_cumulative_hazard_underflows`, `test_exponential_interval_at_tiny_times`, `test_weibull_interval_at_tiny_times`, `test_exponential_left_row_in_the_upper_tail`, `test_weibull_left_row_in_the_upper_tail`, `test_exponential_interval_deep_in_the_upper_tail`, `test_weibull_interval_deep_in_the_upper_tail`, `test_exponential_narrow_interval`, `test_weibull_narrow_interval_uses_cumulative_hazard_increments`, `test_*_under_debug_nans` (numerics) |
| 4.2/7.1/12.2 | `log_interval_prob(lower, upper)` (Slice B by 12.2 rev 54; call convention and reversed-endpoint value Q-065): values; `+inf` upper is `log_survival`; `0` lower is `log_cdf`; `(0,+inf]` is `0`; equal endpoints `-inf`; reversed NaN; improper `(c,+inf]` = `log S(c)`; broadcasting; agrees with the interval row; `jit` and gradient | `test_finite_interval_matches_reference`, `test_upper_infinite_endpoint_is_log_survival`, `test_zero_lower_endpoint_is_log_cdf`, `test_whole_line_is_log_one`, `test_equal_endpoints_give_zero_probability`, `test_reversed_endpoints_are_nan`, `test_improper_family_interval_to_infinity_includes_the_cure_mass`, `test_endpoints_broadcast_against_the_batch`, `test_member_agrees_with_the_interval_row_contribution`, `test_member_under_jit_and_its_gradient` (interval_prob) |

Not tested for `observation.censoring`:

- **Left-censored at `c = 0`** (`(0, 0]`): collapsed and presumably invalid
  (7.1 "Input intervals of collapsed width are invalid"), but 3.3 and 6.2 do
  not say which mask the flag form sets for it; covered for the bounds form
  only (`bounds-equal`).
- **Untruncated `uninformative` left/interval rows:** impossible without
  truncation (a left or finite interval set cannot cover `(0, +inf]`); they
  are `observation.truncation` cases.
- **Which member the evaluator calls** (`log_interval_prob`,
  `log_conditional_interval_prob` or `log_cdf`) for an untruncated censored
  row: unspecified; tested only through values (Q-065).
- **Static-metadata pruning itself:** optional (7.2); only its invariance is
  tested.
- **R fixtures (flexsurv, every kind):** 12.2 Slice B gate, but fixtures are
  maintainer-supplied and none are committed yet.
- **Derivatives at `-inf` or overflowing rows, at support-change boundaries,
  and in data times:** outside 7.3's guarantee.

### `observation.truncation`: `test_observation_truncation_*.py` (Slice B, rev 51)

Left, right and two-sided truncation with every censoring kind. Tests with
only exact/right rows carry `feature("observation.truncation")`; tests with
left or interval rows carry `feature("observation.censoring",
"observation.truncation")` ("+ censoring" below). Hand algebra: every row is
`g(H(lo), H(hi)) - g(H(L), H(U))` with `g(x, y) = -x + log(-expm1(-(y-x)))`
(exact rows `log f(t) - g(H(L), H(U))`), cross-checked against
`observation_reference`.

| Spec | Requirement | Tests |
| --- | --- | --- |
| 2/7.1 | exact/right rows under left, right, two-sided truncation (Exponential, Weibull `k>1`, `k<1`), hand-derived; memorylessness; right with finite `U`; per-row windows; bounds = flag form; scalar truncation broadcast; `(D,1)` draws; hand helper = mpmath | `test_truncated_exact_and_right_rows_are_hand_derived`, `test_exponential_left_truncation_is_memoryless_by_hand`, `test_right_censored_beyond_entry_with_right_truncation_by_hand`, `test_per_row_truncation_windows`, `test_bounds_form_with_truncation_equals_flag_form`, `test_scalar_truncation_broadcasts_to_rows`, `test_draw_column_against_truncated_observations`, `test_hand_values_agree_with_mpmath` (values) |
| 1/2/4.2/12.1 (+ censoring) | left/interval rows with partial overlap at `L` and `U`; full matrix in one batch; mixed kinds and windows; generic defaults; improper family keeps the cure mass | `test_truncated_left_and_interval_rows_are_hand_derived`, `test_full_mixed_matrix_in_one_batch`, `test_mixed_kinds_and_mixed_windows_in_one_batch`, `test_custom_family_on_generic_defaults_matches_weibull`, `test_improper_family_with_right_truncation` (values) |
| 4.4 | PH-Weibull likelihood equals `Weibull(k, b0 exp(-eta/k))` on truncated exact/right rows; on the full matrix (+ censoring) | `test_ph_weibull_truncated_likelihood_equals_the_equivalent_weibull`, `test_ph_weibull_full_matrix_equals_the_equivalent_weibull` (values) |
| 2/3.3/7.1 | right at entry exactly zero and `uninformative` (finite/infinite `U`, `c=L=0` with finite `U`, `from_bounds(0, inf)` with `L=0`); host checkpoint passes | `test_right_censored_at_entry_is_exactly_zero_and_uninformative` (diagnostics) |
| 2/7.1 (+ censoring) | left at `c>=U`, intervals equal to or containing the window, interval-flag `(L, +inf]`: exactly zero, `uninformative`; zero also where `Z` is tiny | `test_censored_set_covering_the_window_is_exactly_zero_and_uninformative`, `test_covering_rows_are_zero_for_extreme_parameters` (diagnostics) |
| 2/3.3/6.2/7.1 | invalid: exact at/before entry and after `U` (`exact_outside_truncation`), right before entry (`censored_before_entry`), right at/after `U`, invalid windows (`invalid_truncation`: negative, infinite, NaN `L`; NaN `U`; `L>=U`), bounds `(0, inf]` after entry; NaN on exactly `invalid_rows`, triaged to data; under `jit` and traced construction; parameter issues on truncated rows; valid batch triages clean | `test_invalid_truncated_exact_or_right_row_is_nan`, `test_invalid_truncation_under_jit_and_traced_construction`, `test_parameter_issue_on_a_truncated_row_is_nan`, `test_valid_truncated_batch_has_no_nan_and_no_triage_mask` (diagnostics) |
| 2/6.2/7.1 (+ censoring) | empty intersections of left/interval rows (ending at/before `L`, starting at `U`, left at/before `L`); interval-flag `(c, +inf]` before entry is `censored_before_entry` | `test_invalid_truncated_censored_row_is_nan` (diagnostics) |
| 7.1/12.1 | `Z=0` is NaN with valid data and parameters, triaged as `other` (Q-064); zero numerator `-inf` vs zero denominator NaN in one bounded-support batch (+ censoring) | `test_zero_denominator_is_nan_and_triaged_as_other`, `test_bounded_support_separates_zero_numerator_from_zero_denominator` (diagnostics) |
| 3.2/6.2/7.2/12.1 | eager/`jit`/`vmap` agreement: prepared data, traced flag construction, traced bounds full matrix (+ censoring), pruning invariance for untruncated rows, `vmap` over draws and over datasets with different windows, indexing, compilation reuse (prepared and traced), `jit(vmap)` full matrix (+ censoring) | `test_prepared_truncated_data_under_jit_matches_hand_values`, `test_traced_construction_with_truncation_matches_eager`, `test_traced_bounds_construction_of_the_full_matrix_matches_eager`, `test_untruncated_rows_agree_with_and_without_truncation_metadata`, `test_vmap_over_draws_and_datasets`, `test_indexing_truncated_data_keeps_windows`, `test_new_truncation_values_reuse_compilation`, `test_traced_construction_with_truncation_reuses_compilation`, `test_full_matrix_under_jit_of_vmap` (transformations) |
| 7.1/7.3/12.1 | first and second derivatives with `log Z`: Exponential hand-derived per window (exact/right; full matrix + censoring), per-row windows Jacobian both modes; Weibull vs mpmath (exact/right; full matrix + censoring); four Hessian compositions (+ censoring); `jit`/`vmap` Hessians; `vmap` over datasets with different windows; `check_grads` (float64) per window, full-matrix per-row (+ censoring), Exponential and `from_log` under `jit`; isolation of rows without parameter issues (four violation classes, both modes, second order); uninformative row has zero derivatives | `test_exponential_exact_right_derivatives_are_hand_derived`, `test_exponential_full_matrix_derivatives_are_hand_derived`, `test_exponential_per_row_jacobian_with_per_row_windows`, `test_weibull_exact_right_hessian_matches_mpmath`, `test_weibull_full_matrix_hessian_matches_mpmath`, `test_forward_and_reverse_hessians_agree_with_mpmath`, `test_hessian_under_jit_and_vmap`, `test_gradient_under_vmap_over_datasets_with_different_windows`, `test_check_grads_*`, `test_rows_without_parameter_issues_keep_derivatives`, `test_uninformative_row_has_zero_derivatives` (derivatives) |
| 2/12.1 | properties (hypothesis): left truncation = untruncated minus `log S(L)`; Exponential memorylessness; right at entry exactly zero; two-sided conditioning for every kind incl. partial overlaps (+ censoring); left+right partition the window (+ censoring); covering sets exactly zero (+ censoring) | `test_left_truncation_conditions_on_survival_to_entry`, `test_exponential_left_truncation_is_memoryless`, `test_right_censoring_at_entry_is_exactly_zero`, `test_two_sided_truncation_conditions_on_the_window`, `test_left_and_right_at_the_same_time_partition_the_window`, `test_censored_sets_covering_the_window_are_exactly_zero` (properties) |
| 7.1/7.2/12.1/12.2 | tiny `Z` in log space (entry where `e**-L` underflows, Weibull `Z ~ e**-900`, tiny upper truncation with a large positive contribution), narrow two-sided windows, censored rows in a deep window (+ censoring); eager `jax_debug_nans` on truncated batches (Weibull, Exponential, custom) and on the full matrix (+ censoring) | `test_exponential_entry_deep_in_the_tail_is_evaluated_in_log_space`, `test_weibull_entry_deep_in_the_tail`, `test_tiny_upper_truncation_gives_a_large_positive_contribution`, `test_weibull_tiny_upper_truncation`, `test_narrow_two_sided_window`, `test_censored_rows_in_a_deep_window`, `test_*_under_debug_nans` (numerics) |
| 4.2/12.2 (+ censoring) | `log_conditional_interval_prob` hook (Q-066): spec definition on inside, partial, one-sided and untruncated sets (built-ins, generic default); untruncated = `log_interval_prob`; whole window = 0; equals censored row contributions; Weibull narrow deep window; broadcasting; `jit(grad)`; a correct custom override gives the Exponential likelihood | `test_hook_values_match_the_spec_definition`, `test_untruncated_hook_is_the_interval_probability`, `test_hook_on_the_whole_window_is_zero`, `test_hook_agrees_with_censored_row_contributions`, `test_weibull_hook_keeps_precision_for_a_narrow_deep_window`, `test_hook_broadcasts_against_the_batch`, `test_hook_gradient_under_jit`, `test_a_correct_override_gives_the_exponential_likelihood` (hook) |

Not tested for `observation.truncation`:

- **Which mask an empty intersection sets** (`empty_observation` is the
  likely one, but 2 says only "invalid data"): only `invalid_rows` is
  asserted. Exact-time rows outside the window are asserted as
  `exact_outside_truncation`, invalid windows as `invalid_truncation`.
- **Whether the likelihood routes rows through an overridden
  `log_conditional_interval_prob`** (4.2 says families shape the likelihood
  "only through the members above", but not which rows use which member):
  only a correct override is tested, by values (Q-066).
- **Static-metadata pruning and separate compilation for different
  truncation flags:** optional and permitted (7.2); only invariance and reuse
  for equal metadata are tested.
- **Truncation-conditioned sampling** (8.6): feature `sampling`.
- **R fixtures (flexsurv, every kind and truncation):** maintainer-supplied;
  none committed yet.
- **Derivatives in truncation endpoints:** outside 7.3's guarantee.
- **AFT wrapper under truncation:** Slice D.

### `family.lognormal`: `test_family_lognormal_*.py` (rev 55)

12.2 (rev 55) brings the log-normal family forward "under the full contract
of sections 4.1, 4.2, 4.5, 5 and 7", gated by the 12.1 rows for a built-in
family; its 8.1 analytical summaries come with the matching `summary.*`
feature. Tests carry `feature("family.lognormal")`; Slice B rows add
`observation.censoring` / `observation.truncation` ("+ censoring", "+
truncation"); summary tests add the summary feature ("+ mean", "+ quantile",
"+ rmst", and "+ containers" for `SummaryInfo`). References:
`family_reference` (new: 50-digit mpmath direct methods with a
condition-number tolerance, `log Phi` split as in 4.1), the log-normal branch
of `observation_reference`, and the log-normal closed forms of
`summary_reference` (`Phi⁻¹` by a well-scaled `findroot`, both tails); all
checked against independent mpmath quadrature before use.

| Spec | Requirement | Tests |
| --- | --- | --- |
| 4.1/4.2/10.2 | `survjax.LogNormal`, a `SurvivalDistribution`; keyword-only, both parameters required; leaves `meanlog`, `sdlog`; no `.shape` | `test_lognormal_is_exported_at_top_level`, `test_lognormal_is_a_survival_distribution`, `test_lognormal_rejects_positional_parameters`, `test_lognormal_requires_both_parameters`, `test_lognormal_stores_meanlog_and_sdlog_as_its_only_leaves`, `test_lognormal_has_no_shape_attribute` (construction) |
| 4.1/5 | batch broadcasting, incompatible batches and non-broadcasting `t` `TypeError`, `(D, N)` batches; no value checks (eager, `jit`); traced and `vmap` construction; indexing, `tree.map` slicing, `vmap` over the distribution | `test_scalar_lognormal_has_empty_batch_shape`, `test_lognormal_parameters_are_broadcast_at_construction`, `test_lognormal_incompatible_parameter_batches_are_rejected`, `test_lognormal_draw_by_individual_batch`, `test_time_that_does_not_broadcast_is_rejected`, `test_lognormal_constructor_accepts_invalid_values_*`, `test_lognormal_construction_*`, `test_lognormal_indexing_selects_batch_entries`, `test_lognormal_tree_map_slicing`, `test_vmap_over_lognormal_matches_batched_call` (construction) |
| 3.3/4.2 | parameter dtype (integer, weak typing, promotion, boolean `TypeError`); `reference_time()` shape, dtype, positive, finite, `jit` | `test_integer_parameters_use_default_float_dtype`, `test_python_scalar_adopts_float32_array_dtype`, `test_mixed_float_parameters_share_promoted_dtype`, `test_boolean_parameter_is_rejected`, `test_lognormal_reference_time_*` (construction) |
| 4.1 | `from_log` equivalents: class, natural leaves, values, overflow reported, chain-rule gradients (Q-069) | `test_lognormal_from_log_*` (construction), `test_from_log_overflow_is_reported_as_parameter_issue` (constraints), `test_from_log_gradient_follows_the_chain_rule` (derivatives) |
| 4.1/4.7/12.1 | `meanlog` `real`, `sdlog` `positive`; masks, union, `ok`, shapes, `jit`/`vmap`, indexing, host checkpoint; violations leave other entries' direct values unchanged | `test_lognormal_declares_meanlog_and_sdlog`, `test_meanlog_declaration_is_real`, `test_sdlog_declaration_is_positive`, `test_parameter_issues_*`, `test_valid_lognormal_reports_no_issue`, `test_indexed_parameter_issues_equal_the_parents_indexed`, `test_host_checkpoint_raises_on_a_violation`, `test_declared_violation_leaves_other_entries_direct_values_unchanged` (constraints) |
| 2/4.1/8.1 | all seven direct methods: hand values at `t = exp(mu)`; interior, paired and scalar-time batches against mpmath | `test_values_at_the_median_time_are_hand_derived`, `test_interior_values_match_mpmath`, `test_paired_batch_values_match_mpmath`, `test_scalar_time_against_a_batch` (closed_forms) |
| 4.1/4.2/12.1 | "log-normal tails against mpmath over `x` in [0,10] (float64) and [0,7] (float32)": `log S` (`log1p(-ndtr(-x))`) and, "symmetrically", `log F`, relative, zero absolute tolerance; non-unit parameters; upper tail to `x = -37` (`log S`, `H`, `log f`, `log h`, `h`); lower tail of `log F`, `log f`, `log h`; `S` underflows to `0` | `test_log_survival_keeps_relative_precision_for_positive_x`, `test_log_cdf_keeps_relative_precision_symmetrically`, `test_tails_with_non_unit_parameters`, `test_upper_tail_where_survival_is_tiny`, `test_lower_tail_where_the_cdf_is_tiny`, `test_survival_underflows_to_zero_not_nan_in_upper_tail` (closed_forms) |
| 4.2/12.1 | identities: `logaddexp(log S, log F) = 0`; `h = -d log S/dt`, `log_prob = log h + log S` | `test_log_survival_and_log_cdf_partition_probability`, `test_density_and_hazard_agree_with_the_derivative_of_log_survival` (closed_forms) |
| 4.5 | `t<0`, `t=0` (base values; `log f(0) = log h(0) = -inf`), `t=+inf` (`log h = -inf` from the override), NaN `t`, NaN parameters, mixed vectors; zero gradients of base-supplied values (both modes) and `t<0` rows; proper `log_survival_at_infinity()` | `test_negative_time_gives_policy_value`, `test_values_at_zero`, `test_values_at_infinity`, `test_nan_time_gives_nan`, `test_nan_parameters_give_nan_for_every_time`, `test_mixed_vector_of_boundary_and_interior_times`, `test_base_supplied_values_have_zero_parameter_gradients`, `test_negative_time_family_rows_have_zero_parameter_gradients`, `test_log_survival_at_infinity_is_the_proper_default` (boundaries) |
| 4.5/12.1 | `t` dtype rule (eager, traced), float64 conversion without warning; `jit`/`vmap` agreement; eager `jax_debug_nans` on boundary, tail and interior times | `test_time_with_a_non_numeric_dtype_is_rejected`, `test_boolean_time_is_rejected_during_tracing`, `test_float64_time_converts_without_warning_in_32_bit_mode`, `test_direct_methods_agree_under_jit_and_vmap`, `test_direct_methods_under_debug_nans` (boundaries) |
| 5/12.1 | every `_grid` twin equals `m(t[:, None, None])`, `(K, D, N)`; `K = N`; from `t = 0` against mpmath; unsorted and repeated `t`; non-1-D and boolean `t` `TypeError`; `jit` | `test_grid_equals_direct_method_with_explicit_axes`, `test_grid_with_as_many_times_as_individuals`, `test_survival_grid_matches_mpmath_and_starts_at_one`, `test_grid_times_need_not_be_sorted_or_distinct`, `test_grid_time_that_is_not_one_dimensional_is_rejected`, `test_boolean_grid_time_is_rejected`, `test_grid_under_jit_matches_eager` (grids) |
| 2/5/7.1 | exact/right rows against mpmath and by hand; `c = 0` exactly zero; raw form = `SurvivalData` form; `(D, 1)` draws; per-row pairing; tails (right to `x = -30`, exact rows at `x = ±10, ±20`, tiny right rows) | `test_exact_and_right_rows_match_mpmath`, `test_exact_row_at_the_median_time_is_hand_derived`, `test_right_censored_at_zero_is_exactly_zero`, `test_raw_form_equals_survival_data_form`, `test_draw_column_against_observations`, `test_per_row_parameters_pair_with_rows`, `test_right_row_deep_in_the_upper_tail`, `test_exact_row_in_both_tails`, `test_right_row_near_entry_keeps_relative_precision` (likelihood) |
| 6.2/7.1 | invalid data and parameter-issue rows NaN (zero, negative, `inf` `sdlog`; NaN, `inf` `meanlog`), attributed by the recipe; valid rows unchanged; clean triage | `test_invalid_data_rows_are_nan_and_attributed_to_data`, `test_parameter_issue_rows_are_nan_and_attributed_to_parameters`, `test_valid_batch_has_no_nan_and_no_triage_mask` (likelihood) |
| 7.2/7.3/12.1 | `jit` (prepared and traced construction), `vmap` over draws, eager `jax_debug_nans` on a valid mixed batch with gradient | `test_likelihood_agrees_under_jit`, `test_likelihood_under_vmap_over_parameter_draws`, `test_likelihood_under_debug_nans` (likelihood) |
| 4.4 | PH over `LogNormal` (no closure) keeps the PH formulas | `test_ph_wrapper_of_lognormal_keeps_the_ph_formulas` (likelihood) |
| 7.1/12.2 | until Slice B: left/interval rows and truncated rows NaN, reported unsupported (`until("observation.censoring")`, `until("observation.truncation")`) | `test_left_and_interval_rows_are_nan_until_censoring_is_enabled`, `test_truncated_rows_are_nan_until_truncation_is_enabled` (likelihood) |
| 2/7.1/12.1 (+ censoring, + truncation) | left and interval rows against mpmath; `log F` tiny and near one; interval additivity; truncated exact/right rows (left, right, two-sided); partial-overlap interval (+ both) | `test_left_and_interval_rows_match_mpmath`, `test_left_row_where_the_cdf_is_tiny`, `test_left_row_where_the_cdf_is_near_one`, `test_interval_rows_are_bounded_by_their_end_points`, `test_truncated_exact_and_right_rows_match_mpmath`, `test_truncated_interval_row_with_partial_overlap` (likelihood) |
| 7.3/12.1 | direct-method gradients (`log S`, `log F`, `log f`, `log h`) and Hessians vs mpmath, both modes; `check_grads(order=2)` (float64) on all seven methods | `test_direct_method_gradients_match_mpmath`, `test_direct_method_hessians_match_mpmath`, `test_check_grads_order_two_direct_methods` (derivatives) |
| 7.3/12.1 | likelihood gradient (both modes) and Hessian (four compositions) vs mpmath; per-row diagonal Hessian; `jit`/`vmap`; `check_grads` shared, per-row, `jit`; isolation of rows without parameter issues (zero, negative, NaN, `inf` `sdlog`) to second order, both modes | `test_likelihood_gradient_matches_mpmath`, `test_likelihood_hessian_matches_mpmath`, `test_per_row_parameters_give_a_block_diagonal_hessian`, `test_likelihood_gradient_agrees_under_jit_and_vmap`, `test_check_grads_order_two_likelihood`, `test_rows_without_parameter_issues_keep_their_derivatives` (derivatives) |
| 7.3 (+ censoring, + truncation) | left/interval gradients and Hessian; truncated gradients with `log Z` | `test_left_and_interval_gradients_match_mpmath`, `test_left_and_interval_hessian_matches_mpmath`, `test_truncated_row_gradients_include_log_z` (derivatives) |
| 8.1/8.4/8.7 (+ mean) | mean `exp(mu + sigma²/2)`, batched; hand derivatives; violation NaN | `test_mean_closed_form`, `test_mean_is_vectorized_over_the_batch`, `test_mean_derivatives_are_hand_derived`, `test_parameter_violation_gives_nan_mean` (summaries) |
| 8.1/8.5 rev 56 (+ mean, + containers) | analytical mean: `converged`, NaN error estimate, zero counts; violation `invalid_input` | `test_mean_with_info_is_an_analytical_answer`, `test_parameter_violation_mean_reports_invalid_input` (summaries) |
| 8.1/8.2/8.4 (+ quantile) | quantile `exp(mu + sigma Φ⁻¹(p))`, median `exp(mu)`; `H⁻¹` in both regimes (`e` from `1e-30` to 80, 600 in float64); `H(H⁻¹(e)) = e`; endpoints; invalid `p`, `e`; hand derivatives of quantile and `H⁻¹`; endpoint NaN tangents; violations NaN; `converged` (+ containers) | `test_quantile_closed_form`, `test_median_is_exp_meanlog`, `test_inverse_cumulative_hazard_closed_form_in_both_regimes`, `test_cumulative_hazard_inverts_the_inverse_cumulative_hazard`, `test_quantile_and_hazard_level_endpoints`, `test_invalid_probability_and_hazard_level_give_nan`, `test_quantile_derivatives_are_hand_derived`, `test_inverse_cumulative_hazard_derivatives`, `test_endpoint_quantiles_have_nan_tangents`, `test_parameter_violation_gives_nan_quantile`, `test_closed_form_quantile_reports_converged` (summaries) |
| 8.1/8.4/8.7 (+ rmst) | RMST closed form; large cutoff below and near the mean; `tau` policy and violations; `d/dtau = S(tau)` and parameter derivatives vs mpmath; `tau_grid` axes; `converged` (+ containers) | `test_rmst_closed_form`, `test_rmst_at_a_large_cutoff_approaches_the_mean_from_below`, `test_rmst_value_policy`, `test_rmst_derivatives`, `test_rmst_cutoff_grid_with_explicit_axes`, `test_closed_form_rmst_reports_converged` (summaries) |

Not tested for `family.lognormal`:

- **`check_family(LogNormal(...))`:** 12.1 asks for it on every built-in, but
  `check_family` "arrives with Slice D and must then pass on both families"
  (12.2 rev 57; Q-070 settled); its identities are tested directly.
- **The declaration order of `meanlog` and `sdlog` in `parameter_constraints`:**
  not fixed by the spec; only the key set is asserted.
- **The value of a violating entry's own direct methods** (for example
  `sdlog = -1`): "whatever the family's formula gives"; only the other
  entries are asserted.
- **Error estimate and counts of the closed-form quantile and RMST:** rev 56
  states the "analytical answer" `SummaryInfo` for `mean_with_info`; for the
  other summaries only `converged` is asserted.
- **Sampling** (8.6 closed-form draws): feature `sampling`, not requested.
- **AFT over a log-normal baseline:** Slice D.
- **Committed R fixtures for the family** (12.1 "Independent references"):
  maintainer-supplied, none committed.

### `family.loglogistic`: `test_family_loglogistic_*.py` (rev 55)

Same structure, markers and references as `family.lognormal`
(`family_reference` log-logistic branch: `log S = -softplus(u)`, `log F =
-softplus(-u)`; log-logistic branch of `observation_reference`; quantile,
`H⁻¹` and quadrature RMST and RMST-derivative references in
`summary_reference`), all checked against independent mpmath quadrature
before use.

| Spec | Requirement | Tests |
| --- | --- | --- |
| 4.1/4.2/10.2 | `survjax.LogLogistic`, a `SurvivalDistribution`; keyword-only, both parameters required; leaves `shape`, `scale`; `.shape` is the parameter, `batch_shape` the array shape | `test_loglogistic_is_exported_at_top_level`, `test_loglogistic_is_a_survival_distribution`, `test_loglogistic_rejects_positional_parameters`, `test_loglogistic_requires_both_parameters`, `test_loglogistic_stores_shape_and_scale_as_its_only_leaves`, `test_shape_attribute_is_the_parameter_not_the_batch_shape` (construction) |
| 3.3/4.1/4.2/5 | broadcasting, `TypeError` for incompatible batches and `t`, `(D, N)`; no value checks; traced/`vmap` construction; indexing, slicing, `vmap`; dtype rules; `reference_time()`; `from_log` (Q-069) | `test_scalar_loglogistic_has_empty_batch_shape`, `test_loglogistic_parameters_are_broadcast_at_construction`, `test_loglogistic_incompatible_parameter_batches_are_rejected`, `test_loglogistic_posterior_style_construction_gives_draw_by_individual_batch`, `test_time_that_does_not_broadcast_is_rejected`, `test_loglogistic_constructor_accepts_invalid_values_*`, `test_loglogistic_construction_*`, `test_loglogistic_indexing_selects_batch_entries`, `test_loglogistic_tree_map_slicing`, `test_vmap_over_loglogistic_matches_batched_call`, `test_integer_parameters_use_default_float_dtype`, `test_python_scalar_adopts_float32_array_dtype`, `test_mixed_float_parameters_share_promoted_dtype`, `test_boolean_parameter_is_rejected`, `test_loglogistic_reference_time_*`, `test_loglogistic_from_log_*` (construction) |
| 4.1/4.7/12.1 | `shape`, `scale` `positive`; masks, union, `ok`; `beta <= 1` valid; shapes, `jit`/`vmap`, indexing, host checkpoint; other entries unchanged; `from_log` overflow | `test_loglogistic_declares_shape_and_scale`, `test_both_declarations_are_positive`, `test_parameter_issues_mark_each_violated_field`, `test_heavy_tailed_shapes_are_valid_parameters`, `test_parameter_issues_*`, `test_indexed_parameter_issues_equal_the_parents_indexed`, `test_host_checkpoint_raises_on_a_violation`, `test_declared_violation_leaves_other_entries_direct_values_unchanged`, `test_from_log_overflow_is_reported_as_parameter_issue` (constraints) |
| 2/4.1/8.1 | hand values at `t = alpha` and for `beta = 1`; all methods at interior times for `beta` 0.5, 1, 3; paired batches | `test_values_at_the_scale_are_hand_derived`, `test_unit_shape_is_hand_derived`, `test_interior_values_match_mpmath`, `test_paired_batch_values_match_mpmath` (closed_forms) |
| 4.1/4.2/12.1 | "both accurate in both tails": `log S`, `log F`, `H`, `log f`, `log h`, `h` for `|u|` to 700 (float64) / 80 (float32), relative, zero absolute tolerance; other shapes; `S` underflows to 0; `logaddexp(log S, log F) = 0`; `h = -d log S/dt` | `test_log_survival_is_accurate_in_both_tails`, `test_log_cdf_is_accurate_in_both_tails`, `test_other_methods_in_both_tails`, `test_tails_with_other_shapes`, `test_survival_underflows_to_zero_not_nan_in_upper_tail`, `test_log_survival_and_log_cdf_partition_probability`, `test_density_and_hazard_agree_with_the_derivative_of_log_survival` (closed_forms) |
| 4.5 | `t<0`, NaN `t`, NaN parameters; `t = 0` limits by shape (`log f(0) = log h(0)`: `+inf`, `-log alpha`, `-inf`); `t = +inf` (`log h = -inf`); mixed vectors; zero gradients of base values (both modes, incl. the plain-softplus `S(0)` case); finite family limits with the value's derivative (`beta = 1` in `scale`; `h(0) = 0` at `beta = 2`; `h(+inf) = 0`); proper `log_survival_at_infinity()` | `test_negative_time_gives_policy_value`, `test_values_at_zero`, `test_values_at_infinity`, `test_nan_time_gives_nan`, `test_nan_parameters_give_nan_for_every_time`, `test_mixed_vector_of_boundary_and_interior_times`, `test_base_supplied_values_have_zero_parameter_gradients`, `test_finite_density_and_hazard_limits_at_zero_have_the_scale_derivative`, `test_zero_hazard_limits_have_zero_parameter_gradients`, `test_negative_time_family_rows_have_zero_parameter_gradients`, `test_log_survival_at_infinity_is_the_proper_default` (boundaries) |
| 4.5/12.1 | `t` dtype rule (eager, traced), float64 conversion; `jit`/`vmap`; eager `jax_debug_nans` for `beta` 0.5, 1, 3 | `test_time_with_a_non_numeric_dtype_is_rejected`, `test_boolean_time_is_rejected_during_tracing`, `test_float64_time_converts_without_warning_in_32_bit_mode`, `test_direct_methods_agree_under_jit_and_vmap`, `test_direct_methods_under_debug_nans` (boundaries) |
| 5/12.1 | `_grid` twins, `K = N`, from `t = 0`, non-1-D `TypeError`, unsorted/repeated, `jit` | `test_grid_*`, `test_survival_grid_matches_mpmath_and_starts_at_one` (grids) |
| 2/5/7.1/7.2/7.3/12.1 | exact/right rows (`beta` 0.5, 1, 1.6, 5) and hand values; `c = 0`; raw form; `(D, 1)`; per-row; both tails; data and parameter-issue NaN rows triaged; clean batches; `jit`, `vmap`, `jax_debug_nans`; PH over `LogLogistic` keeps the PH formulas; until Slice B left/interval/truncated rows NaN | `test_exact_and_right_rows_match_mpmath`, `test_rows_at_the_scale_are_hand_derived`, `test_right_censored_at_zero_is_exactly_zero`, `test_raw_form_equals_survival_data_form`, `test_draw_column_against_observations`, `test_per_row_parameters_pair_with_rows`, `test_rows_in_both_tails`, `test_invalid_data_rows_are_nan_and_attributed_to_data`, `test_parameter_issue_rows_are_nan_and_attributed_to_parameters`, `test_valid_batch_has_no_nan_and_no_triage_mask`, `test_likelihood_agrees_under_jit`, `test_likelihood_under_vmap_over_parameter_draws`, `test_likelihood_under_debug_nans`, `test_ph_wrapper_of_loglogistic_keeps_the_ph_formulas`, `test_left_and_interval_rows_are_nan_until_censoring_is_enabled`, `test_truncated_rows_are_nan_until_truncation_is_enabled` (likelihood) |
| 2/7.1/12.1 (+ censoring, + truncation) | left/interval rows; left rows in both tails; additivity; truncated exact/right rows; partial-overlap interval (+ both) | `test_left_and_interval_rows_match_mpmath`, `test_left_rows_in_both_tails`, `test_interval_rows_are_additive`, `test_truncated_exact_and_right_rows_match_mpmath`, `test_truncated_interval_row_with_partial_overlap` (likelihood) |
| 7.3/12.1 | direct-method gradients and Hessians vs mpmath; `check_grads` (float64); likelihood gradients (`beta` 0.7, 1.6) and four Hessian compositions; per-row diagonal Hessian; `jit`/`vmap`; `check_grads` shared, per-row, `jit`; `from_log` chain rule; isolation (zero, negative, NaN, `inf` `scale`) to second order; censored and truncated gradients/Hessian (+ censoring, + truncation) | `test_direct_method_gradients_match_mpmath`, `test_direct_method_hessians_match_mpmath`, `test_check_grads_order_two_direct_methods`, `test_likelihood_gradient_matches_mpmath`, `test_likelihood_hessian_matches_mpmath`, `test_per_row_parameters_give_a_diagonal_hessian`, `test_likelihood_gradient_agrees_under_jit_and_vmap`, `test_check_grads_order_two_likelihood`, `test_from_log_gradient_follows_the_chain_rule`, `test_rows_without_parameter_issues_keep_their_derivatives`, `test_left_and_interval_gradients_match_mpmath`, `test_left_and_interval_hessian_matches_mpmath`, `test_truncated_row_gradients_include_log_z` (derivatives) |
| 8.1/8.4 (+ mean) | `alpha*(π/beta)/sin(π/beta)` for `beta > 1` (incl. 1.05); hand derivatives; "known infinite for `beta<=1`": `+inf` (0.3, 0.5, 1), per entry; NaN tangents in `scale` and `shape` ("built-ins provide them", rev 56), finite tangent of a finite neighbour; rev 57 isolation of the finite entry through a shared `scale` (`beta = [3, 0.8]`), forward and reverse; violations NaN | `test_mean_closed_form_for_shape_above_one`, `test_mean_derivatives_are_hand_derived`, `test_mean_is_plus_infinity_for_shape_at_most_one`, `test_known_infinite_mean_is_decided_per_entry`, `test_known_infinite_mean_has_nan_tangents`, `test_finite_entries_keep_finite_tangents_beside_known_infinite_ones`, `test_finite_mean_keeps_its_scale_derivative_beside_a_known_infinite_entry_with_shared_scale`, `test_parameter_violation_gives_nan_mean` (summaries) |
| 8.1/8.5 rev 56 (+ mean, + containers) | `known_infinite`, NaN error estimate, zero counts; finite closed form `converged` with no work; per-entry statuses; violations `invalid_input` | `test_known_infinite_mean_reports_known_infinite_and_no_work`, `test_finite_mean_is_an_analytical_answer`, `test_mean_statuses_are_per_entry`, `test_parameter_violation_mean_reports_invalid_input` (summaries) |
| 8.1/8.2/8.4 (+ quantile) | quantile closed form (`beta` 0.5, 1, 3), median `alpha`; `H⁻¹ = alpha expm1(e)**(1/beta)` (`e` 1e-30 to 80, 700 float64); `H(H⁻¹(e)) = e`; endpoints and invalid inputs; hand derivatives (`dq/dp = 1/f(q)`, `dt/de = 1/h(t)`); violations; `converged` (+ containers) | `test_quantile_closed_form`, `test_median_is_the_scale`, `test_inverse_cumulative_hazard_closed_form`, `test_cumulative_hazard_inverts_the_inverse_cumulative_hazard`, `test_quantile_and_hazard_level_endpoints_and_invalid_inputs`, `test_quantile_derivatives_are_hand_derived`, `test_inverse_cumulative_hazard_derivative_in_the_hazard_level`, `test_parameter_violation_gives_nan_quantile`, `test_closed_form_quantile_reports_converged` (summaries) |
| 8.1/8.3/8.4 (+ rmst) | "generic quadrature" against mpmath at `10*rtol`, heavy tails (`beta` 0.5, 1.05) to `tau = 1e4`; `tau` policy, violations; quadrature tangents in `shape`, `scale` and `d/dtau = S(tau)`; `converged` with 2 to `max_levels` levels (+ containers) | `test_rmst_matches_mpmath_quadrature`, `test_rmst_value_policy`, `test_rmst_derivatives_match_quadrature`, `test_rmst_reports_converged_quadrature` (summaries) |

Not tested for `family.loglogistic`: `check_family` (Slice D, 12.2 rev 57;
Q-070 settled); the declaration order; a violating entry's own values; error
estimate and counts of the closed-form quantile; the known-infinite entries'
own reverse-mode derivatives (unspecified, 8.4 rev 57); sampling; AFT;
R fixtures. The custom-family log-logistic mean tests in
`test_summary_mean_values.py` are kept: they test the `mean_with_info`
override contract and the generic heavy-tail path, which the built-in does
not exercise.

### Post-release features: deliberately not tested

- **PH wrappers of a family with a `mean_with_info` override:** not stated
  whether the wrapper uses the override (Q-067).
- **`check_family` checking an override's NaN tangents (8.1 rev 56):**
  `check_family` is Slice D and in no feature of the role table.
- **Higher-order RMST derivatives at `tau=0`:** "carry no guarantee" (8.4,
  rev 51).
- **"All subintervals of a broadcast entry share the same refinement round"
  beyond counts, and the arithmetic-failure NaN error estimate (8.3/8.5,
  rev 52):** only observable through `num_levels` (tested); no public family
  in reach produces an arithmetic failure distinct from a non-finite
  integrand.
- **The exact `num_levels` of a converged default-tolerance case:** depends
  on the node layout, which the spec leaves open; only `[2, max_levels]` is
  asserted.
- **`sample`:** feature `sampling`, not requested yet; its absence stays
  covered by `test_absent_first_release_methods` until `sampling` is enabled.
- **`HazardFamily`, AFT:** Slice D (8.8: "Enabling summaries and sampling
  does not bring forward any other Slice D feature"). Rev 55 brings the
  log-normal and log-logistic families forward separately
  (`family.lognormal`, `family.loglogistic`, above).
- **Executable documentation of convergence limitations:** documentation gate.

## Revision 37 (first public release): remaining items

Revision 37's `ProportionalHazards` and staged RMST requirements are covered
by A11 and A12 (Q-049). Nothing is tested for:

- **Release demos (11.2, 11.3) and the simulation study:** end-to-end
  evidence owned by the use-case design notes, not spec tests.
