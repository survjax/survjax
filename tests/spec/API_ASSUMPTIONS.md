# API assumptions

Entries Q-001, Q-002, ... (see CLAUDE.md rule 4). Each records the spec sections
checked, the guess made (or the question left open), and the tests that depend
on it. Guessed exception types and call shapes are centralised in
`helpers.py`, so resolving an entry usually means editing one line there.

## Q-001: delivery step numbering A1/A2/A3

- **Checked:** 12.2 and the maintainer's delivery schedule in AGENTS.md.
- **Resolved:** the maintainer fixes A1 to representation/dtype, A2 to
  constructors/boundaries, and A3 to indexing, diagnostics and metadata.
- **Depends:** every `step` marker and `SPEC_THROUGH` filtering.
## Q-002: normalized fields are attributes

- **Checked:** 3.2, 6.3.
- **Guess:** the five named public fields are array attributes. The issue
  bitfield has no public field name; tests inspect its PyTree leaf and decode
  it only through the specified `data.issues` property.
- **Depends:** `helpers.field`, `helpers.assert_fields`, and the field checks
  throughout A1–A3.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.
## Q-003: exception types and host checkpoint

- **Checked:** 3.3 and 6.2, revisions 8–9.
- **Resolved:** invalid observation values return data with issues; only an
  explicit concrete `data.issues.raise_if_invalid()` raises the public
  `survjax.InvalidDataError` (`ValueError` subclass). Structural argument
  misuse raises `TypeError`; unavailable explicit dtype requests raise
  `ValueError` and not `InvalidDataError`.
- **Depends:** A2 issue assertions, A3 checkpoint tests, misuse tests and
  `helpers.raises_dtype_request`.
## Q-004: "one warning per call site" for float64 under 32-bit mode

- **Checked:** 3.3.
- **Guess:** the warning category is unconstrained; "call site" means the
  warning is attributed (via `stacklevel`) to the user's line that called the
  constructor, and repeated calls from that line warn once under Python's
  default warning filter. A single call warns once even when several fields
  arrive as float64. Tests count only warnings whose filename is the test file.
- **Depends:** `test_float64_input_warns_once_per_call`,
  `test_pandas_float64_series_converts_to_float32_with_one_warning`,
  `test_repeated_calls_from_one_call_site_warn_once`,
  `test_two_call_sites_warn_once_each`, and for distribution parameters
  (spec 3.3 bullet 4) `test_float64_parameters_warn_once_per_call`.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.

## Q-005: immutability error

- **Checked:** 3.2 ("immutable registered PyTree").
- **Guess:** assigning an attribute raises `AttributeError` (as a frozen
  dataclass does; `helpers.IMMUTABLE`).
- **Depends:** `test_data_is_immutable`.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.

## Q-006: `time2` on rows that are not interval censored

- **Status:** resolved by spec rev 7 (3.1) (maintainer
  decision): "`time2` is read only on interval rows; its values on other rows
  are ignored and set no issue." The rev 4 guess (NaN accepted and ignored) is
  now a special case of spec text.
- **Depends:** now spec-backed: `test_time2_values_on_non_interval_rows_are_ignored`,
  `test_time2_values_on_non_interval_rows_set_no_issue`, and the tests that put
  NaN in `time2` on non-interval rows (`_mixed()` in
  `test_a1_representation.py` and others).

## Q-007: withdrawn

- **Status:** withdrawn by spec rev 7. The feature this entry concerned was
  removed from the specification and the library by maintainer decision; it
  has no special handling and no tests. The number is kept so that later
  entries keep their numbers.

## Q-008: right censoring at or before entry when the set is `(0, +inf]` (spec inconsistency)

- **Status:** resolved by spec rev 5 (sections 2, 3.3, 6.2; D-003). `c<L` is
  invalid (`censored_before_entry`) for any censored row with `upper=+inf`,
  however constructed; `c=L` is valid and `uninformative`, including `c=L=0`
  and finite right truncation; `from_bounds(0, inf)` is uninformative when
  `L=0` and invalid when `L>0`.
- **Previously (rev 4):** section 2 made `c<=L` invalid, while 3.3 made right
  censoring at 0 with no truncation valid and uninformative.
- **Depends:** now tested directly: the censored-before-entry block in
  `test_a2_boundaries.py`, and the `censored_before_entry` and `uninformative`
  blocks in `test_a3_data_issues.py`. `uninformative` on `from_bounds` rows is
  still only observable as "not rejected" (Q-009).

## Q-009: issue-returning convenience constructors

- **Checked:** 6.2, 10.2 and 12.1, revision 9.
- **Resolved:** only `SurvivalData(...)` and `SurvivalData.from_bounds(...)`
  construct observations. Each returns data; `data.issues` exposes diagnostics.
  `helpers.flag_data_and_issues` and `helpers.bounds_data_and_issues` are test
  helpers that compose these public operations, not survjax API assumptions.
- **Depends:** the A3 diagnostic attribution tests and the two tests asserting
  that the removed convenience methods are absent.
## Q-010: which DataIssues field reports each rule

- **Checked:** 2, 3.1, 3.3, 6.2 (rev 5, including D-005 in 20f9359). The spec
  now fixes `invalid_time` for any 3.3 violation by the flag constructor's
  `time` on any row (including an interval's lower endpoint) and by
  `from_bounds` `lower`;
  `censored_before_entry` for `upper=+inf` and `lower<L`; `uninformative` for
  full coverage; and `missing_time2`, `invalid_time2`, `invalid_interval` for
  upper endpoints, with presence and domain checked before interval order.
- **Still guessed (mapping by field name):**
  - `invalid_indicator`: an indicator not exactly 0/1 (including NaN);
  - `conflicting_indicators`: more than one true flag on a row;
  - `invalid_truncation`: `L` negative, infinite or NaN; `U` NaN; `L >= U`;
  - `exact_outside_truncation`: exact `t` outside `L<t<=U`;
  - `empty_observation`: a censored set whose intersection with `(L, U]` is
    empty (right at `c>=U`, left at `c<=L`, interval `b<=L` or `a>=U`).
  Tests assert the named mask is set on the offending row. Other masks are
  asserted clear only where rev 5 fixes the ordering (a malformed lower or
  upper endpoint is not `invalid_interval`; `c=L` is not
  `censored_before_entry`). A NaN `time` or `from_bounds` `lower` sets
  `invalid_time` (settled in b682059).
- **Depends:** the tests in `test_a3_data_issues.py` whose docstrings cite
  Q-010, plus `test_distinct_rules_on_distinct_rows_decode_independently` and
  `test_two_rules_on_one_row_are_both_decoded`.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.

## Q-011: meaning of `ok`

- **Checked:** 6.2, revision 9.
- **Resolved:** `ok` is exactly `~jnp.any(invalid_rows)`, including when
  `uninformative` is present.
- **Depends:** the `ok` tests and diagnostic helpers in A3.
## Q-012: traced ordinary construction and checkpoint misuse

- **Checked:** 3.2, 6.2, 10.2 and 13, revision 10.
- **Resolved:** both ordinary constructors accept traced observation inputs.
  Calling `raise_if_invalid()` on traced diagnostics raises a plain
  `ValueError`, not `InvalidDataError`. `TracedInputError` is retired and not
  exported (revision 10 resolves D-006).
- **Depends:** `test_traced_diagnostics_cannot_raise_at_host_checkpoint`,
  `test_traced_invalid_diagnostics_misuse_is_not_invalid_data_error`,
  `test_traced_input_error_is_retired`.
## Q-013: decoding the bitfield of an existing SurvivalData

- **Checked:** 3.2, 6.2 and 13, revision 10.
- **Resolved:** the `data.issues` property decodes diagnostics on constructed,
  indexed, rebuilt and transformed data. Revision 10 rules out a callable
  `data.issues()` alias; `test_issues_has_no_callable_alias` checks that
  calling it raises `TypeError` (Python's error for calling a non-callable).
- **Depends:** `test_a3_issue_decoding.py` and the A2 issue checks.
## Q-014: observing static metadata

- **Status:** resolved by spec rev 7 (3.2) (maintainer
  decision): "The auxiliary data holds only these flags; datasets with equal
  flags have equal tree structures", and left and right truncation are
  separate flags (any `left_trunc > 0`; any `right_trunc < +inf`).
- **Still true:** the flags have no public accessor, so tests observe them
  through `jax.tree_util.tree_structure` and jit retrace counts; that is now
  spec-backed rather than guessed.
- **Depends:** every test in `test_a3_static_metadata.py`,
  `test_data_passes_through_jit_unchanged`,
  `test_indexing_carries_every_leaf_including_issue_bitfield`.

## Q-015: "passing its default value" for truncation arguments

- **Checked:** 3.1 (signatures: `left_trunc=None, right_trunc=None`), 3.2
  (`left_trunc` "`0` when absent", `right_trunc` "`+inf` when absent"), 3.3
  (amended in 91f5f62).
- **Resolved (maintainer, 2026-09-25):** the default value is the constructor
  default `None`. `0` and `+inf` are the values the constructed object stores.
- **Remaining note:** read this way, the sentence only says that
  `left_trunc=None` behaves like omitting the argument, which Python already
  guarantees. Passing Python `0` or `inf` explicitly gives the omitted dtype
  because of the weak-typing sentence, not this one. Passing a strongly typed
  zero (for example `np.zeros(n)` float64 next to float32 time under 64-bit
  mode) promotes to float64 under JAX promotion, which differs from omitting
  the argument. The spec allows this, and no test asserts otherwise.
- **Depends:** `test_passing_none_truncation_matches_omitting_it`. The Python
  `0`/`inf` tests now cite the weak-typing sentence.

## Q-016: pandas input accepted directly

- **Checked:** 3.1, 3.3 ("Array input with dtype float64 (NumPy, pandas,
  JAX)").
- **Resolved (maintainer, 2026-09-25):** a `pandas.Series` is accepted
  directly. The warning attribution still follows Q-004.
- **Depends:** `test_pandas_float64_series_converts_to_float32_with_one_warning`.

## Q-017: numeric rendering of host-checkpoint counts and indices

- **Checked:** 6.2 and 12.1, revision 9.
- **Guess:** the required count and first offending index appear as decimal
  integers in `str(InvalidDataError)`. No words or punctuation are asserted.
- **Depends:** `test_host_checkpoint_reports_count_and_first_offending_index`.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.

## Q-018: observing host callbacks and transfers (spec 7.2)

- **Checked:** 7.2, 12.3 (performance amendment, accepted in rev 11).
- **Guess:** "no host callbacks" is observable as no primitive whose name
  contains `callback` in the jaxpr of compiled construction plus `data.issues`
  (covers `pure_callback`, `io_callback`, `debug_callback`); "no host
  transfers" is observable as a warmed jitted call on device-resident inputs
  succeeding under `jax.transfer_guard("disallow")`. Both rely on JAX
  behaviour rather than a survjax name.
- **Status of the requirement:** accepted in spec rev 11 with the wording the
  tests rely on; the tests are tagged `rev 11`. The observation method above
  remains a guess.
- **Depends:** `test_compiled_construction_and_diagnostics_install_no_host_callback`,
  `test_warm_compiled_construction_and_diagnostics_make_no_host_transfer`,
  `test_compiled_construction_reuses_compilation_for_new_observation_values`.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.

## Q-019: does the base enforce spec 4.5 for custom families?

- **Status:** resolved by spec rev 12 (4.2, 4.5; maintainer decision,
  2026-10-02): the base applies the value policy for every registered family,
  including overridden methods. Family code is called only with `t` in
  `[0, +inf]`; for `t<0` and NaN `t` the base substitutes a safe input under
  `stop_gradient`; at `0` and `+inf` the family supplies the values, except
  that `log_prob(+inf)` is always `-inf`.
- **Rev 26 (decision 0040):** the base now also supplies `log S`, `S`,
  `log F` and `H` at `t=0` and `t=+inf` (the latter from
  `log_survival_at_infinity()`, default `-inf`), and `log_survival`,
  `log_cdf` and `cumulative_hazard` are called only for finite `t>0`. Only
  `log_prob` and `log_hazard` overrides are still called at the boundaries.
  This needed no new guess; see `test_a4_base_boundaries.py`.
- **Previously (rev 11):** the `log_survival` contract covered only the limits
  at `0` and `+inf`, so it was open whether `t<0` and NaN were the family's or
  the base's duty, and the literal default formulas gave NaN at `+inf`
  (Weibull-like `k>1`) and a finite `log_hazard` at NaN `t`.
- **Depends:** `test_a4_policy_enforcement.py` (unguarded families
  `naive_power`, `naive_log_prob_override`). The guarded families of the
  earlier A4 tests remain valid under rev 12; their tests are unchanged.

## Q-020: `batch_shape`, and the error for shape mismatches

- **Status:** resolved by spec rev 16 (4.1, 5; maintainer decision,
  2026-10-02): `batch_shape` is a property returning a tuple of Python ints,
  computed from the current parameter leaves (excluding component axes), so it
  is the per-entry shape under `vmap` and follows `jax.tree.map` slicing.
  Incompatible parameter batch shapes at construction, and a time argument
  that does not broadcast against `batch_shape` in a direct method, raise
  `TypeError` (static misuse, as for `SurvivalData`). Section 5 now also
  states `TypeError` for a distribution batch against data of a different
  length, which A6 likelihood tests will use.
- **Depends:** the batch-shape tests in `test_a4_parameters.py` and
  `test_a4_indexing.py`; `test_incompatible_batch_shapes_are_rejected`,
  `test_component_parameter_with_incompatible_batch_is_rejected`,
  `test_mismatched_lengths_do_not_broadcast` and the other
  `test_mismatched_time_*` tests in `test_a4_log_prob_default.py`.

## Q-021: no `.shape` attribute on distributions

- **Status:** resolved by spec rev 17 (4.1; maintainer decision,
  2026-10-02): the base class defines no `shape` attribute; on a family
  without a `shape` field, `.shape` raises `AttributeError` (the message may
  point to `batch_shape`). A Weibull-style `shape` field is the parameter.
  Considered and rejected: a parameters container in place of attributes, and
  NumPyro's `concentration` naming.
- **Depends:** `test_distribution_without_shape_field_has_no_shape_attribute`,
  `test_missing_shape_attribute_raises_attribute_error`,
  `test_weibull_style_shape_keyword_is_the_parameter`.

## Q-022: nested distribution fields

- **Status:** resolved by spec rev 18 (4.2; maintainer decision,
  2026-10-02): a nested distribution's batch shape counts like a parameter's;
  the outer `batch_shape` is their broadcast; nested distributions are
  broadcast at construction (keeping component axes and dtypes), so indexing,
  tree-map slicing and `vmap` reach nested leaves; nested static fields are
  part of the outer structure; `survjax.distribution()` takes no arguments.
- **Depends:** `test_a4_nested.py`, and the scalar-batch nested tests in
  `test_a4_registration.py` and `test_a4_log_prob_default.py` (still valid).

## Q-023: parameter dtype policy details

- **Status:** resolved by spec rev 15 (3.3; maintainer decision,
  2026-10-02): `register_family` normalizes each family's own parameter
  leaves, component parameters included, to one floating dtype by JAX
  promotion (weak typing, at least float32, default float for integer- or
  scalar-only input); float64 arrays under 32-bit mode convert with one warning
  per call site (Q-004); boolean, complex, string and object inputs raise
  `TypeError` (a dtype check, not a value check). Nested distribution fields
  keep their own dtypes. Constructors take no `dtype` keyword.
- **Depends:** the dtype block of `test_a4_parameters.py`, including
  `test_object_parameter_is_rejected`,
  `test_constructor_takes_no_dtype_keyword` and
  `test_nested_distribution_keeps_its_own_dtype`.

## Q-024: which generic defaults exist in Slice A

- **Status:** resolved by spec rev 14 (12.2; maintainer decision,
  2026-10-02): Slice A includes the pointwise direct methods `survival`,
  `log_survival`, `log_prob`, `log_cdf`, `cumulative_hazard`, `log_hazard` and
  `hazard`, with their generic defaults and the section 4.5 value policy. The
  maintainer's delivery schedule places them at A4.
- **Still outside Slice A:** `log_interval_prob`,
  `log_conditional_interval_prob` (Slice B); `inverse_cumulative_hazard`,
  `support()`, summaries, sampling and `check_family` (Slice D).
- **`parameter_issues()` resolved by spec rev 28 (decision 0042):** a base
  method returning `ParameterIssues` (4.7), built from declared constraints,
  nested distributions and the new `parameter_conditions()` hook (which
  replaces the `parameter_issues()` row of the 4.2 defaults table). Its
  `invalid_entries` selects the likelihood's NaN rows (7.1; since rev 29
  there is no `invalid_parameters` mask). Slice A
  includes it (12.2); tests are at A6 (Q-035).
- **Depends:** every test in `test_a4_generic_defaults.py` and
  `test_a4_hazard_at_zero_survival.py`, and the `survival`,
  `cumulative_hazard`, `log_cdf` and `hazard` tests in
  `test_a4_policy_enforcement.py`.

## Q-025: constructing registered families

- **Status:** resolved by spec rev 19 (4.2; maintainer decision,
  2026-10-02): registered families, built-in and custom, are constructed with
  keyword arguments only (positional arguments raise `TypeError`); parameter
  fields are required; static fields may have defaults; the recipe may use
  `kw_only=True`.
- **Still a test-side observation, not a guess:** `test_unflatten_bypasses_init`
  checks that unflattening stores the given leaves unchanged (object identity),
  which is what "bypassing `__init__` on reconstruction" implies.
- **Depends:** the construction block of `test_a4_registration.py`; every A4
  test constructs by keyword.

## Q-026: finite invalid parameters at `t<0` (spec 4.5, rev 20)

- **Status:** resolved by spec rev 21 (4.5; maintainer decision,
  2026-10-08): "The substituted input itself is not specified"; for invalid
  parameters at which the family is finite at some inputs and NaN at others,
  the result for `t<0` and NaN `t` "may be NaN or the policy value, and may
  differ between methods". Rejected alternative: fix the input as
  `reference_time()` under `stop_gradient`. No test depends on it.
- **Checked:** 4.5 (rev 20: "if the family's value at the substituted input is
  NaN, the result is NaN; otherwise the policy value applies ... invalid
  parameters for which the family is finite at the substituted input give the
  policy value"), 4.2, 7.2.
- **Question:** the spec does not say which safe input the base substitutes
  for `t<0` and NaN `t` (only that it is safe and under `stop_gradient`). For
  invalid parameters that are finite in some places and NaN in others (for
  example a constant-hazard `rate<0`: `log S = -r t` is `0` at `t=0` but its
  `log_prob` is `log(-|r|) = NaN` for every `t`; or Weibull `shape<0` at
  `t=0`), whether `t<0` gives NaN or the policy value depends on that choice,
  and may differ between methods of the same distribution. Should the spec fix
  the substituted input (e.g. `t=0`), or state that the outcome for such
  parameters is unspecified?
- **Guess made:** none. Tests use only parameters for which the family is NaN
  at every input in `[0, +inf]` (`rate=NaN`, `scale=NaN`), and valid
  parameters.
- **Depends:** none (`test_a4_nan_precedence.py` avoids the case).

## Q-027: built-in families are `SurvivalDistribution` subclasses

- **Status:** resolved by spec rev 22 (4.1; maintainer decision,
  2026-10-11): "Built-in families are registered subclasses of
  `SurvivalDistribution` and implement the interface of section 4.2".
- **Depends:** `test_exponential_is_a_survival_distribution`,
  `test_weibull_is_a_survival_distribution`.

## Q-028: what `from_log` guarantees about overflow

- **Status:** resolved by spec rev 22 (4.1; maintainer decision,
  2026-10-11). The overflow-avoidance claim is dropped. `from_log` returns the
  same class with `exp` of the log parameters stored as natural leaves; values
  and parameter gradients equal the ordinary constructor's wherever `exp` is
  finite; overflowing log parameters give infinite parameters and fall under
  the invalid-parameters row of 4.5. Rejected: a log-space representation in
  the leaves.
- **Previously (rev 21):** 4.1 said `from_log` avoids float32 overflow by
  computing in log space, which natural-parameter leaves cannot deliver.
- **Depends:** `test_a5_from_log.py` (finite cases only; the overflow case
  has no specified value).

## Q-029: scope of "avoid NaN parameter gradients where the value is finite"

- **Status:** resolved by spec rev 22 (4.5; maintainer decision,
  2026-10-11): the rule covers `t=0` and `t=+inf` for built-ins, with
  gradients "equal to the derivative of the value as a function of the
  parameters" (for example `0` for `S(0)`, `S(+inf)`, `log F(+inf)`). A
  derivative across a discontinuity (Weibull `log f(0)` in `k` at `k=1`) is
  outside the guarantee. Custom families meet the rule when their
  `log_survival` does.
- **Depends:** the `t=0` gradient tests and the rev 22 `t=+inf` gradient
  tests in `test_a5_boundaries.py`.

## Q-030: grid method signature and availability

- **Status:** resolved by spec rev 22 (5; maintainer decision, 2026-10-11):
  the base provides the grid methods for every registered family; the time
  vector is the first positional argument, named `t`, and follows the 4.5
  conversion and dtype rules; a `t` that is not one-dimensional raises
  `TypeError`.
- **Rev 24:** the same rules now apply to the `_grid` twin of every
  pointwise direct method (section 5, decision 0039), so no new guess.
- **Depends:** `test_a5_survival_grids.py`, including
  `test_grid_has_no_component_axes` (custom family) and the rev 22
  grid-argument tests.

## Q-031: shape of `reference_time()`

- **Status:** resolved by spec rev 22 (4.2; maintainer decision,
  2026-10-11): "an array of shape `batch_shape` in the parameter dtype,
  positive and finite for valid parameters (unspecified for invalid ones)".
- **Depends:** `test_exponential_reference_time_is_positive_finite_per_batch_entry`,
  `test_weibull_reference_time_is_positive_finite_per_batch_entry`.

## Q-032: `log_prob(+inf)` with NaN parameters

- **Status:** resolved by spec rev 22 (4.5; maintainer decision,
  2026-10-11): `log_prob(+inf)` is `-inf` unless the family's
  `log_survival(+inf)` is NaN, in which case it is NaN.
- **Rev 26:** reworded without a new guess: `log_prob(+inf)` is base-supplied
  and is NaN when the family's value at the substituted input is NaN, or when
  `log_survival_at_infinity()` is NaN (decision 0040). The dependent tests
  now cite rev 26; `test_a4_base_boundaries.py` adds the
  `log_survival_at_infinity()` NaN case.
- **Depends:** `test_nan_rate_gives_nan_for_every_time`,
  `test_nan_weibull_scale_gives_nan_for_every_time` (now including
  `t=+inf`), and the rev 22 tests in `test_a4_nan_precedence.py`.

## Q-033: the `bc` fixed vector `(1,10,-0.5,-1)`

- **Status:** resolved by spec rev 22 (4.5; maintainer decision,
  2026-10-11). The question was mistaken: section 11.1 defines the vector as
  `(k, Good-group scale, beta_Medium, beta_Poor)`, and the test's reading was
  correct. Section 4.5 now gives a self-contained example and points to 11.1.
- **Depends:** `test_survival_grid_from_zero_with_weibull_shape_one`.

## Q-034: read-only `parameter_constraints` (spec 4.7, rev 28)

- **Checked:** 4.2, 4.7.
- **Guess:** "a read-only mapping" rejects item assignment with `TypeError`,
  as `types.MappingProxyType` and every `Mapping` without `__setitem__` do.
- **Depends:** `test_parameter_constraints_is_read_only`.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.

## Q-035: delivery step of the section 4.7 tests

- **Checked:** 12.2 (rev 28 adds parameter constraints and `ParameterIssues`
  to Slice A); the delivery schedule in `CLAUDE.md` names no step for 4.7.
- **Resolved:** maintainer decision, 2026-10-19: mark them `A6`, with the
  likelihood (7.1) and result containers (6.3), so they do not reopen A4/A5.
  The schedule table in `CLAUDE.md` is not yet updated.
- **Depends:** every test in `test_a6_constraints.py`,
  `test_a6_parameter_issues.py` and `test_a6_likelihood_parameter_issues.py`.

## Q-036: constraint declaration details the spec leaves open (spec 4.7, rev 28)

- **Checked:** 4.2, 4.7, 11.3.
- **Guesses:**
  - `survjax.param(constraint=...)` stores any value; the `TypeError` for a
    non-callable constraint or an excessive `event_dim` comes from
    `register_family` ("At decoration"), so the tests call
    `register_family` on an already defined class.
  - `parameter_constraints` holds the declared object itself (`is`
    comparison for custom families). For the built-ins the tests check keys
    and `positive` behaviour, not identity with `survjax.constraints.positive`.
  - The NumPyro constraint module is `numpyro.distributions.constraints`;
    the test skips when NumPyro is not installed.
  - `ordered` of a length-1 vector and `simplex` corner cases at the
    tolerance edge are not tested.
- **Depends:** `test_register_family_rejects_*`,
  `test_parameter_constraints_*`, `test_numpyro_constraint_object_is_accepted`,
  `test_built_in_constraints_behave_as_positive`.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.

## Q-037: how Slice A fails explicitly for unimplemented modes (spec 12.2, rev 28) — resolved by rev 31

- **Resolved by rev 31 (decision 0044):** unsupported rows return NaN in both
  entry points, eagerly and under `jit`/`vmap`, without a value-dependent
  exception; supported rows keep their values and derivatives; no data or
  parameter issue is added, and the 7.1 recipe gains a mode-support check
  (`helpers.supported_mode`, `helpers.unsupported_nan`). The
  `NotImplementedError` alternative is withdrawn.
- **Depends:** nothing guessed remains; `test_a6_unsupported_modes.py` follows
  the spec text.

## Q-038: likelihood module placement (spec 6.3, 10.2, rev 28; narrowed by rev 29) — resolved by rev 35

- **Checked:** 6.2, 6.3, 7.1, 10.2.
- **Resolved by rev 29 (decision 0043):** `LikelihoodIssues` and
  `log_likelihood_with_info` are removed, so the guess about the info class
  name is withdrawn and `test_likelihood_issues_fields_by_name` deleted.
- **Resolved by rev 35 (decision 0048):** there is no top-level
  `survjax.log_likelihood`; `dist.log_likelihood` is the only public entry
  point and `likelihood.py` "holds the private evaluator". The guess that
  `survjax.likelihood.log_likelihood` is the top-level object is withdrawn
  and `test_likelihood_module_holds_the_entry_point` deleted. Since the module
  is private, the tests no longer import it: `test_no_top_level_log_likelihood`,
  `test_likelihood_has_no_with_info_variant` and
  `test_likelihood_issues_container_is_removed` check only `survjax` (and the
  method's receiver).
- **Depends:** nothing guessed remains.


## Q-039: name and schema of the committed `bc`-like R fixture (spec 10.4, 11.1, 12.2; rev 33) — resolved by rev 34

- **Resolved by rev 34 (decision 0047):** 11.1 now says "The spec tester
  defines the file name and schema; the generator writes that schema", fixes
  the per-row survival values at "each row's own time" and adds the
  flexsurv demo-grid curves (141 times, three groups). The schema below and in
  `tests/spec/fixtures/README.md` is therefore the definition, not a guess.

- **Checked:** 10.3, 10.4, 11.1, 12.1, 12.2. The spec requires the fixture
  (synthetic `bc`-like data; per-row log likelihoods and survival values from
  flexsurv and survreg at the three 11.1 vectors; float64 `atol=1e-10`,
  `rtol=1e-8`) but fixes no file name, format or layout.
- **Guess (maintainer-approved approach, 2026-10-22: the spec tester defines
  the schema):** one JSON file `tests/spec/fixtures/bc_synthetic_weibull.json`,
  laid out as in `tests/spec/fixtures/README.md`: `data` (`time` in years,
  `event` 0/1, `group` Good/Medium/Poor), `parameters` (`fixed_a`, `fixed_b`,
  `r_optimum` as `shape`, `good_scale`, `beta_medium`, `beta_poor`),
  `results.<vector>.<flexsurv|survreg>.<loglik|survival>` (686 per-row
  values, contributions on the time scale), and `grid` (141 times, `(141, 3)`
  flexsurv survival per vector). The spec tester has not seen any generator
  or fixture; if the committed one differs, only `_load` in
  `test_a8_bc_fixtures.py` and the README need to change.
- **Formerly guessed, now in 11.1:** the per-row `survival` value is
  `S_i(t_i)` at the row's own time; the grid is the 11.1 demo grid.
- **Depends:** every test in `test_a8_bc_fixtures.py`.

## Q-040: how the wheel smoke test receives the wheel (spec 10.2, 10.3, 12.2; rev 34)

- **Checked:** 10.2, 10.3, 12.1 (Extension and install), 12.2. The spec
  requires a Slice A wheel smoke test "outside the source tree" and forbids
  network in regular tests, but does not say who builds the wheel or how a
  test finds it.
- **Guess:** the runner builds the wheel (for example `uv build --wheel`) and
  passes its path in the environment variable `SURVJAX_WHEEL`. Without it the
  tests **fail** (as a missing fixture does), so the gate cannot pass without
  a wheel. "Installation" is unpacking the pure-Python wheel into a temporary
  directory put first on `sys.path` of a fresh interpreter whose working
  directory is empty; runtime dependencies come from the test environment.
  The wheel is expected to be `py3-none-any` (10.2: no compiled extensions).
- **Depends:** every test in `test_a9_wheel_smoke.py`.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.

## Q-041: version and license metadata (spec 10.1, 10.2; rev 4)

- **Checked:** 10.1 ("matching metadata"), 10.2 ("Initial version
  `0.0.1.dev0`", compatibility policy).
- **Guess:** the spec fixes only the *initial* version, so the installed
  version must be a valid PEP 440 version `>= 0.0.1.dev0`. (Until
  2026-11-09 the test pinned exactly `0.0.1.dev0` "until the first release";
  the release bump to `0.0.1a2` failed the A12 gate, and the spec does not
  require the pin.) Should the spec state a version scheme for releases
  (for example 0.0.x pre-releases), tighten this test. The license
  metadata contains "Apache" in `License-Expression`, `License` or a
  classifier (the field is not fixed by the spec).
- **Depends:** `test_distribution_version_is_not_before_the_initial_version`,
  `test_distribution_license_metadata_is_apache`.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.

## Q-042: module placement of the concordance API (spec 9.2, 10.2; rev 34)

- **Checked:** 6.3, 9.2, 10.2 (layout: `metrics.py  # concordance, Brier,
  IBS, AUC`; "Re-export ... KM and metric entry points"; "result and
  configuration types stay discoverable in their modules").
- **Guess:** `survjax.concordance_index` and
  `survjax.concordance_index_with_info` are the same objects as
  `survjax.metrics.concordance_index(_with_info)`; the info class is
  `survjax.metrics.ConcordanceInfo` (helper `concordance_info_class`). Not
  asserted: whether `ConcordanceInfo` is also exported at top level.
- **Depends:** `test_metrics_module_holds_the_entry_points`,
  `test_concordance_info_is_discoverable_in_its_module`, every
  `isinstance(..., concordance_info_class())` check in
  `test_a10_concordance_api.py`.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.

## Q-043: field names of `ConcordanceInfo` (spec 9.2; rev 34)

- **Checked:** 6.3 ("Containers are accessed by field name"), 9.1, 9.2
  ("concordant, discordant, tied-score and comparable counts plus invalid,
  unsupported and no-comparable-pair masks").
- **Guess:** counts `concordant`, `discordant`, `tied_score`, `comparable`;
  masks `invalid`, `unsupported`, `no_comparable_pairs`
  (`helpers.CONCORDANCE_COUNTS`, `helpers.CONCORDANCE_MASKS`).
- **Depends:** every A10 test that reads the info.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.

## Q-044: how an exact count is read from `ConcordanceInfo` (spec 9.1; rev 34)

- **Checked:** 9.1 ("Integer counts use exact arithmetic without a capacity
  cap: per-row counts in `int32`, totals in a two-word (hi/lo)
  accumulator"), 9.2, 6.3. The spec fixes the accumulator but not how a
  total is exposed in the container.
- **Guess (one helper, `helpers.count`):** a count field is either an
  integer array, or an object with integer arrays `hi` and `lo` whose
  value is `hi * 2**32 + (lo mod 2**32)`. Counts have the result shape
  (one entry per concordance value). Counts of results that are NaN because
  of invalid or unsupported rows are not asserted.
- **Depends:** every A10 test that calls `count`.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.

## Q-045: shape of the `invalid` and `unsupported` masks (spec 9.1, 9.2; rev 34)

- **Checked:** 6.1, 6.2, 9.1 ("Other rows give NaN plus an `unsupported`
  diagnostic"), 9.2. The masks may be per result or per row; the spec does
  not say.
- **Guess (minimal):** `helpers.result_mask` accepts either: a mask of the
  result shape is used as is; one with a trailing observation axis is
  reduced with `any` over it. Tests assert only the per-result reduction,
  and that a NaN result is explained by `invalid` or `unsupported` and a
  valid one by neither. `no_comparable_pairs` is per result.
- **Also assumed:** an invalid or unsupported row makes the whole
  concordance value that contains it NaN (rows "are never dropped", 9.1;
  "invalid runtime values give NaN plus issues", 6.1); non-finite scores
  (`NaN`, `±inf`) are invalid runtime values ("Scores are finite", 9.2) and
  set `invalid`.
- **Depends:** `test_a10_concordance_diagnostics.py`, the mask checks in
  `test_a10_concordance_api.py`, `test_a10_concordance_values.py`,
  `test_a10_concordance_shapes.py` and `test_a10_concordance_properties.py`.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.

## Q-046: name and schema of the concordance R fixture (spec 9.2, 10.4, 12.1; rev 34)

- **Checked:** 9.2 (cross-check against `survival::concordance(...,
  reverse = TRUE, timewt = "n")`), 10.3 (survival 3.7-0), 10.4, 12.1
  ("committed R fixtures (... concordance ...)"). The spec fixes no file
  name or layout; 11.1 lets the spec tester define the `bc` fixture schema,
  and the same approach is taken here.
- **Guess:** one JSON file `tests/spec/fixtures/concordance_synthetic.json`,
  laid out as in `tests/spec/fixtures/README.md`: `meta.call` and versions;
  `cases[]` with `time`, `event` (0/1), `scores` (`D` vectors) and `r[]`
  (`concordance`, `concordant`, `discordant`, `tied_x`, `tied_y`,
  `tied_xy` per vector). The call adds `timefix = FALSE`, from the
  maintainer's draft A10 outline (not in the spec text). Mapping:
  `tied_score = tied.x`, `comparable = concordant + discordant + tied.x`.
  A missing fixture fails the tests.
- **Depends:** every test in `test_a10_concordance_fixtures.py`.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.

## Q-047: step A10 and its source (maintainer replan, 2026-10-23)

- **Checked:** the role file's delivery table (A0–A9, no A10); spec 12.2
  (rev 34) lists concordance under Slice E; `design/proposals/concordance-at-end-of-slice-a.md`
  (draft, 2026-10-23, not yet accepted into the spec).
- **Instruction:** the maintainer asked for A10 tests "with concordance
  index" now. The tests mark `step("A10")` and assert only sections 9.1–9.2
  (and 6.1–6.3, 10.2) of rev 34, which the draft says it does not change.
  Not assumed from the draft: anything about documentation, the held-out
  example, or the 12.1/12.2 wording it proposes. When the amendment is
  accepted, the role file's table should gain the A10 row; the tests'
  revision tags will be updated only if 9.1–9.2 change.
- **Depends:** every `test_a10_*.py`.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.

## Q-048: step for `data.shape`, `len(data)` and `bool(data)` (spec 3.2, rev 36)

- **Checked:** 3.2 (rev 36, "Size"), 12.1 (Construction row), 12.2 (Slice A
  lists "`shape` and `len`"); the role file's delivery table names no step
  for them.
- **Guess:** marked `step("A3")`, with indexing and static metadata, because
  the 12.1 evidence requires them after indexing and under `jit`/`vmap`.
  Missing attributes are tested with `hasattr` (that is, `AttributeError`),
  as for Q-021.
- **Depends:** every test in `test_a3_shape_and_len.py`.
- **Status:** confirmed by the maintainer against the implementation (dev/main 4239d61, full gate in both dtype modes), 2026-11-09.

## Q-049: steps for the rev 37 first-release items (spec 4.4, 8.7, 12.2)

- **Checked:** 12.2 (rev 37, "First public release" items 2 and 3), the role
  file's delivery table (A0–A9) and Q-047 (A10).
- **Resolved** by the maintainer's handoff of 2026-11-06 (spec rev 41):
  **A11** is `ProportionalHazards` excluding RMST (construction, parameter
  issues, direct methods and grids, likelihood, PH identities, ordinary-range
  derivatives, isolation, custom baseline); **A12** is native staged RMST and
  the PH–RMST integration (section 8.7 and the 12.1 "Staged RMST" row,
  including `rmst` on a PH wrapper of a custom baseline). Files are
  `test_a11_*.py` and `test_a12_*.py`.
- **Depends:** every `step("A11")` and `step("A12")` marker.

## Q-050: constructing `ProportionalHazards` (spec 4.2, 4.4, 4.7; rev 41)

- **Status:** resolved by spec rev 42 (4.4; decision 0053, maintainer
  confirmation 2026-11-06): both wrappers are registered families with a nested
  field `baseline` and a parameter field `log_hazard_ratio` / `log_time_ratio`,
  constructed with keyword arguments only; a positional baseline raises
  `TypeError`. The 4.4 example now uses keywords. Adds
  `test_positional_baseline_is_rejected`.
- **Checked:** 4.2 ("Registered families, built-in and custom, are
  constructed with keyword arguments only"), 4.4 (example
  `survjax.ProportionalHazards(base, log_hazard_ratio=X @ beta)`, baseline
  positional), 4.7 (nested keys "prefixed by the field name and a dot";
  4.4 names the prefix `baseline.`), 10.2 (wrappers re-exported at top level).
- **Question:** is the wrapper keyword-only like every registered family, or
  does it accept the baseline positionally as the 4.4 example shows?
- **Guess:** the nested field is named `baseline`, the predictor field
  `log_hazard_ratio`, and `helpers.ph` passes both by keyword, which works
  under either reading. No test calls the wrapper positionally or asserts that
  a positional call raises.
- **Depends:** every A11/A12 test through `helpers.ph`;
  `test_baseline_and_linear_predictor_are_stored_under_constructor_names`,
  `test_non_distribution_baseline_is_rejected_naming_the_field` and the
  `baseline.`-key tests in `test_a11_ph_parameter_issues.py`.

## Q-051: PH hazard limits at `t=0` and `t=+inf` (spec 4.4, 4.5; rev 41)

- **Status:** resolved by spec rev 43 (4.4; decision 0054): the table
  gains `log_hazard` rows (PH `log h0(t) + eta`, AFT `log h0(t*exp(-eta)) - eta`),
  evaluated through the baseline's `log_hazard`, so the baseline's 4.5
  boundary limits carry through. Adds the `log_hazard`/`hazard` boundary tests
  in `test_a11_ph_direct_methods.py`.
- **Checked:** 4.4 (the table gives PH formulas for `log_survival` and
  `log_prob` only; "The direct methods and the likelihood keep the PH
  formulas"), 4.5 (`log_hazard`/`hazard` at `t=0` and `t=+inf` come from a
  `log_hazard` override; "the generic default gives NaN" at `+inf` and "no
  value or gradient guarantee" at `t=0`), 4.4 identity ("every method ...
  equals the same method of `Weibull(...)`").
- **Question:** does `ProportionalHazards` override `log_hazard` (as
  `log h0(t) + eta`, so the baseline's limits pass through), or use the
  generic default? The identity would need the former at the boundaries.
- **Guess:** none. `log_hazard`/`hazard` at `t=0` and `t=+inf` are not
  tested. `log_prob` at `t=0` is tested through the 4.4 formula
  `log h0(t) + eta + exp(eta) log S0(t)` with the Weibull limits of 4.5
  (`test_density_limit_at_zero_for_weibull_baseline`); interior hazards are
  tested everywhere.
- **Depends:** `test_density_limit_at_zero_for_weibull_baseline`.

## Q-052: factor for the staged RMST derivative tolerances (spec 8.7, 12.1, 13; rev 41)

- **Status:** resolved by spec rev 44 (12.1, 13; decision 0055): the
  factor is the ordinary 12.1 `rtol` (1e-8 float64, 1e-4 float32); `eta`
  derivatives have scale `|rmst|`; higher orders use the same log units at
  scale `|rmst|`. The guess below became the requirement; no assertion changed.
- **Checked:** 12.1 ("errors in `k*d rmst/dk` and `b*d rmst/db` (or
  `r*d rmst/dr` for Exponential) have an absolute tolerance scaled by
  `|rmst|` ... The `tau` derivative uses absolute scale 1. Factors are fixed
  after measuring `igamma_grad_a`"), 8.7 (review measurements: first
  derivatives to 2e-13 relative in float64, float32 well-scaled
  derivatives 1e-6 to 1e-5), 13.
- **Question:** the factor is an implementation measurement that the spec
  does not state.
- **Guess:** factor = the 12.1 ordinary `rtol` (1e-8 float64, 1e-4 float32),
  so `|theta (got - ref)| <= rtol |rmst|` and `|d/dtau - S(tau)| <= rtol`.
  This sits two to three orders above the quoted review measurements and was
  not tuned to any survjax output. The same scale is applied to `eta`
  (already a log-scale parameter: scale `|rmst|`), and, as my own choice, to
  second and third derivatives in the same log units (each variable,
  `tau` included, multiplied by its own value; scale `|rmst|`). If the
  measured factors are larger, change `helpers.rmst_rtol`.
- **Depends:** `helpers.assert_log_parameter_derivative`,
  `helpers.assert_cutoff_derivative`, `assert_higher_derivative` in
  `test_a12_rmst_derivatives.py`, and the derivative checks in
  `test_a12_rmst_policy.py`.

## Q-053: must some Weibull shape Hessian raise? (spec 8.7, 12.1; rev 41)

- **Status:** resolved by spec rev 45 (12.1; decision 0056): Weibull
  Hessians in `k` (alone and through PH) raise `NotImplementedError` or return
  the correct value, never a wrong value; no invocation is required to raise.
  The tests already did this; retagged.
- **Checked:** 8.7 (shape higher derivatives "raise `NotImplementedError` ...
  Some differentiation orders, or tail branches that do not use `igamma`,
  may instead return the correct value; they never return a wrong value"),
  12.1 ("`NotImplementedError` from a Weibull Hessian in `k` (alone and
  through PH)").
- **Question:** 12.1 reads as requiring a raise; 8.7 permits a correct
  value instead. Is an implementation that returns correct shape Hessians
  everywhere conforming?
- **Guess:** tests accept either a `NotImplementedError` or a value equal to
  the mpmath reference (Q-052 scale), for interior, heavy-tail and both power
  tails, all four mode compositions, alone and through PH. No test requires
  at least one raise.
- **Depends:** `test_weibull_shape_hessian_raises_or_is_correct`,
  `test_weibull_full_hessian_raises_or_is_correct`,
  `test_ph_weibull_shape_hessian_raises_or_is_correct` (removed at rev 50,
  restored at rev 51 with `until("summary.rmst")`; Q-055).

## Q-054: "not provided" means absent (spec 8.7, 10.2; rev 41)

- **Status:** resolved by spec rev 46 (8.7; decision 0057): "not
  provided" means absent (`AttributeError`, no stubs; `SummaryOptions` and
  `SummaryInfo` not exported from any public module); a non-broadcasting `tau`
  raises `TypeError`; `tau` follows the 4.5 conversion and dtype rules. Adds
  `test_cutoff_that_does_not_broadcast_is_rejected`,
  `test_cutoff_with_a_non_numeric_dtype_is_rejected`,
  `test_boolean_cutoff_is_rejected_during_tracing` and
  `test_float64_cutoff_converts_without_warning_in_32_bit_mode`; the absence
  test now expects `AttributeError` on access.
- **Checked:** 8.7 ("`rmst_with_info`, `SummaryOptions` and `SummaryInfo`
  are not provided"; "`sample` and `inverse_cumulative_hazard` are not
  provided"; "There is no `rmst_grid`"), contrasted with "raise
  `NotImplementedError`" for `mean`, `median`, `quantile`; 10.2 ("Result and
  configuration types stay discoverable in their modules").
- **Guess:** absent attributes: no such method on Exponential, Weibull or PH
  instances; no `SummaryOptions`/`SummaryInfo` at top level, in
  `survjax.diagnostics` or `survjax.distributions` (when importable). A stub
  raising `NotImplementedError` would fail these tests.
- **Not tested:** the error type for a `tau` that does not broadcast against
  `batch_shape` (8.7 says only "ordinary trailing-axis rules"; 4.1's
  `TypeError` is stated for direct methods), and the `tau` dtype rules
  (boolean or complex `tau`).
- **Depends:** `test_absent_first_release_methods` (`rmst_with_info`,
  `inverse_cumulative_hazard`, `sample`, each with its `until`),
  `test_there_is_no_rmst_grid`, `test_summary_configuration_types_are_not_provided`
  (`until("summary.containers")`); restored at rev 51 after the rev 50
  removal (Q-055).

## Q-055: step for the post-release summary stage, and superseded A12 tests (spec 8.8, 12.2; rev 50)

- **Status:** superseded by spec rev 51 (decision 0058) and the maintainer's
  staging scheme (2026-11-09): there is no step A13. Post-release tests carry
  `feature(...)` markers (`summary.containers`, `summary.rmst`,
  `summary.mean`, `summary.quantile`, `sampling`), filtered by
  `SPEC_FEATURES`; the A12 rejection and absence tests removed at rev 50 are
  restored under `step("A12")` with `until(<feature>)`. Earlier resolution
  (2026-11-07, rev 50): step A13, superseded A12 tests deleted.
- **Checked:** 8.8 Gate ("replace shape-Hessian rejection with correctness
  checks, and replace custom-RMST/mean/quantile and diagnostic-absence checks
  with the new surface requirements ... the historical rows remain
  unchanged"), 12.2 ("Post-release summary stage").
- **Consequence (rev 51):** with `SPEC_FEATURES` unset, the suite checks the
  first-release rows again; enabling a feature drops the A12 tests it lifts
  and runs its feature tests.
- **Depends:** every `feature(...)` and `until(...)` marker; `conftest.FEATURES`.

## Q-056: `SummaryOptions` / `SummaryInfo` access conventions (spec 8.5, 8.7; rev 50, 51)

- **Checked:** 8.5 (rev 50 schema; "`options=survjax.SummaryOptions(...)`"),
  8.7 ("Slice D adds a keyword-only `options` with a default"), 8.8.
- **Rev 51:** the rev 48–50 sentence "Export `SummaryOptions` and
  `SummaryInfo` at the package top level" and the explicit signatures
  `rmst(tau, *, options=None)`, `mean(*, options=None)`,
  `quantile(p, *, options=None)` with "`None` selects section 8.5 defaults"
  are gone. Guess kept: `SummaryInfo` is also `survjax.SummaryInfo`; every
  summary method and `_with_info` twin (`median` and `median_with_info`
  included) takes keyword-only `options=None`, and `None`, omission and
  `SummaryOptions()` agree.
- **Guess:** both are constructed with the listed keyword names; fields are
  attributes of the same names; `SummaryInfo.Status` is a class attribute;
  assigning to a field of the frozen containers raises `AttributeError` (as
  `dataclasses.FrozenInstanceError` does); status arrays have an integer
  dtype that compares equal to the `IntEnum` values; a `SummaryOptions` is a
  valid `jit` argument whose work limits read back as Python `int` under
  tracing. Counts are compared as numbers (their dtype is not asserted).
- **Depends:** `test_summary_containers_schema.py`, the `*_options_*` and
  `*_with_info_*` tests in `test_summary_{rmst,mean,quantile}_surface.py`,
  `helpers.status`, `helpers.assert_status`, and every feature test that
  reads `info`.
- **Status:** confirmed by the maintainer, 2026-11-09.

## Q-057: status of invalid-input, zero-cutoff and endpoint entries (spec 4.5, 8.5, 8.7, 8.8; rev 50, 51, 54)

- **Status:** resolved by spec rev 54 (8.5; design f87c3d5): "Invalid
  arguments (`tau<0` or non-finite, `p` outside `[0,1]`, negative or NaN `e`)
  and entries with parameter issues have status `invalid_input` and NaN
  values. Exact endpoint values (`rmst(0)=0`, endpoint quantiles from
  `support()`) have status `converged`." The "not `converged`" tests now
  assert `invalid_input` (`test_parameter_violations_report_invalid_input`
  in the three policy files, `test_invalid_cutoff_reports_invalid_input` on
  every family, `test_invalid_probability_reports_invalid_input`), and new
  tests assert `converged` for `rmst(0)` and endpoint quantiles
  (`test_zero_cutoff_reports_converged`, `test_cutoff_statuses_are_per_entry`,
  `test_endpoint_quantiles_report_converged`,
  `test_bounded_support_endpoint_quantiles_report_converged`; the improper
  `p = 1` test is `known_infinite` since rev 56, Q-068).
- **Settled by spec rev 56 (8.5; design 720524a):** "On an improper family, a
  quantile that is `+inf` because `p >= 1 - S(+inf)` has status
  `known_infinite`; `quantile(1)=+inf` on a proper family is an endpoint and
  `converged`." New tests:
  `test_improper_family_quantile_beyond_the_cure_level_reports_known_infinite`,
  `test_known_infinite_quantile_is_decided_per_entry_of_the_batch`. The
  improper `p = 1` case moved from `converged` to `known_infinite` (Q-068).
- **Checked:** 8.5 (status list), 4.5 summary rows, 8.7 value policy ("NaN
  wherever `parameter_issues()` reports a violation; with no `SummaryInfo`,
  NaN is the release's invalid-input signal"). Rev 50's 8.8 sentence
  "Declared parameter violations yield NaN values and invalid-input status for
  all three summaries" was removed at rev 51.
- **Rev 51 change:** the parameter-violation tests no longer assert
  `invalid_input`; like invalid `tau`/`p`, the violating entry must be NaN and
  not `converged` (valid entries `converged`). Should the spec restore the
  `invalid_input` requirement, tighten
  `test_parameter_violations_are_not_reported_converged` in the three policy
  files.
- **Question (rev 51, answered by rev 54 except the improper interior
  `+inf`):** is `invalid_input` also the required status for invalid `tau`
  or `p`? What status do `rmst(0)`, `quantile(0)`, `quantile(1)` and an
  improper-family `+inf` quantile report?
- **Depends:** `test_invalid_probability_reports_invalid_input`,
  `test_invalid_cutoff_reports_invalid_input`,
  `test_parameter_violations_report_invalid_input` (rmst, mean, quantile
  policy files), `test_parameter_violation_status_under_jit`, and the
  `converged` endpoint tests listed under Status.

## Q-058: tolerances for generic summary values and derivatives (spec 8.3-8.5, 12.1; rev 50, 54)

- **Status:** resolved by spec rev 54 (12.1 Summaries row; design f87c3d5):
  "Generic quadrature and solver results, and their derivatives, are compared
  at `10*rtol` relative (this table's ordinary `rtol` per dtype), since a
  converged estimate may sit at the stopping tolerance; closed forms keep the
  ordinary targets." This is the guess below (`helpers.GENERIC_FACTOR = 10`);
  no assertion changed, only the comments citing it.
- **Interpretation kept:** for derivatives, "relative" is read in the 12.1
  log-parameter units already used for staged RMST (`|theta (got - want)| <=
  10*rtol*|value|`; the `tau` derivative at absolute `10*rtol`), not relative
  to the derivative itself, which would fail on roundoff where a true
  derivative is near zero (for example `dq/dk` near `e = 1`). Tighten
  `assert_generic_log_derivative` if the maintainer means the latter.
- **Checked:** 12.1 (ordinary targets; staged-RMST log-unit derivative
  scales), 8.3 (rev 52: "Convergence requires one adjacent pair of combined
  estimates to satisfy `abs(Q_l - Q_(l-1)) <= atol + rtol*abs(Q_l)` ... This
  is a heuristic"), 8.4 ("Derivative accuracy is
  limited by the quadrature tolerance and is documented method by method"),
  8.5 defaults (`rtol` 1e-8 / 1e-4 equal to the 12.1 `rtol`).
- **Question:** the spec gives no derivative or value target for generic
  quadrature and solver results.
- **Guess:** generic values are compared relatively at `10 * rtol` (12.1
  `rtol`), no absolute floor; generic derivatives in the 12.1 log units at
  `10 * rtol * |value|` (the `tau` derivative at absolute `10 * rtol`).
  The factor 10 allows a converged estimate to sit at the stopping
  tolerance. Built-in closed forms and Weibull RMST (8.8 rev 51: "Section
  8.7's tail accuracy ... remain[s] in force") keep the 12.1 scales. Change `helpers.GENERIC_FACTOR`
  if the documented accuracy differs.
- **Depends:** `helpers.generic_rtol`, `assert_generic_close`,
  `assert_generic_log_derivative`; generic tests in
  `test_summary_{rmst,mean,quantile}_values.py` and
  `test_summary_{rmst,mean,quantile}_derivatives.py`.
- **Status:** confirmed by the maintainer, 2026-11-09 (derivative tolerance in the existing log-parameter units).

## Q-059: `support()` return convention (spec 4.2, 8.1-8.3; rev 50)

- **Checked:** 4.2 ("`support()` | `(0, +inf)`. Bounded-support families
  override"), 8.1, 8.2, 8.3.
- **Guess:** `support()` returns a tuple `(lower, upper)` of arrays
  broadcastable to `batch_shape` in the parameter dtype.
- **Depends:** `spec_families.uniform_support` and the bounded-support tests
  in `test_summary_{rmst,mean,quantile}_values.py`.
- **Status:** confirmed by the maintainer, 2026-11-09.

## Q-060: how a family reports a known infinite mean (spec 4.2, 8.1, 8.3; rev 50, 51, 54)

- **Status:** resolved by spec rev 54 (8.1; design f87c3d5): "A family
  reports it by overriding `mean_with_info`; `mean` returns the value part of
  `mean_with_info`, so one override serves both. The generic mean returns
  `+inf` with status `known_infinite`, without quadrature, wherever
  `log_survival_at_infinity() > -inf`." The improper-family rule-out test is
  replaced by direct assertions (`+inf`, `known_infinite`, zero counts, NaN
  error estimate, per-entry decision); its tangent test now requires NaN
  (8.4 rev 54). Override tests use the custom families
  `known_mean_log_logistic` (the 8.1 log-logistic row through a
  `mean_with_info` override, `beta` 0.5, 1 and 2) and
  `mean_with_info_sentinel` (dispatch only); their override signature was
  Q-067 and is spec since rev 56.
- **Checked:** 4.2 ("Families may override, including to report a known
  infinite mean"), 8.1, 8.3 ("A family's known-infinite mean status is still
  needed").
- **Previously (rev 50, 51):** the override mechanism was unspecified, so the
  only test (`test_improper_family_mean_is_never_finite`, removed) accepted
  `+inf` with `known_infinite` or NaN with a failure status, and the tangent
  test only non-finite.
- **Depends:** `test_improper_family_mean_is_plus_infinity`,
  `test_improper_family_mean_is_known_infinite`,
  `test_improper_family_mean_uses_no_quadrature`,
  `test_known_infinite_mean_is_decided_per_entry`,
  `test_log_logistic_mean_override_*`,
  `test_mean_returns_the_value_part_of_a_mean_with_info_override`,
  `test_improper_mean_has_a_nan_tangent`.

## Q-061: count and failure-status details (spec 8.2, 8.3, 8.5; rev 50, 52, 53, 54)

- **Status:** resolved by spec rev 54 (8.2; design f87c3d5): "Non-convergence
  gives NaN and a failure status: `failed_bracket` when the bracket step limit
  (`max_bracket_steps`) is reached without bracketing the root,
  `work_exhausted` when `max_solver_steps` is reached inside a bracket."
  `test_bracket_limit_reports_failed_bracket` (renamed from
  `test_bracket_limit_failure_is_reported`) now asserts `failed_bracket` and
  `num_bracket_steps == 1`; the new
  `test_solver_step_limit_inside_a_bracket_reports_work_exhausted` asserts
  `work_exhausted` and `num_solver_steps == 1` (the counts equal the limit
  that was "reached"; inference from "Counts record ... steps used").
- **Checked:** 8.2, 8.3, 8.5.
- **Rev 52 resolves the quadrature part:** `max_levels` evaluated levels
  (0 to `max_levels - 1`), `num_levels` counts rounds per broadcast entry
  (not summed over split intervals), convergence needs one adjacent pair of
  summed estimates, and `error_estimate` is the last combined-estimate
  difference (also on work exhaustion; NaN with fewer than two estimates or a
  non-finite integrand). Tests now assert these directly.
- **Rev 53 resolves the returned quadrature value:** 8.3 now says "A
  converged entry returns `Q_l` of the converging pair. An entry with a
  failure status (`non_finite_arithmetic` or `work_exhausted`) returns NaN."
  The earlier guess (returned value is `Q_l`) is confirmed, so
  `test_error_estimate_is_in_time_units` no longer depends on this question;
  the NaN values of work-exhausted and non-finite entries are now spec, not
  inference. 8.5's "NaN when not meaningful" now points to the quadrature
  paragraph (no test change).
- **Question (rev 53, answered by rev 54):** does exhausting
  `max_bracket_steps` report `failed_bracket` or `work_exhausted`?
- **Depends:** `test_generic_quantile_counts`,
  `test_bracket_limit_reports_failed_bracket`,
  `test_solver_step_limit_inside_a_bracket_reports_work_exhausted`.

## Q-062: tangents at finite endpoint quantiles and invalid inputs (spec 8.4, 8.7, 8.8; rev 50, 51, 54)

- **Status:** resolved by spec rev 54 (8.4; design f87c3d5): "Entries that
  failed or did not converge get NaN tangents, never zeros. So do endpoint
  quantiles (`p` = 0 or 1), `inverse_cumulative_hazard` at `e` = 0 or
  `+inf`, known-infinite means and invalid-input entries, even where a
  one-sided or constant derivative exists. Gradients are guaranteed only
  inside the open domain (`0<p<1`, `0<e<+inf`)." Tests now require NaN (not
  merely non-finite) and cover `p = 0`, a bounded `quantile(1) = end`,
  `H⁻¹(0)`, `H⁻¹(+inf)`, invalid `tau`, `p` and `e`, and parameter-issue
  entries of all three summaries. `rmst(0)` keeps its 8.4 first derivatives
  (not in the list).
- **Settled by spec rev 56 (8.1, 8.4; design 720524a):** "The override is
  responsible for the section 8.4 NaN tangents of a known infinite mean;
  built-ins provide them", and "known-infinite means and quantiles" get NaN
  tangents. The custom `known_mean_log_logistic` now supplies them itself;
  `test_mean_carries_the_tangents_of_a_mean_with_info_override` checks that
  `mean()` carries them, and `test_known_infinite_quantile_has_nan_tangents`
  covers the improper `+inf` quantile.
- **Checked:** 8.4 ("Entries that failed or did not converge get NaN
  tangents, never zeros. Endpoint quantiles and known-infinite means get NaN
  or infinite tangents"), 8.8 rev 51 (once a method is enabled, 8.7's
  derivative restrictions, including "section 8.4's NaN-tangent requirement
  is deferred", no longer apply).
- **Question (rev 51, answered by rev 54):** `quantile(0) = 0` and a bounded
  `quantile(1) = end` have finite true derivatives; must their tangents still
  be NaN or infinite? Are invalid-input entries (`tau < 0`, `p > 1`,
  parameter violations) "failed" for the NaN-tangent rule?
- **Depends:** `test_non_converged_entry_has_a_nan_tangent`,
  `test_non_finite_failure_entry_has_a_nan_tangent`,
  `test_endpoint_quantile_has_nan_tangents`,
  `test_bounded_support_upper_endpoint_quantile_has_a_nan_tangent`,
  `test_inverse_cumulative_hazard_endpoints_have_nan_tangents`,
  `test_invalid_*_entries_have_nan_tangents`,
  `test_parameter_violation_entry_has_a_nan_*_tangent`,
  `test_improper_mean_has_a_nan_tangent`.

## Q-063: public `inverse_cumulative_hazard` signature and input policy (spec 4.5, 8.1, 8.2, 8.4, 8.8; rev 51, 54)

- **Status:** resolved by spec rev 54 (8.1, 8.5; design f87c3d5):
  `inverse_cumulative_hazard(e, *, options=None)`; "`e=0` gives the lower and
  `e=+inf` the upper end of `support()`; negative or NaN `e` gives NaN; no
  `_with_info` twin (failures are NaN; `quantile_with_info` reports the same
  solver's diagnostics)"; it "takes `options` for its solver limits". New
  tests: invalid `e` NaN (eager, `jit`, per entry), `e=+inf` endpoints
  (default, bounded, improper), keyword-only `options`, `None` defaults,
  solver limits through `options`, absent `_with_info` twin (`AttributeError`,
  as 8.7 defines absence).
- **Dtype question settled by design d104627 (8.1, within rev 54):** "the
  section 4.5 dtype rule applies to `e` as to `t`". New tests:
  `test_hazard_level_with_a_non_numeric_dtype_is_rejected`,
  `test_boolean_hazard_level_is_rejected_during_tracing`,
  `test_float64_hazard_level_converts_without_warning_in_32_bit_mode`. Which
  feature enables it stays `summary.quantile` (the role table).
- **Checked:** 8.1 ("`inverse_cumulative_hazard(e)` | the `t` with `H(t)=e`,
  for `e>=0`"), 8.2 (solver; improper `+inf`), 8.4 (implicit-function
  tangents), 8.5 ("Summary methods take `options`"), 8.8 (lists
  `inverse_cumulative_hazard` separately from "summaries, their `_with_info`
  variants"; "Section 8.7's ... value policy and shape rules remain in force
  and apply in the same way to `p` for quantiles"), 4.5 (rows for `t`, `tau`
  and `p`, none for `e`).
- **Question:** does `inverse_cumulative_hazard` take `options` and have an
  `inverse_cumulative_hazard_with_info` twin (it uses the same solver and
  limits)? What does it return for negative, NaN or `+inf` `e`, and does the
  4.5 dtype rule (`TypeError` for boolean/complex/string/object) apply to `e`?
  Which feature enables it (the role table puts it under `summary.quantile`)?
- **Guess still made (tested):** `e` broadcasts against `batch_shape` like a
  direct-method argument (`e_grid[:, None, None]` gives `(m, S, 2)`); 8.1 now
  fixes the signature, endpoints, invalid-`e` values, `options`, the absent
  twin and (d104627) the dtype rule.
- **Depends:** the `inverse_cumulative_hazard` tests in
  `test_summary_quantile_surface.py`, `_values.py`, `_policy.py`
  (`test_inverse_cumulative_hazard_grid_with_explicit_axes`, the invalid-`e`
  and `options` tests) and `_derivatives.py`.

## Q-064: the triage recipe's support check after Slice B (spec 7.1, 12.2; rev 51)

- **Checked:** 7.1 ("The documentation shows this **Slice A** recipe";
  "Its mask expands as later slices implement more modes"), 12.2 (Slice B:
  "Left and interval censoring; all truncation combinations").
- **Question:** the spec gives only the Slice A form of `supported_mode`. Once
  Slice B is in, is the support check simply all-true (every v0 observation
  mode implemented), and does the documented recipe keep a `supported_mode`
  term at all?
- **Guess:** `helpers.supported_mode` drops the kind condition when
  `observation.censoring` is enabled and the truncation condition when
  `observation.truncation` is enabled (`SPEC_FEATURES`), so with both enabled
  every row is supported and every unexplained NaN is `other` (`Z=0` or
  numerical failure).
- **Depends:** `helpers.supported_mode`, `helpers.nan_causes`,
  `helpers.unsupported_nan`, and the Slice B feature tests that triage NaN
  rows.
- **Status:** confirmed by the maintainer, 2026-11-09.

## Q-065: calling `log_interval_prob` directly (spec 4.2, 4.5, 7.1, 12.2; rev 51, 54)

- **Status:** settled by spec rev 54 (12.2; design f87c3d5): Slice B is now
  "Left and interval censoring; all truncation combinations;
  `log_interval_prob` and the `log_conditional_interval_prob` hook", so the
  member is part of Slice B, and of `observation.censoring` as guessed. No
  test changed (module docstring retagged).
- **Still guessed (unchanged spec text):** positional `(lower, upper)`
  broadcasting like a direct-method argument, and NaN for reversed endpoints
  (7.1 says only "invalid"; 4.5 rules out raising and issue masks). Which
  member the likelihood uses for an untruncated censored row stays
  untested.
- **Checked:** 4.2 (member table: "`log_interval_prob(lower, upper)` | Stable
  log difference with tail selection ..."; overrides "receive endpoints `0`
  and `+inf`"), 4.5 (direct methods "never raise on values and carry no issue
  masks"), 7.1 ("Direct interval probability with equal endpoints is zero;
  reversed endpoints are invalid"), 12.2 (Slice B lists only the
  `log_conditional_interval_prob` hook).
- **Question:** is `dist.log_interval_prob` a public callable method of every
  registered family, and in which feature does it land? How do its endpoints
  broadcast? What value does it return for reversed endpoints (`lower >
  upper`): NaN, or something else, given that it cannot raise or set an issue?
  Which member does the likelihood use for an untruncated censored row?
- **Guess (tested):** a public method on every family, called positionally
  `(lower, upper)`; endpoints broadcast against `batch_shape` like a
  direct-method `t`; enabled with `observation.censoring`; equal endpoints
  give `-inf`; reversed endpoints give NaN. Nothing is asserted about which
  member the likelihood dispatches to.
- **Depends:** `test_observation_censoring_interval_prob.py`.
- **Status:** confirmed by the maintainer, 2026-11-09.

## Q-066: calling `log_conditional_interval_prob` and its use by the likelihood (spec 4.2, 12.2; rev 51)

- **Checked:** 4.2 ("`log_conditional_interval_prob(lower, upper, left_trunc,
  right_trunc)` | Generic `log P(max(lower,L)<T<=min(upper,U)) - log Z`.
  Families override it ... The likelihood never dispatches on family type";
  "a family shapes the likelihood only through the members above"), 12.2
  (Slice B: "`log_conditional_interval_prob` hook").
- **Question:** is the hook a public callable on every registered family,
  and do its four arguments broadcast against `batch_shape`? Which rows does
  the likelihood evaluate through it: every censored row, only truncated
  censored rows, or also `log Z` for exact rows? Does a family's override
  replace the generic path for all of them? Which feature enables it
  (it is needed only with truncation, but the test rows include interval
  sets)?
- **Guess (tested):** a public method on every family, called positionally
  `(lower, upper, left_trunc, right_trunc)`, broadcasting like a direct-method
  argument; enabled when both Slice B features are enabled; its value is the
  4.2 definition, `0` on the whole window, and equal to the corresponding
  censored row's likelihood contribution. A correct custom override leaves
  the likelihood equal to the closed form; nothing asserts that the
  likelihood calls the override.
- **Depends:** `test_observation_truncation_hook.py`.

## Q-067: writing a `mean_with_info` override in a custom family (spec 4.2, 8.1, 8.4, 8.5; rev 54)

- **Status:** settled by spec rev 56 (8.1; design 720524a): "The override has
  the signature `mean_with_info(self, *, options=None) -> tuple[Array,
  SummaryInfo]` and may ignore `options`. An analytical answer reports status
  `converged` (or `known_infinite`), a NaN `error_estimate` and zero work
  counts. The override is responsible for the section 8.4 NaN tangents of a
  known infinite mean; built-ins provide them and `check_family` checks
  them." 4.2 now points to the same override (design d104627). The guessed
  signature and `SummaryInfo` contents are spec; the custom
  `known_mean_log_logistic` now supplies its own NaN tangents; new tests
  `test_built_in_analytical_mean_reports_no_work` and
  `test_mean_carries_the_tangents_of_a_mean_with_info_override`.
- **Still not stated (not asserted):** whether a `ProportionalHazards`
  wrapper of a family with a `mean_with_info` override uses it (4.4 routes PH
  means through a PH closure or the generic method; the override is not a
  closure). How `SummaryInfo` is constructed (keyword fields, `int32` codes)
  remains Q-056.
- **Checked:** 8.1 (rev 54: "A family reports it by overriding
  `mean_with_info`; `mean` returns the value part of `mean_with_info`, so one
  override serves both"), 8.5 (`SummaryInfo` schema; "Construction stores
  these fields; the evaluating backend supplies consistent shapes, dtypes and
  values"), 8.4 (NaN tangents for known-infinite means), 4.2 (override table).
- **Question:** what signature must the override have, and how does it build
  its `SummaryInfo`? Does `mean(options=...)` forward `options` to it? Does a
  `ProportionalHazards` wrapper of such a family use the baseline's override?
  Must survjax force a NaN tangent on an override's `+inf` value (8.4), or is
  that the family's duty?
- **Guess (tested):** `def mean_with_info(self, *, options=None)` returning
  `(values, survjax.SummaryInfo(status=..., error_estimate=...,
  num_levels=..., num_solver_steps=..., num_bracket_steps=...))` with `int32`
  status codes from `SummaryInfo.Status`, zero counts and a NaN error
  estimate, all of shape `batch_shape` (`spec_families._override_info`).
  `mean()` and `mean(options=SummaryOptions())` return the override's value
  part. Nothing is asserted about PH wrappers of such a family or about the
  tangent of an override's value.
- **Depends:** `spec_families.known_mean_log_logistic`,
  `spec_families.mean_with_info_sentinel`; `test_log_logistic_mean_override_*`,
  `test_mean_returns_the_value_part_of_a_mean_with_info_override`.

## Q-068: status of `quantile(1)` on an improper family (spec 8.5; rev 54, 56)

- **Status:** settled by spec rev 57 (8.5; design 83a8e82): "On an improper
  family, a quantile that is `+inf` because `p >= 1 - S(+inf)` has status
  `known_infinite`, including `quantile(1)` (this rule takes precedence over
  the endpoint rule)". The reading below was right; the test keeps its
  assertion and is retagged `rev 57`.
- **Checked:** 8.5 (rev 54: "Exact endpoint values (`rmst(0)=0`, endpoint
  quantiles from `support()`) have status `converged`"; rev 56: "On an
  improper family, a quantile that is `+inf` because `p >= 1 - S(+inf)` has
  status `known_infinite`; `quantile(1)=+inf` on a proper family is an
  endpoint and `converged`"), 8.4 (both endpoint and known-infinite quantiles
  get NaN tangents, so the tangent is NaN either way).
- **Question:** on an improper family with the default support, `quantile(1)
  = +inf` is both a `support()` endpoint (rev 54: `converged`) and a quantile
  with `p >= 1 - S(+inf)` (rev 56: `known_infinite`). Which status applies?
- **Reading (tested):** `known_infinite`. Rev 56 restricts the `converged`
  endpoint case to "a proper family", which only makes sense if the improper
  `p = 1` falls under the improper rule. The rev 54 test
  `test_improper_family_upper_endpoint_quantile_reports_converged` is renamed
  `test_improper_family_upper_endpoint_quantile_reports_known_infinite`. If
  the maintainer means `converged`, flip that one assertion.
- **Depends:** `test_improper_family_upper_endpoint_quantile_reports_known_infinite`.

## Q-069: `from_log` keywords of the brought-forward families (spec 4.1; rev 55)

- **Checked:** 4.1 ("Log-parameter constructors take unconstrained log
  parameters ... `Weibull.from_log(log_shape=..., log_scale=...)`,
  `Exponential.from_log(log_rate=...)` and equivalents for the other
  families"; "Values and parameter gradients equal those of the ordinary
  constructor wherever `exp` is finite"), 4.6, 12.2 (rev 55: the log-normal
  family is enabled under the full contract of 4.1).
- **Question:** what are the keyword names of `LogNormal.from_log`? `meanlog`
  is already unconstrained (`real`), so does it keep its name, and is the
  scale `log_sdlog`?
- **Guess:** `LogNormal.from_log(meanlog=..., log_sdlog=...)`, with
  `sdlog = exp(log_sdlog)` and `meanlog` stored unchanged
  (`helpers.lognormal_from_log`); `LogLogistic.from_log(log_shape=...,
  log_scale=...)`, the Weibull keywords, since its constructor keywords are
  also `shape` and `scale` (`helpers.loglogistic_from_log`).
- **Depends:** `test_lognormal_from_log_*`, `test_loglogistic_from_log_*`
  (construction), and in both families' files
  `test_from_log_overflow_is_reported_as_parameter_issue` (constraints),
  `test_from_log_gradient_follows_the_chain_rule` (derivatives).

## Q-070: `check_family` for the brought-forward families (spec 4.2, 12.1, 12.2; rev 55, 56)

- **Status:** settled by spec rev 57 (12.2; design 83a8e82): each
  brought-forward family is gated by the 12.1 built-in rows "except
  `check_family`, which arrives with Slice D and must then pass on both
  families". `check_family` is in neither `family.lognormal` nor
  `family.loglogistic`, so no test calls it; nothing changed.
- **Checked:** 12.1 (Families and wrappers: "`check_family` on every
  built-in"), 12.2 (rev 55: each brought-forward family "is gated by the
  section 12.1 rows that apply to a built-in family"; Slice D still lists
  `check_family`), 8.1 (rev 56: "`check_family` checks them", the NaN
  tangents of a known infinite mean), the role table (no feature names
  `check_family`).
- **Question:** is `survjax.testing.check_family` part of `family.lognormal`
  / `family.loglogistic` (so `check_family(LogNormal(...))` must pass), or
  does it stay in Slice D with its own feature?
- **Guess:** none; `check_family` is not called. The identities it checks
  (`logaddexp(log S, log F) = 0`, `log_prob` consistency, boundary limits,
  finite parameter gradients) are covered directly by the family tests.
- **Depends:** nothing.

## Q-071: a cut search that runs reports at least one solver step (spec 8.3, 8.5; rev 58)

- **Checked:** 8.3 (rev 58: "The cut is needed only where `log S(tau)` is
  below the floor; otherwise there is no search ... Any iterate with
  `floor - 1 <= log S(c) <= floor` is accepted ... Its steps count against
  `max_solver_steps` and are reported in `num_solver_steps` (zero where no
  search runs)"), 8.2 (the iteration: Newton steps in `u = log t`, bisection
  fallback; the bracket "starts at `log reference_time()`"), 8.5 (counts
  "record ... solver steps ... used per entry").
- **Question:** the spec fixes neither where the cut search starts nor
  whether an accepted starting point counts as a step. Could a search that
  runs report `num_solver_steps == 0`?
- **Guess:** no: where `log S(tau)` is far below the floor, the search reports
  between 1 and `max_solver_steps` steps. The cases are chosen so that
  neither bracket end, nor the arithmetic or logarithmic midpoint of the
  bracket, lies in the acceptance window, so only an unusual start could be
  accepted without a step.
- **Depends:** `test_cut_search_steps_are_counted`,
  `test_cut_search_counts_are_per_entry` (`test_summary_rmst_tail_cut.py`).

## Q-072: the cut search's `work_exhausted` cannot be forced (spec 8.3, 8.5; rev 58)

- **Checked:** 8.3 (rev 58: "an entry whose search exhausts them has status
  `work_exhausted` and a NaN value"), 8.2, 8.5 (`max_solver_steps` is a
  static `SummaryOptions` field, default 64; no minimum is stated).
- **Question:** no static `max_solver_steps` is guaranteed to exhaust the cut
  search, because the starting point is open (Q-071) and any iterate in the
  acceptance window is accepted. For exponential- and Weibull-like families
  `log H` is linear in `log t`, so one Newton step reaches the root exactly.
  Should the spec fix the start (for example `min(tau, s)`, like 8.2's
  `log reference_time()`), so that a test can force exhaustion? Is
  `max_solver_steps=1` (or 0) a valid option value?
- **Guess:** `SummaryOptions(max_solver_steps=1)` is valid. The tight-limit
  tests use a Gompertz-like family (`spec_families.gompertz_like`, `log H`
  nonlinear in `log t`, window about `1/88` (float32) or `1/709` (float64)
  wide in `t` on the bracket `[1, 50]`), where one step is very unlikely to
  suffice. They assert only what holds either way: at most one step;
  `converged` with the right value or `work_exhausted` with NaN; the other
  entries unchanged in values, counts, status and shared-parameter
  derivatives. The NaN-tangent check runs only if the entry is exhausted and
  is otherwise skipped (reported as a skip).
- **Depends:** `test_cut_search_under_a_one_step_limit_is_consistent`,
  `test_cut_search_limit_isolates_other_entries`,
  `test_exhausted_cut_search_entry_has_a_nan_tangent`.

## Q-073: name and schema of the Slice B flexsurv fixture (spec 10.4, 12.1, 12.2; rev 59)

- **Checked:** 10.4 (rev 59, "Slice B likelihood fixture": "The spec tester
  defines the file name and schema; the generator writes that schema"), 10.5,
  12.1 ("Independent references"), 12.2 (Slice B gate), 2, 4.1.
- **Definition (not a guess, as for Q-039):** one JSON file
  `tests/spec/fixtures/slice_b_likelihood.json`, laid out as in
  `tests/spec/fixtures/README.md`: one shared dataset `data` (`cell`, `kind`,
  `time`, `time2`, `left_trunc`, `right_trunc`; `null` for `+inf` / absent),
  and `families.<weibull|lognormal|loglogistic>.vectors[]` with `name`,
  `parameters` in survjax's natural names, and per-row `loglik`.
- **Readings the spec leaves open:**
  - One dataset shared by all families and vectors (10.4 says "A synthetic
    dataset"). The generator must then keep `H <= 15` at every point under
    every vector.
  - "Finite intervals are not narrower than `1e-4` times their upper bound"
    is applied to interval rows and also to finite truncation windows
    (`U - L >= 1e-4 U`), because `Z` is a difference of probabilities in
    flexsurv too. If only interval rows are meant, drop the window half of
    `test_fixture_intervals_are_not_too_narrow`.
  - Every cell has at least one row; 10.4 fixes no count.
  - The fixture's values are also checked against 50-digit mpmath of spec 2
    at the 12.1 float64 tolerances (a malformed-fixture guard). If flexsurv
    itself is less accurate than that within `H <= 15` and width `1e-4`,
    survjax could not meet the same target either, and 10.4's targets would
    need revisiting.
- **Depends:** every test in `test_observation_censoring_flexsurv_fixture.py`.

## Q-074: markers and dtype scope of the Slice B flexsurv fixture tests (spec 10.4, 12.1, 12.2; rev 59)

- **Checked:** role table (`observation.censoring`, `observation.truncation`,
  `family.lognormal`, `family.loglogistic`), 10.4 ("Targets: section 12.1
  float64 tolerances"), 12.1, 12.2.
- **Guess (maintainer instruction for Weibull, extended to the other
  families):** untruncated exact/right rows carry `observation.censoring`
  for all three families, not only Weibull, so the fixture as a whole is part
  of the Slice B batch and an already-enabled `family.lognormal` /
  `family.loglogistic` gate does not start failing while the fixture is
  still missing. Truncated exact/right rows: `observation.truncation`;
  truncated left/interval rows: both; plus the family feature for log-normal
  and log-logistic. The fixture integrity tests (no survjax calls) carry
  `observation.censoring`. The whole-batch tests carry both Slice B features.
- **Guess:** the survjax comparisons run only in float64 and are skipped in
  float32 (reported as skips). 10.4 sets only float64 targets, and float32
  rounding of the endpoints of a `1e-4`-wide interval (relative `6e-8`)
  changes its log mass by up to about `6e-8 / 1e-4 = 6e-4`, beyond the float32
  `rtol=1e-4`. Float32 Slice B values stay covered by the mpmath tests.
- **Depends:** every survjax comparison test in
  `test_observation_censoring_flexsurv_fixture.py`.
