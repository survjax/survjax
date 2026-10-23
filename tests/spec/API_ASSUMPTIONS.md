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

## Q-005: immutability error

- **Checked:** 3.2 ("immutable registered PyTree").
- **Guess:** assigning an attribute raises `AttributeError` (as a frozen
  dataclass does; `helpers.IMMUTABLE`).
- **Depends:** `test_data_is_immutable`.

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

## Q-037: how Slice A fails explicitly for unimplemented modes (spec 12.2, rev 28) — resolved by rev 31

- **Resolved by rev 31 (decision 0044):** unsupported rows return NaN in both
  entry points, eagerly and under `jit`/`vmap`, without a value-dependent
  exception; supported rows keep their values and derivatives; no data or
  parameter issue is added, and the 7.1 recipe gains a mode-support check
  (`helpers.supported_mode`, `helpers.unsupported_nan`). The
  `NotImplementedError` alternative is withdrawn.
- **Depends:** nothing guessed remains; `test_a6_unsupported_modes.py` follows
  the spec text.

## Q-038: likelihood module placement (spec 6.3, 10.2, rev 28; narrowed by rev 29)

- **Checked:** 6.2, 6.3, 7.1, 10.2.
- **Resolved by rev 29 (decision 0043):** `LikelihoodIssues` and
  `log_likelihood_with_info` are removed, so the guess about the info class
  name is withdrawn and `test_likelihood_issues_fields_by_name` deleted.
  `test_likelihood_has_no_with_info_variant` and
  `test_likelihood_issues_container_is_removed` assert that neither name is
  exposed at top level or in `survjax.likelihood`.
- **Still a guess:** `survjax.likelihood` (named in the 10.2 layout) holds the
  same `log_likelihood` object as the top level.
- **Depends:** `test_likelihood_module_holds_the_entry_point`,
  `test_likelihood_has_no_with_info_variant`,
  `test_likelihood_issues_container_is_removed`.


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
