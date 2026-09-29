# API assumptions

Entries Q-001, Q-002, ... (see CLAUDE.md rule 4). Each records the spec sections
checked, the guess made (or the question left open), and the tests that depend
on it. Guessed exception types and call shapes are centralised in
`helpers.py`, so resolving an entry usually means editing one line there.

## Q-001: delivery step numbering A1/A2/A3

- **Checked:** 12.2 (Slice table), CLAUDE.md.
- **Gap:** spec 12.2 defines Slices A–F; it has no numbered steps A1, A2, A3.
  The maintainer assigned "A1, A2 and A3 (SurvivalData representation, dtype
  policy, constructors, DataIssues, indexing, static metadata)".
- **Guess:** A1 = normalized representation, PyTree storage and dtype policy
  (3.2 storage, 3.3 dtype); A2 = both constructors and the boundary rules
  (3.1, 3.3 boundaries, section 2 structural cases); A3 = `DataIssues` and
  `with_issues`, eager versus traced construction, indexing, static metadata and
  compile counts (3.2, 6.2, 6.3).
- **Depends:** every `@pytest.mark.step(...)` in `test_a1_*`, `test_a2_*`,
  `test_a3_*`, and therefore `SPEC_THROUGH` filtering.
- **Status (2026-09-29):** deferred by the maintainer. Fixing the step plan
  needs a role change and a matching spec change, so the guess above stays in
  force for now.

## Q-002: normalized fields are attributes

- **Checked:** 3.2 ("five public fields" with names), 6.3.
- **Guess:** `data.exact`, `data.lower`, `data.upper`, `data.left_trunc`,
  `data.right_trunc` are attributes returning arrays. The issue bitfield leaf
  has no specified public name; tests compare it only as one of the
  `jax.tree_util.tree_leaves` and read its meaning through `data.issues()`
  (Q-013, resolved).
- **Depends:** nearly all tests (`helpers.field`, `helpers.assert_fields`).

## Q-003: exception type for rejected construction

- **Status:** resolved by spec rev 7 (maintainer
  decision; 6.2 "Exception types", 3.3). Invalid observation rows raise
  `chronox_jax.InvalidDataError` (a `ValueError` subclass) from the ordinary
  constructors and `raise_if_invalid()`; argument misuse raises `TypeError`;
  an unavailable explicit dtype request raises `ValueError`, not
  `InvalidDataError`.
- **In the tests:** `helpers.INVALID_DATA`, `helpers.MISUSE` and
  `helpers.raises_dtype_request()`. `INVALID_DATA` falls back to a class that
  is never raised if `chronox_jax.InvalidDataError` is missing, so only the
  tests expecting it fail (plus `test_invalid_data_error_is_a_public_value_error`).
  Messages are still not inspected.
- **Classification calls made from the spec:** string and object-with-None
  indicators are nonnumeric fields (misuse, `TypeError`); an omitted `time2`
  on an interval row is a row issue (`missing_time2`, so `InvalidDataError`).
- **Depends:** every `pytest.raises(INVALID_DATA)`, `pytest.raises(MISUSE)`
  and `raises_dtype_request()` in `test_a1_dtype.py`, `test_a2_*.py` and
  `test_a3_*.py`.

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
  `test_two_call_sites_warn_once_each`.

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

## Q-009: pure constructors `with_issues` and `from_bounds_with_issues`

- **Status:** resolved by spec rev 6 (6.2). Both pure constructors exist, take
  exactly the arguments of their ordinary counterparts, and return
  `(data, DataIssues)`; argument misuse (`event` with a censoring flag,
  nonnumeric fields, incompatible shapes) raises
  in all four constructors, independent of values and tracing.
- **Remaining guess:** both are reached as attributes of `SurvivalData`
  (`helpers.with_issues`, `helpers.from_bounds_with_issues`), as the spec's
  dotted names suggest, and the pair is a Python `tuple`
  (`test_with_issues_returns_data_and_issues_pair`,
  `test_from_bounds_with_issues_returns_data_and_issues_pair`).
- **Depends:** every test in `test_a3_data_issues.py` and
  `test_a3_bounds_issues.py`, the pure-constructor tests in
  `test_a3_tracing.py` and `test_a3_static_metadata.py`, and
  `test_indexing_carries_every_leaf_including_issue_bitfield`.

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

- **Checked:** 6.2 ("`ok` is the scalar reduction"; "a scalar validity
  reduction").
- **Guess:** `ok` is true exactly when no row is in `invalid_rows`, so rows
  that are only `uninformative` leave it true.
- **Depends:** `test_ok_is_true_with_only_uninformative_rows`,
  `test_ok_is_false_when_any_row_is_invalid`, `_assert_rule` and
  `_assert_uninformative` helpers in `test_a3_data_issues.py`.

## Q-012: error type for the ordinary constructors under tracing

- **Status:** resolved by spec rev 7 (6.2) (maintainer
  decision). A traced observation input to an ordinary constructor raises
  `chronox_jax.TracedInputError`, a subclass of `TypeError`, at trace time; its
  message names the pure counterpart (`with_issues` or
  `from_bounds_with_issues`). Argument misuse raises `TypeError`, not
  `TracedInputError`.
- **In the tests:** `helpers.TRACED_INPUT` (falls back to a class that is
  never raised if `chronox_jax.TracedInputError` is missing) with a message
  match on the counterpart's name; `helpers.raises_misuse()` expects
  `TypeError` and rejects `TracedInputError`, so the two misuse-first tests now
  distinguish the errors.
- **Depends:** the trace-time rejection tests in `test_a3_tracing.py` and
  `test_a3_data_issues.py`,
  `test_ordinary_constructor_raises_misuse_error_before_traced_input_error`,
  `test_bounds_constructor_raises_misuse_error_before_traced_input_error`,
  `test_traced_input_error_is_a_public_type_error`.

## Q-013: decoding the bitfield of an existing SurvivalData

- **Status:** resolved by spec rev 7 (maintainer
  decision): `data.issues() -> DataIssues` decodes the bitfield of any
  `SurvivalData`, including slices, data returned from transformed functions
  and data rebuilt from leaves (3.2, 6.2).
- **Depends:** every test in `test_a3_issue_decoding.py`.

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
