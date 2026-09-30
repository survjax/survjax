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
  `chronox_jax.InvalidDataError` (`ValueError` subclass). Structural argument
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

## Q-009: issue-returning convenience constructors

- **Checked:** 6.2, 10.2 and 12.1, revision 9.
- **Resolved:** only `SurvivalData(...)` and `SurvivalData.from_bounds(...)`
  construct observations. Each returns data; `data.issues` exposes diagnostics.
  `helpers.flag_data_and_issues` and `helpers.bounds_data_and_issues` are test
  helpers that compose these public operations, not Chronox API assumptions.
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

- **Checked:** 3.2, 6.2 and 13, revision 9.
- **Resolved:** both ordinary constructors accept traced observation inputs;
  the old `TracedInputError` rejection requirement is superseded. Calling
  `raise_if_invalid()` on traced diagnostics must fail explicitly.
- **Open:** the exception type for that misuse and the export lifecycle of
  `TracedInputError` are unspecified. Tests assert explicit failure without
  choosing a subtype, and do not require or reject the legacy export.
- **Depends:** `test_traced_diagnostics_cannot_raise_at_host_checkpoint` in
  `test_a3_tracing.py`; no test depends on the legacy name.
## Q-013: decoding the bitfield of an existing SurvivalData

- **Checked:** 3.2 and 6.2, revision 9.
- **Resolved:** the `data.issues` property decodes diagnostics on constructed,
  indexed, rebuilt and transformed data. The old callable spelling is
  superseded; no test requires or forbids a compatibility alias.
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
