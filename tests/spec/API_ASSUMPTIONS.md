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

## Q-002: normalized fields are attributes

- **Checked:** 3.2 ("five public fields" with names), 6.3.
- **Guess:** `data.exact`, `data.lower`, `data.upper`, `data.left_trunc`,
  `data.right_trunc` are attributes returning arrays. The issue bitfield leaf
  has no specified public name; tests reach it only through
  `jax.tree_util.tree_leaves` (it is the one integer leaf).
- **Depends:** nearly all tests (`helpers.field`, `helpers.assert_fields`).

## Q-003: exception type for rejected construction

- **Checked:** 3.1, 3.3, 6.1, 6.2 ("raises descriptive structural errors";
  "fails clearly"; static misuse "raises").
- **Guess:** `ValueError` or `TypeError` (`helpers.REJECTED`). Tests accept
  either and do not inspect messages.
- **Depends:** every `pytest.raises(REJECTED)` in `test_a1_dtype.py`,
  `test_a2_*.py`, and `test_raise_if_invalid_*` in `test_a3_data_issues.py`.

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

- **Checked:** 3.1 ("`time2` is an interval's upper endpoint"), 6.2
  (`missing_time2`, `invalid_time2`).
- **Gap:** the spec does not say what `time2` must hold on exact, right- or
  left-censored rows when an array `time2` is supplied for mixed data.
- **Guess:** NaN is accepted there and ignored (R's `Surv(type="interval")`
  convention). Tests never put a finite non-NaN `time2` on a non-interval row.
- **Depends:** `_mixed()` in `test_a1_representation.py` and all tests using it,
  `test_fields_are_usable_under_vmap`; in A2 `test_all_flags_false_row_is_exact`,
  `test_event_with_all_false_interval_flag_is_rejected` (NaN there must not
  be the reason for rejection), `test_interval_with_nan_time2_is_rejected`
  (row 0), `test_mixed_truncated_rows_normalize_identically`,
  `test_constructors_normalize_identically_property`,
  `test_outside_interval_censored_raises_unsupported`.

## Q-007: unsupported-feature error for `outside_interval_censored`

- **Checked:** 3.1 ("recognized only to raise an explicit unsupported-feature
  error; it is not in the normal signature"), 1 (outside v0).
- **Guess:** passing `outside_interval_censored=` as a keyword to
  `SurvivalData(...)` raises `NotImplementedError` (`helpers.UNSUPPORTED`),
  whatever its values. Whether `from_bounds` also recognizes it is not stated;
  not tested.
- **Depends:** `test_outside_interval_censored_raises_unsupported`,
  `test_all_false_outside_interval_censored_still_raises_unsupported`.

## Q-008: right censoring at or before entry when the set is `(0, +inf]` (spec inconsistency)

- **Checked:** 2 (structural cases), 3.2 (derived kinds), 3.3.
- **Conflict:** section 2 says right censoring at `c <= L` is invalid
  (`censored_before_entry`). With no truncation `L = 0`, so right censoring at
  `c = 0` satisfies `c <= L`. Section 3.3 says the same row ("right censoring
  at 0, or `from_bounds(0, inf)`") is valid and flagged `uninformative`.
- **Related:** kinds are derived views (`upper == inf` means right), so
  `from_bounds(0, inf)` and an interval `(a, +inf]` with `a <= L` are also
  "right censored at or before entry" under section 2 and "a censored set
  covering the whole window" under section 2's uninformative bullet. The spec
  does not say which rule wins, or whether the 3.3 exception applies only when
  `L = 0`.
- **Tests:** none assert either outcome. `test_a2_boundaries.py` and
  `test_a3_data_issues.py` test censored-before-entry only with `L > 0` and
  finite `c`, and uninformative only for left-censored or finite-interval rows.

## Q-009: `SurvivalData.with_issues` signature, and no bounds counterpart

- **Checked:** 6.2 (`SurvivalData.with_issues(...) -> (data, DataIssues)`), 3.1.
- **Guess:** `with_issues` is a classmethod/staticmethod taking exactly the flag
  constructor's arguments, `with_issues(time, event=None, *, right_censored=...,
  left_censored=..., interval_censored=..., time2=..., left_trunc=...,
  right_trunc=..., dtype=...)`, and returns a 2-tuple (`helpers.with_issues`).
- **Gap:** the spec names no pure counterpart for `from_bounds` (for example
  `from_bounds_with_issues`). Bounds-constructor rows are therefore tested
  only through the eager constructor. Also unstated: whether `with_issues`
  raises (static misuse) or reports issues for `event` combined with flags, or
  for `interval_censored` without `time2`; not tested.
- **Depends:** every test in `test_a3_data_issues.py`,
  `test_indexing_carries_every_leaf_including_issue_bitfield`, and
  `_traced`/`test_eager_with_issues_computes_flags_from_values` in
  `test_a3_static_metadata.py`.

## Q-010: which DataIssues field reports each rule

- **Checked:** 2, 3.1, 3.3, 6.2. The spec names the fields but assigns only
  some rules to a field explicitly (`invalid_time` for right/left censoring at
  +inf; `censored_before_entry` for `c<=L`; `uninformative` for full coverage).
- **Guess (mapping by field name):**
  - `invalid_indicator`: an indicator not exactly 0/1 (including NaN);
  - `conflicting_indicators`: more than one true flag on a row;
  - `missing_time2`: interval row whose `time2` is NaN;
  - `invalid_time`: exact time zero, negative or infinite; negative censoring
    time; censoring at +inf;
  - `invalid_interval`: interval row with `time >= time2`;
  - `invalid_truncation`: `L` negative, infinite or NaN; `U` NaN; `L >= U`;
  - `exact_outside_truncation`: exact `t` outside `L<t<=U`;
  - `censored_before_entry`: right censoring at `c <= L` (with `L > 0`);
  - `empty_observation`: a censored set whose intersection with `(L, U]` is
    empty (right at `c>=U`, left at `c<=L`, interval `b<=L` or `a>=U`).
  Tests assert the named mask is set on the offending row and never assert
  that other masks are clear there (rules may overlap). NaN exact time is
  tested only through `invalid_rows`. `invalid_time2` has no test: the spec
  does not say which input sets it rather than `missing_time2` or
  `invalid_interval`.
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

## Q-012: error for the ordinary constructor under tracing

- **Checked:** 6.2 ("raises at trace time and points to `with_issues`").
- **Guess:** the exception type is open (tests accept any `Exception`); "points
  to `with_issues`" means the message contains the text `with_issues`. The
  spec says this of "the ordinary constructor"; the tests assume
  `from_bounds` behaves the same.
- **Depends:** `test_ordinary_constructor_under_jit_raises_pointing_to_with_issues`,
  `test_ordinary_constructor_under_vmap_raises_pointing_to_with_issues`,
  `test_bounds_constructor_under_jit_raises_pointing_to_with_issues`.

## Q-013: decoding the bitfield of an existing SurvivalData

- **Checked:** 3.2 ("`DataIssues` decodes the bitfield"), 6.2.
- **Gap:** no public call is named that turns a `SurvivalData` (for example a
  slice, or data rebuilt from leaves) back into `DataIssues`. Tests decode only
  the `DataIssues` returned by `with_issues`, and check that indexing carries
  the bitfield leaf along unchanged.
- **Depends:** none (a gap, not a guess).

## Q-014: observing static metadata

- **Checked:** 3.2 (flags in PyTree auxiliary data; "datasets with different
  flags have different tree structures and compile separately").
- **Guess:** the auxiliary data holds nothing that varies with values other
  than these flags, so equal flags mean equal `jax.tree_util.tree_structure`
  (and one jit trace for equal shapes and dtypes). The flags have no public
  accessor, and the tests do not read them. Left and right truncation are
  tested separately against "none" but not against each other, because "whether
  any left or right truncation is present" could be one flag or two.
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
