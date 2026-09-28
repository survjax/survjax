"""A3: DataIssues, ``SurvivalData.with_issues`` and eager versus traced construction (spec 3.2, 6.2, 6.3).

Each attribution test builds three rows: two valid exact rows and one row that
breaks a single rule, and checks the rule's mask and ``invalid_rows``. Other
masks are not asserted false on the broken row, because one bad row can break
several rules at once (for example t=0 is both an invalid time and outside
(0, inf]). The exceptions are the orderings rev 5 fixes: a malformed interval
lower or upper endpoint is not also ``invalid_interval``, and right censoring at
entry is not ``censored_before_entry``.
"""

import jax
import numpy as np
import pytest
from helpers import (
    FIELDS,
    INF,
    MASK_FIELDS,
    NAN,
    REJECTED,
    RULE_FIELDS,
    assert_mask,
    b,
    f,
    leaves,
    mask,
    one_hot,
    with_issues,
)

import chronox_jax as cx

NONE = np.zeros(3, dtype=bool)
ROW1 = one_hot(3, 1)


def _assert_rule(issues, name, expected=ROW1):
    assert_mask(issues, name, expected)
    assert_mask(issues, "invalid_rows", expected)
    assert not bool(np.asarray(issues.ok))


def _assert_uninformative(issues, expected=ROW1):
    assert_mask(issues, "uninformative", expected)
    assert_mask(issues, "invalid_rows", NONE)
    assert bool(np.asarray(issues.ok))


def _flags(time, *, right=None, left=None, interval=None, time2=None, lt=None, rt=None):
    """with_issues via explicit flags; row kinds default to exact."""
    kw = {}
    if right is not None:
        kw["right_censored"] = np.asarray(right)
    if left is not None:
        kw["left_censored"] = np.asarray(left)
    if interval is not None:
        kw["interval_censored"] = np.asarray(interval)
        kw["time2"] = f(time2)
    if lt is not None:
        kw["left_trunc"] = f(lt)
    if rt is not None:
        kw["right_trunc"] = f(rt)
    return with_issues(f(time), **kw)


# ------------------------------------------------------- container shape


@pytest.mark.step("A3")
def test_with_issues_returns_data_and_issues_pair():
    """spec 6.2, rev 4: `SurvivalData.with_issues(...) -> (data, DataIssues)`."""
    result = with_issues(f([1.0, 2.0]), b([True, False]))
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], cx.SurvivalData)


@pytest.mark.step("A3")
def test_with_issues_data_matches_ordinary_constructor_on_valid_input():
    """spec 6.2, rev 4: with_issues returns normalized data, the same as the eager constructor for valid input."""
    time, event, lt = f([1.0, 2.0, 3.0]), b([True, False, True]), f([0.5, 0.0, 1.0])
    data, _ = with_issues(time, event, left_trunc=lt)
    eager = cx.SurvivalData(time, event, left_trunc=lt)
    for name in FIELDS:
        np.testing.assert_array_equal(np.asarray(getattr(data, name)), np.asarray(getattr(eager, name)), err_msg=name)


@pytest.mark.step("A3")
def test_with_issues_does_not_raise_on_invalid_rows():
    """spec 6.2, rev 4: with_issues is pure: invalid rows give issues, not exceptions, even eagerly."""
    _, issues = with_issues(f([1.0, -1.0, 2.0]), np.array([1, 2, 0], dtype=np.int32))
    assert not bool(np.asarray(issues.ok))


@pytest.mark.step("A3")
def test_with_issues_raises_on_nonnumeric_time():
    """spec 6.2, rev 5: nonnumeric time fields cannot form an object and raise instead of setting a row mask."""
    with pytest.raises(REJECTED):
        with_issues(np.array(["1.0", "2.0"]))


@pytest.mark.step("A3")
def test_with_issues_raises_on_incompatible_shapes():
    """spec 6.2/3.2, rev 5: incompatible shapes cannot form an object and raise instead of setting a row mask."""
    with pytest.raises(REJECTED):
        with_issues(f([1.0, 2.0, 3.0]), right_censored=b([[True], [False], [True]]))


@pytest.mark.step("A3")
def test_issue_masks_have_observation_shape():
    """spec 6.2, rev 4: per-rule masks have the data shape."""
    _, issues = with_issues(f([1.0, 2.0, 3.0]), b([True, False, True]))
    for name in MASK_FIELDS:
        assert np.shape(getattr(issues, name)) == (3,), name


@pytest.mark.step("A3")
def test_issue_masks_keep_multidimensional_observation_shape():
    """spec 6.2, rev 4: per-rule masks have the data shape, (2, 3) here."""
    _, issues = with_issues(f([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    for name in MASK_FIELDS:
        assert np.shape(getattr(issues, name)) == (2, 3), name


@pytest.mark.step("A3")
def test_ok_is_a_scalar():
    """spec 6.2, rev 4: `ok` is the scalar validity reduction."""
    _, issues = with_issues(f([1.0, 2.0, 3.0]))
    assert np.shape(issues.ok) == ()


@pytest.mark.step("A3")
def test_valid_data_has_no_issues():
    """spec 6.2, rev 4: valid data sets no rule mask, no invalid row, and `ok` is true."""
    _, issues = with_issues(
        f([1.0, 2.0, 3.0]), b([True, False, True]), left_trunc=f([0.5, 0.5, 0.0]), right_trunc=f([5.0, INF, INF])
    )
    for name in MASK_FIELDS:
        assert_mask(issues, name, NONE)
    assert bool(np.asarray(issues.ok))


# ------------------------------------------------------ invalid_indicator


@pytest.mark.step("A3")
def test_event_code_two_sets_invalid_indicator():
    """spec 6.2/3.1, rev 4: other indicator codes set `invalid_indicator`."""
    _, issues = with_issues(f([1.0, 2.0, 3.0]), np.array([1, 2, 0], dtype=np.int32))
    _assert_rule(issues, "invalid_indicator")


@pytest.mark.step("A3")
def test_r_style_coding_flags_each_two_as_invalid_indicator():
    """spec 3.1, rev 4: R-style 1/2 event coding is not inferred; each 2 is an invalid indicator."""
    _, issues = with_issues(f([1.0, 2.0, 3.0]), np.array([1, 2, 2], dtype=np.int32))
    _assert_rule(issues, "invalid_indicator", one_hot(3, 1, 2))


@pytest.mark.step("A3")
def test_nan_event_sets_invalid_indicator():
    """spec 6.2/3.1, rev 4: a NaN indicator sets `invalid_indicator`."""
    _, issues = with_issues(f([1.0, 2.0, 3.0]), f([1.0, NAN, 0.0]))
    _assert_rule(issues, "invalid_indicator")


@pytest.mark.step("A3")
def test_fractional_flag_sets_invalid_indicator():
    """spec 6.2/3.1, rev 4: a censoring flag other than exactly 0 or 1 sets `invalid_indicator`."""
    issues = _flags([1.0, 2.0, 3.0], right=f([0.0, 0.5, 0.0]))[1]
    _assert_rule(issues, "invalid_indicator")


# ------------------------------------------------- conflicting_indicators


@pytest.mark.step("A3")
def test_two_true_flags_set_conflicting_indicators():
    """spec 6.2/3.1, rev 4: more than one true flag on a row sets `conflicting_indicators`."""
    issues = _flags([1.0, 2.0, 3.0], right=b([False, True, False]), left=b([False, True, False]))[1]
    _assert_rule(issues, "conflicting_indicators")


# ----------------------------------------------------------- missing_time2


@pytest.mark.step("A3")
def test_nan_time2_on_interval_row_sets_missing_time2():
    """spec 6.2/3.1, rev 4: an interval row without an upper endpoint sets `missing_time2`."""
    issues = _flags([1.0, 2.0, 3.0], interval=b([False, True, False]), time2=[NAN, NAN, NAN])[1]
    _assert_rule(issues, "missing_time2")


@pytest.mark.step("A3")
def test_omitted_time2_on_interval_row_sets_missing_time2():
    """spec 6.2, rev 5: an active interval row with omitted `time2` sets `missing_time2`."""
    _, issues = with_issues(f([1.0, 2.0, 3.0]), interval_censored=b([False, True, False]))
    _assert_rule(issues, "missing_time2")


@pytest.mark.step("A3")
def test_nan_time2_on_inactive_rows_does_not_set_missing_time2():
    """spec 6.2, rev 5: `missing_time2` is for active interval rows only."""
    issues = _flags([1.0, 2.0, 3.0], interval=b([False, True, False]), time2=[NAN, 4.0, NAN])[1]
    assert_mask(issues, "missing_time2", NONE)


@pytest.mark.step("A3")
def test_nan_time2_is_not_also_invalid_interval():
    """spec 6.2, rev 5: presence is evaluated before interval order, so a missing upper is not a reversed interval."""
    issues = _flags([1.0, 2.0, 3.0], interval=b([False, True, False]), time2=[NAN, NAN, NAN])[1]
    assert_mask(issues, "invalid_interval", NONE)


# ----------------------------------------------------------- invalid_time2


@pytest.mark.step("A3")
def test_negative_time2_sets_invalid_time2():
    """spec 6.2, rev 5: an active interval `time2` that is negative sets `invalid_time2`."""
    issues = _flags([1.0, 2.0, 3.0], interval=b([False, True, False]), time2=[NAN, -1.0, NAN])[1]
    _assert_rule(issues, "invalid_time2")


@pytest.mark.step("A3")
def test_minus_infinite_time2_sets_invalid_time2():
    """spec 6.2, rev 5: an active interval `time2` of `-inf` sets `invalid_time2`."""
    issues = _flags([1.0, 2.0, 3.0], interval=b([False, True, False]), time2=[NAN, -INF, NAN])[1]
    _assert_rule(issues, "invalid_time2")


@pytest.mark.step("A3")
def test_negative_time2_is_not_also_invalid_interval():
    """spec 6.2, rev 5: endpoint domains are evaluated before interval order; one malformed upper is not a reversed interval."""
    issues = _flags([1.0, 2.0, 3.0], interval=b([False, True, False]), time2=[NAN, -1.0, NAN])[1]
    assert_mask(issues, "invalid_interval", NONE)
    assert_mask(issues, "missing_time2", NONE)


@pytest.mark.step("A3")
def test_minus_infinite_time2_is_not_also_invalid_interval_or_missing():
    """spec 6.2, rev 5: `-inf` is a domain error in `time2`, neither missing nor a reversed interval."""
    issues = _flags([1.0, 2.0, 3.0], interval=b([False, True, False]), time2=[NAN, -INF, NAN])[1]
    assert_mask(issues, "invalid_interval", NONE)
    assert_mask(issues, "missing_time2", NONE)


# ------------------------------------------------------------ invalid_time


@pytest.mark.step("A3")
def test_right_censoring_at_infinity_sets_invalid_time():
    """spec 3.3, rev 4: right censoring at +inf is invalid (`invalid_time`)."""
    issues = _flags([1.0, INF, 3.0], right=b([False, True, False]))[1]
    _assert_rule(issues, "invalid_time")


@pytest.mark.step("A3")
def test_left_censoring_at_infinity_sets_invalid_time():
    """spec 3.3, rev 4: left censoring at +inf is invalid (`invalid_time`)."""
    issues = _flags([1.0, INF, 3.0], left=b([False, True, False]))[1]
    _assert_rule(issues, "invalid_time")


@pytest.mark.step("A3")
def test_zero_exact_time_sets_invalid_time():
    """spec 3.3/6.2, rev 5: a zero exact time violates 3.3 and sets `invalid_time`."""
    _, issues = with_issues(f([1.0, 0.0, 3.0]))
    _assert_rule(issues, "invalid_time")


@pytest.mark.step("A3")
def test_negative_exact_time_sets_invalid_time():
    """spec 3.3/6.2, rev 5: a negative exact time violates 3.3 and sets `invalid_time`."""
    _, issues = with_issues(f([1.0, -1.0, 3.0]))
    _assert_rule(issues, "invalid_time")


@pytest.mark.step("A3")
def test_infinite_exact_time_sets_invalid_time():
    """spec 3.3/6.2, rev 5: an infinite exact time violates 3.3 and sets `invalid_time`."""
    _, issues = with_issues(f([1.0, INF, 3.0]))
    _assert_rule(issues, "invalid_time")


@pytest.mark.step("A3")
def test_negative_censoring_time_sets_invalid_time():
    """spec 3.3/6.2, rev 5: a negative censoring time violates 3.3 and sets `invalid_time`."""
    _, issues = with_issues(f([1.0, -1.0, 3.0]), b([True, False, True]))
    _assert_rule(issues, "invalid_time")


@pytest.mark.step("A3")
def test_negative_interval_lower_endpoint_sets_invalid_time():
    """spec 6.2, rev 5: `time` uses `invalid_time` on any row, including an interval row's lower endpoint."""
    issues = _flags([1.0, -1.0, 3.0], interval=b([False, True, False]), time2=[NAN, 2.0, NAN])[1]
    _assert_rule(issues, "invalid_time")


@pytest.mark.step("A3")
def test_infinite_interval_lower_endpoint_sets_invalid_time():
    """spec 6.2/3.3, rev 5: censored lower bounds must be finite; an interval row's `time=+inf` sets `invalid_time`."""
    issues = _flags([1.0, INF, 3.0], interval=b([False, True, False]), time2=[NAN, INF, NAN])[1]
    _assert_rule(issues, "invalid_time")


@pytest.mark.step("A3")
def test_infinite_interval_lower_endpoint_is_not_also_invalid_interval():
    """spec 6.2, rev 5: domains are evaluated before order, so a malformed lower is not a reversed interval (inf >= inf)."""
    issues = _flags([1.0, INF, 3.0], interval=b([False, True, False]), time2=[NAN, INF, NAN])[1]
    assert_mask(issues, "invalid_interval", NONE)


@pytest.mark.step("A3")
def test_negative_interval_lower_endpoint_is_not_also_invalid_interval():
    """spec 6.2, rev 5: a malformed lower endpoint is not also labelled a reversed interval (here -1 > -2)."""
    issues = _flags([1.0, -1.0, 3.0], interval=b([False, True, False]), time2=[NAN, -2.0, NAN])[1]
    assert_mask(issues, "invalid_interval", NONE)


@pytest.mark.step("A3")
def test_nan_exact_time_sets_invalid_time():
    """spec 6.2, rev 5: a NaN `time` sets `invalid_time` (no separate missing-value mask for the first endpoint)."""
    _, issues = with_issues(f([1.0, NAN, 3.0]))
    _assert_rule(issues, "invalid_time")


@pytest.mark.step("A3")
def test_nan_right_censoring_time_sets_invalid_time():
    """spec 6.2, rev 5: a NaN `time` sets `invalid_time` on a censored row too."""
    issues = _flags([1.0, NAN, 3.0], right=b([False, True, False]))[1]
    _assert_rule(issues, "invalid_time")


@pytest.mark.step("A3")
def test_nan_interval_lower_endpoint_sets_invalid_time():
    """spec 6.2, rev 5: a NaN `time` sets `invalid_time`, including an interval row's lower endpoint."""
    issues = _flags([1.0, NAN, 3.0], interval=b([False, True, False]), time2=[NAN, 2.0, NAN])[1]
    _assert_rule(issues, "invalid_time")


@pytest.mark.step("A3")
def test_nan_interval_lower_endpoint_is_not_also_invalid_interval():
    """spec 6.2, rev 5: presence is evaluated before order, so a NaN lower is not a reversed interval."""
    issues = _flags([1.0, NAN, 3.0], interval=b([False, True, False]), time2=[NAN, 2.0, NAN])[1]
    assert_mask(issues, "invalid_interval", NONE)


# -------------------------------------------------------- invalid_interval


@pytest.mark.step("A3")
def test_equal_interval_endpoints_set_invalid_interval():
    """spec 3.1/6.2, rev 5: an interval row with `lower >= upper` (here equal) sets `invalid_interval`."""
    issues = _flags([1.0, 2.0, 3.0], interval=b([False, True, False]), time2=[NAN, 2.0, NAN])[1]
    _assert_rule(issues, "invalid_interval")


@pytest.mark.step("A3")
def test_reversed_interval_endpoints_set_invalid_interval():
    """spec 3.1/6.2, rev 5: an interval row with `lower >= upper` (here reversed) sets `invalid_interval`."""
    issues = _flags([1.0, 2.0, 3.0], interval=b([False, True, False]), time2=[NAN, 1.5, NAN])[1]
    _assert_rule(issues, "invalid_interval")


@pytest.mark.step("A3")
def test_infinite_time2_is_a_valid_interval_upper():
    """spec 6.2, rev 5: a `+inf` interval `time2` is valid."""
    issues = _flags([1.0, 2.0, 3.0], interval=b([False, True, False]), time2=[NAN, INF, NAN])[1]
    assert_mask(issues, "invalid_interval", NONE)
    assert_mask(issues, "invalid_rows", NONE)


# ------------------------------------------------------ invalid_truncation


@pytest.mark.step("A3")
def test_negative_left_truncation_sets_invalid_truncation():
    """spec 3.3, rev 4: lower truncation must be nonnegative (`invalid_truncation`)."""
    issues = _flags([1.0, 2.0, 3.0], lt=[0.0, -0.5, 0.0])[1]
    _assert_rule(issues, "invalid_truncation")


@pytest.mark.step("A3")
def test_infinite_left_truncation_sets_invalid_truncation():
    """spec 3.3, rev 4: lower truncation must be finite (`invalid_truncation`)."""
    issues = _flags([1.0, 2.0, 3.0], lt=[0.0, INF, 0.0])[1]
    _assert_rule(issues, "invalid_truncation")


@pytest.mark.step("A3")
def test_nan_left_truncation_sets_invalid_truncation():
    """spec 3.2, rev 4: NaN truncation bounds are invalid (`invalid_truncation`)."""
    issues = _flags([1.0, 2.0, 3.0], lt=[0.0, NAN, 0.0])[1]
    _assert_rule(issues, "invalid_truncation")


@pytest.mark.step("A3")
def test_nan_right_truncation_sets_invalid_truncation():
    """spec 3.2, rev 4: NaN truncation bounds are invalid (`invalid_truncation`)."""
    issues = _flags([1.0, 2.0, 3.0], rt=[INF, NAN, INF])[1]
    _assert_rule(issues, "invalid_truncation")


@pytest.mark.step("A3")
def test_equal_truncation_bounds_set_invalid_truncation():
    """spec 3.3, rev 4: L<U is required (`invalid_truncation`)."""
    issues = _flags([1.0, 2.0, 3.0], right=b([False, True, False]), lt=[0.0, 3.0, 0.0], rt=[INF, 3.0, INF])[1]
    _assert_rule(issues, "invalid_truncation")


@pytest.mark.step("A3")
def test_reversed_truncation_bounds_set_invalid_truncation():
    """spec 3.3, rev 4: L<U is required (`invalid_truncation`)."""
    issues = _flags([1.0, 2.0, 3.0], right=b([False, True, False]), lt=[0.0, 4.0, 0.0], rt=[INF, 3.0, INF])[1]
    _assert_rule(issues, "invalid_truncation")


# ------------------------------------------------ exact_outside_truncation


@pytest.mark.step("A3")
def test_exact_time_at_left_truncation_sets_exact_outside_truncation():
    """spec 3.3, rev 4: membership is L<t<=U; t=L sets `exact_outside_truncation`."""
    issues = _flags([1.0, 0.5, 3.0], lt=[0.0, 0.5, 0.0])[1]
    _assert_rule(issues, "exact_outside_truncation")


@pytest.mark.step("A3")
def test_exact_time_below_left_truncation_sets_exact_outside_truncation():
    """spec 3.3, rev 4: membership is L<t<=U; t<L sets `exact_outside_truncation`."""
    issues = _flags([1.0, 0.25, 3.0], lt=[0.0, 0.5, 0.0])[1]
    _assert_rule(issues, "exact_outside_truncation")


@pytest.mark.step("A3")
def test_exact_time_above_right_truncation_sets_exact_outside_truncation():
    """spec 3.3, rev 4: membership is L<t<=U; t>U sets `exact_outside_truncation`."""
    issues = _flags([1.0, 4.0, 3.0], rt=[INF, 3.5, INF])[1]
    _assert_rule(issues, "exact_outside_truncation")


@pytest.mark.step("A3")
def test_exact_time_at_right_truncation_is_valid():
    """spec 3.3, rev 4: membership is L<t<=U; t=U is inside."""
    issues = _flags([1.0, 3.5, 3.0], rt=[INF, 3.5, INF])[1]
    assert_mask(issues, "exact_outside_truncation", NONE)
    assert_mask(issues, "invalid_rows", NONE)


# --------------------------------------------------- censored_before_entry


@pytest.mark.step("A3")
def test_right_censoring_before_entry_sets_censored_before_entry():
    """spec 2/6.2, rev 5: right-censored at c<L is invalid (`censored_before_entry`)."""
    issues = _flags([1.0, 0.5, 3.0], right=b([False, True, False]), lt=[0.0, 1.0, 0.0])[1]
    _assert_rule(issues, "censored_before_entry")


@pytest.mark.step("A3")
def test_interval_to_infinity_before_entry_sets_censored_before_entry():
    """spec 3.1/6.2, rev 5: an interval-flag row with `time2=+inf` and `time<L` sets `censored_before_entry`."""
    issues = _flags(
        [1.0, 0.5, 3.0], interval=b([False, True, False]), time2=[NAN, INF, NAN], lt=[0.0, 1.0, 0.0]
    )[1]
    _assert_rule(issues, "censored_before_entry")


@pytest.mark.step("A3")
def test_right_censoring_at_entry_does_not_set_censored_before_entry():
    """spec 6.2, rev 5: `censored_before_entry` applies to `lower<L`, not `lower=L`."""
    issues = _flags([1.0, 1.0, 3.0], right=b([False, True, False]), lt=[0.0, 1.0, 0.0])[1]
    assert_mask(issues, "censored_before_entry", NONE)


# -------------------------------------------------------- empty_observation


@pytest.mark.step("A3")
def test_right_censoring_at_upper_truncation_sets_empty_observation():
    """spec 2, rev 4: right-censored at c>=U is an empty intersection (`empty_observation`, Q-010)."""
    issues = _flags([1.0, 3.0, 2.0], right=b([False, True, False]), rt=[INF, 3.0, INF])[1]
    _assert_rule(issues, "empty_observation")


@pytest.mark.step("A3")
def test_left_censoring_at_left_truncation_sets_empty_observation():
    """spec 2, rev 4: (0, c] with c<=L is an empty intersection (`empty_observation`, Q-010)."""
    issues = _flags([1.0, 0.5, 2.0], left=b([False, True, False]), lt=[0.0, 0.5, 0.0])[1]
    _assert_rule(issues, "empty_observation")


@pytest.mark.step("A3")
def test_interval_ending_at_left_truncation_sets_empty_observation():
    """spec 2, rev 4: an interval ending at or before L is an empty intersection (`empty_observation`, Q-010)."""
    issues = _flags(
        [1.0, 0.25, 2.0], interval=b([False, True, False]), time2=[NAN, 0.5, NAN], lt=[0.0, 0.5, 0.0]
    )[1]
    _assert_rule(issues, "empty_observation")


@pytest.mark.step("A3")
def test_interval_starting_at_right_truncation_sets_empty_observation():
    """spec 2, rev 4: (a, b] with a>=U is an empty intersection (`empty_observation`, Q-010)."""
    issues = _flags(
        [1.0, 3.0, 2.0], interval=b([False, True, False]), time2=[NAN, 4.0, NAN], rt=[INF, 3.0, INF]
    )[1]
    _assert_rule(issues, "empty_observation")


# ------------------------------------------------------------ uninformative


@pytest.mark.step("A3")
def test_right_censoring_at_entry_is_uninformative_and_valid():
    """spec 2/6.2, rev 5: right-censored at c=L>0 is valid and flagged `uninformative`."""
    issues = _flags([1.0, 1.0, 3.0], right=b([False, True, False]), lt=[0.0, 1.0, 0.0])[1]
    _assert_uninformative(issues)


@pytest.mark.step("A3")
def test_right_censoring_at_zero_without_truncation_is_uninformative_and_valid():
    """spec 3.3, rev 5: `(c, +inf]` with c=L=0 is valid and flagged `uninformative`."""
    issues = _flags([1.0, 0.0, 3.0], right=b([False, True, False]))[1]
    _assert_uninformative(issues)


@pytest.mark.step("A3")
def test_right_censoring_at_entry_with_finite_right_truncation_is_uninformative_and_valid():
    """spec 3.3, rev 5: `(c, +inf]` with c=L is uninformative including finite right truncation."""
    issues = _flags(
        [1.0, 1.0, 3.0], right=b([False, True, False]), lt=[0.0, 1.0, 0.0], rt=[INF, 3.0, INF]
    )[1]
    _assert_uninformative(issues)


@pytest.mark.step("A3")
def test_interval_to_infinity_at_entry_is_uninformative_and_valid():
    """spec 3.1, rev 5: an interval-flag row with `time2=+inf` and `time=L` is valid and `uninformative`."""
    issues = _flags(
        [1.0, 1.0, 3.0], interval=b([False, True, False]), time2=[NAN, INF, NAN], lt=[0.0, 1.0, 0.0]
    )[1]
    _assert_uninformative(issues)


@pytest.mark.step("A3")
def test_interval_to_infinity_at_entry_with_finite_right_truncation_is_uninformative_and_valid():
    """spec 3.1, rev 5: `time2=+inf` with `time=L` is `uninformative` for any valid truncation window (here finite U)."""
    issues = _flags(
        [1.0, 1.0, 3.0],
        interval=b([False, True, False]),
        time2=[NAN, INF, NAN],
        lt=[0.0, 1.0, 0.0],
        rt=[INF, 3.0, INF],
    )[1]
    _assert_uninformative(issues)


@pytest.mark.step("A3")
def test_left_censoring_at_upper_truncation_is_uninformative_and_valid():
    """spec 2/6.2, rev 4: left-censored at c>=U is valid and flagged `uninformative`, not in `invalid_rows`."""
    issues = _flags([1.0, 3.0, 2.0], left=b([False, True, False]), rt=[INF, 3.0, INF])[1]
    _assert_uninformative(issues)


@pytest.mark.step("A3")
def test_left_censoring_beyond_upper_truncation_is_uninformative_and_valid():
    """spec 2/6.2, rev 4: left-censored at c>U is valid and flagged `uninformative`."""
    issues = _flags([1.0, 4.0, 2.0], left=b([False, True, False]), rt=[INF, 3.0, INF])[1]
    _assert_uninformative(issues)


@pytest.mark.step("A3")
def test_interval_containing_window_is_uninformative_and_valid():
    """spec 2/6.2, rev 4: an interval containing (L, U] is valid and flagged `uninformative`."""
    issues = _flags(
        [1.0, 0.5, 2.0],
        interval=b([False, True, False]),
        time2=[NAN, 4.0, NAN],
        lt=[0.0, 1.0, 0.0],
        rt=[INF, 3.0, INF],
    )[1]
    _assert_uninformative(issues)


@pytest.mark.step("A3")
def test_partially_overlapping_rows_are_not_uninformative():
    """spec 2, rev 4: only censored sets covering the whole window are uninformative."""
    issues = _flags(
        [2.0, 2.0, 0.5],
        right=b([True, False, False]),
        left=b([False, True, False]),
        interval=b([False, False, True]),
        time2=[NAN, NAN, 2.0],
        lt=[1.0, 1.0, 1.0],
        rt=[5.0, 5.0, 5.0],
    )[1]
    assert_mask(issues, "uninformative", NONE)
    assert_mask(issues, "invalid_rows", NONE)


@pytest.mark.step("A3")
def test_eager_constructor_accepts_uninformative_rows():
    """spec 2/6.2, rev 4: uninformative rows are valid, so the eager constructor does not raise."""
    data = cx.SurvivalData(f([1.0, 3.0]), left_censored=b([False, True]), right_trunc=f([INF, 3.0]))
    assert np.shape(data.lower) == (2,)


# -------------------------------------------- union, reduction, bitfield


def _one_rule_per_row():
    """Rows 0-4 each break a different rule; row 5 is valid.

    row 0: right flag = 2              -> invalid_indicator
    row 1: right and left flags true   -> conflicting_indicators
    row 2: exact at 1, L = -1          -> invalid_truncation
    row 3: exact at 0.25, L = 0.5      -> exact_outside_truncation
    row 4: right at 0.25, L = 0.5      -> censored_before_entry
    row 5: exact at 1                  -> valid
    """
    return _flags(
        [1.0, 1.0, 1.0, 0.25, 0.25, 1.0],
        right=np.array([2, 1, 0, 0, 1, 0], dtype=np.int32),
        left=np.array([0, 1, 0, 0, 0, 0], dtype=np.int32),
        lt=[0.0, 0.0, -1.0, 0.5, 0.5, 0.0],
    )


@pytest.mark.step("A3")
def test_distinct_rules_on_distinct_rows_decode_independently():
    """spec 3.2/12.1, rev 4: issue bitfield decoding: each rule's mask marks exactly its own row."""
    _, issues = _one_rule_per_row()
    assert_mask(issues, "invalid_indicator", one_hot(6, 0))
    assert_mask(issues, "conflicting_indicators", one_hot(6, 1))
    assert_mask(issues, "invalid_truncation", one_hot(6, 2))
    assert_mask(issues, "exact_outside_truncation", one_hot(6, 3))
    assert_mask(issues, "censored_before_entry", one_hot(6, 4))
    assert_mask(issues, "invalid_rows", one_hot(6, 0, 1, 2, 3, 4))


@pytest.mark.step("A3")
def test_two_rules_on_one_row_are_both_decoded():
    """spec 3.2/12.1, rev 4: the bitfield preserves several pieces of raw-input evidence on one row."""
    issues = _flags(
        [1.0, 2.0, 3.0],
        right=b([False, True, False]),
        left=b([False, True, False]),
        lt=[0.0, -1.0, 0.0],
    )[1]
    assert_mask(issues, "conflicting_indicators", ROW1)
    assert_mask(issues, "invalid_truncation", ROW1)
    assert_mask(issues, "invalid_rows", ROW1)


@pytest.mark.step("A3")
def test_invalid_rows_is_the_union_of_the_ten_rule_masks():
    """spec 6.2, rev 4: the union of the ten rule masks is `invalid_rows`."""
    _, issues = _one_rule_per_row()
    union = np.zeros(6, dtype=bool)
    for name in RULE_FIELDS:
        union |= mask(issues, name)
    assert_mask(issues, "invalid_rows", union)


@pytest.mark.step("A3")
def test_uninformative_is_not_part_of_invalid_rows():
    """spec 6.2, rev 4: `uninformative` is reported separately and is not part of `invalid_rows`."""
    # row 0 uninformative (left at c=U), row 1 invalid (exact at 0), row 2 valid
    issues = _flags([3.0, 0.0, 2.0], left=b([True, False, False]), rt=[3.0, INF, INF])[1]
    assert_mask(issues, "uninformative", one_hot(3, 0))
    assert_mask(issues, "invalid_rows", one_hot(3, 1))


@pytest.mark.step("A3")
def test_ok_is_false_when_any_row_is_invalid():
    """spec 6.2, rev 4: `ok` is the scalar validity reduction."""
    _, issues = _one_rule_per_row()
    assert not bool(np.asarray(issues.ok))


@pytest.mark.step("A3")
def test_ok_is_true_with_only_uninformative_rows():
    """spec 6.2, rev 4: `ok` reduces `invalid_rows`, which excludes `uninformative` (Q-011)."""
    issues = _flags([3.0, 4.0], left=b([True, True]), rt=[3.0, 3.0])[1]
    assert bool(np.asarray(issues.ok))


# --------------------------------------------------------- raise_if_invalid


@pytest.mark.step("A3")
def test_raise_if_invalid_raises_for_invalid_rows():
    """spec 6.2, rev 4: `DataIssues.raise_if_invalid()` is an explicit host-only checkpoint."""
    _, issues = _one_rule_per_row()
    with pytest.raises(REJECTED):
        issues.raise_if_invalid()


@pytest.mark.step("A3")
def test_raise_if_invalid_passes_for_valid_data():
    """spec 6.2, rev 4: the checkpoint does not raise for valid data."""
    _, issues = with_issues(f([1.0, 2.0]), b([True, False]))
    issues.raise_if_invalid()


@pytest.mark.step("A3")
def test_raise_if_invalid_passes_for_uninformative_rows():
    """spec 6.2, rev 4: uninformative rows are valid, so the checkpoint does not raise."""
    issues = _flags([3.0, 1.0], left=b([True, False]), rt=[3.0, INF])[1]
    issues.raise_if_invalid()


# ------------------------------------------------------ eager versus traced


def _mixed_inputs():
    """Four rows: valid exact, invalid indicator, conflicting flags, valid left-censored."""
    return (
        f([1.0, 2.0, 3.0, 2.0]),
        np.array([0, 2, 1, 0], dtype=np.int32),
        np.array([0, 0, 1, 1], dtype=np.int32),
        f([0.5, 0.0, 0.0, 0.0]),
    )


def _build(time, right, left, lt):
    return with_issues(time, right_censored=right, left_censored=left, left_trunc=lt)


@pytest.mark.step("A3")
def test_ordinary_constructor_under_jit_raises_pointing_to_with_issues():
    """spec 6.2, rev 4: called with traced values the constructor raises at trace time and points to `with_issues` (Q-012)."""
    with pytest.raises(Exception, match="with_issues"):
        jax.jit(lambda t: cx.SurvivalData(t).lower)(f([1.0, 2.0]))


@pytest.mark.step("A3")
def test_ordinary_constructor_under_vmap_raises_pointing_to_with_issues():
    """spec 6.2, rev 4: traced observation values under vmap also raise at trace time (Q-012)."""
    with pytest.raises(Exception, match="with_issues"):
        jax.vmap(lambda t: cx.SurvivalData(t).lower)(f([[1.0, 2.0], [3.0, 4.0]]))


@pytest.mark.step("A3")
def test_bounds_constructor_under_jit_raises_pointing_to_with_issues():
    """spec 6.2, rev 4: the eager constructors raise at trace time and point to `with_issues` (Q-012)."""
    with pytest.raises(Exception, match="with_issues"):
        jax.jit(lambda lo, hi: cx.SurvivalData.from_bounds(lo, hi).lower)(f([1.0]), f([2.0]))


@pytest.mark.step("A3")
def test_with_issues_under_jit_matches_eager():
    """spec 6.2, rev 4: with_issues is pure and works under jit, with the same data and issues as eagerly."""
    inputs = _mixed_inputs()
    eager_data, eager_issues = _build(*inputs)
    jit_data, jit_issues = jax.jit(_build)(*inputs)
    for name in FIELDS:
        np.testing.assert_array_equal(np.asarray(getattr(jit_data, name)), np.asarray(getattr(eager_data, name)), err_msg=name)
    for name in MASK_FIELDS:
        assert_mask(jit_issues, name, mask(eager_issues, name))
    assert bool(np.asarray(jit_issues.ok)) == bool(np.asarray(eager_issues.ok))


@pytest.mark.step("A3")
def test_with_issues_under_vmap_matches_eager_per_dataset():
    """spec 6.2, rev 4: with_issues works under vmap; each batch entry matches its eager result."""
    time, right, left, lt = _mixed_inputs()
    batch = tuple(np.stack([x, x[::-1]]) for x in (time, right, left, lt))
    _, batched = jax.vmap(_build)(*batch)
    assert np.shape(batched.ok) == (2,)
    for i in range(2):
        _, eager = _build(*(x[i] for x in batch))
        for name in MASK_FIELDS:
            np.testing.assert_array_equal(mask(batched, name)[i], mask(eager, name), err_msg=name)
        assert bool(np.asarray(batched.ok)[i]) == bool(np.asarray(eager.ok))


@pytest.mark.step("A3")
def test_data_issues_is_a_pytree_returned_from_jit():
    """spec 6.3, rev 4: DataIssues is a registered PyTree, so it can be returned from jit."""
    issues = jax.jit(lambda t, e: with_issues(t, e)[1])(f([1.0, 2.0]), np.array([1, 2], dtype=np.int32))
    assert len(leaves(issues)) > 0
    assert_mask(issues, "invalid_indicator", one_hot(2, 1))
