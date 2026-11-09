"""observation.censoring: NaN rows and their diagnosis on left- and interval-censored data (spec 3.1, 3.3, 6.2, 7.1; rev 51).

7.1: "Every row in `data.issues.invalid_rows` yields NaN", and rows marked by
`dist.parameter_issues().invalid_entries` are NaN; with left/interval rows
implemented, the 7.1 triage recipe (``helpers.nan_causes``; its support
check expands with the enabled features, Q-064) leaves no unexplained NaN on
valid censored rows. Each invalid-data case pairs a valid left-censored row
0 (Exponential ``r=0.5``, ``c=2``: ``log(-expm1(-1))``) with an invalid row 1.
Which mask an invalid row sets is A2/A3 territory; here the rule is that NaN
rows are exactly ``invalid_rows`` and are attributed to data, never to
``other``. Masks are named only where the spec fixes them (6.2).
"""

import jax
import numpy as np
import pytest
from helpers import INF, NAN, assert_close, b, f, flag_data, ll, mask, nan_causes

import survjax

FEATURE = pytest.mark.feature("observation.censoring")

ROW0 = float(np.log(-np.expm1(-1.0)))


def _dist():
    return survjax.Exponential(rate=f(0.5))


def _assert_row1_invalid(value, data):
    value = np.asarray(value)
    invalid = np.asarray(data.issues.invalid_rows).astype(bool)
    np.testing.assert_array_equal(invalid, [False, True])
    np.testing.assert_array_equal(np.isnan(value), invalid)
    assert_close(value[0], ROW0)
    bad_data, bad_params, other = nan_causes(_dist(), data, value)
    np.testing.assert_array_equal(bad_data, invalid)
    assert not bad_params.any()
    assert not other.any()


INVALID = {
    # 3.3: "left censoring at `+inf` [is] invalid (`invalid_time`)".
    "left-at-inf": (lambda: flag_data(["left", "left"], [2.0, INF]), "invalid_time"),
    # 3.3: censoring endpoints are nonnegative; 6.2: `time` violating 3.3 sets `invalid_time`.
    "left-negative": (lambda: flag_data(["left", "left"], [2.0, -1.0]), "invalid_time"),
    # 6.2: "A NaN `time` sets `invalid_time`".
    "left-nan": (lambda: flag_data(["left", "left"], [2.0, NAN]), "invalid_time"),
    # 3.1: "interval rows require `time < time2`"; 6.2 `invalid_interval`.
    "interval-reversed": (lambda: flag_data(["left", "interval"], [2.0, 3.0], [NAN, 2.0]), "invalid_interval"),
    "interval-equal": (lambda: flag_data(["left", "interval"], [2.0, 3.0], [NAN, 3.0]), "invalid_interval"),
    # 6.2: "an active interval row has omitted or NaN `time2`".
    "interval-nan-time2": (lambda: flag_data(["left", "interval"], [2.0, 1.0], [NAN, NAN]), "missing_time2"),
    # 6.2: "an active interval `time2` ... is negative or `-inf`".
    "interval-negative-time2": (lambda: flag_data(["left", "interval"], [2.0, 1.0], [NAN, -1.0]), "invalid_time2"),
    # 6.2: "`conflicting_indicators` reports only rows on which more than one censoring flag is true".
    "conflicting-flags": (
        lambda: survjax.SurvivalData(
            f([2.0, 1.0]),
            left_censored=b([True, True]),
            interval_censored=b([False, True]),
            time2=f([NAN, 3.0]),
        ),
        "conflicting_indicators",
    ),
    # 3.1: "Equal endpoints on a non-exact row are invalid".
    "bounds-equal": (
        lambda: survjax.SurvivalData.from_bounds(f([0.0, 2.0]), f([2.0, 2.0]), exact=b([False, False])),
        None,
    ),
    # 6.2: "a non-exact `from_bounds` row has NaN `upper`" sets `missing_time2`.
    "bounds-nan-upper": (
        lambda: survjax.SurvivalData.from_bounds(f([0.0, 1.0]), f([2.0, NAN]), exact=b([False, False])),
        "missing_time2",
    ),
    # 6.2: `from_bounds` upper negative sets `invalid_time2`.
    "bounds-negative-upper": (
        lambda: survjax.SurvivalData.from_bounds(f([0.0, 1.0]), f([2.0, -1.0]), exact=b([False, False])),
        "invalid_time2",
    ),
}


@FEATURE
@pytest.mark.parametrize("case", list(INVALID))
def test_invalid_censored_row_is_nan_and_attributed_to_data(case):
    """spec 6.2 and 7.1, rev 51: an invalid left/interval row is NaN, the NaN rows equal `invalid_rows`, and triage blames the data."""
    build, rule = INVALID[case]
    data = build()
    _assert_row1_invalid(ll(_dist(), data), data)
    if rule is not None:
        assert mask(data.issues, rule)[1], rule


@FEATURE
@pytest.mark.parametrize("case", ["left-at-inf", "interval-reversed", "conflicting-flags"])
def test_invalid_censored_row_under_jit(case):
    """spec 6.2 and 7.1, rev 51: prepared data in `jit` give the same NaN rows and the valid row keeps its value."""
    data = INVALID[case][0]()
    value = np.asarray(jax.jit(lambda d: _dist().log_likelihood(d))(data))
    _assert_row1_invalid(value, data)


@FEATURE
def test_invalid_censored_row_from_traced_construction():
    """spec 6.2 and 7.1, rev 51: construction inside `jit` records the issue and the likelihood row is NaN, without exceptions."""

    def run(time, time2, left, interval):
        data = survjax.SurvivalData(time, left_censored=left, interval_censored=interval, time2=time2)
        return _dist().log_likelihood(data), data.issues.invalid_rows

    value, invalid = jax.jit(run)(f([2.0, 3.0]), f([NAN, 2.0]), b([True, False]), b([False, True]))
    np.testing.assert_array_equal(np.asarray(invalid), [False, True])
    np.testing.assert_array_equal(np.isnan(np.asarray(value)), [False, True])
    assert_close(np.asarray(value)[0], ROW0)


@FEATURE
def test_valid_mixed_batch_has_no_nan_and_no_triage_mask():
    """spec 7.1, rev 51: with censoring implemented, valid exact/right/left/interval rows produce no NaN; every recipe mask is empty."""
    data = flag_data(["exact", "right", "left", "interval"], [1.0, 2.0, 1.5, 0.5], [NAN, NAN, NAN, 2.5])
    value = ll(_dist(), data)
    assert not np.isnan(value).any()
    for m in nan_causes(_dist(), data, value):
        assert not m.any()


@FEATURE
def test_valid_censored_rows_add_no_issue():
    """spec 6.2, rev 51: valid untruncated left and interval rows are valid data and not `uninformative`."""
    data = flag_data(["left", "interval"], [1.5, 0.5], [NAN, 2.5])
    assert bool(np.asarray(data.issues.ok))
    assert not mask(data.issues, "uninformative").any()


@FEATURE
@pytest.mark.parametrize("bad", [0.0, -1.0, NAN, INF], ids=["zero", "negative", "nan", "posinf"])
def test_parameter_issue_on_a_censored_row_is_nan(bad):
    """spec 4.7 and 7.1, rev 51: a declared parameter violation makes its left/interval row NaN; other rows keep their values.

    Rows: left at ``2``, interval ``(1, 3]``, left at ``2``; rates
    ``[0.5, bad, 0.5]``.
    """
    data = flag_data(["left", "interval", "left"], [2.0, 1.0, 2.0], [NAN, 3.0, NAN])
    dist = survjax.Exponential(rate=f([0.5, bad, 0.5]))
    value = ll(dist, data)
    np.testing.assert_array_equal(np.isnan(value), [False, True, False])
    assert_close(value[[0, 2]], [ROW0, ROW0])
    bad_data, bad_params, other = nan_causes(dist, data, value)
    assert not bad_data.any()
    np.testing.assert_array_equal(bad_params, [False, True, False])
    assert not other.any()


@FEATURE
def test_indexing_keeps_issues_of_censored_rows():
    """spec 3.2 and 7.1, rev 51: `data[idx]` keeps the invalid interval row's evidence; its likelihood row is NaN."""
    data = flag_data(["left", "interval", "left"], [2.0, 3.0, 2.0], [NAN, 2.0, NAN])
    sub = data[1:]
    value = ll(_dist(), sub)
    np.testing.assert_array_equal(np.isnan(value), [True, False])
    np.testing.assert_array_equal(np.asarray(sub.issues.invalid_rows), [True, False])
    assert_close(value[1], ROW0)
