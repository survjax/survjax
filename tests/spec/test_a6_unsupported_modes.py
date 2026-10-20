"""A6: explicit failure for observation modes Slice A does not implement (spec 12.2, 6.1, 7.1; rev 28, 29).

Slice A supports exact and right-censored rows without truncation; "unimplemented
observation modes fail explicitly instead of producing plausible values"
(12.2). The spec does not say how (Q-037). What it rules out is a plausible
value, so each case accepts exactly two outcomes on the unsupported row:

- concrete (eager) data: ``NotImplementedError``, or NaN on that row;
- data constructed from traced inputs inside ``jit``: NaN on that row. Traced
  construction sets every may-contain flag (3.2), and traced exact/right data
  must evaluate (6.1, raw form under ``jit``), so no trace-time decision can
  tell the two apart; a value-based NaN is the only explicit failure left.

Row 0 of every case is a valid exact row; row 1 is the unsupported one. Whether
the supported row keeps its value is not specified and not asserted.
"""

import jax
import numpy as np
import pytest
from helpers import INF, b, f

import survjax


def _dist():
    return survjax.Exponential(rate=f(0.5))


def _left_censored():
    return survjax.SurvivalData(f([2.0, 3.0]), left_censored=b([False, True]))


def _interval_censored():
    return survjax.SurvivalData(
        f([2.0, 1.0]), interval_censored=b([False, True]), time2=f([2.0, 3.0])
    )


def _bounds_left():
    return survjax.SurvivalData.from_bounds(f([2.0, 0.0]), f([2.0, 3.0]), exact=b([True, False]))


def _bounds_interval():
    return survjax.SurvivalData.from_bounds(f([2.0, 1.0]), f([2.0, 3.0]), exact=b([True, False]))


def _left_truncated_exact():
    return survjax.SurvivalData(f([2.0, 2.0]), b([True, True]), left_trunc=f([0.0, 1.0]))


def _left_truncated_right():
    return survjax.SurvivalData(f([2.0, 2.0]), b([True, False]), left_trunc=f([0.0, 1.0]))


def _right_truncated_exact():
    return survjax.SurvivalData(f([2.0, 2.0]), b([True, True]), right_trunc=f([INF, 5.0]))


CASES = {
    "left-censored": _left_censored,
    "interval-censored": _interval_censored,
    "bounds-left": _bounds_left,
    "bounds-interval": _bounds_interval,
    "left-truncated-exact": _left_truncated_exact,
    "left-truncated-right": _left_truncated_right,
    "right-truncated-exact": _right_truncated_exact,
}


def _assert_fails_explicitly(call):
    try:
        out = call()
    except NotImplementedError:
        return
    value = np.asarray(out)
    assert np.isnan(value[1]), f"unsupported row produced {value[1]!r}"


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_unsupported_row_is_valid_data(case):
    """spec 3.2 and 12.2, rev 28: "The data representation accommodates every v0 kind from the start": these rows are valid data."""
    data = CASES[case]()
    assert bool(np.asarray(data.issues.ok))


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_unsupported_mode_fails_explicitly(case):
    """spec 12.2, rev 28: unimplemented modes "fail explicitly instead of producing plausible values" (Q-037)."""
    data = CASES[case]()
    _assert_fails_explicitly(lambda: survjax.log_likelihood(_dist(), data))


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_unsupported_mode_fails_explicitly_in_method(case):
    """spec 7.1 and 12.2, rev 29: the `dist.log_likelihood` method also fails explicitly (Q-037)."""
    data = CASES[case]()
    _assert_fails_explicitly(lambda: _dist().log_likelihood(data))


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_unsupported_mode_from_traced_construction_is_nan(case):
    """spec 3.2, 6.1 and 12.2, rev 28: with traced construction the unsupported row is NaN (Q-037).

    The case's normalized fields are passed into ``jit`` and rebuilt with the
    bounds form (exact rows have ``upper == lower``), so construction sees
    traced inputs.
    """
    data = CASES[case]()

    def build_and_evaluate(lower, upper, exact, lt, rt):
        traced = survjax.SurvivalData.from_bounds(
            lower, upper, exact=exact, left_trunc=lt, right_trunc=rt
        )
        return survjax.log_likelihood(_dist(), traced)

    value = jax.jit(build_and_evaluate)(
        data.lower, data.upper, data.exact, data.left_trunc, data.right_trunc
    )
    assert np.isnan(np.asarray(value)[1])
