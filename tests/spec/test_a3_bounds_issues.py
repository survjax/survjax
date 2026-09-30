"""A3: bounds constructor diagnostics (spec 6.2, rev 9).

As in ``test_a3_data_issues.py``, each attribution test builds three rows: two
valid rows and one row that breaks a single rule. Other masks are asserted
clear only where the spec fixes the ordering (a malformed endpoint is not also
``invalid_interval``; an exact row's NaN upper is not ``missing_time2``).
"""

import chronox_jax as cx
import jax
import numpy as np
import pytest
from helpers import (
    FIELDS,
    INF,
    MASK_FIELDS,
    MISUSE,
    NAN,
    assert_mask,
    assert_same_leaves,
    b,
    bounds_data_and_issues,
    f,
    mask,
    one_hot,
)

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


def _bounds(lower, upper, *, exact=None, lt=None, rt=None):
    """Construct bounds data on three rows; rows 0 and 2 are valid by default."""
    kw = {}
    if exact is not None:
        kw["exact"] = b(exact)
    if lt is not None:
        kw["left_trunc"] = f(lt)
    if rt is not None:
        kw["right_trunc"] = f(rt)
    return bounds_data_and_issues(f(lower), f(upper), **kw)[1]


# ------------------------------------------------------- container, parity


@pytest.mark.step("A3")
def test_bounds_constructor_returns_data_with_diagnostics_property():
    """spec 6.2, rev 9: the bounds form returns data and exposes diagnostics through a property."""
    data = cx.SurvivalData.from_bounds(f([1.0, 2.0]), f([2.0, INF]))
    assert isinstance(data, cx.SurvivalData)
    assert np.shape(data.issues.invalid_rows) == (2,)



@pytest.mark.step("A3")
def test_bounds_constructor_has_no_issue_returning_convenience():
    """spec 6.2, rev 9: the bounds form is the only public bounds observation constructor."""
    assert not hasattr(cx.SurvivalData, "from_bounds_with_issues")



@pytest.mark.step("A3")
def test_bounds_data_and_issues_matches_from_bounds_on_uninformative_input():
    """spec 6.2, rev 6: identical fields and issue bits also for accepted rows that carry the uninformative bit."""
    args = (f([0.0, 1.0]), f([INF, 2.0]))
    data, issues = bounds_data_and_issues(*args)
    assert_mask(issues, "uninformative", one_hot(2, 0))
    assert_same_leaves(data, cx.SurvivalData.from_bounds(*args))


@pytest.mark.step("A3")
def test_bounds_data_and_issues_does_not_raise_on_invalid_rows():
    """spec 6.2, rev 9: the bounds constructor records invalid rows without raising."""
    _, issues = bounds_data_and_issues(f([1.0, 3.0]), f([2.0, 2.0]))
    assert not bool(np.asarray(issues.ok))


@pytest.mark.step("A3")
def test_bounds_data_and_issues_valid_data_has_no_issues():
    """spec 6.2, rev 6: valid bounds data set no rule mask, no invalid row, and `ok` is true."""
    issues = _bounds([1.0, 2.0, 0.0], [1.0, INF, 3.0], exact=[True, False, False], lt=[0.5, 0.5, 0.0])
    for name in MASK_FIELDS:
        assert_mask(issues, name, NONE)
    assert bool(np.asarray(issues.ok))


@pytest.mark.step("A3")
def test_bounds_data_and_issues_under_jit_matches_eager():
    """spec 6.2, rev 9: bounds construction and diagnostics agree eagerly and under jit."""
    inputs = (f([1.0, 3.0, 0.5, 1.0]), f([1.0, 2.0, NAN, INF]), b([True, False, False, False]), f([0.0, 0.0, 0.0, 2.0]))

    def build(lower, upper, exact, lt):
        return bounds_data_and_issues(lower, upper, exact=exact, left_trunc=lt)

    eager_data, eager_issues = build(*inputs)
    jit_data, jit_issues = jax.jit(build)(*inputs)
    for name in FIELDS:
        np.testing.assert_array_equal(np.asarray(getattr(jit_data, name)), np.asarray(getattr(eager_data, name)), err_msg=name)
    for name in MASK_FIELDS:
        assert_mask(jit_issues, name, mask(eager_issues, name))


# ----------------------------------------------------------- invalid_time (lower)


@pytest.mark.step("A3")
def test_bounds_negative_lower_sets_invalid_time():
    """spec 6.2, rev 6: `from_bounds` `lower` uses `invalid_time`; censored lower bounds must be nonnegative."""
    _assert_rule(_bounds([1.0, -1.0, 1.0], [2.0, 2.0, 2.0]), "invalid_time")


@pytest.mark.step("A3")
def test_bounds_infinite_lower_sets_invalid_time():
    """spec 6.2, rev 6: `from_bounds` `lower` uses `invalid_time`; censored lower bounds must be finite."""
    _assert_rule(_bounds([1.0, INF, 1.0], [2.0, INF, 2.0]), "invalid_time")


@pytest.mark.step("A3")
def test_bounds_nan_lower_sets_invalid_time():
    """spec 6.2, rev 6: `from_bounds` `lower` uses `invalid_time`, including when it is NaN."""
    _assert_rule(_bounds([1.0, NAN, 1.0], [2.0, 2.0, 2.0]), "invalid_time")


@pytest.mark.step("A3")
def test_bounds_exact_lower_at_zero_sets_invalid_time():
    """spec 6.2, rev 6: exact times must be finite and positive; an exact `lower` of 0 sets `invalid_time`."""
    _assert_rule(_bounds([1.0, 0.0, 1.0], [1.0, 0.0, 1.0], exact=[True, True, True]), "invalid_time")


@pytest.mark.step("A3")
def test_bounds_infinite_lower_is_not_also_invalid_interval():
    """spec 6.2, rev 6: domains are evaluated before order, so a malformed lower is not a reversed interval (inf >= inf)."""
    assert_mask(_bounds([1.0, INF, 1.0], [2.0, INF, 2.0]), "invalid_interval", NONE)


# ----------------------------------------------------------- missing_time2 (upper)


@pytest.mark.step("A3")
def test_bounds_nan_upper_on_censored_row_sets_missing_time2():
    """spec 6.2, rev 6: a non-exact `from_bounds` row with NaN `upper` sets `missing_time2`."""
    _assert_rule(_bounds([1.0, 1.0, 1.0], [2.0, NAN, 2.0]), "missing_time2")


@pytest.mark.step("A3")
def test_bounds_nan_upper_on_exact_row_sets_no_issue():
    """spec 6.2, rev 6: an exact `from_bounds` row may have NaN `upper` and does not set `missing_time2`."""
    issues = _bounds([1.0, 1.5, 1.0], [2.0, NAN, 2.0], exact=[False, True, False])
    assert_mask(issues, "missing_time2", NONE)
    assert_mask(issues, "invalid_rows", NONE)


@pytest.mark.step("A3")
def test_bounds_nan_upper_is_not_also_invalid_interval():
    """spec 6.2, rev 6: presence is evaluated before order, so a missing upper is not a reversed interval."""
    assert_mask(_bounds([1.0, 1.0, 1.0], [2.0, NAN, 2.0]), "invalid_interval", NONE)


# ----------------------------------------------------------- invalid_time2 (upper)


@pytest.mark.step("A3")
def test_bounds_negative_upper_sets_invalid_time2():
    """spec 6.2, rev 6: a supplied `from_bounds` upper that is negative sets `invalid_time2`."""
    _assert_rule(_bounds([1.0, 0.0, 1.0], [2.0, -1.0, 2.0]), "invalid_time2")


@pytest.mark.step("A3")
def test_bounds_minus_infinite_upper_sets_invalid_time2():
    """spec 6.2, rev 6: a supplied `from_bounds` upper of `-inf` sets `invalid_time2`."""
    _assert_rule(_bounds([1.0, 0.0, 1.0], [2.0, -INF, 2.0]), "invalid_time2")


@pytest.mark.step("A3")
def test_bounds_exact_row_with_different_upper_sets_invalid_time2():
    """spec 6.2, rev 6: an exact `from_bounds` row whose upper is neither its lower nor NaN sets `invalid_time2`."""
    issues = _bounds([1.0, 1.5, 1.0], [2.0, 2.0, 2.0], exact=[False, True, False])
    _assert_rule(issues, "invalid_time2")


@pytest.mark.step("A3")
def test_bounds_negative_upper_is_not_also_invalid_interval_or_missing():
    """spec 6.2, rev 6: a domain error in `upper` is neither missing nor a reversed interval (here 0 > -1)."""
    issues = _bounds([1.0, 0.0, 1.0], [2.0, -1.0, 2.0])
    assert_mask(issues, "invalid_interval", NONE)
    assert_mask(issues, "missing_time2", NONE)


# ----------------------------------------------------------- invalid_interval


@pytest.mark.step("A3")
def test_bounds_equal_endpoints_on_censored_row_set_invalid_interval():
    """spec 6.2, rev 6: a non-exact `from_bounds` row with `lower >= upper` (here equal) sets `invalid_interval`."""
    _assert_rule(_bounds([1.0, 2.0, 1.0], [2.0, 2.0, 2.0]), "invalid_interval")


@pytest.mark.step("A3")
def test_bounds_reversed_endpoints_set_invalid_interval():
    """spec 6.2, rev 6: a non-exact `from_bounds` row with `lower >= upper` (here reversed) sets `invalid_interval`."""
    _assert_rule(_bounds([1.0, 3.0, 1.0], [2.0, 2.0, 2.0]), "invalid_interval")


@pytest.mark.step("A3")
def test_bounds_infinite_upper_is_valid():
    """spec 6.2, rev 6: a `+inf` non-exact bounds upper is valid."""
    issues = _bounds([1.0, 2.0, 1.0], [2.0, INF, 2.0])
    assert_mask(issues, "invalid_interval", NONE)
    assert_mask(issues, "invalid_rows", NONE)


# ----------------------------------------------------------- entry and coverage


@pytest.mark.step("A3")
def test_bounds_right_censoring_before_entry_sets_censored_before_entry():
    """spec 2/6.2, rev 6: a censored row with `upper=+inf` and `lower<L` sets `censored_before_entry`, however constructed."""
    issues = _bounds([1.0, 0.5, 1.0], [2.0, INF, 2.0], lt=[0.0, 1.0, 0.0])
    _assert_rule(issues, "censored_before_entry")


@pytest.mark.step("A3")
def test_bounds_zero_to_infinity_with_left_truncation_sets_censored_before_entry():
    """spec 3.3, rev 6: `from_bounds(0, inf)` is invalid when L>0 (`lower<L`, so `censored_before_entry`)."""
    issues = _bounds([1.0, 0.0, 1.0], [2.0, INF, 2.0], lt=[0.0, 1.0, 0.0])
    _assert_rule(issues, "censored_before_entry")


@pytest.mark.step("A3")
def test_bounds_zero_to_infinity_without_truncation_is_uninformative_and_valid():
    """spec 3.3, rev 6: `from_bounds(0, inf)` is uninformative when L=0."""
    _assert_uninformative(_bounds([1.0, 0.0, 1.0], [2.0, INF, 2.0]))


@pytest.mark.step("A3")
def test_bounds_right_censoring_at_entry_is_uninformative_and_valid():
    """spec 2/6.2, rev 6: `(L, +inf]` from bounds is valid and `uninformative`, not `censored_before_entry`."""
    issues = _bounds([1.0, 1.0, 1.0], [2.0, INF, 2.0], lt=[0.0, 1.0, 0.0])
    _assert_uninformative(issues)
    assert_mask(issues, "censored_before_entry", NONE)


# ----------------------------------------------------------- argument misuse


@pytest.mark.step("A3")
def test_bounds_data_and_issues_raises_on_nonnumeric_lower():
    """spec 6.2, rev 7: argument misuse (nonnumeric fields) raises in both constructors."""
    with pytest.raises(MISUSE):
        bounds_data_and_issues(np.array(["1.0", "2.0"]), f([2.0, 3.0]))


@pytest.mark.step("A3")
def test_bounds_data_and_issues_raises_on_incompatible_shapes():
    """spec 6.2, rev 7: argument misuse (incompatible shapes) raises in both constructors."""
    with pytest.raises(MISUSE):
        bounds_data_and_issues(f([1.0, 2.0, 3.0]), f([[4.0], [5.0], [6.0]]))


@pytest.mark.step("A3")
def test_bounds_data_and_issues_raises_on_incompatible_shapes_under_jit():
    """spec 6.2, rev 7: argument misuse raises independent of tracing."""
    with pytest.raises(MISUSE):
        jax.jit(lambda lo, hi: bounds_data_and_issues(lo, hi)[1].ok)(f([1.0, 2.0, 3.0]), f([[4.0], [5.0], [6.0]]))
