"""Slice B: log-normal and log-logistic rows in the cells 10.4 keeps out of the flexsurv fixture (spec 2, 3.3, 6.2, 7.1, 10.4, 12.1, 12.2; rev 60).

Spec 10.4 (rev 59/60): the cells flexsurv "cannot express ... or defines ...
differently" ("left or interval censoring with left truncation; ... censored
sets reaching beyond `U` ...; rows censored at or before entry; observations
outside the window") are not compared with flexsurv; "Independent mathematics
covers them (section 12.1)". The Exponential and Weibull cases are in
``test_observation_truncation_*.py``; this file covers the other two fixture
families (TESTMAP, "Slice B flexsurv fixture", formerly listed as gaps).

Spec 2: a censored row contributes
``log P(max(lower, L) < T <= min(upper, U)) - log Z`` with
``Z = P(L < T <= U)``; a set covering the whole window contributes "exactly
zero" and is ``uninformative``; right-censored at entry (``c=L``) is exactly
zero; right-censored before entry is ``censored_before_entry``; empty
intersections are invalid data. 3.3: exact times need ``L<t<=U``
(``exact_outside_truncation``).

References: 50-digit mpmath of the spec 2 table (``observation_reference``,
log-normal and log-logistic branches) at inputs rounded to the active dtype.
Invalid and structural cases pair a valid exact row 0 at ``t=1`` in the
window ``(0.5, 3]`` (its value from the same reference) with the case row 1,
as in ``test_observation_truncation_diagnostics.py``.

Markers: every test lists its family feature and ``observation.truncation``;
tests with left or interval rows add ``observation.censoring`` (role table).
"""

import numpy as np
import observation_reference as oref
import pytest
from helpers import INF, NAN, assert_close, b, f, flag_data, ll, mask, nan_causes

import survjax

CENSORING = "observation.censoring"
TRUNCATION = "observation.truncation"
FAMILY_FEATURE = {"lognormal": "family.lognormal", "loglogistic": "family.loglogistic"}

# Natural parameters in the order of ``observation_reference``: log-normal
# ``(meanlog, sdlog)``, log-logistic ``(shape, scale)``; a light and a heavy
# spread for each, and a log-logistic shape below 1.
VECTORS = {
    "lognormal": [(0.4, 0.9), (-0.3, 1.6)],
    "loglogistic": [(1.6, 2.0), (0.7, 1.5)],
}

L0, U0 = 0.5, 3.0
WINDOWS = {"left": (L0, INF), "right": (0.0, U0), "two-sided": (L0, U0)}


def _dist(family, params):
    p, q = (f(x) for x in params)
    if family == "lognormal":
        return survjax.LogNormal(meanlog=p, sdlog=q)
    return survjax.LogLogistic(shape=p, scale=q)


def _with_family(cases, *features):
    """``(family, params, case)`` for every family, vector and case, marked with the family feature and ``features``."""
    return [
        pytest.param(
            family,
            params,
            case,
            marks=pytest.mark.feature(FAMILY_FEATURE[family], *features),
            id=f"{family}-{params[0]}-{params[1]}-{case}",
        )
        for family, vectors in VECTORS.items()
        for params in vectors
        for case in cases
    ]


def _row0(family, params):
    """mpmath value of the valid exact row 0 (``t=1`` in ``(0.5, 3]``)."""
    return oref.contribution(family, params, 1.0, exact=True, left=L0, right=U0)


def _two(kind, t, t2=NAN, lt=L0, rt=U0):
    """Row 0 valid exact at 1 in ``(0.5, 3]``; row 1 of ``kind`` with its own window."""
    return flag_data(["exact", kind], [1.0, t], [NAN, t2], left_trunc=[L0, lt], right_trunc=[U0, rt])


# ================================================== left and interval rows with left truncation


# (kind, time, time2) with the set each row stands for; every set meets every window of
# ``LEFT_WINDOWS`` in a set of positive probability.
LEFT_TRUNCATED_CENSORED = [
    ("left", 1.2, NAN),  # (0, 1.2] -> (0.5, 1.2]
    ("left", 2.5, NAN),  # (0, 2.5] -> (0.5, 2.5]
    ("interval", 0.2, 1.4),  # partial overlap at L -> (0.5, 1.4]
    ("interval", L0, 2.2),  # starts at L
    ("interval", 0.8, 2.0),  # inside the window
]
LEFT_WINDOWS = {"left": WINDOWS["left"], "two-sided": WINDOWS["two-sided"]}


def _reference_rows(family, params, rows, left, right):
    out = []
    for kind, t, t2 in rows:
        if kind == "exact":
            out.append(oref.contribution(family, params, t, exact=True, left=left, right=right))
        elif kind == "right":
            out.append(oref.contribution(family, params, t, INF, left=left, right=right))
        elif kind == "left":
            out.append(oref.contribution(family, params, 0.0, t, left=left, right=right))
        else:
            out.append(oref.contribution(family, params, t, t2, left=left, right=right))
    return out


def _batch(rows, left, right):
    kinds, time, time2 = zip(*rows)
    n = len(rows)
    return flag_data(list(kinds), list(time), list(time2), left_trunc=[left] * n, right_trunc=[right] * n)


@pytest.mark.parametrize("family, params, window", _with_family(list(LEFT_WINDOWS), CENSORING, TRUNCATION))
def test_left_and_interval_rows_with_left_truncation_match_mpmath(family, params, window):
    """spec 2/7.1/10.4/12.1, rev 60: left rows `(0, c]` and interval rows (inside, starting at `L`, overlapping `L`) under `L>0` give `log P(max(lower, L) < T <= min(upper, U)) - log Z`; one mixed batch with an exact row."""
    left, right = LEFT_WINDOWS[window]
    rows = [("exact", 1.0, NAN), *LEFT_TRUNCATED_CENSORED]
    out = ll(_dist(family, params), _batch(rows, left, right))
    want = _reference_rows(family, params, rows, left, right)
    assert np.isfinite(out).all()
    assert_close(out, want)


# ================================================== censored sets reaching beyond U


# Sets with ``a < U < b`` (partial overlap at ``U``): intersected with the window, ``P(a<T<=U)``.
BEYOND_UPPER = [
    ("interval", 1.0, 4.0),
    ("interval", 2.5, 10.0),
    ("interval", 1.5, INF),  # interval flag with ``time2=+inf``: ``(1.5, +inf]`` intersected with the window
]
UPPER_WINDOWS = {"right": WINDOWS["right"], "two-sided": WINDOWS["two-sided"]}


@pytest.mark.parametrize("family, params, window", _with_family(list(UPPER_WINDOWS), CENSORING, TRUNCATION))
def test_censored_sets_reaching_beyond_upper_are_intersected_with_the_window(family, params, window):
    """spec 2/7.1/10.4/12.1, rev 60: an interval `(a, b]` with `a<U<b` contributes `log P(a<T<=U) - log Z` (flexsurv does not intersect, 10.4); mpmath of the intersection."""
    left, right = UPPER_WINDOWS[window]
    rows = [("exact", 1.0, NAN), *BEYOND_UPPER]
    out = ll(_dist(family, params), _batch(rows, left, right))
    want = _reference_rows(family, params, rows, left, right)
    assert np.isfinite(out).all()
    assert (out[1:] < 0).all()
    assert_close(out, want)


COVERING = {
    "left-at-upper": lambda: _two("left", U0),
    "left-beyond-upper": lambda: _two("left", 5.0),
    "left-beyond-upper-right-only": lambda: _two("left", 5.0, lt=0.0),
    "interval-equal-to-window": lambda: _two("interval", L0, U0),
    "interval-containing-window": lambda: _two("interval", 0.2, 4.0),
    "interval-to-infinity-at-entry": lambda: _two("interval", L0, INF),
}


def _assert_uninformative(family, params, data):
    value = ll(_dist(family, params), data)
    assert value[1] == 0.0, value
    assert_close(value[0], _row0(family, params))
    np.testing.assert_array_equal(mask(data.issues, "uninformative"), [False, True])
    assert not np.asarray(data.issues.invalid_rows).any()
    data.issues.raise_if_invalid()


@pytest.mark.parametrize("family, params, case", _with_family(list(COVERING), CENSORING, TRUNCATION))
def test_censored_set_covering_the_window_is_exactly_zero(family, params, case):
    """spec 2/7.1/10.4, rev 60: a left row at `c>=U` and intervals equal to or containing `(L, U]` contribute exactly zero (not a positive value, as flexsurv's would be, 10.4) and are `uninformative`."""
    _assert_uninformative(family, params, COVERING[case]())


# Windows far in the upper tail: log-normal ``(0, 0.5)`` over ``(e**4, e**4.5]`` (``z`` from 8 to 9,
# ``Z ~ 6e-16``); log-logistic ``(4, 1)`` over ``(1e3, 2e3]`` (``Z ~ 1e-12``).
DEEP = {
    "lognormal": ((0.0, 0.5), (float(np.exp(4.0)), float(np.exp(4.5)))),
    "loglogistic": ((4.0, 1.0), (1e3, 2e3)),
}


@pytest.mark.parametrize(
    "family",
    [pytest.param(family, marks=pytest.mark.feature(FAMILY_FEATURE[family], CENSORING, TRUNCATION), id=family) for family in DEEP],
)
def test_covering_rows_are_exactly_zero_in_a_deep_window(family):
    """spec 7.1, rev 60: the zero is "detected structurally rather than by arithmetic cancellation", so it holds where `Z` is tiny: left at `U`, an interval containing the window, right at entry."""
    params, (left, right) = DEEP[family]
    data = flag_data(
        ["left", "interval", "right"],
        [right, 0.5 * left, left],
        [NAN, 2.0 * right, NAN],
        left_trunc=[left] * 3,
        right_trunc=[right] * 3,
    )
    np.testing.assert_array_equal(ll(_dist(family, params), data), [0.0, 0.0, 0.0])


# ================================================== censored at or before entry; outside the window


def _assert_row1_invalid(family, params, data):
    dist = _dist(family, params)
    value = ll(dist, data)
    invalid = np.asarray(data.issues.invalid_rows).astype(bool)
    np.testing.assert_array_equal(invalid, [False, True])
    np.testing.assert_array_equal(np.isnan(value), invalid)
    assert_close(value[0], _row0(family, params))
    bad_data, bad_params, other = nan_causes(dist, data, value)
    np.testing.assert_array_equal(bad_data, invalid)
    assert not bad_params.any()
    assert not other.any()


AT_ENTRY_EXACT_RIGHT = {
    "right-at-entry": lambda: _two("right", L0),
    "right-at-entry-no-upper": lambda: _two("right", L0, rt=INF),
}


@pytest.mark.parametrize("family, params, case", _with_family(list(AT_ENTRY_EXACT_RIGHT), TRUNCATION))
def test_right_censored_at_entry_is_exactly_zero(family, params, case):
    """spec 2/3.3/7.1/10.4, rev 60: right-censored at `c=L` contributes "exactly zero, for any valid `L` and `U`" and is `uninformative`."""
    _assert_uninformative(family, params, AT_ENTRY_EXACT_RIGHT[case]())


# (builder, mask named by the spec or None).
BEFORE_ENTRY_EXACT_RIGHT = {
    "right-before-entry": (lambda: _two("right", 0.2), "censored_before_entry"),
    "bounds-zero-to-inf-after-entry": (
        lambda: survjax.SurvivalData.from_bounds(
            f([1.0, 0.0]), f([1.0, INF]), exact=b([True, False]), left_trunc=f([L0, L0]), right_trunc=f([U0, U0])
        ),
        "censored_before_entry",
    ),
}
BEFORE_ENTRY_CENSORED = {
    "left-at-entry": (lambda: _two("left", L0), None),
    "left-before-entry": (lambda: _two("left", 0.3), None),
    "interval-ending-at-entry": (lambda: _two("interval", 0.1, L0), None),
    "interval-ending-before-entry": (lambda: _two("interval", 0.1, 0.3), None),
    "interval-to-infinity-before-entry": (lambda: _two("interval", 0.2, INF), "censored_before_entry"),
}
OUTSIDE_EXACT_RIGHT = {
    "exact-at-entry": (lambda: _two("exact", L0), "exact_outside_truncation"),
    "exact-before-entry": (lambda: _two("exact", 0.2), "exact_outside_truncation"),
    "exact-after-upper": (lambda: _two("exact", 4.0), "exact_outside_truncation"),
    "right-at-upper": (lambda: _two("right", U0), None),
    "right-beyond-upper": (lambda: _two("right", 4.0), None),
}
OUTSIDE_CENSORED = {
    "interval-starting-at-upper": (lambda: _two("interval", U0, 4.0), None),
    "interval-beyond-upper": (lambda: _two("interval", 4.0, 6.0), None),
}


def _check_invalid(family, params, cases, case):
    build, rule = cases[case]
    data = build()
    _assert_row1_invalid(family, params, data)
    if rule is not None:
        assert mask(data.issues, rule)[1], rule


@pytest.mark.parametrize("family, params, case", _with_family(list(BEFORE_ENTRY_EXACT_RIGHT), TRUNCATION))
def test_right_censored_before_entry_is_invalid(family, params, case):
    """spec 2/3.3/6.2/7.1/10.4, rev 60: right-censored before entry (and `from_bounds(0, inf)` with `L>0`) is `censored_before_entry`: NaN, triaged to data; the valid row keeps its mpmath value."""
    _check_invalid(family, params, BEFORE_ENTRY_EXACT_RIGHT, case)


@pytest.mark.parametrize("family, params, case", _with_family(list(BEFORE_ENTRY_CENSORED), CENSORING, TRUNCATION))
def test_left_and_interval_rows_at_or_before_entry_are_invalid(family, params, case):
    """spec 2/3.1/6.2/7.1/10.4, rev 60: left rows at or before `L` and intervals ending at or before `L` are empty intersections; `(c, +inf]` with `c<L` is `censored_before_entry`; NaN, triaged to data."""
    _check_invalid(family, params, BEFORE_ENTRY_CENSORED, case)


@pytest.mark.parametrize("family, params, case", _with_family(list(OUTSIDE_EXACT_RIGHT), TRUNCATION))
def test_exact_and_right_rows_outside_the_window_are_invalid(family, params, case):
    """spec 2/3.3/6.2/7.1/10.4, rev 60: exact times outside `L<t<=U` are `exact_outside_truncation`; right rows at or beyond `U` are empty intersections; NaN, triaged to data."""
    _check_invalid(family, params, OUTSIDE_EXACT_RIGHT, case)


@pytest.mark.parametrize("family, params, case", _with_family(list(OUTSIDE_CENSORED), CENSORING, TRUNCATION))
def test_interval_rows_outside_the_window_are_invalid(family, params, case):
    """spec 2/6.2/7.1/10.4, rev 60: intervals starting at or beyond `U` are empty intersections: NaN, triaged to data."""
    _check_invalid(family, params, OUTSIDE_CENSORED, case)
