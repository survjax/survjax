"""observation.censoring: cross-kind identities, property-based (spec 2, 3.1, 7.1, 12.1; rev 51).

12.1 (Cross-kind identities): "interval `(c, +inf]` equals right-censored at
`c`; `(0, c]` equals left-censored at `c`; `logaddexp` of adjacent intervals
equals their union; full coverage gives exactly zero". Hypothesis with
``derandomize=True``. Without truncation ``Z = 1`` and each censored
contribution is ``log P(lower < T <= upper)``, so for ``0 < a < b < c``:

- ``P(0<T<=a) + P(a<T<=b) = P(0<T<=b)``;
- ``P(a<T<=b) + P(b<T<=c) = P(a<T<=c)``;
- ``P(a<T<=b) + P(b<T<=inf) = P(a<T<=inf)``;
- ``P(0<T<=c) + P(c<T<=inf) = 1``.

Draws keep every mass moderate (``t/b`` in about ``[0.3, 3.9]``, so ``H``
between about ``8e-3`` and ``220``, and endpoint ratios at least ``1.05``),
so the 12.1 ordinary tolerances apply to the log values.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers import INF, NAN, assert_close, b, f, flag_data, ll
from hypothesis import given, settings
from hypothesis import strategies as st

import survjax

FEATURE = pytest.mark.feature("observation.censoring")
PROFILE = settings(derandomize=True, max_examples=40, deadline=None)


@st.composite
def weibull_and_times(draw):
    """A Weibull law and three increasing times ``a < b < c``."""
    k = draw(st.floats(0.4, 4.0))
    s = draw(st.floats(0.5, 4.0))
    a = draw(st.floats(0.3, 1.5)) * s
    b_ = a * draw(st.floats(1.05, 1.6))
    c = b_ * draw(st.floats(1.05, 1.6))
    return k, s, a, b_, c


def _dist(k, s):
    return survjax.Weibull(shape=f(k), scale=f(s))


def _one(dist, kind, t, t2=NAN):
    return ll(dist, flag_data([kind], [t], [t2]))[0]


def _rounded_times(*ts):
    return [float(np.asarray(f(t))) for t in ts]


@FEATURE
@PROFILE
@given(weibull_and_times())
def test_interval_to_infinity_equals_right_censoring(case):
    """spec 3.1 and 12.1, rev 51: an interval-flag row `(c, +inf]` "normalizes to right censoring at `c`" and has its contribution."""
    k, s, _, _, c = case
    (c,) = _rounded_times(c)
    dist = _dist(k, s)
    np.testing.assert_array_equal(_one(dist, "interval", c, INF), _one(dist, "right", c))


@FEATURE
@PROFILE
@given(weibull_and_times())
def test_bounds_from_zero_equals_left_censoring(case):
    """spec 3.1, 3.2 and 12.1, rev 51: `from_bounds(0, c)` is left-censored at `c` and has the left-flag contribution."""
    k, s, _, _, c = case
    (c,) = _rounded_times(c)
    dist = _dist(k, s)
    bounds = ll(dist, survjax.SurvivalData.from_bounds(f([0.0]), f([c]), exact=b([False])))[0]
    np.testing.assert_array_equal(bounds, _one(dist, "left", c))


@FEATURE
@PROFILE
@given(weibull_and_times())
def test_left_plus_adjacent_interval_is_the_larger_left(case):
    """spec 2 and 12.1, rev 51: `logaddexp(left(a), interval(a, b)) = left(b)`."""
    k, s, a, b_, _ = case
    a, b_ = _rounded_times(a, b_)
    dist = _dist(k, s)
    union = jnp.logaddexp(_one(dist, "left", a), _one(dist, "interval", a, b_))
    assert_close(union, _one(dist, "left", b_))


@FEATURE
@PROFILE
@given(weibull_and_times())
def test_adjacent_intervals_add_to_their_union(case):
    """spec 2 and 12.1, rev 51: `logaddexp(interval(a, b), interval(b, c)) = interval(a, c)`."""
    k, s, a, b_, c = case
    a, b_, c = _rounded_times(a, b_, c)
    dist = _dist(k, s)
    union = jnp.logaddexp(_one(dist, "interval", a, b_), _one(dist, "interval", b_, c))
    assert_close(union, _one(dist, "interval", a, c))


@FEATURE
@PROFILE
@given(weibull_and_times())
def test_interval_plus_right_tail_is_the_earlier_right(case):
    """spec 2 and 12.1, rev 51: `logaddexp(interval(a, b), right(b)) = right(a)`."""
    k, s, a, b_, _ = case
    a, b_ = _rounded_times(a, b_)
    dist = _dist(k, s)
    union = jnp.logaddexp(_one(dist, "interval", a, b_), _one(dist, "right", b_))
    assert_close(union, _one(dist, "right", a))


@FEATURE
@PROFILE
@given(weibull_and_times())
def test_left_plus_right_at_the_same_time_covers_everything(case):
    """spec 2 and 12.1, rev 51: `logaddexp(left(c), right(c)) = log 1 = 0` (the two sets partition `(0, +inf]`)."""
    k, s, _, _, c = case
    (c,) = _rounded_times(c)
    dist = _dist(k, s)
    assert_close(jnp.logaddexp(_one(dist, "left", c), _one(dist, "right", c)), 0.0)


@FEATURE
@PROFILE
@given(st.floats(0.4, 4.0), st.floats(0.5, 4.0))
def test_full_coverage_is_exactly_zero(k, s):
    """spec 2, 7.1 and 12.1, rev 51: sets covering the whole untruncated window, `(0, +inf]` in the bounds and interval-flag forms, give exactly zero.

    7.1: "exactly zero, detected structurally rather than by arithmetic
    cancellation"; both rows are right-censored at ``c = L = 0`` (3.2, 3.3).
    """
    dist = _dist(k, s)
    bounds = ll(dist, survjax.SurvivalData.from_bounds(f([0.0]), f([INF]), exact=b([False])))
    flag = ll(dist, flag_data(["interval"], [0.0], [INF]))
    assert bounds[0] == 0.0
    assert flag[0] == 0.0
