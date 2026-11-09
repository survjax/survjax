"""observation.truncation: conditioning identities, property-based (spec 2, 7.1, 12.1; rev 51).

Hypothesis with ``derandomize=True``. With ``Z = P(L<T<=U)``, the spec 2
table gives, for any valid row:

- conditioning: the truncated contribution equals the untruncated
  contribution of the row's set intersected with ``(L, U]`` (exact rows: the
  density), minus the untruncated contribution of ``(L, U]`` itself
  (``log Z``; the right-censored row at ``L`` when ``U = +inf``);
- partition: for ``L < c < U``, left at ``c`` and right at ``c`` split the
  window, so ``logaddexp`` of their truncated contributions is ``0``;
- full coverage: right at ``L``, left at ``c >= U`` and a finite interval
  containing ``(L, U]`` are exactly zero (7.1: "detected structurally");
- Exponential memorylessness: exact at ``t`` or right at ``c`` with entry
  ``L`` (no upper truncation) equals the untruncated row at ``t - L``,
  ``c - L``.

Draws keep masses moderate (``t/b`` in about ``[0.05, 10]``), so the 12.1
ordinary tolerances apply.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers import INF, NAN, assert_close, f, flag_data, ll
from hypothesis import given, settings
from hypothesis import strategies as st

import survjax

TRUNC = pytest.mark.feature("observation.truncation")
BOTH = pytest.mark.feature("observation.censoring", "observation.truncation")
PROFILE = settings(derandomize=True, max_examples=40, deadline=None)


def _rd(*xs):
    return [float(np.asarray(f(x))) for x in xs]


@st.composite
def window_case(draw, upper=True):
    """A Weibull law, a window ``(L, U]`` and a time ``c`` with ``L < c < U``."""
    k = draw(st.floats(0.4, 4.0))
    s = draw(st.floats(0.5, 4.0))
    lt = s * draw(st.floats(0.05, 1.2))
    rt = lt * draw(st.floats(1.3, 3.0)) if upper else INF
    frac = draw(st.floats(0.1, 0.9))
    c = lt + frac * ((rt if upper else 3.0 * lt) - lt)
    lt, c = _rd(lt, c)
    if upper:
        (rt,) = _rd(rt)
    return k, s, lt, rt, c


def _one(dist, kind, t, t2=NAN, lt=0.0, rt=INF):
    return ll(dist, flag_data([kind], [t], [t2], left_trunc=[lt], right_trunc=[rt]))[0]


@TRUNC
@PROFILE
@given(window_case(upper=False))
def test_left_truncation_conditions_on_survival_to_entry(case):
    """spec 2, rev 51: with `U = +inf`, exact and right rows equal their untruncated contributions minus `log S(L)` (right at `L`)."""
    k, s, lt, _, c = case
    dist = survjax.Weibull(shape=f(k), scale=f(s))
    log_z = _one(dist, "right", lt)
    assert_close(_one(dist, "exact", c, lt=lt), _one(dist, "exact", c) - log_z)
    assert_close(_one(dist, "right", c, lt=lt), _one(dist, "right", c) - log_z)


@TRUNC
@PROFILE
@given(window_case(upper=False), st.floats(0.1, 3.0))
def test_exponential_left_truncation_is_memoryless(case, rate):
    """spec 2, rev 51: Exponential with entry `L`: exact at `t` equals untruncated exact at `t - L`; right at `c` equals right at `c - L`."""
    _, _, lt, _, c = case
    dist = survjax.Exponential(rate=f(rate))
    (shift,) = _rd(c - lt)
    assert_close(_one(dist, "exact", c, lt=lt), _one(dist, "exact", shift))
    assert_close(_one(dist, "right", c, lt=lt), _one(dist, "right", shift))


@TRUNC
@PROFILE
@given(window_case())
def test_right_censoring_at_entry_is_exactly_zero(case):
    """spec 2 and 7.1, rev 51: right-censored at `c=L` is exactly zero "for any valid `L` and `U`"."""
    k, s, lt, rt, _ = case
    dist = survjax.Weibull(shape=f(k), scale=f(s))
    assert _one(dist, "right", lt, lt=lt, rt=rt) == 0.0
    assert _one(dist, "right", lt, lt=lt) == 0.0


@BOTH
@PROFILE
@given(window_case())
def test_two_sided_truncation_conditions_on_the_window(case):
    """spec 2, rev 51: every kind's truncated contribution is the untruncated contribution of its intersection minus the untruncated `(L, U]`."""
    k, s, lt, rt, c = case
    dist = survjax.Weibull(shape=f(k), scale=f(s))
    log_z = _one(dist, "interval", lt, rt)
    lo = 0.5 * lt  # an interval starting before entry: partial overlap at L
    hi = 2.0 * rt  # an interval ending after U: partial overlap at U
    lo, hi = _rd(lo, hi)
    assert_close(_one(dist, "exact", c, lt=lt, rt=rt), _one(dist, "exact", c) - log_z)
    assert_close(_one(dist, "right", c, lt=lt, rt=rt), _one(dist, "interval", c, rt) - log_z)
    assert_close(_one(dist, "left", c, lt=lt, rt=rt), _one(dist, "interval", lt, c) - log_z)
    assert_close(_one(dist, "interval", lo, c, lt=lt, rt=rt), _one(dist, "interval", lt, c) - log_z)
    assert_close(_one(dist, "interval", c, hi, lt=lt, rt=rt), _one(dist, "interval", c, rt) - log_z)


@BOTH
@PROFILE
@given(window_case())
def test_left_and_right_at_the_same_time_partition_the_window(case):
    """spec 2 and 12.1, rev 51: `logaddexp(left(c), right(c)) = 0` under `(L, U]` for `L < c < U`."""
    k, s, lt, rt, c = case
    dist = survjax.Weibull(shape=f(k), scale=f(s))
    total = jnp.logaddexp(_one(dist, "left", c, lt=lt, rt=rt), _one(dist, "right", c, lt=lt, rt=rt))
    assert_close(total, 0.0)


@BOTH
@PROFILE
@given(window_case())
def test_censored_sets_covering_the_window_are_exactly_zero(case):
    """spec 2, 7.1 and 12.1, rev 51: left at `U` or beyond and finite intervals containing `(L, U]` give exactly zero."""
    k, s, lt, rt, _ = case
    dist = survjax.Weibull(shape=f(k), scale=f(s))
    lo, hi = _rd(0.5 * lt, 2.0 * rt)
    assert _one(dist, "left", rt, lt=lt, rt=rt) == 0.0
    assert _one(dist, "left", hi, lt=lt, rt=rt) == 0.0
    assert _one(dist, "interval", lo, hi, lt=lt, rt=rt) == 0.0
    assert _one(dist, "interval", lt, rt, lt=lt, rt=rt) == 0.0
