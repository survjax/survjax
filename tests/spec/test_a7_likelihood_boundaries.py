"""A7: likelihood boundary cases for exact and right-censored rows (spec 2, 4.5, 7.1, 7.2, 12.1, 12.2; rev 33).

Hand values (``Z = 1``, no truncation):

- Exact rows at the smallest positive normal time ``tiny`` (``finfo.tiny``):
  Exponential ``log r - r tiny``; Weibull
  ``log k - log b + (k-1) log(tiny/b) - (tiny/b)**k``. With ``k<1`` this is
  large and positive; with ``k>1`` large and negative; finite in every case.
- Right-censored rows at ``c=0`` are uninformative: ``log S(0) = 0`` comes
  from the base (4.5), and the inactive density piece uses
  ``reference_time()`` (7.2), so even where ``f(0)`` is infinite (Weibull
  ``k<1``) the contribution is exactly zero with zero, finite gradients.
- ``mixture_cure`` (``spec_families.py``), improper with ``S(+inf) = c``:
  "Sets closed at `+inf` include `T=+inf`, so for an improper family
  `P(c<T<=+inf)=S(c)`" (2). A right-censored row contributes
  ``log S(c) = log(cure + (1-cure) exp(-rate c))``, tending to ``log cure``,
  not ``log(S(c) - S(+inf))``. An exact row contributes
  ``log((1-cure) rate) - rate t``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import (
    FLOAT,
    assert_close,
    assert_tail_close,
    b,
    f,
    ll,
    mp_exponential_contribution,
    mp_weibull_contribution,
    nan_causes,
)
from tolerances import TOLERANCES

import survjax

EXACT = True
RIGHT = False
TINY = float(np.finfo(FLOAT).tiny)
RTOL = TOLERANCES[FLOAT]["rtol"]


def _weibull(k, s, t, exact):
    return ll(survjax.Weibull(shape=f(k), scale=f(s)), survjax.SurvivalData(f(t), b(exact)))


# ------------------------------------------------------- smallest positive time


@pytest.mark.step("A7")
def test_exponential_exact_at_smallest_normal_time():
    """spec 3.3/7.1/12.1, rev 33: an exact time `t=tiny>0` is valid; `log f = log r - r tiny`, here `log 0.5`."""
    value = ll(survjax.Exponential(rate=f(0.5)), survjax.SurvivalData(f(TINY), b(EXACT)))
    expected, rtol = mp_exponential_contribution(0.5, TINY, exact=True)
    assert_tail_close(value, expected, rtol)


@pytest.mark.step("A7")
@pytest.mark.parametrize("k", [0.5, 1.0, 2.5])
def test_weibull_exact_at_smallest_normal_time(k):
    """spec 3.3/4.1/7.1/12.1, rev 33: Weibull exact row at `t=tiny`, `b=2`: finite, against mpmath."""
    value = _weibull(k, 2.0, TINY, EXACT)
    expected, rtol = mp_weibull_contribution(k, 2.0, TINY, exact=True)
    assert np.isfinite(value)
    assert_tail_close(value, expected, rtol)


@pytest.mark.step("A7")
def test_exponential_right_censored_at_smallest_normal_time():
    """spec 7.1/12.1, rev 33: right-censored at `c=tiny`, `r=1`: contribution `-tiny`, negative, compared with zero `atol`."""
    value = ll(survjax.Exponential(rate=f(1.0)), survjax.SurvivalData(f(TINY), b(RIGHT)))
    assert value < 0
    assert_tail_close(value, -TINY, RTOL)


@pytest.mark.step("A7")
def test_weibull_right_censored_at_smallest_normal_time_with_shape_below_one():
    """spec 7.1/12.1, rev 33: `k=0.5, b=1, c=tiny`: contribution `-sqrt(tiny)`, a normal number, not rounded to zero."""
    value = _weibull(0.5, 1.0, TINY, RIGHT)
    expected, rtol = mp_weibull_contribution(0.5, 1.0, TINY, exact=False)
    assert value < 0
    assert_tail_close(value, expected, rtol)


@pytest.mark.step("A7")
@pytest.mark.parametrize("k", [0.5, 1.0, 2.5])
def test_weibull_exact_at_smallest_normal_time_has_finite_gradients(k):
    """spec 4.5/7.3, rev 33: at `t=tiny` the exact-row gradients are finite (`d/dk = 1/k + log z - z**k log z`).

    ``z = tiny/2``; ``z**k log z`` is finite (it underflows to zero for
    ``k >= 1``). The value is checked against the hand derivative with the
    ``1 + |log z|`` amplification of ``log z``.
    """
    k_, s, t = (float(np.asarray(v, dtype=FLOAT)) for v in (k, 2.0, TINY))
    log_z = np.log(t) - np.log(s)
    data = survjax.SurvivalData(f(t), b(EXACT))
    grad = jax.grad(
        lambda kk, ss: survjax.log_likelihood(survjax.Weibull(shape=kk, scale=ss), data),
        argnums=(0, 1),
    )
    dk, db = grad(jnp.asarray(f(k_)), jnp.asarray(f(s)))
    h = np.exp(k_ * log_z)
    eps = float(np.finfo(FLOAT).eps)
    rtol = max(RTOL, 16.0 * eps * (1.0 + abs(log_z)) * abs(log_z) / abs(1.0 / k_ + log_z))
    assert np.isfinite(np.asarray(dk)) and np.isfinite(np.asarray(db))
    assert_tail_close(dk, 1.0 / k_ + log_z - h * log_z, rtol)
    assert_close(db, (k_ / s) * (h - 1.0))


# ------------------------------------------------------- right-censored at zero


@pytest.mark.step("A7")
@pytest.mark.parametrize("k", [0.5, 1.0, 2.5])
def test_weibull_right_censored_at_zero_is_exactly_zero(k):
    """spec 2/4.5/7.1, rev 33: `c=0=L` is uninformative: exactly `0`, also where `f(0)` is infinite (`k<1`)."""
    value = _weibull(k, 2.0, 0.0, RIGHT)
    np.testing.assert_array_equal(value, 0.0)


@pytest.mark.step("A7")
@pytest.mark.parametrize("k", [0.5, 1.0, 2.5])
def test_weibull_right_censored_at_zero_has_zero_gradients(k):
    """spec 4.5/7.2, rev 33: the inactive density piece does not leak: zero, finite gradients in `k` and `b` at `c=0`.

    For ``k=0.5`` the density at ``0`` is infinite, so a density evaluated at
    the censoring time instead of ``reference_time()`` would give NaN here.
    """
    data = survjax.SurvivalData(f(0.0), b(RIGHT))
    grad = jax.grad(
        lambda kk, ss: survjax.log_likelihood(survjax.Weibull(shape=kk, scale=ss), data),
        argnums=(0, 1),
    )
    for g in grad(jnp.asarray(f(k)), jnp.asarray(f(2.0))):
        np.testing.assert_array_equal(np.asarray(g), 0.0)


@pytest.mark.step("A7")
def test_exponential_right_censored_at_zero_has_zero_gradient_in_mixed_batch():
    """spec 4.5/7.2, rev 33: the `c=0` row keeps value and gradient zero next to exact rows sharing the rate.

    Rows: right at ``0``, exact at ``1``. Total ``log r - r``; ``d/dr = 1/r - 1``
    at ``r=0.5`` is ``1``, all from the exact row.
    """
    data = survjax.SurvivalData(f([0.0, 1.0]), b([RIGHT, EXACT]))

    def row(r, i):
        return survjax.log_likelihood(survjax.Exponential(rate=r), data)[i]

    r = jnp.asarray(f(0.5))
    np.testing.assert_array_equal(np.asarray(row(r, 0)), 0.0)
    np.testing.assert_array_equal(np.asarray(jax.grad(row)(r, 0)), 0.0)
    assert_close(jax.grad(row)(r, 1), 1.0)


# ------------------------------------------------------- improper family


def _cure(cure=0.3, rate=0.5):
    return fam.mixture_cure()(cure=f(cure), rate=f(rate))


@pytest.mark.step("A7")
def test_improper_family_right_censored_contribution_is_log_survival():
    """spec 2/7.1, rev 33: `(c, +inf]` includes `T=+inf`: the contribution is `log S(c)`, not `log(S(c) - S(+inf))`.

    ``cure=0.3, rate=0.5, c=2``: ``log(0.3 + 0.7 e^-1)``.
    """
    value = ll(_cure(), survjax.SurvivalData(f(2.0), b(RIGHT)))
    assert_close(value, np.log(0.3 + 0.7 * np.exp(-1.0)))


@pytest.mark.step("A7")
def test_improper_family_right_censored_far_tail_tends_to_log_cure():
    """spec 2/7.1, rev 33: `c=200`: `log(0.3 + 0.7 e^-100)` equals `log 0.3` to working precision; finite, not `-inf` or NaN."""
    value = ll(_cure(), survjax.SurvivalData(f(200.0), b(RIGHT)))
    assert_close(value, np.log(0.3))


@pytest.mark.step("A7")
def test_improper_family_exact_contribution():
    """spec 2/4.2/7.1, rev 33: exact `t=2`: `log f = log(0.7 * 0.5) - 1` through the generic `log_prob` default."""
    value = ll(_cure(), survjax.SurvivalData(f(2.0), b(EXACT)))
    assert_close(value, np.log(0.35) - 1.0)


@pytest.mark.step("A7")
def test_improper_family_right_censored_at_zero_is_exactly_zero():
    """spec 2/4.5/7.1, rev 33: `c=0` is uninformative for an improper family too: `log S(0) = 0` from the base."""
    value = ll(_cure(), survjax.SurvivalData(f(0.0), b(RIGHT)))
    np.testing.assert_array_equal(value, 0.0)


@pytest.mark.step("A7")
def test_improper_family_mixed_rows_have_no_nan_cause():
    """spec 7.1, rev 33: valid exact and right rows of an improper family give finite values; the triage recipe finds no cause."""
    dist = _cure()
    data = survjax.SurvivalData(f([0.0, 1.0, 2.0, 200.0]), b([RIGHT, EXACT, RIGHT, RIGHT]))
    value = ll(dist, data)
    assert np.isfinite(value).all()
    for mask in nan_causes(dist, data, value):
        assert not mask.any()
