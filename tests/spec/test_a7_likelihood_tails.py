"""A7: likelihood tails, tiny masses and no floors (spec 4.1, 7.1, 7.2, 12.1, 12.2; rev 33).

Slice A scope (12.2, rev 33): the numerical-behaviour and stable-tail evidence
of 12.1 applies here to exact and right-censored rows without truncation.
Narrow intervals, cancellation, ``Z=0`` and tiny ``Z`` are Slice B.

Without truncation (``Z = 1``) the contributions are those of
``test_a7_likelihood_mpmath.py``. In the tails:

- A right-censored row contributes ``log S(c) = -H(c)`` exactly. Where ``H``
  is tiny (``S`` rounds to 1) the contribution must keep relative precision
  and never round to zero; where ``S`` underflows it must stay ``-H``, finite,
  never ``-inf`` and never floored (12.1: "tiny nonzero masses, both tails ...
  no floors").
- An exact row contributes ``log f(t)``, finite wherever it is representable
  even where ``f`` itself underflows.

Comparisons of tiny values and derivatives use zero absolute tolerance
(12.1), with the relative tolerances of ``helpers._reference``; the
derivative tolerances are justified in each test.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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
EPS = float(np.finfo(FLOAT).eps)
RTOL = TOLERANCES[FLOAT]["rtol"]
LOG_TINY = float(np.log(np.finfo(FLOAT).tiny))


def _exp(rate, t, exact):
    return ll(survjax.Exponential(rate=f(rate)), survjax.SurvivalData(f(t), b(exact)))


def _weibull(k, s, t, exact):
    return ll(survjax.Weibull(shape=f(k), scale=f(s)), survjax.SurvivalData(f(t), b(exact)))


def _rounded(*values):
    return [float(np.asarray(v, dtype=FLOAT)) for v in values]


# ------------------------------------------------------- tiny cumulative hazard


@pytest.mark.step("A7")
def test_exponential_right_censored_tiny_hazard_keeps_relative_precision():
    """spec 7.1/12.1, rev 33: `r=1e-10, c=1e-10`: contribution `-r c = -1e-20`, negative, not rounded to zero."""
    value = _exp(1e-10, 1e-10, RIGHT)
    expected, rtol = mp_exponential_contribution(1e-10, 1e-10, exact=False)
    assert value < 0
    assert_tail_close(value, expected, rtol)


@pytest.mark.step("A7")
def test_weibull_right_censored_tiny_hazard_keeps_relative_precision():
    """spec 7.1/12.1, rev 33: `k=3, b=1, c=1e-10`: contribution `-z**k = -1e-30` (a normal number in both dtypes)."""
    value = _weibull(3.0, 1.0, 1e-10, RIGHT)
    expected, rtol = mp_weibull_contribution(3.0, 1.0, 1e-10, exact=False)
    assert value < 0
    assert_tail_close(value, expected, rtol)


@pytest.mark.step("A7")
def test_exponential_right_censored_tiny_hazard_rate_gradient():
    """spec 7.3/12.1, rev 33: d/dr of `-r c` is `-c`; at `c=1e-10` it is tiny and compared with zero `atol`.

    The derivative of a product is computed to a few ulps; the 12.1 ``rtol``
    applies.
    """
    (c,) = _rounded(1e-10)
    data = survjax.SurvivalData(f(c), b(RIGHT))
    grad = jax.grad(lambda r: survjax.Exponential(rate=r).log_likelihood(data))
    assert_tail_close(grad(jnp.asarray(f(1e-10))), -c, RTOL)


@pytest.mark.step("A7")
def test_weibull_right_censored_tiny_hazard_parameter_gradients():
    """spec 7.3/12.1, rev 33: `k=3, b=1, c=1e-10`, `H=z**k=1e-30`: d/dk `= -H log z`, d/db `= k H / b`.

    Both are tiny (about ``6.9e-29`` and ``3e-30``) and compared with zero
    absolute tolerance. ``H`` carries a relative error of about
    ``eps * (1 + k |log z|)`` (``k |log z| = 69``), hence the scale.
    """
    k, s, c = _rounded(3.0, 1.0, 1e-10)
    log_z = np.log(c / s)
    h = np.exp(k * log_z)
    data = survjax.SurvivalData(f(c), b(RIGHT))
    grad = jax.grad(
        lambda kk, ss: survjax.Weibull(shape=kk, scale=ss).log_likelihood(data),
        argnums=(0, 1),
    )
    dk, db = grad(jnp.asarray(f(k)), jnp.asarray(f(s)))
    rtol = max(RTOL, 16.0 * EPS * (1.0 + k * abs(log_z)))
    assert_tail_close(dk, -h * log_z, rtol)
    assert_tail_close(db, k * h / s, rtol)


# ------------------------------------------------------- upper tail: S and f underflow


@pytest.mark.step("A7")
def test_exponential_right_censored_where_survival_underflows():
    """spec 7.1/12.1, rev 33: `r=1, c=1e6`: `S` underflows; the contribution is `-1e6`, finite and below `log(tiny)` (no floor)."""
    value = _exp(1.0, 1e6, RIGHT)
    assert np.isfinite(value)
    assert value < LOG_TINY
    assert_tail_close(value, -1e6, RTOL)


@pytest.mark.step("A7")
def test_exponential_exact_where_density_underflows():
    """spec 7.1/12.1, rev 33: `r=2, t=500`: `f = 2 e^-1000` underflows; `log f = log 2 - 1000`."""
    value = _exp(2.0, 500.0, EXACT)
    expected, rtol = mp_exponential_contribution(2.0, 500.0, exact=True)
    assert np.isfinite(value)
    assert_tail_close(value, expected, rtol)


@pytest.mark.step("A7")
def test_weibull_right_censored_where_survival_underflows():
    """spec 7.1/12.1, rev 33: `k=2, b=1, c=100`: contribution `-1e4`, finite, not `-inf`, not floored."""
    value = _weibull(2.0, 1.0, 100.0, RIGHT)
    expected, rtol = mp_weibull_contribution(2.0, 1.0, 100.0, exact=False)
    assert value < LOG_TINY
    assert_tail_close(value, expected, rtol)


@pytest.mark.step("A7")
def test_weibull_exact_where_density_underflows():
    """spec 7.1/12.1, rev 33: `k=2, b=1, t=100`: `log f = log 2 + log 100 - 1e4`, finite."""
    value = _weibull(2.0, 1.0, 100.0, EXACT)
    expected, rtol = mp_weibull_contribution(2.0, 1.0, 100.0, exact=True)
    assert np.isfinite(value)
    assert_tail_close(value, expected, rtol)


@pytest.mark.step("A7")
def test_weibull_exact_with_shape_below_one_where_density_underflows():
    """spec 7.1/12.1, rev 33: `k=0.5, b=1, t=1e6`: `H=1000`, `log f = log 0.5 - 0.5 log 1e6 - 1000`."""
    value = _weibull(0.5, 1.0, 1e6, EXACT)
    expected, rtol = mp_weibull_contribution(0.5, 1.0, 1e6, exact=True)
    assert_tail_close(value, expected, rtol)


@pytest.mark.step("A7")
def test_weibull_exact_with_huge_cumulative_hazard_is_finite():
    """spec 7.1/12.1, rev 33: `k=8, b=1, t=1e4`: `H=1e32` (representable in both dtypes); `log f ~ -1e32`, finite, not `-inf`."""
    value = _weibull(8.0, 1.0, 1e4, EXACT)
    expected, rtol = mp_weibull_contribution(8.0, 1.0, 1e4, exact=True)
    assert np.isfinite(value)
    assert_tail_close(value, expected, rtol)


@pytest.mark.step("A7")
def test_weibull_right_censored_with_huge_cumulative_hazard_is_finite():
    """spec 7.1/12.1, rev 33: `k=8, b=1, c=1e4`: contribution `-1e32`, finite, not `-inf`."""
    value = _weibull(8.0, 1.0, 1e4, RIGHT)
    expected, rtol = mp_weibull_contribution(8.0, 1.0, 1e4, exact=False)
    assert np.isfinite(value)
    assert_tail_close(value, expected, rtol)


@pytest.mark.step("A7")
def test_exponential_upper_tail_rate_gradients():
    """spec 7.3/12.1, rev 33: `r=2`: exact `t=500` gives d/dr `= 1/r - t = -499.5`; right `c=1e6` gives `-c`.

    Smooth O(1)-relative quantities; the 12.1 ``rtol`` applies.
    """
    data = survjax.SurvivalData(f([500.0, 1e6]), b([EXACT, RIGHT]))

    def total(r, i):
        return survjax.Exponential(rate=r).log_likelihood(data)[i]

    r = jnp.asarray(f(2.0))
    assert_tail_close(jax.grad(total)(r, 0), 0.5 - 500.0, RTOL)
    assert_tail_close(jax.grad(total)(r, 1), -1e6, RTOL)


@pytest.mark.step("A7")
def test_weibull_upper_tail_exact_parameter_gradients():
    """spec 7.3/12.1, rev 33: `k=2, b=1, t=100` (`H=1e4`): hand-derived gradients where `f` underflows.

    ``d/dk log f = 1/k + log z - H log z``; ``d/db log f = (k/b)(H - 1)``.
    The dominant terms carry the relative error of ``H`` (``k |log z| = 9.2``
    amplification), hence the scale.
    """
    k, s, t = 2.0, 1.0, 100.0
    log_z = np.log(t / s)
    h = (t / s) ** k
    data = survjax.SurvivalData(f(t), b(EXACT))
    grad = jax.grad(
        lambda kk, ss: survjax.Weibull(shape=kk, scale=ss).log_likelihood(data),
        argnums=(0, 1),
    )
    dk, db = grad(jnp.asarray(f(k)), jnp.asarray(f(s)))
    rtol = max(RTOL, 16.0 * EPS * (1.0 + k * abs(log_z)))
    assert_tail_close(dk, 1.0 / k + log_z - h * log_z, rtol)
    assert_tail_close(db, (k / s) * (h - 1.0), rtol)


# ------------------------------------------------------- overflowing contributions (rev 34)

# Weibull k=50, b=1 at t=1e10: H = (1e10)**50 = 1e500, beyond the finite range
# of float32 and float64. The true log contribution (about -1e500) is finite
# but unrepresentable; spec 7.1 (rev 34) fixes the result as -inf.
OVERFLOW = (50.0, 1.0, 1e10)


@pytest.mark.step("A7")
def test_right_censored_row_whose_contribution_overflows_is_minus_infinity():
    """spec 7.1, rev 34: a valid row whose log contribution is below the dtype's finite range gives `-inf`, not NaN.

    Right-censored, Weibull ``k=50, b=1, c=1e10``: ``-H = -1e500``.
    """
    value = _weibull(*OVERFLOW, RIGHT)
    assert value == -np.inf


@pytest.mark.step("A7")
def test_exact_row_whose_contribution_overflows_is_minus_infinity():
    """spec 7.1, rev 34: exact, Weibull `k=50, b=1, t=1e10`: `log f = log 50 + 49 log 1e10 - 1e500` overflows to `-inf`, not NaN."""
    value = _weibull(*OVERFLOW, EXACT)
    assert value == -np.inf


@pytest.mark.step("A7")
def test_overflowing_row_has_no_nan_cause_and_leaves_other_rows_unchanged():
    """spec 7.1, rev 34: the `-inf` row is valid (no data or parameter issue, no NaN); the interior row keeps its value.

    Shared ``k=50, b=1``; rows: exact ``t=1`` (``log 50 - 1``), right
    ``c=1e10`` (``-inf``).
    """
    k, s, _ = OVERFLOW
    dist = survjax.Weibull(shape=f(k), scale=f(s))
    data = survjax.SurvivalData(f([1.0, 1e10]), b([EXACT, RIGHT]))
    value = ll(dist, data)
    assert value[1] == -np.inf
    assert_close(value[0], np.log(50.0) - 1.0)
    for mask in nan_causes(dist, data, value):
        assert not mask.any()


@pytest.mark.step("A7")
def test_forward_mode_derivatives_of_other_rows_survive_an_overflowing_row():
    """spec 7.3, rev 34: with shared parameters, "Forward-mode derivatives ... are unaffected" by an infinite row.

    Shared ``k=50, b=1``; row 0 exact at ``t=1``
    (``z=1``): ``d/dk log f = 1/k + log z - z**k log z = 1/50``,
    ``d/db log f = (k/b)(z**k - 1) = 0``. Row 1 overflows. ``jacfwd`` of the
    per-row vector gives row 0 these values; row 1's derivatives are outside
    the guarantee and not asserted.
    """
    k, s, _ = OVERFLOW
    data = survjax.SurvivalData(f([1.0, 1e10]), b([EXACT, RIGHT]))

    def rows(kk, ss):
        return survjax.Weibull(shape=kk, scale=ss).log_likelihood(data)

    dk, db = jax.jacfwd(rows, argnums=(0, 1))(jnp.asarray(f(k)), jnp.asarray(f(s)))
    assert_close(np.asarray(dk)[0], 1.0 / k)
    assert_close(np.asarray(db)[0], 0.0)


@pytest.mark.step("A7")
def test_reverse_mode_derivatives_with_per_row_parameters_survive_an_overflowing_row():
    """spec 7.3, rev 34: with per-row parameters, reverse-mode derivatives of the other rows are unaffected by an infinite row.

    Per-row ``k = (2, 50)``, ``b = (1, 1)``; row 0 exact at ``t=0.5``
    (``z=0.5``, ``H=0.25``): ``d/dk_0 = 1/2 + log 0.5 - 0.25 log 0.5``,
    ``d/db_0 = (k/b)(H - 1) = -1.5``. Row 1 (right, ``c=1e10``) overflows.
    ``jax.grad`` of row 0 gives these in entry 0. Entry 1 belongs to the
    infinite row itself (its reverse pass is ``0 x inf``), outside the
    guarantee, so it is not asserted.
    """
    data = survjax.SurvivalData(f([0.5, 1e10]), b([EXACT, RIGHT]))

    def row0(kk, ss):
        return survjax.Weibull(shape=kk, scale=ss).log_likelihood(data)[0]

    dk, db = jax.grad(row0, argnums=(0, 1))(jnp.asarray(f([2.0, 50.0])), jnp.asarray(f([1.0, 1.0])))
    log_z = np.log(0.5)
    assert_close(np.asarray(dk)[0], 0.5 + log_z - 0.25 * log_z)
    assert_close(np.asarray(db)[0], -1.5)


# ------------------------------------------------------- lower tail: times near zero


@pytest.mark.step("A7")
def test_weibull_exact_near_zero_with_decreasing_hazard_is_large_and_finite():
    """spec 4.1/7.1/12.1, rev 33: `k=0.5, b=1, t=1e-30`: `log f = log 0.5 + 0.5 * 69.08 - 1e-15`, about `+33.8`."""
    value = _weibull(0.5, 1.0, 1e-30, EXACT)
    expected, rtol = mp_weibull_contribution(0.5, 1.0, 1e-30, exact=True)
    assert value > 0
    assert_tail_close(value, expected, rtol)


@pytest.mark.step("A7")
def test_weibull_exact_near_zero_with_increasing_hazard_is_finite():
    """spec 4.1/7.1/12.1, rev 33: `k=3, b=1, t=1e-12`: `log f = log 3 + 2 log 1e-12 - 1e-36`, about `-54.2`."""
    value = _weibull(3.0, 1.0, 1e-12, EXACT)
    expected, rtol = mp_weibull_contribution(3.0, 1.0, 1e-12, exact=True)
    assert_tail_close(value, expected, rtol)


@pytest.mark.step("A7")
def test_exponential_exact_near_zero_is_log_rate():
    """spec 7.1/12.1, rev 33: `r=2, t=1e-30`: `log f = log 2 - 2e-30`, equal to `log 2` to working precision."""
    value = _exp(2.0, 1e-30, EXACT)
    expected, rtol = mp_exponential_contribution(2.0, 1e-30, exact=True)
    assert_tail_close(value, expected, rtol)


@pytest.mark.step("A7")
def test_exponential_exact_with_tiny_rate():
    """spec 7.1/12.1, rev 33: `r=1e-30, t=1`: `log f = log 1e-30 - 1e-30`, about `-69.08`; the tiny rate is not floored."""
    value = _exp(1e-30, 1.0, EXACT)
    expected, rtol = mp_exponential_contribution(1e-30, 1.0, exact=True)
    assert_tail_close(value, expected, rtol)


# ------------------------------------------------------- mixed rows are independent


@pytest.mark.step("A7")
def test_tail_rows_do_not_change_interior_rows():
    """spec 7.1/7.2, rev 33: evaluation is per row; tail rows in the same batch leave an interior row's value unchanged.

    Weibull ``k=2, b=1``: rows exact ``t=1e-30``, exact ``t=100``, right
    ``c=100``, right ``c=1e-15`` and the interior exact ``t=1``
    (``log 2 - 1``). The interior row equals its value evaluated alone, at
    12.1 tolerances (bitwise equality across batch layouts is not specified).
    """
    dist = survjax.Weibull(shape=f(2.0), scale=f(1.0))
    data = survjax.SurvivalData(
        f([1e-30, 100.0, 100.0, 1e-15, 1.0]), b([EXACT, EXACT, RIGHT, RIGHT, EXACT])
    )
    mixed = ll(dist, data)
    alone = ll(dist, survjax.SurvivalData(f([1.0]), b([EXACT])))
    assert_close(mixed[4], alone[0])
    assert np.isfinite(mixed).all()
