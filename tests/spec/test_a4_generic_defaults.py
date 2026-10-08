"""A4: generic defaults derived from `log_survival` beyond `log_prob` (spec 4.2, 4.5, 8.1).

Spec rev 14 (12.2) puts the pointwise direct methods ``survival``,
``cumulative_hazard``, ``log_cdf``, ``log_hazard`` and ``hazard``, with their
generic defaults, in Slice A; the delivery schedule places them at A4 (Q-024,
resolved).

Hand derivations for ConstHazard (``log S = -r t``): ``S = exp(-r t)``,
``H = r t``, ``log F = log(-expm1(-r t))``, ``log h = log r``, ``h = r``.
For PowerHazard (``z = t/b``): ``H = z**k``, ``log h = log(k/b) + (k-1) log z``.
"""

import jax
import numpy as np
import pytest
import spec_families as fam
from helpers import INF, NAN, assert_close, f


def _exp(rate=0.5):
    return fam.const_hazard()(rate=f(rate))


# --------------------------------------------------------------- interior values


@pytest.mark.step("A4")
def test_default_survival_is_exp_log_survival():
    """spec 8.1, rev 11: `survival(t)` is `P(T>t)`, i.e. `exp(log_survival(t))`."""
    assert_close(_exp().survival(f([0.5, 2.0])), np.exp(-0.5 * np.array([0.5, 2.0])))


@pytest.mark.step("A4")
def test_default_cumulative_hazard_is_minus_log_survival():
    """spec 4.2, rev 11: default `cumulative_hazard = -log S`."""
    assert_close(_exp().cumulative_hazard(f([0.5, 2.0])), 0.5 * np.array([0.5, 2.0]))


@pytest.mark.step("A4")
def test_default_log_cdf():
    """spec 4.2, rev 11: default `log_cdf = log(-expm1(log S))`."""
    t = np.array([0.5, 2.0, 30.0])
    assert_close(_exp().log_cdf(f(t)), np.log(-np.expm1(-0.5 * t)))


@pytest.mark.step("A4")
def test_default_log_cdf_keeps_precision_for_tiny_cdf():
    """spec 4.2, rev 11: the default `log_cdf` is accurate until `|log S|` underflows.

    ``r=1``, ``t=1e-20``: ``log F = log(-expm1(-x)) = log x + log(1 - x/2 + ...)``,
    which equals ``log(1e-20)`` to within ``1e-20``. A ``log(1 - S)`` form would
    give ``-inf``.
    """
    assert_close(_exp(1.0).log_cdf(f(1e-20)), np.log(1e-20))


@pytest.mark.step("A4")
def test_default_log_hazard_constant_hazard():
    """spec 4.2, rev 11: default `log_hazard = log(-d/dt log S)`; here `log r` at every time."""
    assert_close(_exp().log_hazard(f([0.5, 2.0, 30.0])), np.full(3, np.log(0.5)))


@pytest.mark.step("A4")
def test_default_hazard_constant_hazard():
    """spec 8.1, rev 11: `hazard(t) = f(t)/S(t)`; here `r`."""
    assert_close(_exp().hazard(f([0.5, 2.0])), [0.5, 0.5])


@pytest.mark.step("A4")
def test_default_log_hazard_power_hazard():
    """spec 4.2, rev 11: default `log_hazard` for a Weibull-like family: `log(k/b) + (k-1) log(t/b)`."""
    t = np.array([0.5, 2.0, 10.0])
    out = fam.power_hazard()(shape=f(3.0), scale=f(2.0)).log_hazard(f(t))
    assert_close(out, np.log(3.0 / 2.0) + 2.0 * np.log(t / 2.0))


@pytest.mark.step("A4")
def test_default_log_hazard_with_large_cumulative_hazard():
    """spec 4.2, rev 11: the derivative form stays accurate where `H` is large (no `log_prob - log_survival` cancellation).

    ``k=2``, ``b=1``, ``t=1e3``: ``H = 1e6``; ``log h = log 2 + log 1e3``.
    """
    out = fam.power_hazard()(shape=f(2.0), scale=f(1.0)).log_hazard(f(1e3))
    assert_close(out, np.log(2.0) + np.log(1e3))


# --------------------------------------------------------------- policy: t < 0


@pytest.mark.step("A4")
def test_negative_time_survival_is_one():
    """spec 4.5, rev 11: `t<0` gives `S=1`."""
    np.testing.assert_array_equal(
        np.asarray(_exp().survival(f([-INF, -1.0]))), [1.0, 1.0]
    )


@pytest.mark.step("A4")
def test_negative_time_log_cdf_is_minus_infinity():
    """spec 4.5, rev 11: `t<0` gives `log_cdf=-inf`."""
    np.testing.assert_array_equal(
        np.asarray(_exp().log_cdf(f([-INF, -1.0]))), [-INF, -INF]
    )


@pytest.mark.step("A4")
def test_negative_time_cumulative_hazard_is_zero():
    """spec 4.5, rev 11: `t<0` gives `H=0`."""
    np.testing.assert_array_equal(
        np.asarray(_exp().cumulative_hazard(f([-INF, -1.0]))), [0.0, 0.0]
    )


@pytest.mark.step("A4")
def test_negative_time_hazard_is_zero():
    """spec 4.5, rev 11: `t<0` gives `h=0`."""
    np.testing.assert_array_equal(
        np.asarray(_exp().hazard(f([-INF, -1.0]))), [0.0, 0.0]
    )


@pytest.mark.step("A4")
def test_negative_time_log_hazard_is_minus_infinity():
    """spec 4.5, rev 21: `t<0` gives `log_hazard=-inf`."""
    np.testing.assert_array_equal(
        np.asarray(_exp().log_hazard(f([-INF, -1.0]))), [-INF, -INF]
    )


# --------------------------------------------------------------- policy: t = 0 and t = +inf


@pytest.mark.step("A4")
def test_zero_time_survival_cumulative_hazard_and_log_cdf():
    """spec 4.5, rev 11: at `t=0` the family's limits: `S(0)=1`, `H(0)=0`, `log F(0)=-inf`."""
    dist = _exp()
    np.testing.assert_array_equal(np.asarray(dist.survival(f(0.0))), 1.0)
    np.testing.assert_array_equal(np.asarray(dist.cumulative_hazard(f(0.0))), 0.0)
    np.testing.assert_array_equal(np.asarray(dist.log_cdf(f(0.0))), -INF)


@pytest.mark.step("A4")
def test_infinite_time_survival_is_zero():
    """spec 4.5, rev 11: `t=+inf` gives `log S(+inf)` from the family (`-inf`, so `S=0`, for a proper family)."""
    np.testing.assert_array_equal(np.asarray(_exp().survival(f(INF))), 0.0)


@pytest.mark.step("A4")
def test_infinite_time_cumulative_hazard_is_infinite():
    """spec 4.5, rev 11: `t=+inf` gives `H = -log S(+inf) = +inf` for a proper family."""
    np.testing.assert_array_equal(np.asarray(_exp().cumulative_hazard(f(INF))), INF)


@pytest.mark.step("A4")
def test_infinite_time_log_cdf_is_zero():
    """spec 4.5, rev 11: `t=+inf` gives `log F = log(-expm1(-inf)) = 0` for a proper family."""
    np.testing.assert_array_equal(np.asarray(_exp().log_cdf(f(INF))), 0.0)


# --------------------------------------------------------------- policy: NaN


@pytest.mark.step("A4")
def test_nan_time_survival_is_nan():
    """spec 4.5, rev 11: NaN `t` gives NaN (`survival`)."""
    assert np.isnan(np.asarray(_exp().survival(f(NAN))))


@pytest.mark.step("A4")
def test_nan_time_cumulative_hazard_is_nan():
    """spec 4.5, rev 11: NaN `t` gives NaN (`cumulative_hazard`)."""
    assert np.isnan(np.asarray(_exp().cumulative_hazard(f(NAN))))


@pytest.mark.step("A4")
def test_nan_time_log_cdf_is_nan():
    """spec 4.5, rev 11: NaN `t` gives NaN (`log_cdf`)."""
    assert np.isnan(np.asarray(_exp().log_cdf(f(NAN))))


@pytest.mark.step("A4")
def test_nan_time_log_hazard_is_nan():
    """spec 4.5, rev 11: NaN `t` gives NaN (`log_hazard`).

    The family's ``log_survival`` returns NaN, but its time derivative is ``-r``
    whatever the primal, so ``log(-d/dt log S)`` alone would be finite; the
    policy requires NaN.
    """
    assert np.isnan(np.asarray(_exp().log_hazard(f(NAN))))


@pytest.mark.step("A4")
def test_nan_time_hazard_is_nan():
    """spec 4.5, rev 11: NaN `t` gives NaN (`hazard`).

    The family's ``log_survival`` returns NaN, but its time derivative is ``-r``
    whatever the primal, so ``log(-d/dt log S)`` alone would be finite; the
    policy requires NaN.
    """
    assert np.isnan(np.asarray(_exp().hazard(f(NAN))))


@pytest.mark.step("A4")
def test_invalid_parameter_gives_nan_survival_and_log_cdf_defaults():
    """spec 4.5, rev 11: invalid (NaN) parameters give NaN."""
    dist = fam.power_hazard()(shape=f(NAN), scale=f(2.0))
    assert np.isnan(np.asarray(dist.survival(f(1.0))))
    assert np.isnan(np.asarray(dist.log_cdf(f(1.0))))


# --------------------------------------------------------------- transformations


@pytest.mark.step("A4")
def test_defaults_agree_under_jit():
    """spec 12.1, rev 11: eager and `jit` agree for the generic defaults, including boundary inputs."""
    dist = fam.power_hazard()(shape=f([0.5, 1.0, 3.0]), scale=f(2.0))
    t = f([[-1.0], [0.0], [0.7], [4.0], [INF]])
    for name in ("survival", "cumulative_hazard", "log_cdf"):
        eager = np.asarray(getattr(dist, name)(t))
        compiled = np.asarray(jax.jit(lambda d, s, n=name: getattr(d, n)(s))(dist, t))
        assert_close(compiled, eager, err_msg=name)
