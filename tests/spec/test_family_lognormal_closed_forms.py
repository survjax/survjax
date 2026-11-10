"""Feature `family.lognormal`: closed-form direct methods and their tails (spec 2, 4.1, 4.2, 8.1, 12.1; rev 55).

4.1: `S(t) = Phi(-(log t - mu)/sigma)`; "compute `log S` as `log1p(-ndtr(-x))`
for `x>0` and `log_ndtr(x)` for `x<=0`, where `x=-(log t - mu)/sigma`, and
`log_cdf` symmetrically. JAX's `log_ndtr` loses relative precision for
moderately large positive arguments (up to 3.5e-2 relative error near `x=7.9`
in float64, 4e-2 near `x=5` in float32)". 4.2: `log_survival` "Must keep
relative precision when `|log S|` is small"; "`log_hazard` ... Built-ins
override analytically". 12.1 (Boundaries): "log-normal tails against mpmath
over `x` in [0,10] (float64) and [0,7] (float32)"; "compare log probabilities
when probabilities underflow".

Hand derivation (spec 2: ``H = -log S``, ``f = -dS/dt``, ``h = f/S``), with
``z = (log t - mu)/sigma = -x``: ``log S = log Phi(-z)``, ``log F = log
Phi(z)``, ``log f = -log t - log sigma - log(2 pi)/2 - z**2/2``, ``log h =
log f - log S``. At ``t = exp(mu)`` (``z = 0``): ``S = F = 1/2``, ``H = log 2``,
``log f = -mu - log sigma - log(2 pi)/2``, ``log h = log f + log 2``.

References: ``family_reference.direct`` (mpmath, 50 digits, at the
dtype-rounded inputs) with its condition-number tolerance (at least the 12.1
``rtol``; zero absolute tolerance, so a tail value rounded to zero fails).
"""

import math

import family_reference as fr
import jax
import numpy as np
import pytest
from helpers import X64, assert_close, assert_tail_close, f

import survjax

LOGNORMAL = pytest.mark.feature("family.lognormal")
METHODS = fr.DIRECT_METHODS
X_MAX = 10 if X64 else 7  # 12.1: x in [0, 10] (float64), [0, 7] (float32)


def _dist(mu, sigma):
    return survjax.LogNormal(meanlog=f(mu), sdlog=f(sigma))


def _check(mu, sigma, t, method):
    got = getattr(_dist(mu, sigma), method)(f(t))
    want, rtol = fr.direct("lognormal", (mu, sigma), t, method)
    assert_tail_close(got, want, rtol, err_msg=f"{method} at t={t}")


# --------------------------------------------------------------- hand values at the median


@LOGNORMAL
@pytest.mark.parametrize("mu, sigma", [(0.0, 1.0), (1.5, 0.4), (-2.0, 2.5)])
def test_values_at_the_median_time_are_hand_derived(mu, sigma):
    """spec 2/4.1, rev 55: at `t = exp(mu)`: `S = 1/2`, `log S = log F = -log 2`, `H = log 2`, `log f = -mu - log sigma - log(2π)/2`, `log h = log f + log 2`."""
    d = _dist(mu, sigma)
    t = np.exp(f(mu))
    log_f = -float(f(mu)) - math.log(float(f(sigma))) - 0.5 * math.log(2 * math.pi)
    assert_close(d.survival(t), 0.5)
    assert_close(d.log_survival(t), -math.log(2))
    assert_close(d.log_cdf(t), -math.log(2))
    assert_close(d.cumulative_hazard(t), math.log(2))
    assert_close(d.log_prob(t), log_f)
    assert_close(d.log_hazard(t), log_f + math.log(2))
    assert_close(d.hazard(t), math.exp(log_f + math.log(2)))


# --------------------------------------------------------------- interior values


@LOGNORMAL
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("t", [0.05, 0.7, 1.35, 3.0, 40.0])
def test_interior_values_match_mpmath(t, method):
    """spec 4.1/8.1, rev 55: every direct method of `LogNormal(meanlog=0.3, sdlog=0.8)` at interior times."""
    _check(0.3, 0.8, t, method)


@LOGNORMAL
@pytest.mark.parametrize("method", METHODS)
def test_paired_batch_values_match_mpmath(method):
    """spec 4.1/5, rev 55: batch `(N,)` against times `(N,)` pairs entries."""
    mu, sigma, t = [-1.0, 0.0, 0.5, 2.0], [0.3, 1.0, 2.0, 0.7], [0.2, 4.0, 0.01, 9.0]
    got = np.asarray(getattr(survjax.LogNormal(meanlog=f(mu), sdlog=f(sigma)), method)(f(t)))
    for i in range(4):
        want, rtol = fr.direct("lognormal", (mu[i], sigma[i]), t[i], method)
        assert_tail_close(got[i], want, rtol, err_msg=f"{method}[{i}]")


@LOGNORMAL
@pytest.mark.parametrize("method", METHODS)
def test_scalar_time_against_a_batch(method):
    """spec 5, rev 55: a scalar time against batch `(N,)` gives `(N,)`, entry `i` from its own parameters."""
    mu, sigma = [-1.0, 0.0, 2.0], [0.5, 1.0, 1.5]
    got = np.asarray(getattr(survjax.LogNormal(meanlog=f(mu), sdlog=f(sigma)), method)(f(1.7)))
    assert got.shape == (3,)
    for i in range(3):
        want, rtol = fr.direct("lognormal", (mu[i], sigma[i]), 1.7, method)
        assert_tail_close(got[i], want, rtol, err_msg=f"{method}[{i}]")


# --------------------------------------------------------------- tails over x in [0, 10] / [0, 7] (12.1)


@LOGNORMAL
@pytest.mark.parametrize("x", range(X_MAX + 1))
def test_log_survival_keeps_relative_precision_for_positive_x(x):
    """spec 4.1/4.2/12.1, rev 55: `log S = log1p(-ndtr(-x))` for `x>0`: tiny `|log S|` (`7.6e-24` at `x = 10`) keeps relative precision, where a plain `log_ndtr` errs by up to 4e-2."""
    _check(0.0, 1.0, fr.lognormal_time(0.0, 1.0, x), "log_survival")


@LOGNORMAL
@pytest.mark.parametrize("x", range(X_MAX + 1))
def test_log_cdf_keeps_relative_precision_symmetrically(x):
    """spec 4.1/12.1, rev 55: "`log_cdf` symmetrically": at `z = (log t - mu)/sigma = x > 0`, `|log F| = Phi(-x)` keeps relative precision."""
    _check(0.0, 1.0, fr.lognormal_time(0.0, 1.0, -x), "log_cdf")


@LOGNORMAL
@pytest.mark.parametrize("x", [2.5, 6.0, X_MAX])
def test_tails_with_non_unit_parameters(x):
    """spec 4.1/12.1, rev 55: the same tails for `meanlog = 2`, `sdlog = 0.4` (scale change), for `log S` and `log F`."""
    _check(2.0, 0.4, fr.lognormal_time(2.0, 0.4, x), "log_survival")
    _check(2.0, 0.4, fr.lognormal_time(2.0, 0.4, -x), "log_cdf")


@LOGNORMAL
@pytest.mark.parametrize("method", ["log_survival", "cumulative_hazard", "log_prob", "log_hazard", "hazard"])
@pytest.mark.parametrize("x", [-5.0, -10.0, -20.0, -37.0])
def test_upper_tail_where_survival_is_tiny(x, method):
    """spec 4.1/4.2/12.1, rev 55: upper tail (`x < 0`, `log S = log_ndtr(x)`, `-689` at `x = -37`): log quantities, `H` and the analytic hazard against mpmath."""
    _check(0.0, 1.0, fr.lognormal_time(0.0, 1.0, x), method)


@LOGNORMAL
@pytest.mark.parametrize("method", ["log_cdf", "log_prob", "log_hazard"])
@pytest.mark.parametrize("z", [-5.0, -10.0, -20.0, -37.0])
def test_lower_tail_where_the_cdf_is_tiny(z, method):
    """spec 4.1/12.1, rev 55: lower tail (`z = (log t - mu)/sigma < 0`, `log F = log_ndtr(z)`): `log F`, `log f` and `log h` against mpmath (`h ≈ f` there)."""
    _check(0.0, 1.0, fr.lognormal_time(0.0, 1.0, -z), method)


# --------------------------------------------------------------- identities (4.2 check_family list, 12.1)


@LOGNORMAL
def test_log_survival_and_log_cdf_partition_probability():
    """spec 4.2/12.1, rev 55: `logaddexp(log S, log F) = 0` across the tails and the interior."""
    d = _dist(0.3, 0.8)
    t = f([fr.lognormal_time(0.3, 0.8, x) for x in (X_MAX, 3.0, 0.5, 0.0, -2.0, -6.0, -X_MAX)])
    assert_close(np.logaddexp(np.asarray(d.log_survival(t), np.float64), np.asarray(d.log_cdf(t), np.float64)), 0.0)


@LOGNORMAL
@pytest.mark.parametrize("t", [0.1, 1.0, 3.0, 20.0])
def test_density_and_hazard_agree_with_the_derivative_of_log_survival(t):
    """spec 2/4.2, rev 55: `h = -d log S/dt` (autodiff in `t`), `log h = log_hazard`, and `log_prob = log h + log S`."""
    d = _dist(0.3, 0.8)
    minus_dlogs = -float(jax.grad(lambda tt: d.log_survival(tt))(f(t)))
    assert_close(d.log_hazard(f(t)), math.log(minus_dlogs))
    assert_close(d.log_prob(f(t)), math.log(minus_dlogs) + float(np.asarray(d.log_survival(f(t)))))


@LOGNORMAL
def test_survival_underflows_to_zero_not_nan_in_upper_tail():
    """spec 8.1/12.1, rev 55: `survival` is `P(T>t)`; at `x = -40` (`S ≈ 4e-350`) it underflows to `0`, while `log S` stays finite."""
    d = _dist(0.0, 1.0)
    t = f(fr.lognormal_time(0.0, 1.0, -40.0))
    np.testing.assert_array_equal(np.asarray(d.survival(t)), 0.0)
    want, rtol = fr.direct("lognormal", (0.0, 1.0), float(t), "log_survival")
    assert_tail_close(d.log_survival(t), want, rtol)
