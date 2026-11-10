"""Feature `family.loglogistic`: closed-form direct methods and their tails (spec 2, 4.1, 4.2, 8.1, 12.1; rev 55).

4.1: `S(t) = 1/(1+(t/alpha)**beta)`; "`log S = -softplus(beta*(log t - log
alpha))` and `log F = -softplus(-beta*(log t - log alpha))`, both accurate in
both tails". 4.2: `log_survival` "Must keep relative precision when `|log S|`
is small"; `log_hazard` "Built-ins override analytically". 12.1: "compare log
probabilities when probabilities underflow"; tiny log probabilities use zero
absolute tolerance.

Hand derivation (spec 2), ``u = beta (log t - log alpha)``: ``log S =
-softplus(u)``, ``log F = -softplus(-u)``, ``H = softplus(u)``, ``log f = log
beta - log alpha + (beta-1)(log t - log alpha) - 2 softplus(u)``, ``log h =
log f - log S``. At ``t = alpha`` (``u = 0``): ``S = F = 1/2``, ``H = log 2``,
``log f = log beta - log alpha - 2 log 2``, ``log h = log beta - log alpha -
log 2``. With ``beta = 1``: ``S = alpha/(alpha + t)``, ``h = 1/(alpha + t)``.

References: ``family_reference.direct`` (mpmath, 50 digits, dtype-rounded
inputs, condition-number tolerance, zero absolute tolerance). Tails reach
``|u| = 700`` in float64 and ``80`` in float32 (``exp(-80)`` is still a normal
float32), with ``alpha = 1.5`` so that no tested value is exactly zero.
"""

import math

import family_reference as fr
import jax
import numpy as np
import pytest
from helpers import X64, assert_close, assert_tail_close, f

import survjax

LOGLOGISTIC = pytest.mark.feature("family.loglogistic")
METHODS = fr.DIRECT_METHODS
U_MAX = 700.0 if X64 else 80.0
TAIL_U = [-U_MAX, -100.0 if X64 else -60.0, -30.0, -5.0, 5.0, 30.0, 100.0 if X64 else 60.0, U_MAX]


def _dist(beta, alpha):
    return survjax.LogLogistic(shape=f(beta), scale=f(alpha))


def _check(beta, alpha, t, method):
    got = getattr(_dist(beta, alpha), method)(f(t))
    want, rtol = fr.direct("loglogistic", (beta, alpha), t, method)
    assert_tail_close(got, want, rtol, err_msg=f"{method} at t={t}")


# --------------------------------------------------------------- hand values


@LOGLOGISTIC
@pytest.mark.parametrize("beta, alpha", [(0.5, 1.0), (2.0, 3.0), (7.0, 0.2)])
def test_values_at_the_scale_are_hand_derived(beta, alpha):
    """spec 2/4.1, rev 55: at `t = alpha`: `S = 1/2`, `log S = log F = -log 2`, `H = log 2`, `log f = log beta - log alpha - 2 log 2`, `log h = log beta - log alpha - log 2`."""
    d = _dist(beta, alpha)
    t = f(alpha)
    log_h = math.log(float(f(beta))) - math.log(float(t)) - math.log(2)
    assert_close(d.survival(t), 0.5)
    assert_close(d.log_survival(t), -math.log(2))
    assert_close(d.log_cdf(t), -math.log(2))
    assert_close(d.cumulative_hazard(t), math.log(2))
    assert_close(d.log_prob(t), log_h - math.log(2))
    assert_close(d.log_hazard(t), log_h)
    assert_close(d.hazard(t), math.exp(log_h))


@LOGLOGISTIC
@pytest.mark.parametrize("t", [0.1, 1.0, 4.0, 50.0])
def test_unit_shape_is_hand_derived(t):
    """spec 4.1, rev 55: `beta = 1`: `S = alpha/(alpha + t)`, `h = 1/(alpha + t)`, `log f = log alpha - 2 log(alpha + t)`."""
    alpha = 2.0
    d = _dist(1.0, alpha)
    (tt,) = fr.rounded(t)
    assert_close(d.survival(f(t)), alpha / (alpha + tt))
    assert_close(d.hazard(f(t)), 1.0 / (alpha + tt))
    assert_close(d.log_prob(f(t)), math.log(alpha) - 2 * math.log(alpha + tt))


# --------------------------------------------------------------- interior values


@LOGLOGISTIC
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("beta", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("t", [0.05, 0.7, 2.0, 40.0])
def test_interior_values_match_mpmath(t, beta, method):
    """spec 4.1/8.1, rev 55: every direct method of `LogLogistic(shape=beta, scale=1.5)` at interior times, for decreasing (`beta<1`), constant-shape (`beta=1`) and unimodal (`beta>1`) hazards."""
    _check(beta, 1.5, t, method)


@LOGLOGISTIC
@pytest.mark.parametrize("method", METHODS)
def test_paired_batch_values_match_mpmath(method):
    """spec 4.1/5, rev 55: batch `(N,)` against times `(N,)` pairs entries."""
    beta, alpha, t = [0.7, 1.0, 1.8, 4.0], [0.5, 3.0, 10.0, 1.5], [0.2, 4.0, 12.0, 0.9]
    got = np.asarray(getattr(survjax.LogLogistic(shape=f(beta), scale=f(alpha)), method)(f(t)))
    for i in range(4):
        want, rtol = fr.direct("loglogistic", (beta[i], alpha[i]), t[i], method)
        assert_tail_close(got[i], want, rtol, err_msg=f"{method}[{i}]")


# --------------------------------------------------------------- tails (4.1: "accurate in both tails")


@LOGLOGISTIC
@pytest.mark.parametrize("u", TAIL_U)
def test_log_survival_is_accurate_in_both_tails(u):
    """spec 4.1/4.2, rev 55: `log S = -softplus(u)`: tiny `|log S| ≈ exp(u)` for `u ≪ 0` (relative precision), `log S ≈ -u` for `u ≫ 0`."""
    _check(2.0, 1.5, fr.loglogistic_time(2.0, 1.5, u), "log_survival")


@LOGLOGISTIC
@pytest.mark.parametrize("u", TAIL_U)
def test_log_cdf_is_accurate_in_both_tails(u):
    """spec 4.1, rev 55: `log F = -softplus(-u)`: `log F ≈ u` for `u ≪ 0`, tiny `|log F| ≈ exp(-u)` for `u ≫ 0`, negative and nonzero."""
    _check(2.0, 1.5, fr.loglogistic_time(2.0, 1.5, u), "log_cdf")


@LOGLOGISTIC
@pytest.mark.parametrize("method", ["cumulative_hazard", "log_prob", "log_hazard", "hazard"])
@pytest.mark.parametrize("u", TAIL_U)
def test_other_methods_in_both_tails(u, method):
    """spec 4.1/4.2/12.1, rev 55: `H`, `log f`, the analytic `log h` and `h` in both tails, against mpmath."""
    _check(2.0, 1.5, fr.loglogistic_time(2.0, 1.5, u), method)


@LOGLOGISTIC
@pytest.mark.parametrize("beta", [0.3, 8.0])
@pytest.mark.parametrize("u", [-20.0, 20.0])
def test_tails_with_other_shapes(u, beta):
    """spec 4.1, rev 55: `log S` and `log F` tails for a small and a large `beta` (`t = alpha exp(u/beta)` spans `1e-29` to `1e29` at `beta = 0.3`, inside float32)."""
    t = fr.loglogistic_time(beta, 1.5, u)
    _check(beta, 1.5, t, "log_survival")
    _check(beta, 1.5, t, "log_cdf")


@LOGLOGISTIC
def test_survival_underflows_to_zero_not_nan_in_upper_tail():
    """spec 8.1, rev 55: where `S = exp(-softplus(u))` underflows (`u = 800` float64, `110` float32) it is `0`, while `log S ≈ -u` stays finite."""
    u = 800.0 if X64 else 110.0
    t = fr.loglogistic_time(4.0, 1.5, u)
    d = _dist(4.0, 1.5)
    np.testing.assert_array_equal(np.asarray(d.survival(f(t))), 0.0)
    want, rtol = fr.direct("loglogistic", (4.0, 1.5), t, "log_survival")
    assert_tail_close(d.log_survival(f(t)), want, rtol)


# --------------------------------------------------------------- identities (4.2 check_family list, 12.1)


@LOGLOGISTIC
def test_log_survival_and_log_cdf_partition_probability():
    """spec 4.2/12.1, rev 55: `logaddexp(log S, log F) = 0` across both tails."""
    d = _dist(2.0, 1.5)
    t = f([fr.loglogistic_time(2.0, 1.5, u) for u in TAIL_U])
    assert_close(np.logaddexp(np.asarray(d.log_survival(t), np.float64), np.asarray(d.log_cdf(t), np.float64)), 0.0)


@LOGLOGISTIC
@pytest.mark.parametrize("beta", [0.5, 1.0, 3.0])
def test_density_and_hazard_agree_with_the_derivative_of_log_survival(beta):
    """spec 2/4.2, rev 55: `h = -d log S/dt` (autodiff in `t`), `log h = log_hazard`, `log_prob = log h + log S`."""
    d = _dist(beta, 1.5)
    t = f(2.2)
    minus_dlogs = -float(jax.grad(lambda tt: d.log_survival(tt))(t))
    assert_close(d.log_hazard(t), math.log(minus_dlogs))
    assert_close(d.log_prob(t), math.log(minus_dlogs) + float(np.asarray(d.log_survival(t))))
