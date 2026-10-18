"""A4: the base applies the direct-method value policy for every registered family (spec 4.2, 4.5, rev 12).

Uses families whose code is written only for ``t >= 0`` (``naive_power``,
``naive_log_prob_override`` in ``spec_families.py``). Rev 12: "family code
need only be correct on `[0, +inf]`" (rev 26: "for finite `t>0`"); for ``t<0``
and NaN ``t`` the base substitutes a safe input under ``stop_gradient``, so the
policy values have no NaN parameter gradients. Rev 26: the base also supplies
``log S``, ``S``, ``log F`` and ``H`` at ``t=0`` and ``t=+inf``; the generic
``log_prob`` default at ``t=0`` carries no guarantee, so it is not tested here. ``k=0.5`` makes the naive formula NaN for ``t<0``;
``k=2`` makes it finite but wrong (``-(t/b)**2 < 0``).

``survival``, ``cumulative_hazard``, ``log_cdf`` and ``hazard`` are Slice A
methods by spec rev 14 (Q-024, resolved).
"""

import jax
import numpy as np
import pytest
import spec_families as fam
from helpers import INF, NAN, assert_close, f, log1mexp


def _naive(k=0.5, b=2.0):
    return fam.naive_power()(shape=f(k), scale=f(b))


def _override(k=0.5, b=2.0):
    return fam.naive_log_prob_override()(shape=f(k), scale=f(b))


# --------------------------------------------------------------- values for t < 0


@pytest.mark.step("A4")
def test_unguarded_log_survival_is_zero_for_negative_time():
    """spec 4.2/4.5, rev 12: `log_survival` is called only on `[0, +inf]`; `t<0` gives `log S = 0`."""
    for k in (0.5, 2.0):
        out = _naive(k).log_survival(f([-INF, -1.0, -1e-30]))
        np.testing.assert_array_equal(
            np.asarray(out), [0.0, 0.0, 0.0], err_msg=f"k={k}"
        )


@pytest.mark.step("A4")
def test_unguarded_default_log_prob_is_minus_infinity_for_negative_time():
    """spec 4.5, rev 12: `t<0` gives `log_prob=-inf` for a family whose formula is undefined there."""
    for k in (0.5, 2.0):
        out = _naive(k).log_prob(f([-INF, -1.0]))
        np.testing.assert_array_equal(np.asarray(out), [-INF, -INF], err_msg=f"k={k}")


@pytest.mark.step("A4")
def test_unguarded_survival_is_one_for_negative_time():
    """spec 4.5, rev 12: `t<0` gives `S=1`."""
    np.testing.assert_array_equal(
        np.asarray(_naive().survival(f([-INF, -1.0]))), [1.0, 1.0]
    )


@pytest.mark.step("A4")
def test_unguarded_cumulative_hazard_is_zero_for_negative_time():
    """spec 4.5, rev 12: `t<0` gives `H=0`."""
    np.testing.assert_array_equal(
        np.asarray(_naive(2.0).cumulative_hazard(f([-INF, -1.0]))), [0.0, 0.0]
    )


@pytest.mark.step("A4")
def test_unguarded_log_cdf_is_minus_infinity_for_negative_time():
    """spec 4.5, rev 12: `t<0` gives `log_cdf=-inf`."""
    np.testing.assert_array_equal(
        np.asarray(_naive().log_cdf(f([-INF, -1.0]))), [-INF, -INF]
    )


@pytest.mark.step("A4")
def test_unguarded_hazard_is_zero_for_negative_time():
    """spec 4.5, rev 12: `t<0` gives `h=0`."""
    np.testing.assert_array_equal(
        np.asarray(_naive(2.0).hazard(f([-INF, -1.0]))), [0.0, 0.0]
    )


@pytest.mark.step("A4")
def test_unguarded_log_hazard_is_minus_infinity_for_negative_time():
    """spec 4.5, rev 21: `t<0` gives `log_hazard=-inf` for a family whose formula is undefined there."""
    for k in (0.5, 2.0):
        out = _naive(k).log_hazard(f([-INF, -1.0]))
        np.testing.assert_array_equal(np.asarray(out), [-INF, -INF], err_msg=f"k={k}")


# --------------------------------------------------------------- NaN, zero and +inf


@pytest.mark.step("A4")
def test_unguarded_nan_time_gives_nan():
    """spec 4.5, rev 12: NaN `t` gives NaN for `log_survival` and `log_prob`."""
    dist = _naive()
    assert np.isnan(np.asarray(dist.log_survival(f(NAN))))
    assert np.isnan(np.asarray(dist.log_prob(f(NAN))))


@pytest.mark.step("A4")
def test_unguarded_zero_time_survival_values_come_from_the_base():
    """spec 4.5, rev 26: at `t=0`, `log S=0`, `S=1`, `log_cdf=-inf`, `H=0` "come from the base" (`S(0)=1`, section 2)."""
    for k in (0.5, 1.0, 2.0):
        dist = _naive(k)
        np.testing.assert_array_equal(np.asarray(dist.log_survival(f(0.0))), 0.0)
        np.testing.assert_array_equal(np.asarray(dist.survival(f(0.0))), 1.0)
        np.testing.assert_array_equal(np.asarray(dist.log_cdf(f(0.0))), -INF)
        np.testing.assert_array_equal(np.asarray(dist.cumulative_hazard(f(0.0))), 0.0)


@pytest.mark.step("A4")
def test_unguarded_infinite_time_log_prob_is_minus_infinity():
    """spec 4.5, rev 26: at `t=+inf` the base gives `log_prob=-inf` (rev 12: even where a default formula would give `+inf - inf`)."""
    np.testing.assert_array_equal(np.asarray(_naive(2.0).log_prob(f(INF))), -INF)


@pytest.mark.step("A4")
def test_unguarded_infinite_time_log_survival_comes_from_the_base():
    """spec 4.5, rev 26: at `t=+inf`, `log_survival` equals `log_survival_at_infinity()` (default `-inf`)."""
    np.testing.assert_array_equal(np.asarray(_naive().log_survival(f(INF))), -INF)


@pytest.mark.step("A4")
def test_unguarded_interior_values_are_the_family_formula():
    """spec 4.2, rev 26: for finite `t>0` the family's own formula is used.

    ``k=0.5``, ``b=2``, ``t=1``: ``log S = -(0.5)**0.5``;
    ``log f = log(k/b) + (k-1) log(t/b) - (t/b)**k = log 0.25 - 0.5 log 0.5 - 0.5**0.5``.
    """
    dist = _naive()
    assert_close(dist.log_survival(f(1.0)), -np.sqrt(0.5))
    assert_close(dist.log_prob(f(1.0)), np.log(0.25) - 0.5 * np.log(0.5) - np.sqrt(0.5))


@pytest.mark.step("A4")
def test_unguarded_policy_applies_elementwise_with_batches():
    """spec 4.5/5, rev 26: the policy applies per element of a broadcast `(K, N)` result.

    Batch ``k = (0.5, 1, 2)``, ``b = 2``; times ``(-1, 0, 1)`` as a column;
    ``log_cdf``: rows ``t=-1`` and ``t=0`` are ``-inf`` (policy and base);
    row ``t=1`` is ``log1mexp(-(1/2)**k)``.
    """
    k = np.array([0.5, 1.0, 2.0])
    dist = fam.naive_power()(shape=f(k), scale=f(2.0))
    out = np.asarray(dist.log_cdf(f([[-1.0], [0.0], [1.0]])))
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out[:2], np.full((2, 3), -INF))
    assert_close(out[2], log1mexp(-(0.5**k)))


@pytest.mark.step("A4")
def test_unguarded_policy_agrees_under_jit():
    """spec 4.5/12.1, rev 26: policy values are the same eagerly and under `jit` (`t=0` only for base-supplied `log_survival`)."""
    dist = fam.naive_power()(shape=f([0.5, 1.0, 2.0]), scale=f(2.0))
    times = {
        "log_survival": f([[-INF], [-1.0], [0.0], [1.0], [INF], [NAN]]),
        "log_prob": f([[-INF], [-1.0], [1.0], [INF], [NAN]]),
    }
    for name, t in times.items():
        eager = getattr(dist, name)(t)
        compiled = jax.jit(lambda d, s, n=name: getattr(d, n)(s))(dist, t)
        assert_close(compiled, eager, err_msg=name)


# --------------------------------------------------------------- overrides


@pytest.mark.step("A4")
def test_unguarded_log_prob_override_is_minus_infinity_for_negative_time():
    """spec 4.5, rev 12: the base applies the policy to "methods the family overrides" (`t<0`: `-inf`)."""
    out = _override().log_prob(f([-INF, -1.0]))
    np.testing.assert_array_equal(np.asarray(out), [-INF, -INF])


@pytest.mark.step("A4")
def test_unguarded_log_prob_override_nan_time_gives_nan():
    """spec 4.5, rev 12: NaN `t` gives NaN for an overridden method."""
    assert np.isnan(np.asarray(_override().log_prob(f(NAN))))


@pytest.mark.step("A4")
def test_unguarded_log_prob_override_supplies_interior_values():
    """spec 4.2, rev 26: the override is used for finite `t>0` (same value as the hand-derived density)."""
    assert_close(
        _override().log_prob(f(1.0)), np.log(0.25) - 0.5 * np.log(0.5) - np.sqrt(0.5)
    )


# --------------------------------------------------------------- gradients at policy values


def _param_grads(method, t, k):
    """Gradients of ``dist.<method>(t)`` with respect to (shape, scale) of NaivePower."""

    def value(kk, bb):
        return getattr(fam.naive_power()(shape=kk, scale=bb), method)(f(t))

    return jax.grad(value, argnums=(0, 1))(f(k), f(2.0))


@pytest.mark.step("A4")
def test_negative_time_log_survival_has_zero_parameter_gradients():
    """spec 4.5, rev 12: policy values have no NaN parameter gradients; `log S = 0` is constant, so gradients are 0.

    ``k=0.5`` (naive formula NaN at ``t=-1``) and ``k=2`` (naive formula finite and parameter-dependent).
    """
    for k in (0.5, 2.0):
        for g in _param_grads("log_survival", -1.0, k):
            np.testing.assert_array_equal(np.asarray(g), 0.0, err_msg=f"k={k}")


@pytest.mark.step("A4")
def test_negative_time_survival_has_zero_parameter_gradients():
    """spec 4.5, rev 12: `S=1` for `t<0` has zero parameter gradients."""
    for k in (0.5, 2.0):
        for g in _param_grads("survival", -1.0, k):
            np.testing.assert_array_equal(np.asarray(g), 0.0, err_msg=f"k={k}")


@pytest.mark.step("A4")
def test_negative_time_log_prob_has_zero_parameter_gradients():
    """spec 4.5, rev 20: "With valid parameters the policy values keep zero parameter gradients" (`log_prob=-inf`)."""
    for k in (0.5, 2.0):
        for g in _param_grads("log_prob", -1.0, k):
            np.testing.assert_array_equal(np.asarray(g), 0.0, err_msg=f"k={k}")


@pytest.mark.step("A4")
def test_negative_time_cumulative_hazard_has_zero_parameter_gradients():
    """spec 4.5, rev 20: `H=0` for `t<0` keeps zero parameter gradients with valid parameters."""
    for k in (0.5, 2.0):
        for g in _param_grads("cumulative_hazard", -1.0, k):
            np.testing.assert_array_equal(np.asarray(g), 0.0, err_msg=f"k={k}")


@pytest.mark.step("A4")
def test_negative_time_log_cdf_has_zero_parameter_gradients():
    """spec 4.5, rev 20: `log_cdf=-inf` for `t<0` keeps zero parameter gradients with valid parameters."""
    for k in (0.5, 2.0):
        for g in _param_grads("log_cdf", -1.0, k):
            np.testing.assert_array_equal(np.asarray(g), 0.0, err_msg=f"k={k}")


@pytest.mark.step("A4")
def test_negative_time_hazard_has_zero_parameter_gradients():
    """spec 4.5, rev 20: `h=0` for `t<0` keeps zero parameter gradients with valid parameters."""
    for k in (0.5, 2.0):
        for g in _param_grads("hazard", -1.0, k):
            np.testing.assert_array_equal(np.asarray(g), 0.0, err_msg=f"k={k}")


@pytest.mark.step("A4")
def test_negative_time_log_hazard_has_zero_parameter_gradients():
    """spec 4.5, rev 21: `log_hazard=-inf` for `t<0` keeps zero parameter gradients with valid parameters."""
    for k in (0.5, 2.0):
        for g in _param_grads("log_hazard", -1.0, k):
            np.testing.assert_array_equal(np.asarray(g), 0.0, err_msg=f"k={k}")


@pytest.mark.step("A4")
def test_negative_time_override_has_zero_parameter_gradients():
    """spec 4.5, rev 20: an overridden method is also called only on a safe input under `stop_gradient`;
    with valid parameters its policy value keeps zero parameter gradients."""

    def value(kk, bb):
        return fam.naive_log_prob_override()(shape=kk, scale=bb).log_prob(f(-1.0))

    for g in jax.grad(value, argnums=(0, 1))(f(0.5), f(2.0)):
        np.testing.assert_array_equal(np.asarray(g), 0.0)
