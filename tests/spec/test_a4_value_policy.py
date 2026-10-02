"""A4: direct-method value policy for the default `log_prob` (spec 4.5).

The test families' own ``log_survival`` already meets the policy (Q-019), so
these tests check that the base's default ``log_prob`` turns a policy-compliant
``log_survival`` into the values of the spec 4.5 table, elementwise and
without raising. Hand derivations are in ``test_a4_log_prob_default.py``.
"""

import jax
import numpy as np
import pytest
import spec_families as fam
from helpers import INF, NAN, assert_close, f

# --------------------------------------------------------------- t < 0


@pytest.mark.step("A4")
def test_negative_time_log_prob_is_minus_infinity():
    """spec 4.5, rev 11: `t<0` gives `log_prob=-inf`."""
    out = fam.const_hazard()(rate=f(0.5)).log_prob(f([-1.0, -1e-30]))
    np.testing.assert_array_equal(np.asarray(out), [-INF, -INF])


@pytest.mark.step("A4")
def test_minus_infinity_time_log_prob_is_minus_infinity():
    """spec 4.5, rev 11: methods are total over `[-inf, +inf]`; `t=-inf` is `t<0`."""
    out = fam.const_hazard()(rate=f(0.5)).log_prob(f(-INF))
    np.testing.assert_array_equal(np.asarray(out), -INF)


@pytest.mark.step("A4")
def test_negative_time_log_prob_for_infinite_density_at_zero():
    """spec 4.5, rev 11: `t<0` gives `-inf` even when the density is infinite at `0` (Weibull-like `k<1`)."""
    out = fam.power_hazard()(shape=f(0.5), scale=f(2.0)).log_prob(f([-2.0, -INF]))
    np.testing.assert_array_equal(np.asarray(out), [-INF, -INF])


# --------------------------------------------------------------- t = 0


@pytest.mark.step("A4")
def test_zero_time_infinite_density_for_shape_below_one():
    """spec 4.5, rev 11: at `t=0`, boundary values come from the family; Weibull-like `k<1` gives infinite density.

    ``-d/dt log S = (k/b) z**(k-1)`` diverges as ``z -> 0`` when ``k<1``; ``log S(0) = 0``.
    """
    out = fam.power_hazard()(shape=f(0.5), scale=f(2.0)).log_prob(f(0.0))
    np.testing.assert_array_equal(np.asarray(out), INF)


@pytest.mark.step("A4")
def test_zero_time_density_for_shape_one():
    """spec 4.5, rev 11: Weibull-like `k=1` at `t=0` gives `log(1/b)` (no epsilon, no NaN)."""
    out = fam.power_hazard()(shape=f(1.0), scale=f(2.0)).log_prob(f(0.0))
    assert_close(out, np.log(0.5))


@pytest.mark.step("A4")
def test_zero_time_zero_density_for_shape_above_one():
    """spec 4.5, rev 11: Weibull-like `k>1` at `t=0` gives `-inf`."""
    out = fam.power_hazard()(shape=f(2.0), scale=f(2.0)).log_prob(f(0.0))
    np.testing.assert_array_equal(np.asarray(out), -INF)


@pytest.mark.step("A4")
def test_zero_time_constant_hazard_density():
    """spec 4.5, rev 11: at `t=0` an exponential-like family gives `log r`."""
    out = fam.const_hazard()(rate=f(0.5)).log_prob(f(0.0))
    assert_close(out, np.log(0.5))


@pytest.mark.step("A4")
def test_grid_from_zero_with_shape_one_is_finite():
    """spec 4.5, rev 11: a grid starting at `t=0` with Weibull-like `k=1` is finite throughout: `log(1/b) - t/b`."""
    t = [0.0, 0.5, 1.0, 2.0]
    out = fam.power_hazard()(shape=f(1.0), scale=f(2.0)).log_prob(f(t))
    assert np.all(np.isfinite(np.asarray(out)))
    assert_close(out, np.log(0.5) - np.asarray(t) / 2.0)


# --------------------------------------------------------------- t = +inf


@pytest.mark.step("A4")
def test_infinite_time_log_prob_constant_hazard():
    """spec 4.5, rev 11: `t=+inf` gives `log_prob=-inf`."""
    out = fam.const_hazard()(rate=f(0.5)).log_prob(f(INF))
    np.testing.assert_array_equal(np.asarray(out), -INF)


@pytest.mark.step("A4")
def test_infinite_time_log_prob_with_vanishing_hazard():
    """spec 4.5, rev 11: `t=+inf` gives `-inf` for a decreasing hazard (Weibull-like `k<1`)."""
    out = fam.power_hazard()(shape=f(0.5), scale=f(2.0)).log_prob(f(INF))
    np.testing.assert_array_equal(np.asarray(out), -INF)


@pytest.mark.step("A4")
def test_infinite_time_log_prob_with_diverging_hazard():
    """spec 4.5, rev 11: `t=+inf` gives `-inf`, not NaN, when the hazard diverges (Weibull-like `k>1`).

    Evaluated literally, ``log(-d/dt log S) + log S`` is ``+inf + (-inf) = NaN``
    here; the policy requires ``-inf``.
    """
    out = fam.power_hazard()(shape=f(2.0), scale=f(2.0)).log_prob(f(INF))
    np.testing.assert_array_equal(np.asarray(out), -INF)


# --------------------------------------------------------------- NaN and mixed


@pytest.mark.step("A4")
def test_nan_time_log_prob_is_nan():
    """spec 4.5, rev 11: NaN `t` gives NaN."""
    out = fam.const_hazard()(rate=f(0.5)).log_prob(f(NAN))
    assert np.isnan(np.asarray(out))


@pytest.mark.step("A4")
def test_nan_time_log_prob_is_nan_for_power_hazard():
    """spec 4.5, rev 11: NaN `t` gives NaN (Weibull-like family)."""
    out = fam.power_hazard()(shape=f(2.0), scale=f(2.0)).log_prob(f(NAN))
    assert np.isnan(np.asarray(out))


@pytest.mark.step("A4")
def test_policy_applies_elementwise_in_one_call():
    """spec 4.5, rev 11: every case is handled per element of one array, without raising.

    Weibull-like ``k=2``, ``b=2`` at ``(-inf, -1, 0, 1, +inf, NaN)``:
    ``(-inf, -inf, -inf, log f(1), -inf, NaN)`` with
    ``log f(1) = log 2 - log 2 + log 0.5 - 0.25``.
    """
    out = np.asarray(
        fam.power_hazard()(shape=f(2.0), scale=f(2.0)).log_prob(
            f([-INF, -1.0, 0.0, 1.0, INF, NAN])
        )
    )
    np.testing.assert_array_equal(out[[0, 1, 2, 4]], [-INF, -INF, -INF, -INF])
    assert_close(out[3], np.log(0.5) - 0.25)
    assert np.isnan(out[5])


@pytest.mark.step("A4")
def test_policy_values_agree_under_jit():
    """spec 4.5/12.1, rev 11: boundary values are the same eagerly and under `jit`."""
    dist = fam.power_hazard()(shape=f([0.5, 1.0, 2.0]), scale=f(2.0))
    t = f([[-1.0], [0.0], [INF], [NAN]])
    assert_close(jax.jit(lambda d, s: d.log_prob(s))(dist, t), dist.log_prob(t))


@pytest.mark.step("A4")
def test_boundary_values_agree_under_vmap():
    """spec 4.5/12.1, rev 11: boundary values are the same eagerly and under `vmap` over time."""
    dist = fam.power_hazard()(shape=f(0.5), scale=f(2.0))
    t = f([-1.0, 0.0, 1.0, INF, NAN])
    assert_close(jax.vmap(dist.log_prob)(t), dist.log_prob(t))


# --------------------------------------------------------------- invalid parameters


@pytest.mark.step("A4")
def test_invalid_parameter_gives_nan_log_prob():
    """spec 4.5, rev 11: invalid parameters give NaN, without raising.

    With ``r=-1`` the family's ``log S(t) = t`` grows, so ``-d/dt log S = -1`` and
    ``log(-1)`` is NaN.
    """
    out = fam.const_hazard()(rate=f(-1.0)).log_prob(f(1.0))
    assert np.isnan(np.asarray(out))


@pytest.mark.step("A4")
def test_nan_parameter_gives_nan_log_prob():
    """spec 4.5, rev 11: invalid (NaN) parameters give NaN at an interior time."""
    out = fam.power_hazard()(shape=f(NAN), scale=f(2.0)).log_prob(f(1.0))
    assert np.isnan(np.asarray(out))


@pytest.mark.step("A4")
def test_invalid_parameter_does_not_affect_other_batch_entries():
    """spec 4.5, rev 11: an invalid batch entry gives NaN only in its own position."""
    out = np.asarray(fam.const_hazard()(rate=f([0.5, -1.0, 2.0])).log_prob(f(1.0)))
    assert np.isnan(out[1])
    assert_close(out[[0, 2]], [np.log(0.5) - 0.5, np.log(2.0) - 2.0])
