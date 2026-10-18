"""A4: direct-method value policy for the default `log_prob` (spec 4.5).

The test families' own ``log_survival`` already meets the policy (Q-019), so
these tests check that the base's default ``log_prob`` turns a policy-compliant
``log_survival`` into the values of the spec 4.5 table, elementwise and
without raising. Rev 26: the density limit at ``t=0`` comes from a
``log_prob`` override; the generic default carries no guarantee there. Hand derivations are in ``test_a4_log_prob_default.py``.
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
#
# Rev 26: at ``t=0`` the density limits are family-specific: "an override of
# `log_prob` ... is called at `t=0` and supplies them". The generic default's
# value at ``t=0`` "carries no value or gradient guarantee", so these tests use
# ``power_log_prob_override`` (an ``xlogy``-style analytic ``log_prob``).


def _power_override(k, b=2.0):
    return fam.power_log_prob_override()(shape=f(k), scale=f(b))


@pytest.mark.step("A4")
def test_zero_time_log_prob_override_infinite_density_for_shape_below_one():
    """spec 4.5, rev 26: at `t=0` a `log_prob` override supplies the density limit; Weibull-like `k<1` gives `+inf`.

    ``(k-1) log z`` with ``k-1 < 0`` and ``z -> 0`` gives ``+inf``.
    """
    np.testing.assert_array_equal(
        np.asarray(_power_override(0.5).log_prob(f(0.0))), INF
    )


@pytest.mark.step("A4")
def test_zero_time_log_prob_override_density_for_shape_one():
    """spec 4.5, rev 26: Weibull-like `k=1` at `t=0`: the override's `log(1/b)` (``xlogy(0, 0) = 0``)."""
    assert_close(_power_override(1.0).log_prob(f(0.0)), np.log(0.5))


@pytest.mark.step("A4")
def test_zero_time_log_prob_override_zero_density_for_shape_above_one():
    """spec 4.5, rev 26: Weibull-like `k>1` at `t=0`: the override's `-inf`."""
    np.testing.assert_array_equal(
        np.asarray(_power_override(2.0).log_prob(f(0.0))), -INF
    )


@pytest.mark.step("A4")
def test_zero_time_log_prob_override_is_called():
    """spec 4.5, rev 26: "an override of `log_prob` ... is called at `t=0`": the sentinel override's value is returned."""
    out = fam.log_prob_override()(rate=f(0.5)).log_prob(f(0.0))
    np.testing.assert_array_equal(np.asarray(out), fam.OVERRIDE_SENTINEL)


@pytest.mark.step("A4")
def test_grid_from_zero_with_shape_one_is_finite():
    """spec 4.5, rev 26: a time vector starting at `t=0` with Weibull-like `k=1` (override) is finite: `log(1/b) - t/b`."""
    t = [0.0, 0.5, 1.0, 2.0]
    out = _power_override(1.0).log_prob(f(t))
    assert np.all(np.isfinite(np.asarray(out)))
    assert_close(out, np.log(0.5) - np.asarray(t) / 2.0)


# --------------------------------------------------------------- t = +inf


@pytest.mark.step("A4")
def test_infinite_time_log_prob_constant_hazard():
    """spec 4.5, rev 26: `t=+inf` gives `log_prob=-inf` (base-supplied)."""
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
    """spec 4.5, rev 26: every case is handled per element of one array, without raising.

    Weibull-like ``k=2``, ``b=2`` at ``(-inf, -1, 1, +inf, NaN)``:
    ``(-inf, -inf, log f(1), -inf, NaN)`` with
    ``log f(1) = log 2 - log 2 + log 0.5 - 0.25``. ``t=0`` is left out: the
    generic default carries no guarantee there (rev 26).
    """
    out = np.asarray(
        fam.power_hazard()(shape=f(2.0), scale=f(2.0)).log_prob(
            f([-INF, -1.0, 1.0, INF, NAN])
        )
    )
    np.testing.assert_array_equal(out[[0, 1, 3]], [-INF, -INF, -INF])
    assert_close(out[2], np.log(0.5) - 0.25)
    assert np.isnan(out[4])


@pytest.mark.step("A4")
def test_policy_values_agree_under_jit():
    """spec 4.5/12.1, rev 26: policy values are the same eagerly and under `jit` (`t=0` excluded: no generic-default guarantee)."""
    dist = fam.power_hazard()(shape=f([0.5, 1.0, 2.0]), scale=f(2.0))
    t = f([[-1.0], [1.0], [INF], [NAN]])
    assert_close(jax.jit(lambda d, s: d.log_prob(s))(dist, t), dist.log_prob(t))


@pytest.mark.step("A4")
def test_boundary_values_agree_under_vmap():
    """spec 4.5/12.1, rev 26: policy values are the same eagerly and under `vmap` over time (`t=0` excluded)."""
    dist = fam.power_hazard()(shape=f(0.5), scale=f(2.0))
    t = f([-1.0, 1.0, INF, NAN])
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
