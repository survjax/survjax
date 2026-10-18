"""A4: `log_hazard` and `hazard` where `S(t)=0` and at `t=+inf` (spec 4.2, 4.5, rev 13, 26).

Rev 13: the generic default ``log(-d/dt log S)`` applies where ``S(t)>0`` and
gives NaN where ``S(t)=0`` beyond a finite support end; a family that
overrides ``log_hazard`` gives its analytic value. Rev 26: at ``t=+inf``,
"`log_hazard` and `hazard` come from an override of `log_hazard`, called at
`+inf`; the generic default gives NaN" (proper or improper family). The generic ``log_hazard`` and ``hazard`` defaults
are Slice A methods by spec rev 14 (Q-024, resolved).
"""

import jax
import numpy as np
import pytest
import spec_families as fam
from helpers import INF, assert_close, f

# --------------------------------------------------------------- proper families at +inf


@pytest.mark.step("A4")
def test_default_log_hazard_is_nan_at_infinity():
    """spec 4.2/4.5, rev 26: at `t=+inf` the generic `log_hazard` default gives NaN.

    The derivative form would give ``log r`` for this exponential-like family;
    rev 13 and rev 26 require NaN.
    """
    assert np.isnan(np.asarray(fam.const_hazard()(rate=f(0.5)).log_hazard(f(INF))))


@pytest.mark.step("A4")
def test_default_hazard_is_nan_at_infinity():
    """spec 4.5, rev 26: at `t=+inf`, `hazard` from the generic default is NaN."""
    assert np.isnan(np.asarray(fam.const_hazard()(rate=f(0.5)).hazard(f(INF))))


@pytest.mark.step("A4")
def test_default_log_hazard_is_nan_at_infinity_for_unguarded_family():
    """spec 4.5, rev 26: the same for a Weibull-like family (`k=2`, derivative form `+inf`) and `k=0.5` (`-inf`)."""
    for k in (0.5, 2.0):
        dist = fam.naive_power()(shape=f(k), scale=f(2.0))
        assert np.isnan(np.asarray(dist.log_hazard(f(INF)))), f"k={k}"


@pytest.mark.step("A4")
def test_default_log_hazard_is_nan_only_where_survival_is_zero():
    """spec 4.2/4.5, rev 26: elementwise: `log r` at finite `t`, NaN at `+inf`, in one call."""
    out = np.asarray(fam.const_hazard()(rate=f(0.5)).log_hazard(f([1.0, INF, 30.0])))
    assert_close(out[[0, 2]], [np.log(0.5), np.log(0.5)])
    assert np.isnan(out[1])


@pytest.mark.step("A4")
def test_default_log_hazard_with_tiny_but_positive_survival_is_not_nan():
    """spec 4.2, rev 13: `S(t)>0` is the mathematical condition, read from `log S`, not from an underflowed `exp(log S)`.

    ``r=1``, ``t=1e3``: ``log S = -1e3`` is finite although ``exp(-1e3)`` underflows
    to 0 in both dtypes; ``log h = log 1 = 0``.
    """
    assert_close(fam.const_hazard()(rate=f(1.0)).log_hazard(f(1e3)), 0.0)


# --------------------------------------------------------------- overrides


@pytest.mark.step("A4")
def test_log_hazard_override_supplies_value_at_infinity():
    """spec 4.5, rev 26: at `t=+inf` the `log_hazard` override is called and supplies the limit (`log r`)."""
    assert_close(fam.log_hazard_override()(rate=f(0.5)).log_hazard(f(INF)), np.log(0.5))


@pytest.mark.step("A4")
def test_hazard_uses_log_hazard_override_at_infinity():
    """spec 4.5, rev 26: `hazard` at `t=+inf` comes from the `log_hazard` override (`r`)."""
    assert_close(fam.log_hazard_override()(rate=f(0.5)).hazard(f(INF)), 0.5)


@pytest.mark.step("A4")
def test_log_hazard_override_still_follows_policy_for_negative_and_nan_time():
    """spec 4.5, rev 12/13: the override is wrapped by the base: `t<0` gives `h=0`, NaN `t` gives NaN."""
    dist = fam.log_hazard_override()(rate=f(0.5))
    np.testing.assert_array_equal(np.asarray(dist.hazard(f(-1.0))), 0.0)
    assert np.isnan(np.asarray(dist.log_hazard(f(np.nan))))


@pytest.mark.step("A4")
def test_log_hazard_override_is_minus_infinity_for_negative_time():
    """spec 4.5, rev 21: the base applies the `t<0` row to a `log_hazard` override: `log_hazard=-inf`, not `log r`."""
    out = fam.log_hazard_override()(rate=f(0.5)).log_hazard(f([-INF, -1.0]))
    np.testing.assert_array_equal(np.asarray(out), [-INF, -INF])


# --------------------------------------------------------------- beyond a finite support end


@pytest.mark.step("A4")
def test_default_log_hazard_inside_bounded_support():
    """spec 4.2, rev 13: inside the support the default is `log(-d/dt log S)`; for uniform on `(0, c]`, `-log(c - t)`."""
    out = fam.bounded_support()(end=f(4.0)).log_hazard(f([1.0, 3.0]))
    assert_close(out, -np.log([3.0, 1.0]))


@pytest.mark.step("A4")
def test_default_log_hazard_is_nan_at_and_beyond_support_end():
    """spec 4.5, rev 13: where `S(t)=0` beyond a finite support end (`t=c`, `t>c`, `t=+inf`), the default gives NaN.

    There ``log S`` is identically ``-inf``; the derivative form would give
    ``log 0 = -inf``, i.e. a hazard of 0.
    """
    out = np.asarray(fam.bounded_support()(end=f(4.0)).log_hazard(f([4.0, 6.0, INF])))
    assert np.all(np.isnan(out))


@pytest.mark.step("A4")
def test_default_hazard_is_nan_beyond_support_end():
    """spec 4.5, rev 13: `hazard` beyond a finite support end is NaN from the generic default."""
    assert np.isnan(np.asarray(fam.bounded_support()(end=f(4.0)).hazard(f(6.0))))


@pytest.mark.step("A4")
def test_zero_survival_rule_agrees_under_jit():
    """spec 4.5/12.1, rev 13: the same values eagerly and under `jit`."""
    dist = fam.bounded_support()(end=f([4.0, 8.0]))
    t = f([[1.0], [6.0], [INF]])
    assert_close(jax.jit(lambda d, s: d.log_hazard(s))(dist, t), dist.log_hazard(t))
