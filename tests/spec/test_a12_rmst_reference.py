"""A12: validation of the independent RMST references themselves (spec 8.1, 8.7, 12.1; rev 41).

``rmst_reference`` integrates the physical survival law with mpmath
quadrature. These checks use no survjax code: they show the reference
integration does not miss mass on very large intervals (``tau/b = 1e5``,
``k = 0.2``, where most of the mass is far from ``b``) or near ``t = 0``, by
comparing it with the analytically independent closed form
``b Gamma(1+1/k) P(1/k, (tau/b)**k)`` (8.7), with high-precision numerical
derivatives of that closed form, with a quadrature on refined breakpoints, and
with the exponential identity.
"""

import mpmath
import pytest
import rmst_reference as ref

K_GRID = (0.2, 0.5, 1.0, 2.5, 10.0, 50.0)
RATIOS = (1e-2, 1e-1, 1.0, 10.0, 1e3, 1e5)
B = 2.0


def _refined_levels():
    return tuple(sorted(set(ref.LEVELS) | {x * 1.7 for x in ref.LEVELS} | {x / 3.1 for x in ref.LEVELS}))


@pytest.mark.step("A12")
@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("k", K_GRID)
def test_quadrature_value_equals_closed_form(k, ratio):
    """spec 8.1/8.7, rev 41: `int_0^tau S` by quadrature equals `b Gamma(1+1/k) P(1/k, (tau/b)**k)` over the 12.1 range."""
    value = ref.weibull(k, B, ratio * B)[0]  # a 40-digit integral rounded to float64
    with mpmath.workdps(40):
        closed = float(ref.weibull_closed_form(k, B, ratio * B))
    assert value == pytest.approx(closed, rel=1e-15)


@pytest.mark.step("A12")
@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("k", K_GRID)
def test_quadrature_derivatives_equal_closed_form_derivatives(k, ratio):
    """spec 8.4/12.1, rev 41: `int dS/dtheta` equals `d/dtheta` of the closed form (mpmath `diff`, 60 digits); `d/dtau = S(tau)`."""
    tau = ratio * B
    value, dk, db, dtau = ref.weibull(k, B, tau)
    want = [ref.weibull_closed_form_derivative(k, B, tau, o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    # log-parameter units, relative to the value (12.1 scale), far tighter than any test target
    assert abs(k * (dk - want[0])) <= 1e-14 * abs(value)
    assert abs(B * (db - want[1])) <= 1e-14 * abs(value)
    assert abs(dtau - want[2]) <= 1e-14


@pytest.mark.step("A12")
@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("k", K_GRID)
def test_refined_breakpoints_do_not_change_the_reference(k, ratio):
    """spec 12.1, rev 41: refining the split points leaves value and derivatives unchanged (no mass is missed)."""
    coarse = ref.ph_weibull(k, B, 0.0, ratio * B)
    fine = ref.ph_weibull(k, B, 0.0, ratio * B, _refined_levels())
    for c, f_ in zip(coarse, fine, strict=True):
        assert abs(c - f_) <= 1e-14 * max(abs(coarse[0]), 1.0)


@pytest.mark.step("A12")
def test_very_large_interval_with_heavy_tail_keeps_its_mass():
    """spec 12.1, rev 41: `k=0.2`, `tau/b=1e5` (`S(tau) = e**-10`): mass spread over 30 decades of `t` is recovered."""
    value = ref.weibull(0.2, 1.0, 1e5)[0]
    with mpmath.workdps(40):
        closed = float(ref.weibull_closed_form(0.2, 1.0, 1e5))
    assert value == pytest.approx(closed, rel=1e-15)
    assert value < 120.0  # the mean b Gamma(6) = 120 exceeds every RMST
    assert value > 0.99 * 120.0 * float(mpmath.gammainc(5, 0, 10, regularized=True))


@pytest.mark.step("A12")
@pytest.mark.parametrize("ratio", RATIOS)
def test_weibull_shape_one_reference_equals_exponential_identity(ratio):
    """spec 8.7, rev 41: Weibull `k=1` is Exponential `r = 1/b`: `b (1 - exp(-tau/b))`."""
    value, _, db, dtau = ref.weibull(1.0, B, ratio * B)
    exp_value, exp_dr, exp_dtau = ref.exponential(1.0 / B, ratio * B)
    assert value == pytest.approx(exp_value, rel=1e-15)
    assert dtau == pytest.approx(exp_dtau, rel=1e-15, abs=1e-300)
    # b = 1/r: d/db = -(1/b**2) d/dr
    assert db == pytest.approx(-exp_dr / B**2, rel=1e-13, abs=1e-15 * value)


@pytest.mark.step("A12")
def test_ph_law_reference_equals_effective_scale_closed_form():
    """spec 4.4, rev 41: the PH-Weibull law integrated directly equals the closed form at scale `b0 exp(-eta/k)` with its derivatives."""
    for k, b0, eta, tau in [(2.5, 2.0, 0.8, 3.0), (0.5, 2.0, -1.5, 100.0), (10.0, 2.0, 0.3, 1.9)]:
        got = ref.ph_weibull(k, b0, eta, tau)
        orders = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
        for g, o in zip(got, orders, strict=True):
            want = ref.ph_weibull_closed_form_derivative(k, b0, eta, tau, o)
            assert abs(g - want) <= 1e-14 * abs(got[0]), (k, b0, eta, tau, o)


@pytest.mark.step("A12")
def test_exponential_reference_derivative_equals_numerical_derivative():
    """spec 8.7, rev 41: the hand `dR/dr = tau e/r - (1 - e)/r**2` equals mpmath `diff` of `-expm1(-r tau)/r`."""
    for r, tau in [(0.5, 3.0), (2.0, 1e-2), (1e-3, 10.0), (3.0, 1e5)]:
        _, dr, dtau = ref.exponential(r, tau)
        assert dr == pytest.approx(ref.exponential_derivative(r, tau, (1, 0)), rel=1e-14, abs=1e-300)
        assert dtau == pytest.approx(ref.exponential_derivative(r, tau, (0, 1)), rel=1e-14, abs=1e-300)
