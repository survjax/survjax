"""Feature `summary.rmst`: generic RMST values (spec 4.4, 8.1, 8.3, 8.7, 8.8, 12.1; rev 51).

8.1: `rmst(tau) = ∫_0^tau S(t) dt`. 8.3: tanh-sinh on `[0, min(tau, s)]` and
`[min(tau, s), tau]`, "with further splits at known breakpoints (a finite
support end inside the interval ...)". "verification covers heavy tails
(log-logistic `beta` near 1), bounded support and scale changes". 4.4: PH
without a closure uses "the generic method". 8.8: gated by the 12.1
Summaries row: "analytical values; generic fallbacks against known answers;
heavy tails, bounded support ...". Built-in Exponential/Weibull RMST values
stay covered by the A12 regressions (8.8: "The value, tail and
first-derivative regressions of the first-release rows still apply").

References: closed forms (hand algebra in the docstrings) and mpmath
quadrature of the physical law (``rmst_reference``,
``summary_reference``). Inputs are rounded to the active dtype first.
Generic numerical values use ``generic_rtol()``: "Generic quadrature and
solver results ... are compared at `10*rtol` relative" (12.1, rev 54).
"""

import math

import jax.numpy as jnp
import pytest
import rmst_reference as rref
import spec_families as fam
import summary_reference as sref
from helpers import assert_generic_close, f, ph, rounded
from summary_cases import RMST

# --------------------------------------------------------------- generic fallbacks: known answers


@pytest.mark.parametrize("tau", [0.01, 1.0, 4.0, 100.0])
@RMST
def test_generic_rmst_of_a_constant_hazard(tau):
    """spec 8.3/8.8, rev 51: generic RMST of a custom `S = exp(-r t)` is `-expm1(-r tau)/r`."""
    rr, tt = rounded(0.5, tau)
    got = fam.const_hazard()(rate=f(rr)).rmst(f(tt))
    assert_generic_close(got, -math.expm1(-rr * tt) / rr)


@pytest.mark.parametrize(("k", "ratio"), [(0.5, 1.0), (0.5, 100.0), (2.5, 0.1), (2.5, 1.5), (2.5, 1e3), (10.0, 1.0)])
@RMST
def test_generic_rmst_of_a_power_hazard_matches_quadrature(k, ratio):
    """spec 8.3/8.8, rev 51: generic RMST of a custom Weibull-like family against mpmath quadrature of `S(t)`."""
    kk, bb, tt = rounded(k, 2.0, ratio * 2.0)
    got = fam.power_hazard()(shape=f(kk), scale=f(bb)).rmst(f(tt))
    assert_generic_close(got, rref.weibull(kk, bb, tt)[0])


@RMST
def test_generic_rmst_is_vectorized_over_a_batch():
    """spec 8.7/8.8, rev 51: per-entry generic RMST on a batched custom family."""
    rates = rounded(0.1, 1.0, 10.0)
    d = fam.const_hazard()(rate=f(rates))
    assert_generic_close(d.rmst(f(1.0)), [-math.expm1(-r) / r for r in rates])


# --------------------------------------------------------------- PH without closure


@pytest.mark.parametrize("tau", [0.5, 3.0, 50.0])
@RMST
def test_ph_rmst_on_a_baseline_without_closure_matches_the_physical_law(tau):
    """spec 4.4/8.8, rev 51: PH of a custom baseline uses "the generic method"; mpmath of `exp(-exp(eta)(t/b0)**k)`."""
    kk, bb, ee, tt = rounded(1.5, 2.0, 0.3, tau)
    got = ph(fam.power_hazard()(shape=f(kk), scale=f(bb)), f(ee)).rmst(f(tt))
    assert_generic_close(got, rref.ph_weibull(kk, bb, ee, tt)[0])


# --------------------------------------------------------------- bounded support


@pytest.mark.parametrize("tau", [0.5, 1.9, 2.0, 3.0, 100.0])
@RMST
def test_bounded_support_rmst_is_hand_derived(tau):
    """spec 8.3/8.8, rev 51: RMST of uniform `(0, end]`: `∫(1 - t/end) dt = tau - tau**2/(2 end)` inside, `end/2` beyond the support end."""
    end, tt = rounded(2.0, tau)
    want = tt - tt**2 / (2 * end) if tt <= end else end / 2
    assert_generic_close(fam.uniform_support()(end=f(end)).rmst(f(tt)), want)


# --------------------------------------------------------------- improper families


@pytest.mark.parametrize("tau", [0.5, 5.0, 100.0])
@RMST
def test_improper_family_has_finite_rmst(tau):
    """spec 8.1/8.3, rev 51: RMST of an improper family is the finite integral `c tau + (1-c)(1-exp(-r tau))/r`."""
    c, r, tt = rounded(0.3, 1.0, tau)
    got = fam.mixture_cure()(cure=f(c), rate=f(r)).rmst(f(tt))
    assert_generic_close(got, sref.mixture_cure_rmst(c, r, tt))


# --------------------------------------------------------------- heavy tails


@pytest.mark.parametrize("ratio", [10.0, 1e3, 1e5])
@RMST
def test_heavy_tail_generic_rmst(ratio):
    """spec 8.3/8.8, rev 51: heavy-tailed RMST over long intervals (custom log-logistic, `beta = 1.2`) against mpmath quadrature."""
    aa, bb, tt = rounded(2.0, 1.2, ratio * 2.0)
    got = fam.softplus_log_logistic()(alpha=f(aa), beta=f(bb)).rmst(f(tt))
    assert_generic_close(got, sref.log_logistic_rmst(aa, bb, tt))


# --------------------------------------------------------------- scale changes and dtype


@pytest.mark.parametrize("c", [1e-6, 1e6])
@RMST
def test_generic_rmst_scales_with_the_time_scale(c):
    """spec 8.3/8.5, rev 51: scale changes: with `b -> c*b` and `tau -> c*tau` the generic RMST scales by `c` (atol is `1e-10*s` resp. `1e-6*s`)."""
    kk, bb, tt = rounded(2.5, 2.0 * c, 3.0 * c)
    d = fam.power_hazard()(shape=f(kk), scale=f(bb))
    assert_generic_close(d.rmst(f(tt)), rref.weibull(kk, bb, tt)[0])


@RMST
def test_generic_rmst_is_in_the_parameter_dtype():
    """spec 3.3/8.7, rev 51: the generic RMST is in the parameter dtype."""
    assert jnp.result_type(fam.const_hazard()(rate=f(0.5)).rmst(f(1.0))) == jnp.result_type(f(0.0))
