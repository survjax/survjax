"""Feature `summary.quantile`: quantile, median and inverse cumulative hazard values (spec 4.4, 8.1, 8.2, 8.8, 12.1; rev 51).

8.1: `quantile(p) = H⁻¹(-log1p(-p))` for `0<p<1`, "endpoints from
`support()`"; `median() = quantile(0.5)`; `inverse_cumulative_hazard(e)` is
"the `t` with `H(t)=e`, for `e>=0`"; Exponential quantile closed form, Weibull
`b*(-log1p(-p))**(1/k)`. 8.2: one solver for `H(t)=e`; "Improper families:
with `H(+inf) = -log_survival_at_infinity()`, if `e >= H(+inf)`, the result is
`+inf`. Bounded support limits the bracket through `support()`"; built-in
closed forms "exponential `e/r`; Weibull `b*e**(1/k)`". 4.4: PH
`inverse_cumulative_hazard` is the closure's "otherwise `H0⁻¹(e*exp(-eta))`";
median and quantile use the closure, "otherwise via
`inverse_cumulative_hazard`". 8.8: gated by the 12.1 Summaries row
"including ... the section 8.1 endpoints".

References: closed forms (hand algebra in the docstrings, or
``summary_reference`` at 60 digits). Built-in values use the 12.1 ``rtol``
with no absolute floor; generic solver values use ``generic_rtol()`` (Q-058).
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
import summary_reference as sref
from helpers import INF, assert_close, assert_generic_close, assert_rel_close, f, ph, rounded
from summary_cases import FAMILIES, QUANTILE, families

import survjax

P_GRID = (1e-6, 0.1, 0.5, 0.9, 1 - 1e-4)
E_GRID = (1e-6, 0.1, 1.0, 5.0, 30.0)


def _exp_quantile(r, p):
    return -math.log1p(-p) / r


def _ph_law_scale(k, b0, eta):
    """PH-Weibull law equals `Weibull(k, b0*exp(-eta/k))` (spec 4.4)."""
    return b0 * math.exp(-eta / k)


# --------------------------------------------------------------- built-in closed forms


@pytest.mark.parametrize("p", P_GRID)
@QUANTILE
def test_exponential_quantile_closed_form(p):
    """spec 8.1/8.2, rev 51: Exponential quantile `e/r` with `e = -log1p(-p)`."""
    rr, pp = rounded(0.7, p)
    assert_rel_close(survjax.Exponential(rate=f(rr)).quantile(f(pp)), _exp_quantile(rr, pp))


@pytest.mark.parametrize("p", P_GRID)
@pytest.mark.parametrize("k", [0.5, 2.5, 10.0])
@QUANTILE
def test_weibull_quantile_closed_form(k, p):
    """spec 8.1, rev 51: Weibull quantile `b*(-log1p(-p))**(1/k)`, including `p` near 0 and 1."""
    kk, bb, pp = rounded(k, 2.0, p)
    got = survjax.Weibull(shape=f(kk), scale=f(bb)).quantile(f(pp))
    assert_rel_close(got, sref.weibull_quantile(kk, bb, pp))


@pytest.mark.parametrize("c", [1e-6, 1e6])
@QUANTILE
def test_built_in_quantile_scales_with_the_time_scale(c):
    """spec 8.2/12.1, rev 51: scale changes for the built-in quantile ("Tolerances are relative in time")."""
    kk, bb = rounded(2.5, 2.0 * c)
    got = survjax.Weibull(shape=f(kk), scale=f(bb)).quantile(f(0.5))
    assert_rel_close(got, sref.weibull_quantile(kk, bb, float(f(0.5))))


@pytest.mark.parametrize("r", [0.01, 0.7, 50.0])
@QUANTILE
def test_exponential_median_is_log_two_over_rate(r):
    """spec 8.1, rev 51: `median() = quantile(0.5) = -log1p(-0.5)/r = log(2)/r`."""
    (rr,) = rounded(r)
    assert_rel_close(survjax.Exponential(rate=f(rr)).median(), math.log(2.0) / rr)


@pytest.mark.parametrize("k", [0.2, 1.0, 2.5, 50.0])
@QUANTILE
def test_weibull_median_closed_form(k):
    """spec 8.1, rev 51: Weibull median `b*log(2)**(1/k)`."""
    kk, bb = rounded(k, 2.0)
    assert_rel_close(survjax.Weibull(shape=f(kk), scale=f(bb)).median(), bb * math.log(2.0) ** (1 / kk))


@pytest.mark.parametrize("e", E_GRID)
@QUANTILE
def test_exponential_inverse_cumulative_hazard_closed_form(e):
    """spec 8.1/8.2, rev 51: `inverse_cumulative_hazard(e) = e/r`."""
    rr, ee = rounded(0.7, e)
    assert_rel_close(survjax.Exponential(rate=f(rr)).inverse_cumulative_hazard(f(ee)), ee / rr)


@pytest.mark.parametrize("e", E_GRID)
@pytest.mark.parametrize("k", [0.5, 2.5, 10.0])
@QUANTILE
def test_weibull_inverse_cumulative_hazard_closed_form(k, e):
    """spec 8.1/8.2, rev 51: `inverse_cumulative_hazard(e) = b*e**(1/k)`."""
    kk, bb, ee = rounded(k, 2.0, e)
    got = survjax.Weibull(shape=f(kk), scale=f(bb)).inverse_cumulative_hazard(f(ee))
    assert_rel_close(got, bb * ee ** (1 / kk))


# --------------------------------------------------------------- endpoints


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_quantile_endpoints_follow_the_default_support(family):
    """spec 8.1, rev 51: "endpoints from `support()`"; with the default `(0, +inf)` (4.2), `quantile(0)=0` and `quantile(1)=+inf`."""
    got = np.asarray(families()[family]().quantile(f([0.0, 1.0])))
    assert got[0] == 0.0
    assert got[1] == INF


@QUANTILE
def test_bounded_support_quantile_endpoints_come_from_support():
    """spec 4.2/8.1, rev 51: a `support()` override gives `quantile(0)=0` and `quantile(1)=end` (Q-059)."""
    got = np.asarray(fam.uniform_support()(end=f(2.0)).quantile(f([0.0, 1.0])))
    assert got[0] == 0.0
    assert_rel_close(got[1], 2.0)


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_inverse_cumulative_hazard_at_zero_is_zero(family):
    """spec 8.1, rev 51: the `t` with `H(t)=0` is `t=0` (every family here has `H(t)>0` for `t>0`, and `H(0)=0` by 4.5)."""
    assert np.asarray(families()[family]().inverse_cumulative_hazard(f(0.0))) == 0.0


# --------------------------------------------------------------- generic fallbacks


@pytest.mark.parametrize("p", P_GRID)
@QUANTILE
def test_generic_quantile_of_a_constant_hazard(p):
    """spec 8.2/8.8, rev 51: the safeguarded solver on a custom `S = exp(-r t)` gives `-log1p(-p)/r`."""
    rr, pp = rounded(0.5, p)
    assert_generic_close(fam.const_hazard()(rate=f(rr)).quantile(f(pp)), _exp_quantile(rr, pp))


@pytest.mark.parametrize("p", P_GRID)
@pytest.mark.parametrize("k", [0.5, 2.5, 10.0])
@QUANTILE
def test_generic_quantile_of_a_power_hazard(k, p):
    """spec 8.2/8.8, rev 51: generic quantile of a custom Weibull-like family is `b*(-log1p(-p))**(1/k)`."""
    kk, bb, pp = rounded(k, 2.0, p)
    got = fam.power_hazard()(shape=f(kk), scale=f(bb)).quantile(f(pp))
    assert_generic_close(got, sref.weibull_quantile(kk, bb, pp))


@pytest.mark.parametrize("e", E_GRID)
@pytest.mark.parametrize("k", [0.5, 2.5, 10.0])
@QUANTILE
def test_generic_inverse_cumulative_hazard_of_a_power_hazard(k, e):
    """spec 8.1/8.2, rev 51: the generic solver on a custom `H = (t/b)**k` gives `b*e**(1/k)`."""
    kk, bb, ee = rounded(k, 2.0, e)
    got = fam.power_hazard()(shape=f(kk), scale=f(bb)).inverse_cumulative_hazard(f(ee))
    assert_generic_close(got, bb * ee ** (1 / kk))


@QUANTILE
def test_generic_median_of_a_power_hazard():
    """spec 8.1, rev 51: generic median `b*log(2)**(1/k)`."""
    kk, bb = rounded(1.5, 2.0)
    assert_generic_close(fam.power_hazard()(shape=f(kk), scale=f(bb)).median(), bb * math.log(2.0) ** (1 / kk))


@QUANTILE
def test_generic_quantile_is_vectorized_over_a_batch():
    """spec 8.7/8.8, rev 51: per-entry generic quantiles on a batched custom family."""
    rates = rounded(0.1, 1.0, 10.0)
    assert_generic_close(fam.const_hazard()(rate=f(rates)).quantile(f(0.5)), [math.log(2.0) / r for r in rates])


@pytest.mark.parametrize("c", [1e-6, 1e6])
@QUANTILE
def test_generic_quantile_scales_with_the_time_scale(c):
    """spec 8.2, rev 51: scale changes: with `b -> c*b` the generic quantile scales by `c` ("Tolerances are relative in time")."""
    kk, bb = rounded(2.5, 2.0 * c)
    got = fam.power_hazard()(shape=f(kk), scale=f(bb)).quantile(f(0.5))
    assert_generic_close(got, sref.weibull_quantile(kk, bb, float(f(0.5))))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("e", [0.05, 1.0, 8.0])
@QUANTILE
def test_cumulative_hazard_inverts_the_inverse_cumulative_hazard(family, e):
    """spec 8.1, rev 51: "the `t` with `H(t)=e`": `cumulative_hazard(inverse_cumulative_hazard(e)) = e` (generic tolerance on every family)."""
    d = families()[family]()
    (ee,) = rounded(e)
    assert_generic_close(d.cumulative_hazard(d.inverse_cumulative_hazard(f(ee))), ee)


@QUANTILE
def test_quantile_results_are_in_the_parameter_dtype():
    """spec 3.3/8.1, rev 51: quantile, median and inverse cumulative hazard are in the parameter dtype."""
    want = jnp.result_type(f(0.0))
    for d in (survjax.Weibull(shape=f(2.5), scale=f(2.0)), fam.const_hazard()(rate=f(0.5))):
        assert jnp.result_type(d.quantile(f(0.5))) == want
        assert jnp.result_type(d.median()) == want
        assert jnp.result_type(d.inverse_cumulative_hazard(f(0.5))) == want


# --------------------------------------------------------------- PH identities


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("closure", [True, False], ids=["builtin-closure", "custom-generic"])
@QUANTILE
def test_ph_weibull_quantile_equals_the_equivalent_weibull(closure, p):
    """spec 4.4, rev 51: PH-Weibull quantile is that of `Weibull(k, b0*exp(-eta/k))`, through the closure or via `inverse_cumulative_hazard`."""
    kk, bb, ee, pp = rounded(1.5, 2.0, 0.3, p)
    base = (survjax.Weibull if closure else fam.power_hazard())(shape=f(kk), scale=f(bb))
    want = sref.weibull_quantile(kk, _ph_law_scale(kk, bb, ee), pp)
    check = assert_rel_close if closure else assert_generic_close
    check(ph(base, f(ee)).quantile(f(pp)), want)


@pytest.mark.parametrize("e", [0.1, 1.0, 5.0])
@pytest.mark.parametrize("closure", [True, False], ids=["builtin-closure", "custom-generic"])
@QUANTILE
def test_ph_inverse_cumulative_hazard_is_the_baseline_inverse_at_scaled_hazard(closure, e):
    """spec 4.4, rev 51: PH `inverse_cumulative_hazard(e) = H0⁻¹(e*exp(-eta)) = b0*(e*exp(-eta))**(1/k)`."""
    kk, bb, eta, ee = rounded(1.5, 2.0, 0.3, e)
    base = (survjax.Weibull if closure else fam.power_hazard())(shape=f(kk), scale=f(bb))
    want = bb * (ee * math.exp(-eta)) ** (1 / kk)
    check = assert_rel_close if closure else assert_generic_close
    check(ph(base, f(eta)).inverse_cumulative_hazard(f(ee)), want)


@QUANTILE
def test_ph_exponential_quantile_equals_the_equivalent_exponential():
    """spec 4.4, rev 51: PH-Exponential quantile and median equal those of `Exponential(r*exp(eta))`."""
    rr, ee = rounded(0.7, -0.4)
    d = ph(survjax.Exponential(rate=f(rr)), f(ee))
    rate = rr * math.exp(ee)
    assert_rel_close(d.quantile(f(0.3)), _exp_quantile(rate, float(f(0.3))))
    assert_rel_close(d.median(), math.log(2.0) / rate)


@QUANTILE
def test_ph_quantile_with_draw_by_arm_broadcasting():
    """spec 4.4/8.7, rev 51: baseline batch `(S, 1)` with `eta` `(S, 2)`; quantile and median per entry, shape `(S, 2)`."""
    ks = np.asarray(f([[1.5], [2.5], [0.8]]))
    eta = np.asarray(f([[0.0, 0.5], [-0.3, 0.2], [0.1, -1.0]]))
    for make, check in ((survjax.Weibull, assert_rel_close), (fam.power_hazard(), assert_generic_close)):
        d = ph(make(shape=f(ks), scale=f(2.0)), f(eta))
        quant = d.quantile(f(0.5))
        assert np.shape(quant) == (3, 2)
        want = [
            [sref.weibull_quantile(float(ks[i, 0]), _ph_law_scale(float(ks[i, 0]), 2.0, float(eta[i, j])), 0.5) for j in range(2)]
            for i in range(3)
        ]
        check(quant, want)
        assert_close(d.median(), quant)


# --------------------------------------------------------------- bounded support


@pytest.mark.parametrize("p", [0.1, 0.5, 0.99])
@QUANTILE
def test_bounded_support_quantile_is_hand_derived(p):
    """spec 8.2, rev 51: uniform `(0, end]` has `S = 1 - t/end`, so the quantile is `p*end`; "Bounded support limits the bracket through `support()`"."""
    end, pp = rounded(2.0, p)
    assert_generic_close(fam.uniform_support()(end=f(end)).quantile(f(pp)), pp * end)


@pytest.mark.parametrize("e", [0.1, 1.0, 10.0])
@QUANTILE
def test_bounded_support_inverse_cumulative_hazard_is_hand_derived(e):
    """spec 8.1/8.2, rev 51: `H = -log(1 - t/end)`, so `H⁻¹(e) = -end*expm1(-e)`."""
    end, ee = rounded(2.0, e)
    got = fam.uniform_support()(end=f(end)).inverse_cumulative_hazard(f(ee))
    assert_generic_close(got, -end * math.expm1(-ee))


# --------------------------------------------------------------- improper families


@pytest.mark.parametrize("p", [0.1, 0.5, 0.65])
@QUANTILE
def test_improper_family_quantile_below_the_cure_level(p):
    """spec 8.2, rev 51: for `e < H(+inf)` the solver finds `q = log((1-c)/(1-p-c))/r`."""
    c, r, pp = rounded(0.3, 1.0, p)
    got = fam.mixture_cure()(cure=f(c), rate=f(r)).quantile(f(pp))
    assert_generic_close(got, sref.mixture_cure_quantile(c, r, pp))


@pytest.mark.parametrize("p", [0.75, 0.9, 1.0])
@QUANTILE
def test_improper_family_quantile_beyond_the_cure_level_is_infinite(p):
    """spec 8.2, rev 51: "if `e >= H(+inf)`, the result is `+inf`" (`c = 0.3`: `H(+inf) = -log 0.3`, so `p >= 0.7`)."""
    got = np.asarray(fam.mixture_cure()(cure=f(0.3), rate=f(1.0)).quantile(f(p)))
    assert got == INF


@pytest.mark.parametrize("e", [0.1, 1.0])
@QUANTILE
def test_improper_family_inverse_cumulative_hazard_below_the_cure_level(e):
    """spec 8.2, rev 51: `S = c + (1-c)exp(-r t) = exp(-e)` gives `t = log((1-c)/(exp(-e)-c))/r`, the quantile at `p = -expm1(-e)`."""
    c, r, ee = rounded(0.3, 1.0, e)
    got = fam.mixture_cure()(cure=f(c), rate=f(r)).inverse_cumulative_hazard(f(ee))
    assert_generic_close(got, math.log((1 - c) / (math.exp(-ee) - c)) / r)


@pytest.mark.parametrize("e", [1.3, 5.0, 100.0])
@QUANTILE
def test_improper_family_inverse_cumulative_hazard_beyond_the_cure_level_is_infinite(e):
    """spec 8.2, rev 51: `e >= H(+inf) = -log 0.3 ≈ 1.204` gives `+inf`."""
    got = np.asarray(fam.mixture_cure()(cure=f(0.3), rate=f(1.0)).inverse_cumulative_hazard(f(e)))
    assert got == INF
