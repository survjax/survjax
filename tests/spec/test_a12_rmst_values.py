"""A12: staged RMST values (spec 4.4, 8.1, 8.7, 12.1; rev 41).

8.7: "`rmst(tau)` on `Exponential`, `Weibull` and `ProportionalHazards` with
either baseline (by the PH closure of section 4.4)". Values: Exponential
``-expm1(-r*tau)/r``; Weibull ``b*Γ(1+1/k)*P(1/k, (tau/b)**k)``; "Both keep
relative accuracy at the section 12.1 targets over `k` in [0.2, 50] and
`tau/b` in [1e-2, 1e5], in both dtypes. That includes the lower tail where
`(tau/b)**k` underflows (RMST about `tau`) and the upper tail where it
overflows (RMST about the mean `b*Γ(1+1/k)`)."

References are mpmath quadrature of the physical law ``S(t)``
(``rmst_reference``, validated in ``test_a12_rmst_reference.py``), evaluated at
the inputs rounded to the active dtype. Comparisons use the 12.1 ``rtol`` with
zero absolute tolerance ("relative accuracy").
"""

import jax.numpy as jnp
import numpy as np
import pytest
import rmst_reference as ref
from helpers import FLOAT, assert_rel_close, f, ph, rounded

import survjax

K_GRID = (0.2, 0.5, 1.0, 2.5, 10.0, 50.0)
RATIOS = (1e-2, 1e-1, 1.0, 10.0, 1e3, 1e5)
B = 2.0


def _weibull_reference(k, b, tau):
    k, b, tau = rounded(k, b, tau)
    return ref.weibull(k, b, tau)[0]


# --------------------------------------------------------------- Exponential


@pytest.mark.step("A12")
@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("r", [0.25, 3.0])
def test_exponential_rmst_matches_reference(r, ratio):
    """spec 8.7, rev 41: Exponential `-expm1(-r*tau)/r` with relative accuracy for `r*tau` in [1e-2, 1e5]."""
    rr, tau = rounded(r, ratio / r)
    out = survjax.Exponential(rate=f(rr)).rmst(f(tau))
    assert_rel_close(out, ref.exponential(rr, tau)[0])


@pytest.mark.step("A12")
def test_exponential_rmst_is_vectorized_over_rates():
    """spec 8.7, rev 41: per-entry rates with a scalar cutoff."""
    r = np.array([0.1, 1.0, 10.0])
    out = survjax.Exponential(rate=f(r)).rmst(f(2.0))
    assert np.shape(out) == (3,)
    assert_rel_close(out, [ref.exponential(*rounded(ri, 2.0))[0] for ri in r])


# --------------------------------------------------------------- Weibull grid


@pytest.mark.step("A12")
@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("k", K_GRID)
def test_weibull_rmst_matches_quadrature(k, ratio):
    """spec 8.7/12.1, rev 41: Weibull RMST against mpmath quadrature of `S(t)` over `k` in [0.2, 50], `tau/b` in [1e-2, 1e5]."""
    tau = ratio * B
    out = survjax.Weibull(shape=f(k), scale=f(B)).rmst(f(tau))
    assert_rel_close(out, _weibull_reference(k, B, tau))


@pytest.mark.step("A12")
@pytest.mark.parametrize("ratio", [1e-2, 1e-1], ids=["tau/b=0.01", "tau/b=0.1"])
def test_weibull_rmst_where_the_power_underflows_is_about_tau(ratio):
    """spec 8.7, rev 41: `k=50`, `(tau/b)**k` is `1e-100` or `1e-50` (underflows in float32): RMST is about `tau`, not 0."""
    tau = ratio * B
    out = np.asarray(survjax.Weibull(shape=f(50.0), scale=f(B)).rmst(f(tau)))
    assert out > 0
    assert_rel_close(out, _weibull_reference(50.0, B, tau))
    assert_rel_close(out, rounded(tau)[0])  # RMST = tau (1 - O((tau/b)**k)): equal at 12.1 rtol


@pytest.mark.step("A12")
@pytest.mark.parametrize(
    ("k", "ratio"), [(50.0, 10.0), (50.0, 1e5), (10.0, 1e5)], ids=["k50-tau/b10", "k50-tau/b1e5", "k10-tau/b1e5"]
)
def test_weibull_rmst_where_the_power_overflows_is_about_the_mean(k, ratio):
    """spec 8.7, rev 41: `(tau/b)**k` is `1e50`, `1e250` or `1e50` (overflows in float32): RMST is about `b Γ(1+1/k)`."""
    tau = ratio * B
    out = np.asarray(survjax.Weibull(shape=f(k), scale=f(B)).rmst(f(tau)))
    assert np.isfinite(out)
    assert_rel_close(out, _weibull_reference(k, B, tau))
    kk, bb = rounded(k, B)
    from math import gamma

    assert_rel_close(out, bb * gamma(1.0 + 1.0 / kk))


@pytest.mark.step("A12")
@pytest.mark.parametrize("ratio", RATIOS)
def test_weibull_shape_one_equals_exponential(ratio):
    """spec 8.7/12.1, rev 41: "the Exponential special case (Weibull `k=1`)": `b (1 - exp(-tau/b))`."""
    tau = ratio * B
    w = survjax.Weibull(shape=f(1.0), scale=f(B)).rmst(f(tau))
    e = survjax.Exponential(rate=f(1.0 / B)).rmst(f(tau))
    bb, tt = rounded(B, tau)
    assert_rel_close(w, ref.exponential(1.0 / bb, tt)[0])
    assert_rel_close(w, e)


@pytest.mark.step("A12")
@pytest.mark.parametrize("c", [1e-3, 1e3], ids=["c=1e-3", "c=1e3"])
@pytest.mark.parametrize("k", [0.5, 2.5, 10.0])
def test_weibull_scale_change(k, c):
    """spec 12.1, rev 41: "scale changes": `rmst` of `Weibull(k, c b)` at `c tau` is `c` times that of `Weibull(k, b)` at `tau`.

    Substituting ``t = c s`` in ``int_0^{c tau} exp(-(t/(c b))**k) dt``.
    Powers of ten are not exact in binary, so each side is also checked
    against its own reference.
    """
    tau = 3.0
    small = survjax.Weibull(shape=f(k), scale=f(B)).rmst(f(tau))
    big = survjax.Weibull(shape=f(k), scale=f(c * B)).rmst(f(c * tau))
    assert_rel_close(big, _weibull_reference(k, c * B, c * tau))
    assert_rel_close(np.asarray(big) / c, small)


@pytest.mark.step("A12")
def test_weibull_rmst_is_vectorized_over_shapes():
    """spec 8.7, rev 41: per-entry shapes, shared scale and cutoff."""
    k = np.array([0.2, 1.0, 50.0])
    out = survjax.Weibull(shape=f(k), scale=f(B)).rmst(f(3.0))
    assert np.shape(out) == (3,)
    assert_rel_close(out, [_weibull_reference(ki, B, 3.0) for ki in k])


@pytest.mark.step("A12")
def test_rmst_result_dtype_is_the_parameter_dtype():
    """spec 3.3, rev 41: calculations use the common parameter dtype."""
    out = survjax.Weibull(shape=f(2.0), scale=f(1.0)).rmst(f(1.0))
    assert np.dtype(jnp.asarray(out).dtype) == FLOAT


# --------------------------------------------------------------- PH wrappers


PH_CASES = [
    (0.5, 2.0, -1.5, 0.5),  # effective scale b0 exp(-eta/k) = 2 e**3: tau/b = 0.0125
    (0.5, 2.0, 0.8, 50.0),
    (1.0, 2.0, 0.3, 2.0),
    (2.5, 2.0, -0.7, 3.0),
    (2.5, 2.0, 1.2, 0.05),
    (10.0, 2.0, 0.5, 1.9),
    (50.0, 2.0, -2.0, 1e4),
]


@pytest.mark.step("A12")
@pytest.mark.parametrize("case", PH_CASES, ids=str)
def test_ph_weibull_rmst_matches_physical_law_quadrature(case):
    """spec 4.4/8.7, rev 41: PH-Weibull RMST against quadrature of `S(t) = exp(-exp(eta) (t/b0)**k)` directly."""
    k, b0, eta, tau = rounded(*case)
    out = ph(survjax.Weibull(shape=f(k), scale=f(b0)), f(eta)).rmst(f(tau))
    assert_rel_close(out, ref.ph_weibull(k, b0, eta, tau)[0])


@pytest.mark.step("A12")
@pytest.mark.parametrize("case", PH_CASES, ids=str)
def test_ph_weibull_rmst_equals_equivalent_weibull(case):
    """spec 4.4, rev 41: delegated summaries equal those of `Weibull(shape=k, scale=b0*exp(-eta/k))`."""
    k, b0, eta, tau = case
    out = ph(survjax.Weibull(shape=f(k), scale=f(b0)), f(eta)).rmst(f(tau))
    eq = survjax.Weibull(shape=f(k), scale=f(b0) * jnp.exp(-f(eta) / f(k))).rmst(f(tau))
    assert_rel_close(out, eq)


@pytest.mark.step("A12")
@pytest.mark.parametrize(("r", "eta", "tau"), [(0.5, -1.0, 2.0), (2.0, 0.7, 0.01), (0.1, 2.0, 1e3)])
def test_ph_exponential_rmst_is_hand_derived(r, eta, tau):
    """spec 4.4/8.7, rev 41: PH-Exponential is `Exponential(rate=r*exp(eta))`: `-expm1(-r e**eta tau)/(r e**eta)`."""
    rr, ee, tt = rounded(r, eta, tau)
    out = ph(survjax.Exponential(rate=f(rr)), f(ee)).rmst(f(tt))
    assert_rel_close(out, ref.exponential(rr * np.exp(ee), tt)[0])
    assert_rel_close(out, survjax.Exponential(rate=f(rr) * jnp.exp(f(ee))).rmst(f(tt)))


@pytest.mark.step("A12")
def test_ph_rmst_with_draw_by_arm_broadcasting():
    """spec 4.4/8.7, rev 41: baseline `(S, 1)`, `eta` `(S, 2)`: `rmst(5.0)` has shape `(S, 2)` and matches the physical law."""
    k = np.array([[0.8], [2.0]])
    b0 = np.array([[1.0], [3.0]])
    eta = np.array([[0.0, 0.6], [-0.5, 1.0]])
    out = ph(survjax.Weibull(shape=f(k), scale=f(b0)), f(eta)).rmst(f(5.0))
    assert np.shape(out) == (2, 2)
    want = [[ref.ph_weibull(*rounded(k[s, 0], b0[s, 0], eta[s, j], 5.0))[0] for j in range(2)] for s in range(2)]
    assert_rel_close(out, want)


@pytest.mark.step("A12")
def test_ph_weibull_rmst_tail_cases():
    """spec 4.4/8.7, rev 41: PH-Weibull (`k=50`) at effective `tau/b = b0 exp(-eta/k)` of 0.01 and 1e4, both power tails.

    ``eta = 50 log 10`` makes ``exp(eta) = 1e50``, beyond float32 range, though
    the law is the ordinary ``Weibull(50, 0.2)``.
    """
    for eta, tau in [(0.0, 0.02), (50.0 * np.log(10.0), 2e3)]:
        out = ph(survjax.Weibull(shape=f(50.0), scale=f(2.0)), f(eta)).rmst(f(tau))
        assert_rel_close(out, ref.ph_weibull(*rounded(50.0, 2.0, eta, tau))[0])
