"""Features `family.loglogistic` + `summary.*`: summaries of `LogLogistic` (spec 4.5, 8.1-8.5, 8.7, 8.8, 12.1, 12.2; rev 55, 56).

12.2 (rev 55): analytical summaries "are enabled with the matching summary
method", so every test lists `family.loglogistic` and its summary feature.
8.1 table, log-logistic row: mean `alpha*(π/beta)/sin(π/beta)` for `beta>1`;
"known infinite for `beta<=1`"; median/quantile `alpha*(p/(1-p))**(1/beta)`;
RMST "generic quadrature" (8.4: JAX's `betainc` has no parameter
derivatives). 8.2: `H⁻¹(e) = alpha*expm1(e)**(1/beta)`. 8.1 (rev 54, 56): a
known infinite mean is `+inf` "with its own status in `mean_with_info`";
"An analytical answer reports status `converged` (or `known_infinite`), a
NaN `error_estimate` and zero work counts"; the NaN tangents of a known
infinite mean: "built-ins provide them" (8.4: "known-infinite means ... get
NaN tangents"). 8.3: verification covers "heavy tails (log-logistic `beta`
near 1)". 12.1: closed forms at the ordinary targets; generic quadrature and
its derivatives at `10*rtol` (Q-058 log units).

Hand derivatives: mean ``m = alpha x/sin x``, ``x = pi/beta``: ``dm/dalpha =
m/alpha``; ``dm/dbeta = (m/beta)(x cot x - 1)`` (from ``d/dbeta log m =
-(1/beta)(1 - x cot x)``). Quantile ``q = alpha r**(1/beta)``, ``r =
p/(1-p)``: ``dq/dalpha = q/alpha``, ``dq/dbeta = -q log(r)/beta**2``,
``dq/dp = q/(beta p (1-p)) = 1/f(q)``. ``H⁻¹``: ``t = alpha
expm1(e)**(1/beta)``, ``dt/de = t exp(e)/(beta expm1(e)) = 1/h(t)``. RMST:
``d/dtau = S(tau)``; parameter derivatives by quadrature of ``dS/dtheta``
(``summary_reference.log_logistic_rmst_derivative``).

The custom-family log-logistic tests of ``test_summary_mean_values.py`` stay:
they test the `mean_with_info` override contract and the generic heavy-tail
quadrature, which these built-in tests do not exercise.
"""

import math

import family_reference as fr
import jax
import numpy as np
import pytest
import summary_reference as sref
from helpers import (
    INF,
    NAN,
    X64,
    assert_generic_close,
    assert_generic_log_derivative,
    assert_log_parameter_derivative,
    assert_rel_close,
    assert_status,
    f,
    generic_rtol,
    rounded,
)

import survjax

MEAN = pytest.mark.feature("family.loglogistic", "summary.mean")
MEAN_INFO = pytest.mark.feature("family.loglogistic", "summary.containers", "summary.mean")
QUANTILE = pytest.mark.feature("family.loglogistic", "summary.quantile")
QUANTILE_INFO = pytest.mark.feature("family.loglogistic", "summary.containers", "summary.quantile")
RMST = pytest.mark.feature("family.loglogistic", "summary.rmst")
RMST_INFO = pytest.mark.feature("family.loglogistic", "summary.containers", "summary.rmst")


def _dist(beta, alpha):
    return survjax.LogLogistic(shape=f(beta), scale=f(alpha))


def _violations():
    """Middle entry violates `shape` positive (spec 4.1/4.7)."""
    return survjax.LogLogistic(shape=f([2.0, 0.0, 3.0]), scale=f([1.0, 1.0, 2.0]))


def _tangent(fn, x):
    _, t = jax.jvp(fn, (f(x),), (f(1.0),))
    return np.asarray(t, dtype=np.float64)


# =============================================================== mean: beta > 1


@MEAN
@pytest.mark.parametrize("beta", [1.05, 1.5, 2.0, 5.0, 30.0])
def test_mean_closed_form_for_shape_above_one(beta):
    """spec 8.1, rev 55: `alpha*(π/beta)/sin(π/beta)` for `beta > 1`, including the heavy tail `beta = 1.05` (mean about `21 alpha`)."""
    bb, aa = rounded(beta, 2.0)
    assert_rel_close(_dist(bb, aa).mean(), sref.log_logistic_mean(aa, bb))


@MEAN
@pytest.mark.parametrize("mode", ["rev", "fwd"])
@pytest.mark.parametrize("beta", [1.5, 4.0])
def test_mean_derivatives_are_hand_derived(beta, mode):
    """spec 8.4, rev 55: `dm/dalpha = m/alpha`, `dm/dbeta = (m/beta)(x cot x - 1)` with `x = π/beta` (12.1 log units)."""
    bb, aa = rounded(beta, 2.0)
    m = sref.log_logistic_mean(aa, bb)
    x = math.pi / bb
    gk, ga = (jax.grad if mode == "rev" else jax.jacfwd)(lambda k, a: _dist(k, a).mean(), argnums=(0, 1))(f(bb), f(aa))
    assert_log_parameter_derivative(ga, m / aa, aa, m, "dm/dalpha")
    assert_log_parameter_derivative(gk, (m / bb) * (x / math.tan(x) - 1.0), bb, m, "dm/dbeta")


# =============================================================== mean: known infinite for beta <= 1


@MEAN
@pytest.mark.parametrize("beta", [0.3, 0.5, 1.0])
def test_mean_is_plus_infinity_for_shape_at_most_one(beta):
    """spec 8.1, rev 55: "known infinite for `beta<=1`": the mean is `+inf` (including the boundary `beta = 1`)."""
    assert np.asarray(_dist(beta, 2.0).mean()) == INF


@MEAN
def test_known_infinite_mean_is_decided_per_entry():
    """spec 8.1, rev 55: a batch `beta = [0.5, 1, 2]` gives `[+inf, +inf, alpha*π/2]`."""
    (aa,) = rounded(2.0)
    got = np.asarray(survjax.LogLogistic(shape=f([0.5, 1.0, 2.0]), scale=f(aa)).mean())
    assert (got[:2] == INF).all()
    assert_rel_close(got[2], sref.log_logistic_mean(aa, 2.0))


@MEAN
@pytest.mark.parametrize("beta", [0.5, 1.0])
def test_known_infinite_mean_has_nan_tangents(beta):
    """spec 8.1/8.4, rev 56: "known-infinite means" get NaN tangents, which "built-ins provide": in `scale` and in `shape`."""
    assert np.isnan(_tangent(lambda a: _dist(beta, a).mean(), 2.0))
    assert np.isnan(_tangent(lambda k: survjax.LogLogistic(shape=k, scale=f(2.0)).mean(), beta))


@MEAN
def test_finite_entries_keep_finite_tangents_beside_known_infinite_ones():
    """spec 8.4, rev 56: per-entry tangents: with `beta = [0.5, 2]` and per-entry `alpha`, the `beta = 2` entry has the finite `dm/dalpha = π/2` and the other NaN."""
    _, tangent = jax.jvp(
        lambda a: survjax.LogLogistic(shape=f([0.5, 2.0]), scale=a).mean(), (f([2.0, 2.0]),), (f([1.0, 1.0]),)
    )
    tangent = np.asarray(tangent, dtype=np.float64)
    assert np.isnan(tangent[0])
    np.testing.assert_allclose(tangent[1], math.pi / 2, rtol=1e-4 if not X64 else 1e-8)


@MEAN_INFO
@pytest.mark.parametrize("beta", [0.5, 1.0])
def test_known_infinite_mean_reports_known_infinite_and_no_work(beta):
    """spec 8.1/8.5, rev 56: `mean_with_info` gives `+inf` with status `known_infinite`, a NaN `error_estimate` and zero work counts ("an analytical answer"); `mean()` is its value part."""
    d = _dist(beta, 2.0)
    value, info = d.mean_with_info()
    assert np.asarray(value) == INF
    assert_status(info, "known_infinite")
    assert np.isnan(np.asarray(info.error_estimate))
    for name in ("num_levels", "num_solver_steps", "num_bracket_steps"):
        np.testing.assert_array_equal(np.asarray(getattr(info, name)), 0, err_msg=name)
    assert np.asarray(d.mean()) == INF


@MEAN_INFO
def test_finite_mean_is_an_analytical_answer():
    """spec 8.1, rev 56: for `beta > 1` the closed form reports `converged`, a NaN `error_estimate` and zero work counts."""
    _, info = _dist(2.0, 2.0).mean_with_info()
    assert_status(info, "converged")
    assert np.isnan(np.asarray(info.error_estimate))
    for name in ("num_levels", "num_solver_steps", "num_bracket_steps"):
        np.testing.assert_array_equal(np.asarray(getattr(info, name)), 0, err_msg=name)


@MEAN_INFO
def test_mean_statuses_are_per_entry():
    """spec 8.1/8.5, rev 56: `beta = [0.5, 1, 2]`: statuses `[known_infinite, known_infinite, converged]`."""
    values, info = survjax.LogLogistic(shape=f([0.5, 1.0, 2.0]), scale=f(2.0)).mean_with_info()
    assert (np.asarray(values)[:2] == INF).all()
    assert_status(info, ["known_infinite", "known_infinite", "converged"])


@MEAN
def test_parameter_violation_gives_nan_mean():
    """spec 8.7/8.8, rev 55: NaN where `parameter_issues()` reports a violation (`shape = 0`), not `+inf`; the other entries keep their closed forms."""
    got = np.asarray(_violations().mean())
    assert np.isnan(got[1])
    assert np.isfinite(got[[0, 2]]).all()


@MEAN_INFO
def test_parameter_violation_mean_reports_invalid_input():
    """spec 8.5, rev 55: "entries with parameter issues have status `invalid_input` and NaN values"."""
    values, info = _violations().mean_with_info()
    assert np.isnan(np.asarray(values)[1])
    assert_status(info, ["converged", "invalid_input", "converged"])


# =============================================================== quantile, median, inverse cumulative hazard


@QUANTILE
@pytest.mark.parametrize("p", [1e-6, 0.1, 0.5, 0.9, 0.999])
@pytest.mark.parametrize("beta", [0.5, 1.0, 3.0])
def test_quantile_closed_form(beta, p):
    """spec 8.1, rev 55: `quantile(p) = alpha*(p/(1-p))**(1/beta)`, also for `beta <= 1`."""
    bb, aa, pp = rounded(beta, 2.0, p)
    assert_rel_close(_dist(bb, aa).quantile(f(pp)), sref.log_logistic_quantile(aa, bb, pp))


@QUANTILE
@pytest.mark.parametrize("beta", [0.5, 3.0])
def test_median_is_the_scale(beta):
    """spec 8.1, rev 55: `median() = quantile(0.5) = alpha`."""
    (aa,) = rounded(2.5)
    assert_rel_close(_dist(beta, aa).median(), aa)


@QUANTILE
@pytest.mark.parametrize("e", [1e-30, 1e-8, 0.5, 1.0, 30.0, 80.0] + ([700.0] if X64 else []))
def test_inverse_cumulative_hazard_closed_form(e):
    """spec 8.2, rev 55: `H⁻¹(e) = alpha*expm1(e)**(1/beta)`, with `expm1` keeping tiny `e` exact."""
    bb, aa, ee = rounded(2.0, 1.5, e)
    assert_rel_close(_dist(bb, aa).inverse_cumulative_hazard(f(ee)), sref.log_logistic_inverse_cumulative_hazard(aa, bb, ee))


@QUANTILE
@pytest.mark.parametrize("e", [1e-8, 0.5, 30.0])
def test_cumulative_hazard_inverts_the_inverse_cumulative_hazard(e):
    """spec 8.1, rev 55: `H(H⁻¹(e)) = e`."""
    d = _dist(2.0, 1.5)
    (ee,) = rounded(e)
    assert_rel_close(d.cumulative_hazard(d.inverse_cumulative_hazard(f(ee))), ee)


@QUANTILE
def test_quantile_and_hazard_level_endpoints_and_invalid_inputs():
    """spec 8.1/4.5, rev 55: `quantile(0) = 0`, `quantile(1) = +inf`, `H⁻¹(0) = 0`, `H⁻¹(+inf) = +inf`; `p` outside `[0, 1]` and negative or NaN `e` give NaN."""
    d = _dist(2.0, 1.5)
    np.testing.assert_array_equal(np.asarray(d.quantile(f([0.0, 1.0]))), [0.0, INF])
    np.testing.assert_array_equal(np.asarray(d.inverse_cumulative_hazard(f([0.0, INF]))), [0.0, INF])
    assert np.isnan(np.asarray(d.quantile(f([1.5, -0.1, NAN])))).all()
    assert np.isnan(np.asarray(d.inverse_cumulative_hazard(f([-1.0, NAN])))).all()


@QUANTILE
@pytest.mark.parametrize("mode", ["rev", "fwd"])
@pytest.mark.parametrize("p", [0.05, 0.5, 0.95])
def test_quantile_derivatives_are_hand_derived(p, mode):
    """spec 8.4, rev 55: `dq/dalpha = q/alpha`, `dq/dbeta = -q log(r)/beta**2`, `dq/dp = 1/f(q) = q/(beta p (1-p))` (12.1 log units)."""
    bb, aa, pp = rounded(2.0, 1.5, p)
    q = sref.log_logistic_quantile(aa, bb, pp)
    r = pp / (1 - pp)
    gk, ga, gp = (jax.grad if mode == "rev" else jax.jacfwd)(lambda k, a, p_: _dist(k, a).quantile(p_), argnums=(0, 1, 2))(
        f(bb), f(aa), f(pp)
    )
    assert_log_parameter_derivative(ga, q / aa, aa, q, "dq/dalpha")
    assert_log_parameter_derivative(gk, -q * math.log(r) / bb**2, bb, q, "dq/dbeta")
    assert_log_parameter_derivative(gp, q / (bb * pp * (1 - pp)), pp, q, "dq/dp")


@QUANTILE
@pytest.mark.parametrize("mode", ["rev", "fwd"])
@pytest.mark.parametrize("e", [0.01, 1.0, 20.0])
def test_inverse_cumulative_hazard_derivative_in_the_hazard_level(e, mode):
    """spec 8.4, rev 55: `dt/de = 1/h(t)` at `t = H⁻¹(e)` (hand form `t exp(e)/(beta expm1(e))`)."""
    bb, aa, ee = rounded(2.0, 1.5, e)
    t = sref.log_logistic_inverse_cumulative_hazard(aa, bb, ee)
    g = (jax.grad if mode == "rev" else jax.jacfwd)(lambda e_: _dist(bb, aa).inverse_cumulative_hazard(e_))(f(ee))
    want = math.exp(-fr.direct("loglogistic", (bb, aa), t, "log_hazard")[0])
    assert_log_parameter_derivative(g, want, ee, t, "dt/de")


@QUANTILE
def test_parameter_violation_gives_nan_quantile():
    """spec 8.7/8.8, rev 55: quantile and median are NaN on the violating entry."""
    for values in (np.asarray(_violations().quantile(f(0.4))), np.asarray(_violations().median())):
        assert np.isnan(values[1])
        assert np.isfinite(values[[0, 2]]).all()


@QUANTILE_INFO
def test_closed_form_quantile_reports_converged():
    """spec 8.5, rev 55: valid interior closed-form quantiles report `converged`."""
    _, info = _dist(0.5, 2.0).quantile_with_info(f([0.1, 0.5, 0.9]))
    assert_status(info, "converged")


# =============================================================== RMST (generic quadrature)


@RMST
@pytest.mark.parametrize("tau", [0.1, 2.0, 50.0, 1e4])
@pytest.mark.parametrize("beta", [0.5, 1.05, 3.0])
def test_rmst_matches_mpmath_quadrature(beta, tau):
    """spec 8.1/8.3, rev 55: "generic quadrature", against mpmath quadrature of `S`, including heavy tails (`beta` 0.5 and near 1) at large cutoffs (12.1: `10*rtol`)."""
    bb, aa, tt = rounded(beta, 2.0, tau)
    assert_generic_close(_dist(bb, aa).rmst(f(tt)), sref.log_logistic_rmst(aa, bb, tt))


@RMST
def test_rmst_value_policy():
    """spec 4.5/8.7/8.8, rev 55: `rmst(0) = 0`; NaN for negative, NaN and infinite `tau`; NaN on the violating entry."""
    got = np.asarray(_dist(1.5, 2.0).rmst(f([0.0, -1.0, NAN, INF])))
    assert got[0] == 0.0
    assert np.isnan(got[1:]).all()
    assert np.isnan(np.asarray(_violations().rmst(f(2.0)))[1])


@RMST
@pytest.mark.parametrize("mode", ["rev", "fwd"])
@pytest.mark.parametrize("tau", [0.5, 5.0, 100.0])
def test_rmst_derivatives_match_quadrature(tau, mode):
    """spec 8.4, rev 55: the generic tangent `∫ ∂θ S dt` in `shape` and `scale` against mpmath quadrature of `dS/dθ` (Q-058 units at `10*rtol`); `d rmst/d tau = S(tau)` within absolute `10*rtol`."""
    bb, aa, tt = rounded(1.5, 2.0, tau)
    value = sref.log_logistic_rmst(aa, bb, tt)
    gk, ga, gt = (jax.grad if mode == "rev" else jax.jacfwd)(lambda k, a, t_: _dist(k, a).rmst(t_), argnums=(0, 1, 2))(
        f(bb), f(aa), f(tt)
    )
    assert_generic_log_derivative(gk, sref.log_logistic_rmst_derivative(aa, bb, tt, "beta"), bb, value, "d rmst/d beta")
    assert_generic_log_derivative(ga, sref.log_logistic_rmst_derivative(aa, bb, tt, "alpha"), aa, value, "d rmst/d alpha")
    s_tau = fr.direct("loglogistic", (bb, aa), tt, "survival")[0]
    np.testing.assert_allclose(np.asarray(gt, dtype=np.float64), s_tau, rtol=0.0, atol=generic_rtol())


@RMST_INFO
def test_rmst_reports_converged_quadrature():
    """spec 8.3/8.5, rev 55: the generic quadrature converges with at least two levels ("one evaluated level alone cannot establish convergence") and at most `max_levels`."""
    _, info = _dist(1.5, 2.0).rmst_with_info(f([0.5, 5.0, 100.0]))
    assert_status(info, "converged")
    levels = np.asarray(info.num_levels)
    assert ((levels >= 2) & (levels <= 8)).all(), levels
