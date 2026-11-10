"""Features `family.lognormal` + `summary.*`: analytical summaries of `LogNormal` (spec 4.5, 8.1, 8.2, 8.4, 8.5, 8.7, 8.8, 12.1, 12.2; rev 55, 56).

12.2 (rev 55): "Their analytical summaries of section 8.1 are enabled with the
matching summary method (section 8.8)", so every test lists `family.lognormal`
and the summary feature it needs. 8.1 table, log-normal row: mean
`exp(mu+sigma²/2)`; median/quantile `exp(mu+sigma*Φ⁻¹(p))`; RMST `tau*S(tau)
+ exp(mu+sigma²/2)*Φ((log tau-mu-sigma²)/sigma)`. 8.2: closed form `H⁻¹(e) =
exp(mu+sigma*Φ⁻¹(F))` "using `F=-expm1(-e)` for small `e` and `-Φ⁻¹(exp(-e))`
for large `e`". 8.1 rev 56: "An analytical answer reports status `converged`
(or `known_infinite`), a NaN `error_estimate` and zero work counts." 12.1:
"closed forms keep the ordinary targets"; derivatives in the 12.1 log units
(Q-058: ``|theta (got - want)| <= rtol |value|``; ``meanlog`` is a location on
the log-time scale, like ``eta``, so its scale is ``|value|`` with ``theta =
1``).

Hand derivatives: mean ``m = exp(mu + sigma**2/2)``: ``dm/dmu = m``,
``dm/dsigma = sigma m``. Quantile ``q = exp(mu + sigma y)``, ``y = Φ⁻¹(p)``:
``dq/dmu = q``, ``dq/dsigma = q y``, ``dq/dp = 1/f(q)`` (8.4). ``H⁻¹``:
``dt/de = 1/h(t)`` (8.4). RMST: ``d/dtau = S(tau)`` (8.7). Other derivatives
are ``mpmath.diff`` of the closed forms (``summary_reference``).
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
    assert_close,
    assert_cutoff_derivative,
    assert_log_parameter_derivative,
    assert_rel_close,
    assert_status,
    f,
    rounded,
)

import survjax

MEAN = pytest.mark.feature("family.lognormal", "summary.mean")
MEAN_INFO = pytest.mark.feature("family.lognormal", "summary.containers", "summary.mean")
QUANTILE = pytest.mark.feature("family.lognormal", "summary.quantile")
QUANTILE_INFO = pytest.mark.feature("family.lognormal", "summary.containers", "summary.quantile")
RMST = pytest.mark.feature("family.lognormal", "summary.rmst")
RMST_INFO = pytest.mark.feature("family.lognormal", "summary.containers", "summary.rmst")

CASES = [(0.0, 1.0), (1.5, 0.4), (-2.0, 2.0)]


def _dist(mu, sigma):
    return survjax.LogNormal(meanlog=f(mu), sdlog=f(sigma))


def _violations():
    """Middle entry violates `sdlog` positive (spec 4.1/4.7)."""
    return survjax.LogNormal(meanlog=f([0.0, 0.0, 1.0]), sdlog=f([1.0, 0.0, 0.5]))


def _assert_analytical_info(info):
    assert_status(info, "converged")
    assert np.isnan(np.asarray(info.error_estimate)).all()
    for name in ("num_levels", "num_solver_steps", "num_bracket_steps"):
        np.testing.assert_array_equal(np.asarray(getattr(info, name)), 0, err_msg=name)


# =============================================================== mean


@MEAN
@pytest.mark.parametrize("mu, sigma", CASES)
def test_mean_closed_form(mu, sigma):
    """spec 8.1, rev 55: log-normal mean `exp(mu + sigma**2/2)`."""
    m, s = rounded(mu, sigma)
    assert_rel_close(_dist(m, s).mean(), sref.lognormal_mean(m, s))


@MEAN
def test_mean_is_vectorized_over_the_batch():
    """spec 8.1/8.7, rev 55: the mean has `batch_shape`; each entry is its own closed form."""
    mus, sigmas = rounded(0.0, 1.5, -2.0), rounded(1.0, 0.4, 2.0)
    got = survjax.LogNormal(meanlog=f(mus), sdlog=f(sigmas)).mean()
    assert np.shape(got) == (3,)
    assert_rel_close(got, [sref.lognormal_mean(m, s) for m, s in zip(mus, sigmas, strict=True)])


@MEAN
@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_mean_derivatives_are_hand_derived(mode):
    """spec 8.4, rev 55: `dm/dmu = m`, `dm/dsigma = sigma m` (12.1 log units)."""
    mu, sigma = rounded(0.3, 0.8)
    m = sref.lognormal_mean(mu, sigma)
    gm, gs = (jax.grad if mode == "rev" else jax.jacfwd)(lambda a, s: survjax.LogNormal(meanlog=a, sdlog=s).mean(), argnums=(0, 1))(f(mu), f(sigma))
    assert_log_parameter_derivative(gm, m, 1.0, m, "dm/dmu")
    assert_log_parameter_derivative(gs, sigma * m, sigma, m, "dm/dsigma")


@MEAN
def test_parameter_violation_gives_nan_mean():
    """spec 8.7/8.8, rev 55: "NaN wherever `parameter_issues()` reports a violation"; the other entries keep their closed forms."""
    got = np.asarray(_violations().mean())
    assert np.isnan(got[1])
    assert_rel_close(got[[0, 2]], [sref.lognormal_mean(0.0, 1.0), sref.lognormal_mean(1.0, float(f(0.5)))])


@MEAN_INFO
def test_mean_with_info_is_an_analytical_answer():
    """spec 8.1, rev 56: the closed-form mean reports `converged`, a NaN `error_estimate` and zero work counts; `mean()` is its value part."""
    d = _dist(0.3, 0.8)
    value, info = d.mean_with_info()
    _assert_analytical_info(info)
    assert_close(value, d.mean())


@MEAN_INFO
def test_parameter_violation_mean_reports_invalid_input():
    """spec 8.5, rev 55: "entries with parameter issues have status `invalid_input` and NaN values"."""
    values, info = _violations().mean_with_info()
    assert np.isnan(np.asarray(values)[1])
    assert_status(info, ["converged", "invalid_input", "converged"])


# =============================================================== quantile, median, inverse cumulative hazard


@QUANTILE
@pytest.mark.parametrize("p", [1e-6, 0.1, 0.5, 0.9, 0.999])
@pytest.mark.parametrize("mu, sigma", CASES)
def test_quantile_closed_form(mu, sigma, p):
    """spec 8.1, rev 55: `quantile(p) = exp(mu + sigma*Φ⁻¹(p))`."""
    m, s, pp = rounded(mu, sigma, p)
    assert_rel_close(_dist(m, s).quantile(f(pp)), sref.lognormal_quantile(m, s, pp))


@QUANTILE
@pytest.mark.parametrize("mu, sigma", CASES)
def test_median_is_exp_meanlog(mu, sigma):
    """spec 8.1, rev 55: `median() = quantile(0.5) = exp(mu)` (`Φ⁻¹(1/2) = 0`)."""
    (m,) = rounded(mu)
    assert_rel_close(_dist(mu, sigma).median(), math.exp(m))


@QUANTILE
@pytest.mark.parametrize("e", [1e-30, 1e-8, 0.5, 1.0, 30.0, 80.0] + ([600.0] if X64 else []))
def test_inverse_cumulative_hazard_closed_form_in_both_regimes(e):
    """spec 8.2, rev 55: `H⁻¹(e) = exp(mu + sigma Φ⁻¹(F))` with `F = -expm1(-e)` for small `e` (`1e-30` keeps `F` exact) and `-Φ⁻¹(exp(-e))` for large `e` (`exp(-600)` in float64)."""
    mu, sigma, ee = rounded(0.3, 0.8, e)
    assert_rel_close(_dist(mu, sigma).inverse_cumulative_hazard(f(ee)), sref.lognormal_inverse_cumulative_hazard(mu, sigma, ee))


@QUANTILE
@pytest.mark.parametrize("e", [1e-8, 0.5, 30.0])
def test_cumulative_hazard_inverts_the_inverse_cumulative_hazard(e):
    """spec 8.1, rev 55: `inverse_cumulative_hazard(e)` is "the `t` with `H(t)=e`": `H(H⁻¹(e)) = e`."""
    d = _dist(0.3, 0.8)
    (ee,) = rounded(e)
    assert_rel_close(d.cumulative_hazard(d.inverse_cumulative_hazard(f(ee))), ee)


@QUANTILE
def test_quantile_and_hazard_level_endpoints():
    """spec 8.1, rev 55: endpoints from the default `support() = (0, +inf)`: `quantile(0) = 0`, `quantile(1) = +inf`, `H⁻¹(0) = 0`, `H⁻¹(+inf) = +inf`."""
    d = _dist(0.3, 0.8)
    np.testing.assert_array_equal(np.asarray(d.quantile(f([0.0, 1.0]))), [0.0, INF])
    np.testing.assert_array_equal(np.asarray(d.inverse_cumulative_hazard(f([0.0, INF]))), [0.0, INF])


@QUANTILE
def test_invalid_probability_and_hazard_level_give_nan():
    """spec 4.5/8.1/8.8, rev 55: `p` outside `[0, 1]` and negative or NaN `e` give NaN; valid entries keep their closed forms."""
    d = _dist(0.3, 0.8)
    q = np.asarray(d.quantile(f([0.5, 1.5, -0.1, NAN])))
    assert np.isnan(q[1:]).all()
    assert_rel_close(q[0], math.exp(float(f(0.3))))
    t = np.asarray(d.inverse_cumulative_hazard(f([-1.0, NAN, 1.0])))
    assert np.isnan(t[:2]).all()
    assert np.isfinite(t[2])


@QUANTILE
@pytest.mark.parametrize("mode", ["rev", "fwd"])
@pytest.mark.parametrize("p", [0.05, 0.5, 0.95])
def test_quantile_derivatives_are_hand_derived(p, mode):
    """spec 8.4, rev 55: `dq/dmu = q`, `dq/dsigma = q Φ⁻¹(p)`, `dq/dp = 1/f(q)` (12.1 log units)."""
    mu, sigma, pp = rounded(0.3, 0.8, p)
    q = sref.lognormal_quantile(mu, sigma, pp)
    y = (math.log(q) - mu) / sigma
    log_f_q = fr.direct("lognormal", (mu, sigma), q, "log_prob")[0]
    grads = (jax.grad if mode == "rev" else jax.jacfwd)(lambda a, s, p_: survjax.LogNormal(meanlog=a, sdlog=s).quantile(p_), argnums=(0, 1, 2))(
        f(mu), f(sigma), f(pp)
    )
    assert_log_parameter_derivative(grads[0], q, 1.0, q, "dq/dmu")
    assert_log_parameter_derivative(grads[1], q * y, sigma, q, "dq/dsigma")
    assert_log_parameter_derivative(grads[2], math.exp(-log_f_q), pp, q, "dq/dp")


@QUANTILE
@pytest.mark.parametrize("mode", ["rev", "fwd"])
@pytest.mark.parametrize("e", [0.01, 1.0, 20.0])
def test_inverse_cumulative_hazard_derivatives(e, mode):
    """spec 8.4, rev 55: `t = H⁻¹(e)`: `dt/dmu = t`, `dt/dsigma = t (log t - mu)/sigma`, `dt/de = 1/h(t)`."""
    mu, sigma, ee = rounded(0.3, 0.8, e)
    t = sref.lognormal_inverse_cumulative_hazard(mu, sigma, ee)
    y = (math.log(t) - mu) / sigma
    log_h = fr.direct("lognormal", (mu, sigma), t, "log_hazard")[0]
    grads = (jax.grad if mode == "rev" else jax.jacfwd)(
        lambda a, s, e_: survjax.LogNormal(meanlog=a, sdlog=s).inverse_cumulative_hazard(e_), argnums=(0, 1, 2)
    )(f(mu), f(sigma), f(ee))
    assert_log_parameter_derivative(grads[0], t, 1.0, t, "dt/dmu")
    assert_log_parameter_derivative(grads[1], t * y, sigma, t, "dt/dsigma")
    assert_log_parameter_derivative(grads[2], math.exp(-log_h), ee, t, "dt/de")


@QUANTILE
def test_endpoint_quantiles_have_nan_tangents():
    """spec 8.4, rev 55: "endpoint quantiles (`p` = 0 or 1)" get NaN tangents, also on the log-normal closed form."""
    for p in (0.0, 1.0):
        _, tangent = jax.jvp(lambda a: survjax.LogNormal(meanlog=a, sdlog=f(0.8)).quantile(f(p)), (f(0.3),), (f(1.0),))
        assert np.isnan(np.asarray(tangent)), p


@QUANTILE
def test_parameter_violation_gives_nan_quantile():
    """spec 8.7/8.8, rev 55: quantile and median are NaN where `parameter_issues()` reports a violation."""
    for values in (np.asarray(_violations().quantile(f(0.4))), np.asarray(_violations().median())):
        assert np.isnan(values[1])
        assert np.isfinite(values[[0, 2]]).all()


@QUANTILE_INFO
def test_closed_form_quantile_reports_converged():
    """spec 8.5, rev 55: valid interior closed-form quantiles report `converged` (the rev 56 "analytical answer" sentence is stated for `mean_with_info` and is not asserted here)."""
    _, info = _dist(0.3, 0.8).quantile_with_info(f([0.1, 0.5, 0.9]))
    assert_status(info, "converged")


# =============================================================== RMST


@RMST
@pytest.mark.parametrize("tau", [0.1, 1.5, 10.0, 1e3])
@pytest.mark.parametrize("mu, sigma", CASES)
def test_rmst_closed_form(mu, sigma, tau):
    """spec 8.1, rev 55: `rmst(tau) = tau*S(tau) + exp(mu+sigma²/2)*Φ((log tau - mu - sigma²)/sigma)`."""
    m, s, tt = rounded(mu, sigma, tau)
    assert_rel_close(_dist(m, s).rmst(f(tt)), sref.lognormal_rmst(m, s, tt))


@RMST
def test_rmst_at_a_large_cutoff_approaches_the_mean_from_below():
    """spec 8.1, rev 55: `rmst(tau) <= mean` and tends to it: at `tau = exp(mu + 12 sigma)` they agree to the ordinary target."""
    mu, sigma = rounded(0.3, 0.8)
    (tau,) = rounded(math.exp(mu + 12 * sigma))
    got = float(np.asarray(_dist(mu, sigma).rmst(f(tau))))
    assert got <= sref.lognormal_mean(mu, sigma) * (1 + 1e-12)
    assert_rel_close(got, sref.lognormal_rmst(mu, sigma, tau))


@RMST
def test_rmst_value_policy():
    """spec 4.5/8.7/8.8, rev 55: `rmst(0) = 0`; NaN for negative, NaN and infinite `tau`; NaN on the parameter-violation entry."""
    d = _dist(0.3, 0.8)
    got = np.asarray(d.rmst(f([0.0, -1.0, NAN, INF])))
    assert got[0] == 0.0
    assert np.isnan(got[1:]).all()
    assert np.isnan(np.asarray(_violations().rmst(f(2.0)))[1])


@RMST
@pytest.mark.parametrize("mode", ["rev", "fwd"])
@pytest.mark.parametrize("tau", [0.5, 3.0, 20.0])
def test_rmst_derivatives(tau, mode):
    """spec 8.4/8.7, rev 55: `d rmst/d tau = S(tau)` (absolute scale 1); `meanlog` and `sdlog` derivatives against `mpmath.diff` of the closed form (12.1 log units)."""
    mu, sigma, tt = rounded(0.3, 0.8, tau)
    value = sref.lognormal_rmst(mu, sigma, tt)
    gm, gs, gt = (jax.grad if mode == "rev" else jax.jacfwd)(lambda a, s, t_: survjax.LogNormal(meanlog=a, sdlog=s).rmst(t_), argnums=(0, 1, 2))(
        f(mu), f(sigma), f(tt)
    )
    want_m = sref.derivative(sref.lognormal_rmst_mp, (mu, sigma, tt), (1, 0, 0))
    want_s = sref.derivative(sref.lognormal_rmst_mp, (mu, sigma, tt), (0, 1, 0))
    assert_log_parameter_derivative(gm, want_m, 1.0, value, "d rmst/d mu")
    assert_log_parameter_derivative(gs, want_s, sigma, value, "d rmst/d sigma")
    assert_cutoff_derivative(gt, fr.direct("lognormal", (mu, sigma), tt, "survival")[0], "d rmst/d tau")


@RMST
def test_rmst_cutoff_grid_with_explicit_axes():
    """spec 8.7, rev 55: `tau_grid[:, None, None]` against a `(S, 2)` batch gives `(m, S, 2)`."""
    d = survjax.LogNormal(meanlog=f([[0.0, 0.5], [1.0, -0.5], [0.2, 0.3]]), sdlog=f(0.8))
    grid = f([0.5, 2.0, 8.0])
    values = d.rmst(grid[:, None, None])
    assert np.shape(values) == (3, 3, 2)
    for i, tau in enumerate(grid):
        assert_close(np.asarray(values)[i], d.rmst(f(tau)))


@RMST_INFO
def test_closed_form_rmst_reports_converged():
    """spec 8.5, rev 55: valid closed-form RMST entries report `converged` (counts and error estimate not asserted; see the quantile case)."""
    _, info = _dist(0.3, 0.8).rmst_with_info(f([0.5, 3.0]))
    assert_status(info, "converged")
