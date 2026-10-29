"""Shared release generator: use-cases/00-common and 01 details."""

import sys
from pathlib import Path

import impl_tolerances as tol
import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from jax.test_util import check_grads

# Examples are source-checkout demo code, deliberately absent from the wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.weibull_simulation.study import (  # noqa: E402
    calibrate_censoring,
    generate_weibull,
)


def test_exponential_calibration_has_analytical_integral():
    """00-common: at c=2b, integral S/c=(1-exp(-2))/2 for k=1."""
    target = -np.expm1(-2) / 2
    c = calibrate_censoring(1.0, 3.0, target)
    np.testing.assert_allclose(c, 6.0, atol=tol.ATOL[True], rtol=tol.RTOL[True])


def test_heterogeneous_calibration_matches_independent_quadrature():
    """00-common: average every broadcast row, not just average parameters."""
    k, b = np.array([[0.5], [2.7]]), np.array([[2.0, 5.0, 11.0]])
    c = calibrate_censoring(k, b, 0.55)
    k, b = np.broadcast_arrays(k, b)
    with mp.workdps(40):
        expected = (
            sum(
                mp.quad(lambda t, k=shape, b=scale: mp.exp(-((t / b) ** k)), [0, c]) / c
                for shape, scale in zip(k.ravel(), b.ravel(), strict=True)
            )
            / 6
        )
    np.testing.assert_allclose(
        float(expected), 0.55, atol=tol.ATOL[True], rtol=tol.RTOL[True]
    )


@pytest.mark.parametrize("target", [0.25, 0.55])
def test_simulation_calibration_matches_mpmath(target):
    """01 calibration: equal-arm integrated survivals, independently at 40 dps."""
    k, b0, gamma = 1.38, 11.42, np.log(0.7)
    b = b0 * np.exp(-gamma * np.array([0, 1]) / k)
    c = calibrate_censoring(k, b0, target, log_hazard_ratio=gamma * np.array([0, 1]))
    with mp.workdps(40):
        pooled = (
            sum(
                mp.quad(lambda t, b=scale: mp.exp(-((t / b) ** k)), [0, c]) / c
                for scale in b
            )
            / 2
        )
    np.testing.assert_allclose(
        float(pooled), target, atol=tol.ATOL[True], rtol=tol.RTOL[True]
    )


@pytest.mark.parametrize(
    "k,b,target", [(0, 1, 0.5), (1, np.inf, 0.5), ([], [], 0.5), (1, 1, 1)]
)
def test_calibration_rejects_invalid_inputs(k, b, target):
    """00-common: positive finite Weibull parameters and an interior target."""
    with pytest.raises(ValueError):
        calibrate_censoring(k, b, target)


def test_calibration_rejects_unconverged_quadrature():
    """00-common calibration: do not return a root with unverified quadrature."""
    with pytest.raises(RuntimeError, match="quadrature"):
        calibrate_censoring(20.0, 5.0, 0.5)


def test_keys_broadcast_and_transformations_follow_details(x64_enabled):
    """01 seeds: root -> fold_in scenario -> split replicates -> split U/C."""
    keys = jax.random.split(jax.random.fold_in(jax.random.key(20261025), 2), 3)
    k, b = jnp.array([[1.0], [1.38]]), jnp.array([[3.0, 5.0, 8.0]])

    def draw(key):
        return generate_weibull(key, k, b, 10.0, sample_shape=(4,))

    eager = jax.tree.map(lambda *xs: jnp.stack(xs), *(draw(key) for key in keys))
    for result in (jax.vmap(draw)(keys), jax.jit(jax.vmap(draw))(keys)):
        for actual, expected in zip(result, eager, strict=True):
            np.testing.assert_allclose(
                actual, expected, atol=tol.ATOL[x64_enabled], rtol=tol.RTOL[x64_enabled]
            )
    assert eager.time.shape == (3, 4, 2, 3)
    u_key, c_key = jax.random.split(keys[0])
    u = jax.random.uniform(u_key, (4, 2, 3), dtype=k.dtype)
    # Independent inverse-CDF algebra and the exact prescribed censor key.
    np.testing.assert_allclose(
        eager.event_time[0],
        np.asarray(b) * (-np.log(np.asarray(u))) ** (1 / np.asarray(k)),
        atol=tol.ATOL[x64_enabled],
        rtol=tol.RTOL[x64_enabled],
    )
    np.testing.assert_array_equal(
        eager.censor_time[0], 10 * jax.random.uniform(c_key, (4, 2, 3), dtype=k.dtype)
    )
    scalar = generate_weibull(keys[0], 1.38, 11.42, 20.0)
    assert scalar.time.shape == ()
    assert scalar.time.dtype == (jnp.float64 if x64_enabled else jnp.float32)


@pytest.mark.parametrize("target", [0.25, 0.55])
def test_empirical_survival_hr_and_censoring_recover_truth(target):
    """01 DGP: seeded large samples recover survival, HR and pooled censoring."""
    n, k, gamma = 100_000, 1.38, np.log(0.7)
    b = 11.42 * np.exp(-gamma * np.array([0, 1]) / k)
    c = calibrate_censoring(k, b, target)
    root = jax.random.key(20261025)
    scenario = jax.random.fold_in(root, int(target == 0.55))
    key = jax.random.split(scenario, 1)[0]
    data = generate_weibull(key, k, jnp.asarray(b), c, sample_shape=(n,))
    t, censor = np.asarray(data.event_time), np.asarray(data.censor_time)
    np.testing.assert_array_equal(data.time, np.minimum(t, censor))
    np.testing.assert_array_equal(data.event, t <= censor)
    # Bernoulli MCSE, six-sigma bounds fixed before observing the draws.
    assert abs(
        np.mean(~np.asarray(data.event)) - target
    ) < tol.SIMULATION_SIGMAS / np.sqrt(4 * 2 * n)
    for time in (5.0, 10.0, 20.0):
        truth = np.exp(-((time / b) ** k))
        observed = (t > time).mean(axis=0)
        assert np.all(
            abs(observed - truth)
            < tol.SIMULATION_SIGMAS * np.sqrt(truth * (1 - truth) / n)
        )
    survival = (t > 10).mean(axis=0)
    log_hr = np.log(-np.log(survival[1])) - np.log(-np.log(survival[0]))
    # Delta method: Var(log(-log(S_hat))) = (1-S)/(n*S*log(S)^2).
    truth = np.exp(-((10 / b) ** k))
    variance = np.sum((1 - truth) / (n * truth * np.log(truth) ** 2))
    assert abs(log_hr - gamma) < tol.SIMULATION_SIGMAS * np.sqrt(variance)
    # Independence: joint exceedance probability is S_T(10) * P(C>c/2).
    joint = ((t > 10) & (censor > c / 2)).mean(axis=0)
    p = truth / 2
    assert np.all(abs(joint - p) < tol.SIMULATION_SIGMAS * np.sqrt(p * (1 - p) / n))


def test_generator_event_times_have_parameter_derivatives(x64_enabled):
    """Spec 7.3: fixed-key inverse CDF is smooth in positive k/b, to order 2."""

    def total(parameters):
        return generate_weibull(
            jax.random.key(12), parameters[0], parameters[1], 5.0, sample_shape=(3,)
        ).event_time.sum()

    check_grads(
        total,
        (jnp.array([1.38, 2.0]),),
        order=2,
        modes=("fwd", "rev"),
        eps=tol.LIKELIHOOD_GRAD_EPS[x64_enabled],
        atol=tol.DERIVATIVE_ATOL[x64_enabled],
        rtol=tol.DERIVATIVE_RTOL[x64_enabled],
    )


def test_ph_generator_matches_equivalent_weibull_under_transformations(x64_enabled):
    """Spec 4.4 / 01 DGP: PH changes b0 to b0*exp(-eta/k), with row broadcasting."""
    k, b = jnp.array([[1.0], [1.38]]), jnp.array([3.0, 5.0, 8.0])
    eta = jnp.array([[-0.3, 0.0, 0.2], [0.5, -0.7, 0.3]])
    key = jax.random.key(20261025)

    def draw(lp):
        return generate_weibull(key, k, b, 10.0, log_hazard_ratio=lp)

    expected = generate_weibull(key, k, b * jnp.exp(-eta / k), 10.0)
    mapped = jax.tree.map(
        lambda x: x[0], jax.jit(jax.vmap(draw))(jnp.stack([eta, eta]))
    )
    for result in (draw(eta), jax.jit(draw)(eta), mapped):
        for actual, reference in zip(result, expected, strict=True):
            np.testing.assert_allclose(
                actual,
                reference,
                atol=tol.ATOL[x64_enabled],
                rtol=tol.RTOL[x64_enabled],
            )


@pytest.mark.parametrize("eta", [np.nan, 1000.0])
def test_ph_calibration_rejects_unusable_effective_scale(eta):
    """01 DGP: calibration requires finite eta and representable positive scales."""
    with pytest.raises(ValueError):
        calibrate_censoring(1.0, 5.0, 0.5, log_hazard_ratio=eta)


def test_ph_parameter_derivatives_match_inverse_cdf_algebra(x64_enabled):
    """Spec 7.3 / 01 DGP: T=b*exp((log(-log(U))-eta)/k) gives all partials."""
    key, p = jax.random.key(12), jnp.array([1.38, 2.0, 0.3])
    u = np.asarray(jax.random.uniform(jax.random.split(key)[0], (3,), dtype=p.dtype))
    k, b, eta = np.asarray(p)
    a = np.log(-np.log(u)) - eta
    t = b * np.exp(a / k)
    expected_grad = np.array([np.sum(-a * t / k**2), np.sum(t / b), np.sum(-t / k)])
    kk, kb, ke = (
        np.sum(t * (a * a / k**4 + 2 * a / k**3)),
        np.sum(-a * t / (b * k**2)),
        np.sum(t * (1 / k**2 + a / k**3)),
    )
    be, ee = np.sum(-t / (b * k)), np.sum(t / k**2)
    expected_hessian = np.array([[kk, kb, ke], [kb, 0.0, be], [ke, be, ee]])

    def total(parameters):
        return generate_weibull(
            key,
            parameters[0],
            parameters[1],
            5.0,
            sample_shape=(3,),
            log_hazard_ratio=parameters[2],
        ).event_time.sum()

    for derivative in (jax.jacfwd(total), jax.jacrev(total)):
        np.testing.assert_allclose(
            jax.jit(derivative)(p),
            expected_grad,
            atol=tol.ATOL[x64_enabled],
            rtol=tol.RTOL[x64_enabled],
        )
    np.testing.assert_allclose(
        jax.jit(jax.hessian(total))(p),
        expected_hessian,
        atol=tol.ATOL[x64_enabled],
        rtol=tol.RTOL[x64_enabled],
    )
