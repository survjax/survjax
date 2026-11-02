"""Optional checks: JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q <this file>."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import impl_tolerances as tol  # noqa: E402

import survjax  # noqa: E402
from examples.weibull_simulation.study import (  # noqa: E402
    _assess_fit,
    calibrate_censoring,
    fit_replicate,
    generate_weibull,
    negative_log_likelihood,
)

if not jax.config.x64_enabled:
    raise RuntimeError("Run these use-case checks with JAX_ENABLE_X64=1")

ATOL, RTOL = tol.ATOL[True], tol.RTOL[True]


def cohort():
    # Exact/right data: d0=d1=2 events, Y0=6, Y1=12 total follow-up.
    data = survjax.SurvivalData(
        jnp.array([1.0, 2.0, 3.0, 2.0, 4.0, 6.0]),
        event=jnp.array([1, 0, 1, 1, 1, 0], dtype=bool),
    )
    data.issues.raise_if_invalid()
    return data, jnp.array([0, 0, 0, 1, 1, 1])


@pytest.mark.parametrize("model", ["weibull", "exponential"])
def test_objective_and_derivatives_match_closed_form(model):
    """01 models / spec 7.3: sum event*log(h)-H, including PH on each arm."""
    data, arm = cohort()
    theta = (
        jnp.array([np.log(1.4), np.log(3.0), -0.2])
        if model == "weibull"
        else jnp.array([-np.log(3.0), -0.2])
    )

    def objective(p):
        return negative_log_likelihood(p, data, arm, model=model) / len(arm)

    def analytical(p):
        if model == "weibull":
            k, log_b, gamma = jnp.exp(p[0]), p[1], p[2]
            log_h = p[0] - log_b + (k - 1) * (jnp.log(data.lower) - log_b) + gamma * arm
            h_cum = jnp.exp(k * (jnp.log(data.lower) - log_b) + gamma * arm)
        else:
            log_h = p[0] + p[1] * arm
            h_cum = jnp.exp(log_h) * data.lower
        return jnp.mean(h_cum - data.exact * log_h)

    for operation in (lambda f: f, jax.jacfwd, jax.jacrev, jax.hessian):
        actual = jax.jit(operation(objective))(theta)
        expected = operation(analytical)(theta)
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
    expected = analytical(theta)
    with jax.debug_nans(True):
        np.testing.assert_allclose(objective(theta), expected, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(objective))(jnp.stack([theta, theta + 0.1])),
        jax.vmap(analytical)(jnp.stack([theta, theta + 0.1])),
        atol=ATOL,
        rtol=RTOL,
    )
    check_grads(
        objective,
        (theta,),
        order=2,
        modes=("fwd", "rev"),
        eps=tol.LIKELIHOOD_GRAD_EPS[True],
        atol=tol.DERIVATIVE_ATOL[True],
        rtol=tol.DERIVATIVE_RTOL[True],
    )


def test_exponential_fit_matches_analytical_mle_hessian_and_interval():
    """01 fitting: rates d_g/Y_g are 1/3,1/6; gamma=log(.5)."""
    data, arm = cohort()
    result = fit_replicate(data, arm, model="exponential")
    assert (
        result.accepted & result.solver_success & result.estimable & result.hessian_ok
    )
    np.testing.assert_allclose(
        result.theta, [-np.log(3), np.log(0.5)], atol=tol.SIMULATION_FIT_ATOL, rtol=0
    )
    # At any theta, H=[[mu0+mu1,mu1],[mu1,mu1]], mu_g=lambda_g*Y_g.
    mu0, mu1 = np.exp([result.theta[0], result.theta.sum()]) * [6, 12]
    expected_hessian = [[mu0 + mu1, mu1], [mu1, mu1]]
    se = np.sqrt(1 / mu0 + 1 / mu1)
    np.testing.assert_allclose(result.hessian, expected_hessian, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(result.log_hr_se, se, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(
        result.hr_interval,
        np.exp(result.theta[-1] + np.array([-1.96, 1.96]) * se),
        atol=ATOL,
        rtol=RTOL,
    )
    assert np.max(np.abs(result.gradient)) < 1e-5


@pytest.mark.parametrize("model", ["weibull", "exponential"])
def test_models_accept_one_prescribed_seeded_replicate(model):
    """01 DGP/seeds and fitting: a valid mixed cohort meets all acceptance gates."""
    arm = jnp.repeat(jnp.arange(2), 50)
    lp = np.log(0.7) * arm
    c = calibrate_censoring(
        1.38, 11.42, 0.55, log_hazard_ratio=np.log(0.7) * np.array([0, 1])
    )
    key = jax.random.split(jax.random.fold_in(jax.random.key(20261025), 1), 1)[0]
    generated = generate_weibull(key, 1.38, 11.42, c, log_hazard_ratio=lp)
    data = survjax.SurvivalData(generated.time, event=generated.event)
    data.issues.raise_if_invalid()
    result = jax.jit(lambda d: fit_replicate(d, arm, model=model))(data)
    assert (
        result.accepted & result.solver_success & result.estimable & result.hessian_ok
    )
    assert np.isfinite(result.theta).all() and np.isfinite(result.objective)
    assert np.max(np.abs(result.gradient)) < 1e-5 and result.log_hr_se > 0


def test_jitted_vmap_preserves_estimable_and_unestimable_replicates():
    """01 acceptance: no events in one arm is unestimable, not invalid data."""
    data, arm = cohort()
    events = jnp.stack([data.exact, data.exact & (arm == 0)])
    batch = survjax.SurvivalData(jnp.stack([data.lower, data.lower]), event=events)
    batch.issues.raise_if_invalid()

    def fit(d):
        return fit_replicate(d, arm, model="exponential")

    result = jax.jit(jax.vmap(fit))(batch)
    np.testing.assert_array_equal(result.estimable, [True, False])
    np.testing.assert_array_equal(result.accepted, [True, False])
    np.testing.assert_allclose(
        result.theta[0], [-np.log(3), np.log(0.5)], atol=tol.SIMULATION_FIT_ATOL, rtol=0
    )
    assert np.isnan(result.theta[1]).all() and np.isnan(result.hr_interval[1]).all()


@pytest.mark.parametrize(
    "failure",
    [
        "none",
        "solver",
        "gradient",
        "threshold",
        "nan_gradient",
        "theta",
        "objective",
        "hessian",
        "indefinite",
    ],
)
def test_estimable_fit_acceptance_gates_and_rejected_inference(failure):
    """01 acceptance: each failed gate rejects and retains raw diagnostics."""
    theta = jnp.array([np.nan if failure == "theta" else 0.0, 0.0])
    value = jnp.asarray(np.nan if failure == "objective" else 1.0)
    score = {"gradient": 2e-5, "threshold": 1e-5, "nan_gradient": np.nan}.get(
        failure, 0.0
    )
    gradient = jnp.array([score, 0.0])
    # Positive control H^-1=[[.5,-.5],[-.5,1]] gives gamma SE=1.
    hessian = (
        jnp.array([[1.0, 2.0], [2.0, 1.0]])
        if failure == "indefinite"
        else jnp.array([[4.0, 2.0], [2.0, 2.0]])
    )
    if failure == "hessian":
        hessian = hessian.at[0, 0].set(jnp.nan)
    success = jnp.asarray(failure != "solver")
    inputs = (theta, value, gradient, hessian, success)
    for assess in (_assess_fit, jax.jit(_assess_fit)):
        result = assess(*inputs)
        assert result.estimable and bool(result.accepted) == (failure == "none")
        assert result.solver_success == success
        assert bool(result.hessian_ok) == (failure not in ("hessian", "indefinite"))
        for actual, expected in zip(result[:4], inputs[:4], strict=True):
            np.testing.assert_array_equal(actual, expected)
        if failure == "none":
            np.testing.assert_allclose(result.log_hr_se, 1.0, atol=ATOL, rtol=RTOL)
            np.testing.assert_allclose(
                result.hr_interval,
                np.exp(np.array([-1.96, 1.96])),
                atol=ATOL,
                rtol=RTOL,
            )
        else:
            assert np.isnan(result.log_hr_se) and np.isnan(result.hr_interval).all()
