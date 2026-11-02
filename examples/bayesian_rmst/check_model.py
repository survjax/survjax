"""Explicit optional checks: JAX_ENABLE_X64=1 uv run --group examples python -m examples.bayesian_rmst.check_model."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as distributions
from numpyro.infer import Predictive
from numpyro.infer.util import log_density

import survjax
from examples.bayesian_rmst.model import (
    PosteriorDraws,
    check_rmst_routes,
    make_trial,
    model,
    posterior_draws,
    summarize_draws,
    time_grids,
    use_case_key,
)
from examples.weibull_simulation.study import calibrate_censoring, generate_weibull


def check_trial() -> None:
    """02-details: balanced fixed trial, valid float64 rows and stable keys."""
    trial = make_trial()
    # 02-details Data: independent fixed-value oracle using the shared generator.
    expected_arm = jnp.repeat(jnp.array([0.0, 1.0]), 150)
    expected_eta = jnp.log(0.7) * expected_arm
    expected_c = calibrate_censoring(
        1.38, 11.42, 0.55, log_hazard_ratio=jnp.log(0.7) * jnp.array([0.0, 1.0])
    )
    data_key = jax.random.fold_in(jax.random.fold_in(jax.random.key(20261025), 2), 0)
    expected = generate_weibull(
        data_key, 1.38, 11.42, expected_c, log_hazard_ratio=expected_eta
    )
    assert trial.data.shape == (300,)
    assert trial.data.lower.dtype == jnp.float64
    np.testing.assert_array_equal(trial.arm, np.repeat([0.0, 1.0], 150))
    np.testing.assert_array_equal(trial.data.lower, expected.time)
    np.testing.assert_array_equal(trial.data.exact, expected.event)
    assert bool(trial.data.issues.ok)
    assert trial.censor_limit == expected_c
    root = jax.random.key(20261025)
    for stream in range(4):
        expected = jax.random.fold_in(jax.random.fold_in(root, 2), stream)
        np.testing.assert_array_equal(
            jax.random.key_data(use_case_key(stream)), jax.random.key_data(expected)
        )
    print(f"trial checks passed; c={trial.censor_limit:.15g} years")


def check_factor() -> None:
    """Spec 11.3: log density equals priors plus one full-data likelihood."""
    trial = make_trial()
    params = {"log_k": jnp.log(1.2), "log_b0": jnp.log(10.0), "gamma": -0.3}
    actual, trace = log_density(model, (trial.data, trial.arm), {}, params)
    base = survjax.Weibull.from_log(
        log_shape=params["log_k"], log_scale=params["log_b0"]
    )
    survival = survjax.ProportionalHazards(
        base, log_hazard_ratio=params["gamma"] * trial.arm
    )
    prior = (
        distributions.Normal(0.0, 5.0).log_prob(params["log_k"])
        + distributions.Normal(0.0, 5.0).log_prob(params["log_b0"])
        + distributions.Normal(0.0, 5.0).log_prob(params["gamma"])
    )
    expected = prior + survival.log_likelihood(trial.data).sum()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    factor = trace["survival"]
    assert factor["fn"].batch_shape == (300,)
    assert [
        (frame.name, frame.dim, frame.size) for frame in factor["cond_indep_stack"]
    ] == [("observations", -1, 300)]
    assert "rmst_tau" not in trace and "rmst_curve" not in trace
    grid = jnp.linspace(0.0, 5.0, 3)
    result = Predictive(
        model,
        posterior_samples={
            name: jnp.asarray([value]) for name, value in params.items()
        },
    )(jax.random.key(0), trial.data, trial.arm, compute_rmst=True, tau_grid=grid)
    assert result["rmst_tau"].shape == (1, 2)
    assert result["rmst_curve"].shape == (1, 3, 2)
    print(
        f"factor/plate/deterministic-site checks passed; density={float(actual):.15g}"
    )


def check_aggregation_and_grids() -> None:
    """02-details: paired draw arithmetic and half-year follow-up boundary."""
    # Hand arithmetic: paired deltas [2,-2,0], mean=0, sample SD=2, P>0=1/3.
    rmst = jnp.array([[1.0, 3.0], [3.0, 1.0], [5.0, 5.0]])
    draws = PosteriorDraws(
        jnp.array([1.0, 3.0, 5.0]),
        jnp.ones(3),
        rmst,
        jnp.zeros((2, 3, 2)),
        jnp.ones((2, 3, 2)),
    )
    table, probability = summarize_draws(draws)
    np.testing.assert_allclose(probability, 1 / 3, atol=1e-12, rtol=0)
    for key, value in {
        "mean": 0.0,
        "sd": 2.0,
        "lower_95": -1.9,
        "upper_95": 1.9,
    }.items():
        np.testing.assert_allclose(
            table["delta_RMST(5)"][key], value, atol=1e-12, rtol=0
        )
    np.testing.assert_allclose(table["k"]["sd"], 2.0, atol=1e-12, rtol=0)
    # p90([1,6])=5.5; p90([1,6.4])=5.86, rounded down gives the same endpoint.
    for last in (6.0, 6.4):
        data = survjax.SurvivalData(jnp.array([1.0, last]), jnp.ones(2))
        t_grid, tau_grid = time_grids(data)
        np.testing.assert_allclose(t_grid, np.linspace(0, 20, 201), atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            tau_grid, np.linspace(0, 5.5, 101), atol=1e-12, rtol=0
        )
    # A constant observed time has p90=5: equality must be accepted.
    _, boundary_grid = time_grids(
        survjax.SurvivalData(jnp.array([5.0, 5.0]), jnp.ones(2))
    )
    np.testing.assert_allclose(
        boundary_grid, np.linspace(0, 5, 101), atol=1e-12, rtol=0
    )
    try:
        time_grids(survjax.SurvivalData(jnp.array([1.0, 4.0]), jnp.ones(2)))
    except ValueError:
        pass
    else:
        raise AssertionError("tau=5 must lie within observed follow-up")
    print("aggregation and grid checks passed")


def check_posterior_routes() -> None:
    """Spec 11.3: independent exponential laws check time/draw/arm axes."""
    samples = {
        "log_k": jnp.zeros(3),
        "log_b0": jnp.log(jnp.array([2.0, 4.0, 8.0])),
        "gamma": jnp.log(jnp.array([0.5, 1.0, 2.0])),
    }
    t_grid = jnp.array([0.0, 1.0, 5.0, 10.0])
    tau_grid = jnp.array([0.0, 2.0, 5.0, 7.0])
    draws = posterior_draws(samples, t_grid, tau_grid)
    # Shape=1: r=HR_arm/b0, S(t)=exp(-rt), RMST(tau)=(1-exp(-r*tau))/r.
    rates = np.array([[0.5, 0.25], [0.25, 0.25], [0.125, 0.25]])
    expected_rmst = -np.expm1(-np.asarray(tau_grid)[:, None, None] * rates) / rates
    np.testing.assert_allclose(draws.rmst_curve, expected_rmst, atol=1e-12, rtol=0)
    np.testing.assert_allclose(draws.rmst_tau, expected_rmst[2], atol=1e-12, rtol=0)
    expected_survival = np.exp(-np.asarray(t_grid)[:, None, None] * rates)
    np.testing.assert_allclose(draws.survival, expected_survival, atol=1e-12, rtol=0)
    np.testing.assert_allclose(draws.shape, np.ones(3), atol=1e-12, rtol=0)
    np.testing.assert_allclose(draws.hazard_ratio, [0.5, 1.0, 2.0], atol=1e-12, rtol=0)
    error = check_rmst_routes(samples, make_trial(), tau_grid, draws)
    print(f"posterior route checks passed; maximum absolute error={error}")


if __name__ == "__main__":
    check_trial()
    check_aggregation_and_grids()
    check_factor()
    check_posterior_routes()
