"""Exponential values against committed fixed-parameter R outputs (spec 12.1)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

import survjax


@pytest.fixture(params=("slow", "generating", "fast"))
def exponential_case(request, r_fixture):
    """Expected values come from flexsurv/survreg, never from survjax."""
    return r_fixture("exponential")["cases"][request.param]


@pytest.fixture
def observations(r_fixture, x64_enabled):
    data = r_fixture("exponential")["data"]
    dtype = jnp.float64 if x64_enabled else jnp.float32
    return jnp.asarray(data["time"], dtype=dtype), jnp.asarray(
        data["event"], dtype=bool
    )


@pytest.fixture
def distribution(exponential_case, observations):
    time, _ = observations
    return survjax.Exponential(
        rate=jnp.asarray(exponential_case["rate"], dtype=time.dtype)
    )


def test_valid_r_observations_and_rates_have_clear_diagnostics(
    distribution, observations
):
    """Spec 4.7/6.2: these positive finite exact/right inputs are valid."""
    data = survjax.SurvivalData(*observations)
    assert data.issues.ok
    assert not data.issues.invalid_rows.any()
    assert distribution.parameter_issues().ok
    assert not distribution.parameter_issues().invalid_entries.any()


@pytest.mark.parametrize("source", ("flexsurv", "survreg"))
def test_function_prepared_rows_match_r(
    distribution, observations, exponential_case, source, x64_enabled
):
    """Spec 6.1/12.1: prepared-data contributions match every R row."""
    data = survjax.SurvivalData(*observations)
    evaluate = survjax.log_likelihood
    expected = exponential_case[source]["log_likelihood"]
    for mode, run in (
        ("eager", evaluate),
        ("jit", jax.jit(evaluate)),
        ("vmap", jax.vmap(evaluate, in_axes=(None, 0))),
    ):
        actual = run(distribution, data)
        assert actual.shape == expected.shape
        assert actual.dtype == observations[0].dtype
        np.testing.assert_allclose(
            actual,
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            err_msg=mode,
        )


@pytest.mark.parametrize("source", ("flexsurv", "survreg"))
def test_method_prepared_rows_match_r(
    distribution, observations, exponential_case, source, x64_enabled
):
    """Spec 6.1/12.1: the distribution method matches prepared-data R rows."""

    def evaluate(dist, data):
        return dist.log_likelihood(data)

    data = survjax.SurvivalData(*observations)
    expected = exponential_case[source]["log_likelihood"]
    for mode, run in (
        ("eager", evaluate),
        ("jit", jax.jit(evaluate)),
        ("vmap", jax.vmap(evaluate, in_axes=(None, 0))),
    ):
        actual = run(distribution, data)
        assert actual.shape == expected.shape
        assert actual.dtype == observations[0].dtype
        np.testing.assert_allclose(
            actual,
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            err_msg=mode,
        )


@pytest.mark.parametrize("source", ("flexsurv", "survreg"))
def test_function_raw_rows_match_r(
    distribution, observations, exponential_case, source, x64_enabled
):
    """Spec 6.1/12.1: keyword raw observations match every R contribution."""

    def evaluate(dist, time, event):
        return survjax.log_likelihood(dist, time=time, event=event)

    expected = exponential_case[source]["log_likelihood"]
    for mode, run in (
        ("eager", evaluate),
        ("jit", jax.jit(evaluate)),
        ("vmap", jax.vmap(evaluate, in_axes=(None, 0, 0))),
    ):
        actual = run(distribution, *observations)
        assert actual.shape == expected.shape
        assert actual.dtype == observations[0].dtype
        np.testing.assert_allclose(
            actual,
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            err_msg=mode,
        )


@pytest.mark.parametrize("source", ("flexsurv", "survreg"))
def test_method_raw_rows_match_r(
    distribution, observations, exponential_case, source, x64_enabled
):
    """Spec 6.1/12.1: method keyword raw observations match every R row."""

    def evaluate(dist, time, event):
        return dist.log_likelihood(time=time, event=event)

    expected = exponential_case[source]["log_likelihood"]
    for mode, run in (
        ("eager", evaluate),
        ("jit", jax.jit(evaluate)),
        ("vmap", jax.vmap(evaluate, in_axes=(None, 0, 0))),
    ):
        actual = run(distribution, *observations)
        assert actual.shape == expected.shape
        assert actual.dtype == observations[0].dtype
        np.testing.assert_allclose(
            actual,
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            err_msg=mode,
        )


@pytest.mark.parametrize("source", ("flexsurv", "survreg"))
def test_survival_matches_r(
    distribution, observations, exponential_case, source, x64_enabled
):
    """Spec 4.6/12.1: observed-time survival matches both R paths."""

    def evaluate(dist, time):
        return dist.survival(time)

    expected = exponential_case[source]["survival"]
    for mode, run in (
        ("eager", evaluate),
        ("jit", jax.jit(evaluate)),
        ("vmap", jax.vmap(evaluate, in_axes=(None, 0))),
    ):
        actual = run(distribution, observations[0])
        assert actual.shape == expected.shape
        assert actual.dtype == observations[0].dtype
        np.testing.assert_allclose(
            actual,
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            err_msg=mode,
        )


@pytest.mark.parametrize("source", ("flexsurv", "survreg"))
def test_log_survival_matches_r(
    distribution, observations, exponential_case, source, x64_enabled
):
    """Spec 4.6/12.1: observed-time log survival matches both R paths."""

    def evaluate(dist, time):
        return dist.log_survival(time)

    expected = exponential_case[source]["log_survival"]
    for mode, run in (
        ("eager", evaluate),
        ("jit", jax.jit(evaluate)),
        ("vmap", jax.vmap(evaluate, in_axes=(None, 0))),
    ):
        actual = run(distribution, observations[0])
        assert actual.shape == expected.shape
        assert actual.dtype == observations[0].dtype
        np.testing.assert_allclose(
            actual,
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            err_msg=mode,
        )


@pytest.mark.parametrize("source", ("flexsurv", "survreg"))
def test_summed_likelihood_matches_r_total(
    distribution, observations, exponential_case, source, x64_enabled
):
    """Spec 12.1: the separate R package total supplements per-row checks."""
    data = survjax.SurvivalData(*observations)
    actual = jnp.sum(survjax.log_likelihood(distribution, data))
    assert actual.shape == ()
    assert actual.dtype == observations[0].dtype
    np.testing.assert_allclose(
        actual,
        exponential_case[source]["total_log_likelihood"],
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
