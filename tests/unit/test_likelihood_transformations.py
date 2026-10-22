"""Finite-difference checks of public likelihood derivatives (spec 7.3/12.1)."""

import jax
import jax.numpy as jnp
import pytest
from impl_tolerances import DERIVATIVE_ATOL, DERIVATIVE_RTOL, LIKELIHOOD_GRAD_EPS
from jax.test_util import check_grads

import survjax


@pytest.fixture(params=["function", "method"])
def evaluate(request):
    if request.param == "method":
        return lambda dist, *args, **kwargs: dist.log_likelihood(*args, **kwargs)
    return survjax.log_likelihood


@pytest.mark.parametrize("form", ["raw", "prepared"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_exponential_likelihood_derivatives_match_finite_differences(
    evaluate, form, compiled, x64_enabled
):
    """Spec 7.3/12.1: check the rate to order two in forward and reverse modes."""
    time = jnp.array([0.5, 1.0, 1.5])
    event = jnp.array([True, False, True])

    def mean_log_likelihood(rate, t, e):
        dist = survjax.Exponential(rate=rate)
        if form == "raw":
            contributions = evaluate(dist, time=t, event=e)
        else:
            contributions = evaluate(dist, survjax.SurvivalData(t, e))
        return contributions.mean()

    if compiled:
        mean_log_likelihood = jax.jit(mean_log_likelihood)
    # Expected derivatives come from central differences of values/AD derivatives.
    # Only rate is perturbed; observation-time derivatives are not guaranteed.
    check_grads(
        lambda rate: mean_log_likelihood(rate, time, event),
        (jnp.asarray(1.5),),
        order=2,
        modes=("fwd", "rev"),
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
        eps=LIKELIHOOD_GRAD_EPS[x64_enabled],
    )


@pytest.mark.parametrize("form", ["raw", "prepared"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_weibull_likelihood_derivatives_match_finite_differences(
    evaluate, form, compiled, x64_enabled
):
    """Spec 7.3/12.1: check shape and scale jointly to order two in both modes."""
    time = jnp.array([0.5, 1.0, 1.5])
    event = jnp.array([True, False, True])

    def mean_log_likelihood(parameters, t, e):
        dist = survjax.Weibull(shape=parameters[0], scale=parameters[1])
        if form == "raw":
            contributions = evaluate(dist, time=t, event=e)
        else:
            contributions = evaluate(dist, survjax.SurvivalData(t, e))
        return contributions.mean()

    if compiled:
        mean_log_likelihood = jax.jit(mean_log_likelihood)
    # Central differences perturb the joint parameter vector; times and event
    # indicators remain fixed.
    check_grads(
        lambda parameters: mean_log_likelihood(parameters, time, event),
        (jnp.array([1.5, 2.0]),),
        order=2,
        modes=("fwd", "rev"),
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
        eps=LIKELIHOOD_GRAD_EPS[x64_enabled],
    )
