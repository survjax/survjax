"""Synthetic AFT Weibull comparisons to committed R outputs (spec 11.1/12.1)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

import survjax


@pytest.fixture
def reference(r_fixture):
    return r_fixture("bc_like")


@pytest.fixture(params=("fitted", "unit_shape", "alternative"))
def case(request, reference):
    """All expected numerical values are stored R results."""
    return reference["cases"][request.param]


@pytest.fixture
def observations(reference, x64_enabled):
    dtype = jnp.float64 if x64_enabled else jnp.float32
    return (
        jnp.asarray(reference["data"]["time"], dtype=dtype),
        jnp.asarray(reference["data"]["event"], dtype=bool),
    )


@pytest.fixture(params=("natural", "from_log"))
def distributions(request, reference, case, observations):
    """Spec 4.6: log b_g = log b_Good + beta_g; shape is common."""
    dtype = observations[0].dtype
    shape = jnp.asarray(case["shape"], dtype=dtype)
    baseline = jnp.asarray(case["scale_Good"], dtype=dtype)
    beta = jnp.asarray([0, case["beta_Medium"], case["beta_Poor"]], dtype=dtype)
    groups = reference["grid"]["groups"]
    row_group = jnp.asarray(
        [groups.index(group) for group in reference["data"]["group"]]
    )
    if request.param == "natural":
        group_scale = baseline * jnp.exp(beta)
        rows = survjax.Weibull(shape=shape, scale=group_scale[row_group])
        grid = survjax.Weibull(shape=shape, scale=group_scale)
    else:
        log_scale = jnp.log(baseline) + beta
        rows = survjax.Weibull.from_log(
            log_shape=jnp.log(shape), log_scale=log_scale[row_group]
        )
        grid = survjax.Weibull.from_log(log_shape=jnp.log(shape), log_scale=log_scale)
    assert rows.shape.shape == (686,)
    assert rows.scale.shape == (686,)
    assert grid.scale.shape == (3,)
    return rows, grid


def test_valid_aft_inputs_have_clear_diagnostics(distributions, observations):
    """Spec 4.7/6.2: positive finite AFT vectors and observations are valid."""
    data = survjax.SurvivalData(*observations)
    assert data.issues.ok
    assert not data.issues.invalid_rows.any()
    for dist in distributions:
        assert dist.parameter_issues().ok
        assert not dist.parameter_issues().invalid_entries.any()


@pytest.mark.parametrize("source", ("flexsurv", "survreg"))
@pytest.mark.parametrize("entry", ("function", "method"))
@pytest.mark.parametrize("form", ("prepared", "raw"))
def test_likelihood_rows_match_r(
    distributions, observations, case, source, entry, form, x64_enabled
):
    """Spec 6.1/12.1: every ordered contribution matches R in each form."""
    dist, _ = distributions
    time, event = observations
    if form == "prepared":
        args = (survjax.SurvivalData(time, event),)

        def evaluate(distribution, data):
            if entry == "function":
                return survjax.log_likelihood(distribution, data)
            return distribution.log_likelihood(data)

        mapped = jax.vmap(evaluate, in_axes=(0, 0))
    else:
        args = (time, event)

        def evaluate(distribution, time, event):
            if entry == "function":
                return survjax.log_likelihood(distribution, time=time, event=event)
            return distribution.log_likelihood(time=time, event=event)

        mapped = jax.vmap(evaluate, in_axes=(0, 0, 0))

    expected = case[source]["log_likelihood"]
    # The constructor broadcasts the common shape alongside the row scales.
    for mode, run in (
        ("eager", evaluate),
        ("jit", jax.jit(evaluate)),
        ("vmap", mapped),
    ):
        actual = run(dist, *args)
        assert actual.shape == (686,)
        assert actual.dtype == time.dtype
        np.testing.assert_allclose(
            actual,
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            err_msg=mode,
        )


@pytest.mark.parametrize("source", ("flexsurv", "survreg"))
@pytest.mark.parametrize("quantity", ("survival", "log_survival"))
def test_observed_time_predictions_match_r(
    distributions, observations, case, source, quantity, x64_enabled
):
    """Spec 4.6/12.1: pair each row's scale with its own observed time."""
    dist, _ = distributions

    def evaluate(distribution, time):
        return getattr(distribution, quantity)(time)

    expected = case[source][quantity]
    for mode, run in (
        ("eager", evaluate),
        ("jit", jax.jit(evaluate)),
        ("vmap", jax.vmap(evaluate, in_axes=(0, 0))),
    ):
        actual = run(dist, observations[0])
        assert actual.shape == (686,)
        assert actual.dtype == observations[0].dtype
        np.testing.assert_allclose(
            actual,
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            err_msg=mode,
        )


@pytest.mark.parametrize("source", ("flexsurv", "survreg"))
def test_time_first_survival_grid_matches_r(
    distributions, observations, reference, case, source, x64_enabled
):
    """Spec 5/11.1: stored R times cross all Good/Medium/Poor scales."""
    _, dist = distributions
    # Rebuilding R seq with linspace/arange can shift a time by one ULP.
    time = jnp.asarray(reference["grid"]["time"], dtype=observations[0].dtype)

    def evaluate(distribution, times):
        return distribution.survival_grid(times)

    def pointwise(distribution, times):
        return jax.vmap(distribution.survival)(times)

    expected = case[source]["survival_grid"]
    for mode, run in (
        ("eager", evaluate),
        ("jit", jax.jit(evaluate)),
        ("vmap", pointwise),
    ):
        actual = run(dist, time)
        assert actual.shape == (141, 3)
        assert actual.dtype == time.dtype
        np.testing.assert_allclose(
            actual,
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            err_msg=mode,
        )


@pytest.mark.parametrize("source", ("flexsurv", "survreg"))
def test_total_likelihood_matches_r(
    distributions, observations, case, source, x64_enabled
):
    """Spec 12.1: separately recorded R totals supplement per-row evidence."""
    dist, _ = distributions
    actual = jnp.sum(dist.log_likelihood(survjax.SurvivalData(*observations)))
    assert actual.shape == ()
    assert actual.dtype == observations[0].dtype
    np.testing.assert_allclose(
        actual,
        case[source]["total_log_likelihood"],
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
