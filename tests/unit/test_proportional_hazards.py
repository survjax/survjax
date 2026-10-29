"""PH construction and analytical direct-method composition, spec 4.4."""

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import (
    ATOL,
    DERIVATIVE_ATOL,
    DERIVATIVE_RTOL,
    LIKELIHOOD_GRAD_EPS,
    RTOL,
    TAIL_RTOL,
)
from jax.test_util import check_grads

from survjax import Exponential, SurvivalData
from survjax.distributions.wrappers import ProportionalHazards

METHODS = (
    "log_survival survival log_cdf cumulative_hazard log_hazard hazard log_prob".split()
)


def test_broadcast_pytree_indexing_and_dtypes(x64_enabled):
    """Spec 3.3/4.1/4.4: nested leaves broadcast without dtype normalization."""
    dtype = jnp.float64 if x64_enabled else jnp.float32
    baseline = Exponential(rate=jnp.array([[1], [2]], dtype=jnp.float32))
    dist = ProportionalHazards(
        baseline=baseline, log_hazard_ratio=jnp.zeros((2, 2), dtype=dtype)
    )
    assert dist.batch_shape == dist.baseline.batch_shape == (2, 2)
    assert dist.baseline.rate.dtype == jnp.float32
    assert dist.log_hazard_ratio.dtype == dist.reference_time().dtype == dtype
    assert baseline.batch_shape == (2, 1)
    leaves, structure = jax.tree.flatten(dist)
    assert len(leaves) == 2
    rebuilt = jax.tree.unflatten(structure, leaves)
    np.testing.assert_array_equal(rebuilt[1].baseline.rate, [2, 2])
    np.testing.assert_array_equal(
        jax.tree.map(lambda x: x[0], dist).baseline.rate, [1, 1]
    )
    with pytest.raises(FrozenInstanceError):
        dist.log_hazard_ratio = 0


def test_traced_diagnostics_include_predictor_and_prefixed_baseline():
    """Spec 4.7: real predictor and nested positive constraint keep row masks."""

    def issues(rate, eta):
        return ProportionalHazards(
            baseline=Exponential(rate=rate), log_hazard_ratio=eta
        ).parameter_issues()

    rate, eta = (
        jnp.array([1.0, -1.0, 1.0, 1.0]),
        jnp.array([0.0, 0.0, jnp.nan, jnp.inf]),
    )
    for result in (
        issues(rate, eta),
        jax.jit(issues)(rate, eta),
        jax.vmap(issues)(rate, eta),
    ):
        assert set(result.fields) == {"log_hazard_ratio", "baseline.rate"}
        np.testing.assert_array_equal(result.fields["baseline.rate"], [0, 1, 0, 0])
        np.testing.assert_array_equal(result.fields["log_hazard_ratio"], [0, 0, 1, 1])
        np.testing.assert_array_equal(result.invalid_entries, [0, 1, 1, 1])
    dist = ProportionalHazards(baseline=Exponential(rate=rate), log_hazard_ratio=eta)
    assert dist[1].parameter_issues().invalid_entries


def test_static_misuse_raises():
    """Spec 4.1–4.2: registration owns keyword/type/shape validation."""
    with pytest.raises(TypeError):
        ProportionalHazards(Exponential(rate=1), log_hazard_ratio=0)
    for baseline, eta in (
        (1, 0),
        (Exponential(rate=[1, 2]), [0, 0, 0]),
        (Exponential(rate=1), True),
    ):
        with pytest.raises(TypeError):
            ProportionalHazards(baseline=baseline, log_hazard_ratio=eta)


@pytest.mark.parametrize("name", METHODS)
def test_direct_and_grid_exponential_formulas(name, x64_enabled):
    """Spec 4.4/5: r=2, exp(eta)=[1/2,2] gives hazards [1,4]."""
    dist = ProportionalHazards(
        baseline=Exponential(rate=2), log_hazard_ratio=jnp.log(jnp.array([0.5, 2.0]))
    )
    time = jnp.array([0.0, 1.0, jnp.inf])
    hazard = np.array([1.0, 4.0])
    cumulative = np.array([[0.0, 0.0], [1.0, 4.0], [np.inf, np.inf]])
    expected = {
        "log_survival": -cumulative,
        "survival": np.exp(-cumulative),
        "log_cdf": np.array(
            [[-np.inf, -np.inf], np.log(-np.expm1(-hazard)), [0.0, 0.0]]
        ),
        "cumulative_hazard": cumulative,
        "log_hazard": np.broadcast_to(np.log(hazard), (3, 2)),
        "hazard": np.broadcast_to(hazard, (3, 2)),
        "log_prob": np.log(hazard) - cumulative,
    }[name]
    grid = getattr(dist, name + "_grid")
    for actual in (
        grid(time),
        jax.jit(grid)(time),
        jax.vmap(getattr(dist, name))(time),
    ):
        assert actual.shape == (3, 2)
        np.testing.assert_allclose(
            actual, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


def test_likelihood_and_second_order_interior_derivatives(x64_enabled):
    """Spec 6–7/12.1: log L=[log(r)+eta-r*exp(eta), -2*r*exp(eta)]."""
    data = SurvivalData([1.0, 2.0], [1, 0])

    def objective(p):
        dist = ProportionalHazards(
            baseline=Exponential(rate=p[0]), log_hazard_ratio=p[1]
        )
        return dist.log_likelihood(data).mean()

    p = jnp.array([2.0, 0.0])
    dist = ProportionalHazards(baseline=Exponential(rate=p[0]), log_hazard_ratio=p[1])
    with jax.debug_nans(True):
        for actual in (
            dist.log_likelihood(data),
            jax.jit(dist.log_likelihood)(time=data.lower, event=data.exact),
        ):
            np.testing.assert_allclose(
                actual,
                [np.log(2) - 2, -4],
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )
        check_grads(
            objective,
            (p,),
            order=2,
            modes=("fwd", "rev"),
            eps=LIKELIHOOD_GRAD_EPS[x64_enabled],
            atol=DERIVATIVE_ATOL[x64_enabled],
            rtol=DERIVATIVE_RTOL[x64_enabled],
        )


def test_extreme_predictors_do_not_form_overflow_times_zero(x64_enabled):
    """Spec 7.2: exp(eta)*H0 is representable even if exp(eta) is not."""
    eta = 720.0 if x64_enabled else 90.0
    rate = 1e-300 if x64_enabled else 1e-30
    dist = ProportionalHazards(baseline=Exponential(rate=rate), log_hazard_ratio=eta)
    # Analytical log H = eta+log(rate), evaluated in host float64.
    expected = np.exp(eta + np.log(rate))
    np.testing.assert_allclose(
        jax.jit(dist.cumulative_hazard)(1.0),
        expected,
        atol=0,
        rtol=TAIL_RTOL[x64_enabled],
    )
    for predictor in (eta, -eta):

        def fn(e):
            return ProportionalHazards(
                baseline=Exponential(rate=1), log_hazard_ratio=e
            ).survival_grid(jnp.array([0.0, jnp.inf]))

        np.testing.assert_array_equal(jax.jit(fn)(predictor), [1, 0])
        np.testing.assert_array_equal(jax.jit(jax.jacrev(fn))(predictor), [0, 0])
