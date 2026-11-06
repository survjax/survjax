"""PH identities from H=exp(eta)*H0, in values and parameters (spec 4.4/7.3)."""

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
)
from jax.test_util import check_grads

from survjax import Exponential, SurvivalData, Weibull
from survjax.distributions.wrappers import ProportionalHazards

METHODS = (
    "log_survival survival log_cdf cumulative_hazard log_hazard hazard log_prob".split()
)


def _exponential_pair(p):
    # H=exp(eta)*r*t, so the equivalent natural rate is r*exp(eta).
    rate, eta = p
    return (
        ProportionalHazards(baseline=Exponential(rate=rate), log_hazard_ratio=eta),
        Exponential(rate=rate * jnp.exp(eta)),
    )


def _weibull_pair(p):
    # exp(eta)*(t/b)**k = (t/[b*exp(-eta/k)])**k, including k's chain rule.
    shape, scale, eta = p
    return (
        ProportionalHazards(
            baseline=Weibull(shape=shape, scale=scale), log_hazard_ratio=eta
        ),
        Weibull(shape=shape, scale=scale * jnp.exp(-eta / shape)),
    )


FAMILIES = [
    pytest.param(_exponential_pair, [1.3, 0.2], id="exponential"),
    pytest.param(_weibull_pair, [1.4, 2.0, 0.2], id="weibull"),
]


@pytest.mark.parametrize("pair,parameters", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
def test_direct_identity_values_and_first_derivatives(
    pair, parameters, method, x64_enabled
):
    """Spec 4.4/7.3: each direct method obeys the equivalent-family chain rule."""
    p = jnp.asarray(parameters)
    times = jnp.array([0.4, 1.3, 2.1])

    def actual(q):
        return getattr(pair(q)[0], method)(times)

    def equivalent(q):
        return getattr(pair(q)[1], method)(times)

    atol, rtol = ATOL[x64_enabled], RTOL[x64_enabled]
    expected = equivalent(p)
    for value in (actual(p), jax.jit(actual)(p)):
        np.testing.assert_allclose(
            value, expected, atol=atol, rtol=rtol, equal_nan=False
        )
    draws = jnp.stack([p, p + 0.1])
    mapped = jax.jit(jax.vmap(actual))(draws)
    assert mapped.shape == (2, 3)
    np.testing.assert_allclose(
        mapped,
        jax.vmap(equivalent)(draws),
        atol=atol,
        rtol=rtol,
        equal_nan=False,
    )
    for differentiate in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            jax.jit(differentiate(actual))(p),
            jax.jit(differentiate(equivalent))(p),
            atol=atol,
            rtol=rtol,
            equal_nan=False,
        )


@pytest.mark.parametrize(
    "pair", [_exponential_pair, _weibull_pair], ids=["exponential", "weibull"]
)
@pytest.mark.parametrize("method", METHODS)
def test_grid_identity_broadcast_limits_and_field_derivatives(
    pair, method, x64_enabled
):
    """Spec 4.4–4.5/5: (S,1) baseline × (S,2) eta retains time-first grids."""
    eta = jnp.array([[-0.4, 0.2], [0.6, -0.2], [0.3, 0.8]])
    baseline = jnp.array([[0.7], [1.0], [1.7]])
    p = (
        (baseline, eta)
        if pair is _exponential_pair
        else (baseline, jnp.array([[1.4], [2.2], [3.0]]), eta)
    )
    ph, native = pair(p)
    assert ph.batch_shape == native.batch_shape == (3, 2)
    limits = jnp.array([0.0, 0.7, 2.0, jnp.inf])
    atol, rtol = ATOL[x64_enabled], RTOL[x64_enabled]
    expected = getattr(native, method + "_grid")(limits)
    grid = jax.jit(lambda d: getattr(d, method + "_grid")(limits))(ph)
    mapped = jax.jit(jax.vmap(lambda t: getattr(ph, method)(t)))(limits)
    for value in (getattr(ph, method + "_grid")(limits), grid, mapped):
        assert value.shape == (4, 3, 2)
        np.testing.assert_allclose(
            value, expected, atol=atol, rtol=rtol, equal_nan=False
        )
    times = jnp.array([0.6, 1.4])  # K=N deliberately; paired and grid differ.
    paired = getattr(ph, method)(times)
    assert paired.shape == (3, 2)
    np.testing.assert_allclose(
        paired,
        getattr(native, method)(times),
        atol=atol,
        rtol=rtol,
        equal_nan=False,
    )

    def actual(q):
        return getattr(pair(q)[0], method + "_grid")(times)

    def equivalent(q):
        return getattr(pair(q)[1], method + "_grid")(times)

    assert actual(p).shape == (2, 3, 2)
    for differentiate in (jax.jacfwd, jax.jacrev):
        left, right = (
            jax.jit(differentiate(actual))(p),
            jax.jit(differentiate(equivalent))(p),
        )
        for value, expected_field in zip(left, right, strict=True):
            np.testing.assert_allclose(
                value, expected_field, atol=atol, rtol=rtol, equal_nan=False
            )


@pytest.mark.parametrize(
    "pair", [_exponential_pair, _weibull_pair], ids=["exponential", "weibull"]
)
def test_likelihood_identity_both_forms_and_broadcast_jacobians(pair, x64_enabled):
    """Spec 4.4/6.1/7.3: exact/right contributions and every field's chain rule."""
    eta = jnp.array([[-0.4, 0.2], [0.6, -0.2], [0.3, 0.8]])
    baseline = jnp.array([[0.7], [1.0], [1.7]])
    p = (
        (baseline, eta)
        if pair is _exponential_pair
        else (baseline, jnp.array([[1.4], [2.2], [3.0]]), eta)
    )
    time, event = jnp.array([0.5, 2.0]), jnp.array([1, 0])
    data = SurvivalData(time, event)

    def actual(q):
        return pair(q)[0].log_likelihood(data)

    def equivalent(q):
        return pair(q)[1].log_likelihood(time=time, event=event)

    atol, rtol = ATOL[x64_enabled], RTOL[x64_enabled]
    ph, native = pair(p)
    expected = equivalent(p)
    for value in (
        actual(p),
        jax.jit(actual)(p),
        jax.jit(jax.vmap(actual))(p),
        jax.jit(ph.log_likelihood)(time=time, event=event),
        native.log_likelihood(data),
    ):
        assert value.shape == (3, 2)
        np.testing.assert_allclose(
            value, expected, atol=atol, rtol=rtol, equal_nan=False
        )
    for differentiate in (jax.jacfwd, jax.jacrev):
        left, right = (
            jax.jit(differentiate(actual))(p),
            jax.jit(differentiate(equivalent))(p),
        )
        for value, expected_field in zip(left, right, strict=True):
            np.testing.assert_allclose(
                value, expected_field, atol=atol, rtol=rtol, equal_nan=False
            )


@pytest.mark.parametrize("pair,parameters", FAMILIES)
def test_likelihood_hessian_matches_hand_derivation(pair, parameters, x64_enabled):
    """Spec 7.3/12.1: both mixed AD orders and order-2 fwd/rev interior checks."""
    p = jnp.asarray(parameters)
    data = SurvivalData([0.5, 1.5, 2.0], [1, 0, 1])

    def objective(q):
        return pair(q)[0].log_likelihood(data).mean()

    # Two exact rows, three rows total, sum(t)=4. Exponential mean log L is
    # (2/3)*(log(r)+eta) - (4/3)*r*exp(eta); differentiate twice directly.
    if pair is _exponential_pair:
        rate, eta = parameters
        coefficient = 4 * np.exp(eta) / 3
        expected = np.array(
            [[-2 / (3 * rate**2), -coefficient], [-coefficient, -coefficient * rate]]
        )
    else:
        # u=log(t/b), H=exp(eta)*(t/b)**k, c=mean(event)=2/3.
        # L=c*(log(k)-log(b)+eta)+(k-1)*mean(event*u)-mean(H).
        # W=mean(H), A=mean(H*u), B=mean(H*u**2); differentiating gives:
        k, b, eta = parameters
        u = np.log(np.array([0.5, 1.5, 2.0]) / b)
        cumulative = np.exp(eta + k * u)
        w, a, square = (
            cumulative.mean(),
            np.mean(cumulative * u),
            np.mean(cumulative * u * u),
        )
        c = 2 / 3
        cross = (k * a + w - c) / b
        expected = np.array(
            [
                [-c / k**2 - square, cross, -a],
                [cross, k * (c - (k + 1) * w) / b**2, k * w / b],
                [-a, k * w / b, -w],
            ]
        )
    # Entries are O(1); AD/analytic comparisons incur rounding, not finite-difference error.
    for differentiate in (jax.hessian, lambda f: jax.jacrev(jax.jacfwd(f))):
        np.testing.assert_allclose(
            jax.jit(differentiate(objective))(p),
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            equal_nan=False,
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


def test_weibull_zero_grid_survival_has_zero_shape_derivative():
    """Spec 4.5/7.3: S(0)=1, so its shape derivative is exactly zero at k=1 too."""

    def value(shape):
        ph = ProportionalHazards(
            baseline=Weibull(shape=shape, scale=2), log_hazard_ratio=0.4
        )
        return ph.survival_grid(jnp.array([0.0, 1.0]))[0]

    shapes = jnp.array([0.7, 1.0, 1.7])
    np.testing.assert_array_equal(jax.jit(jax.vmap(value))(shapes), 1)
    for differentiate in (jax.jacfwd, jax.jacrev):
        np.testing.assert_array_equal(
            jax.jit(jax.vmap(differentiate(value)))(shapes), 0
        )
