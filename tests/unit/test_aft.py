"""AFT construction and direct identities, spec 4.4–4.5 and 7.3."""

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
)
from jax.test_util import check_grads

from survjax import AcceleratedFailureTime, Exponential, SurvivalData, Weibull

METHODS = (
    "log_survival survival log_cdf cumulative_hazard log_prob log_hazard hazard".split()
)


def test_construction_broadcast_dtype_and_pytree(x64_enabled):
    """Spec 3.3/4.1–4.4: nested leaves keep their dtype while broadcasting."""
    dtype = jnp.float64 if x64_enabled else jnp.float32
    base = Exponential(rate=jnp.array([[1], [2]], dtype=jnp.float32))
    dist = AcceleratedFailureTime(
        baseline=base, log_time_ratio=jnp.zeros((2, 2), dtype=dtype)
    )
    assert dist.batch_shape == dist.baseline.batch_shape == (2, 2)
    assert base.batch_shape == (2, 1)
    assert dist.baseline.rate.dtype == jnp.float32
    assert dist.reference_time().dtype == dtype
    leaves, structure = jax.tree.flatten(dist)
    assert len(leaves) == 2
    rebuilt = jax.tree.unflatten(structure, leaves)
    np.testing.assert_array_equal(rebuilt[1].baseline.rate, [2, 2])
    np.testing.assert_array_equal(
        jax.tree.map(lambda x: x[0], dist).baseline.rate, [1, 1]
    )
    with pytest.raises(FrozenInstanceError):
        dist.log_time_ratio = 1


def test_constructor_misuse():
    """Spec 4.2/4.4: keyword-only construction and registered baseline required."""
    with pytest.raises(TypeError):
        AcceleratedFailureTime(Exponential(rate=1), 0)
    for baseline, eta in (
        (1, 0),
        (Exponential(rate=[1, 2]), [0, 0, 0]),
        (Exponential(rate=1), True),
    ):
        with pytest.raises(TypeError):
            AcceleratedFailureTime(baseline=baseline, log_time_ratio=eta)


def test_nested_diagnostics():
    """Spec 4.7: real predictor and prefixed baseline diagnostics compose."""

    def issues(rate, eta):
        return AcceleratedFailureTime(
            baseline=Exponential(rate=rate), log_time_ratio=eta
        ).parameter_issues()

    rate = jnp.array([1.0, -1.0, 1.0, 1.0])
    eta = jnp.array([0.0, 0.0, jnp.nan, jnp.inf])
    for fn in (issues, jax.jit(issues), jax.vmap(issues)):
        result = fn(rate, eta)
        assert set(result.fields) == {"log_time_ratio", "baseline.rate"}
        np.testing.assert_array_equal(result.fields["baseline.rate"], [0, 1, 0, 0])
        np.testing.assert_array_equal(result.fields["log_time_ratio"], [0, 0, 1, 1])
        np.testing.assert_array_equal(result.invalid_entries, [0, 1, 1, 1])


@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("shape", [0.5, 1.0, 2.0])
def test_weibull_equivalence_and_time_policy(name, shape, x64_enabled):
    """Spec 4.4–4.5: AFT Weibull(k,b,eta) equals Weibull(k,b*exp(eta))."""
    eta = jnp.log(jnp.array([0.5, 2.0]))
    dist = AcceleratedFailureTime(
        baseline=Weibull(shape=shape, scale=2), log_time_ratio=eta
    )
    equivalent = Weibull(shape=shape, scale=jnp.array([1.0, 4.0]))
    time = jnp.array([-jnp.inf, -1.0, 0.0, 0.5, 2.0, jnp.inf, jnp.nan])
    expected = getattr(equivalent, name)(time[:, None])
    for actual in (
        getattr(dist, name)(time[:, None]),
        jax.jit(getattr(dist, name + "_grid"))(time),
        jax.vmap(getattr(dist, name))(time),
    ):
        np.testing.assert_allclose(
            actual, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


@pytest.mark.parametrize("name", METHODS)
def test_exponential_identity(name, x64_enabled):
    """Spec 4.4: r=2, exp(eta)=2 gives an exponential rate of one."""
    dist = AcceleratedFailureTime(
        baseline=Exponential(rate=2), log_time_ratio=jnp.log(2.0)
    )
    time = jnp.array([0.0, 0.5, 2.0, jnp.inf])
    np.testing.assert_allclose(
        jax.jit(getattr(dist, name))(time),
        getattr(Exponential(rate=1), name)(time),
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )


def test_interior_parameter_derivatives(x64_enabled):
    """Spec 7.3: log f = log k-log b-eta+(k-1)*log u-u**k, u=t/b/exp(eta)."""

    def objective(p):
        return AcceleratedFailureTime(
            baseline=Weibull(shape=p[0], scale=p[1]), log_time_ratio=p[2]
        ).log_prob(1.3)

    def identity(p):
        k, b, eta = p
        log_u = jnp.log(1.3) - jnp.log(b) - eta
        return jnp.log(k) - jnp.log(b) - eta + (k - 1) * log_u - jnp.exp(k * log_u)

    p = jnp.array([1.4, 2.0, 0.3])
    for transform in (lambda f: f, jax.jacfwd, jax.jacrev, jax.hessian):
        np.testing.assert_allclose(
            jax.jit(transform(objective))(p),
            transform(identity)(p),
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


def test_invalid_row_isolation(x64_enabled):
    """Spec 7.1–7.3: invalid rows leave valid likelihood derivatives intact."""
    data = SurvivalData([1.0, 2.0], [1, 0])

    def likelihood(eta):
        return AcceleratedFailureTime(
            baseline=Exponential(rate=jnp.array([2.0, -1.0])), log_time_ratio=eta
        ).log_likelihood(data)

    eta = jnp.array([0.0, 0.0])
    result = jax.jit(likelihood)(eta)
    assert np.isnan(result[1])
    np.testing.assert_allclose(
        result[0], np.log(2) - 2, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    # log L0 = log(2)-eta0-2*exp(-eta0): gradient 1, Hessian -2 at eta0=0.
    for derivative in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            jax.jit(derivative(likelihood))(eta)[0],
            [1, 0],
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )
    np.testing.assert_allclose(
        jax.jit(jax.hessian(lambda e: likelihood(e)[0]))(eta),
        [[-2, 0], [0, 0]],
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )


def test_rmst_remains_explicitly_unsupported():
    """Staged AFT plan: generic summary integration belongs to another worktree."""
    with pytest.raises(NotImplementedError):
        AcceleratedFailureTime(baseline=Exponential(rate=1), log_time_ratio=0).rmst(1)


@pytest.mark.parametrize("eta", [-1000.0, 1000.0])
@pytest.mark.parametrize("name", ["log_survival", "log_prob", "log_hazard"])
def test_unrepresentable_rescaling_is_numerical_failure(eta, name):
    """Spec 7.1: lost positive times must not turn into baseline boundaries."""
    dist = AcceleratedFailureTime(
        baseline=Weibull(shape=0.5, scale=1), log_time_ratio=eta
    )
    # log u = -eta exceeds either end of both dtypes' exponent ranges.
    assert jnp.isnan(jax.jit(getattr(dist, name))(1.0))


@pytest.mark.parametrize("time", [-1.0, 0.0, jnp.inf])
def test_policy_boundaries_have_finite_parameter_derivatives(time):
    """Spec 4.5/7.2: inactive log/exp operands retain finite parameter tangents."""

    def fn(p):
        dist = AcceleratedFailureTime(
            baseline=Weibull(shape=p[0], scale=p[1]), log_time_ratio=p[2]
        )
        return dist.survival(time)

    p = jnp.array([1.4, 2.0, 0.3])
    # S before/at zero is identically one; S(infinity) is identically zero.
    for derivative in (jax.jacfwd, jax.jacrev, jax.hessian):
        np.testing.assert_array_equal(
            jax.jit(derivative(fn))(p), jnp.zeros_like(derivative(fn)(p))
        )


def test_rescaling_zero_and_infinity_retains_zero_predictor_tangents():
    """Spec 7.2: the rescaling boundary substitution precedes log/exp."""

    # Exercise the numerical helper directly: outer policy and built-in
    # baselines independently mask boundaries, which can hide this invariant.
    def fn(eta):
        dist = AcceleratedFailureTime(baseline=Exponential(rate=1), log_time_ratio=eta)
        return dist._rescale_time(jnp.array([0.0, jnp.inf]))

    for derivative in (jax.jacfwd, jax.jacrev):
        np.testing.assert_array_equal(jax.jit(derivative(fn))(0.3), [0.0, 0.0])
