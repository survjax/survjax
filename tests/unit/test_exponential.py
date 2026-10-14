"""Exponential regression cases derived from S=exp(-r*t), spec 4.1/4.5/7.3."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

from survjax import Exponential

METHODS = (
    "log_survival",
    "survival",
    "cumulative_hazard",
    "log_cdf",
    "log_prob",
    "log_hazard",
    "hazard",
)


@pytest.mark.parametrize("name", ["log_hazard", "hazard"])
def test_raw_hazard_methods_broadcast_scalar_and_multiaxis_times(name, x64_enabled):
    """Spec 4.1/5: raw formulas broadcast without the time-policy wrapper."""
    dist = Exponential(rate=[2.0, 3.0])
    # Analytically h(t)=r and log h(t)=log(r), at every positive time.
    expected = np.log([2.0, 3.0]) if name == "log_hazard" else [2.0, 3.0]
    raw = getattr(Exponential, name)._survjax_raw

    def raw_value(time):
        return raw(dist, time)

    for time, shape in ((1.0, (2,)), (jnp.ones((2, 1)), (2, 2))):
        for result in (
            raw_value(time),
            jax.jit(raw_value)(time),
            getattr(dist, name)(time),
        ):
            assert result.shape == shape
            np.testing.assert_allclose(
                result,
                np.broadcast_to(expected, shape),
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(raw_value))(jnp.array([1.0, 2.0])),
        np.broadcast_to(expected, (2, 2)),
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )


def test_values_broadcast_and_transform(x64_enabled):
    """Spec 4.1/5/12.1: algebra at rate 2; multiaxis values under jit/vmap."""
    expected = (
        -2.0,
        np.exp(-2),
        2.0,
        np.log(-np.expm1(-2)),
        np.log(2) - 2,
        np.log(2),
        2.0,
    )
    dist = Exponential(rate=[[2.0, 2.0]])
    for name, value in zip(METHODS, expected, strict=True):

        def fn(t, name=name):
            return getattr(dist, name)(t)

        for out in (fn(1.0), jax.jit(fn)(jnp.ones((2, 1))), jax.vmap(fn)(jnp.ones(2))):
            np.testing.assert_allclose(
                out, value, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
            )
    np.testing.assert_allclose(
        Exponential(rate=[2.0, jnp.nan]).survival(1),
        [np.exp(-2), np.nan],
        rtol=RTOL[x64_enabled],
    )
    assert Exponential(rate=-2.0).log_survival(1.0) == 2
    assert np.isnan(Exponential(rate=-2.0).log_prob(1.0))


def test_tails(x64_enabled):
    """Spec 4.2/12.1: tiny H retains relative precision; large-H hazard is analytic."""
    dist = Exponential(rate=2.0)
    np.testing.assert_allclose(
        dist.cumulative_hazard(1e-30), 2e-30, atol=0, rtol=RTOL[x64_enabled]
    )
    np.testing.assert_allclose(
        dist.log_cdf(1e-30),
        np.log(2e-30),
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    np.testing.assert_allclose(
        dist.log_prob(1e7), np.log(2) - 2e7, rtol=RTOL[x64_enabled]
    )
    np.testing.assert_allclose(dist.log_hazard(1e7), np.log(2), rtol=RTOL[x64_enabled])


def test_log_cdf_retains_tail_when_cumulative_hazard_underflows(x64_enabled):
    """Spec 4.2/7.1: log F = log(rate)+log(t) when rate*t underflows."""
    time = jnp.asarray(1e-30)
    actual = Exponential(rate=1e-30).log_cdf(time)
    expected = np.log(1e-30) + np.log(1e-30)
    np.testing.assert_allclose(
        actual, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    if not x64_enabled:
        assert Exponential(rate=1e-30).cumulative_hazard(time) == 0


def test_log_cdf_switch_is_continuous_with_finite_gradients(x64_enabled):
    """Spec 4.3/7.2: both sides of the eps switch retain finite rate gradients."""
    dtype = jnp.float64 if x64_enabled else jnp.float32
    eps = np.finfo(np.float64 if x64_enabled else np.float32).eps
    log_eps = jnp.log(jnp.asarray(eps, dtype=dtype))
    offset = 4 * np.sqrt(eps)
    times = jnp.asarray(
        np.exp(np.asarray([log_eps - offset, log_eps + offset])), dtype=dtype
    )
    log_h = jnp.log(times)  # rate=1 gives log H=log(t).
    assert bool(log_h[0] < log_eps) and bool(log_h[1] > log_eps)

    def log_cdf(rate, time):
        return Exponential(rate=rate).log_cdf(time)

    with jax.debug_nans(True):
        values = jax.jit(lambda t: Exponential(rate=1.0).log_cdf(t))(times)
        gradients = jax.jit(jax.vmap(jax.grad(log_cdf, argnums=0), in_axes=(None, 0)))(
            1.0, times
        )
    np.testing.assert_allclose(values, log_h, atol=eps, rtol=0)
    assert np.isfinite(np.asarray(gradients)).all()


def test_construction_and_reference(x64_enabled):
    """Spec 3.3/4.1/4.2: natural leaves, transforms, dtype and extreme references."""
    dtype = jnp.float64 if x64_enabled else jnp.float32
    assert Exponential(rate=1).rate.dtype == dtype
    limits = np.finfo(np.float64 if x64_enabled else np.float32)
    extreme = Exponential(rate=jnp.array([limits.tiny, limits.max, 2.0], dtype=dtype))
    ref = jax.jit(lambda d: d.reference_time())(extreme)
    assert ref.shape == extreme.batch_shape and ref.dtype == dtype
    assert np.isfinite(ref).all() and (ref > 0).all()
    np.testing.assert_allclose(ref[-1], 0.5)
    with pytest.raises(TypeError):
        Exponential.from_log(0)


@pytest.mark.parametrize("bad", [True, 1j, "1", np.array([1], dtype=object), None])
def test_dtype_misuse(bad):
    """Spec 3.3: reject input kinds before exponentiating log parameters."""
    for build in (
        lambda: Exponential(rate=bad),
        lambda: Exponential.from_log(log_rate=bad),
    ):
        with pytest.raises(TypeError):
            build()


def test_float64_conversion_warning(x64_enabled):
    """Spec 3.3: one conversion warning for stored float64 rate or log-rate."""
    if x64_enabled:
        pytest.skip("conversion warning applies only with x64 disabled")
    for build in (
        lambda: Exponential(rate=np.array([1.0], dtype=np.float64)),
        lambda: Exponential.from_log(log_rate=np.array([0.0], dtype=np.float64)),
    ):
        with pytest.warns(RuntimeWarning, match="float64 parameter") as warnings:
            assert build().rate.dtype == jnp.float32
        assert len(warnings) == 1


def test_from_log_values_and_derivatives(x64_enabled):
    """Spec 4.1/7.3: log f=a-exp(a)*t, derivative=1-exp(a)*t."""
    a = jnp.array([-1.0, 0.5])
    dist = Exponential.from_log(log_rate=a)
    np.testing.assert_allclose(
        jax.tree.leaves(dist)[0], np.exp(np.asarray(a)), rtol=RTOL[x64_enabled]
    )

    def fn(a):
        return Exponential.from_log(log_rate=a).log_prob(1.5).sum()

    np.testing.assert_allclose(
        jax.jit(fn)(a),
        (np.asarray(a) - np.exp(np.asarray(a)) * 1.5).sum(),
        rtol=RTOL[x64_enabled],
    )
    np.testing.assert_allclose(
        jax.grad(fn)(a), 1 - np.exp(np.asarray(a)) * 1.5, rtol=RTOL[x64_enabled]
    )
    assert Exponential.from_log(log_rate=0).rate.dtype == jnp.asarray(0.0).dtype
    assert (
        Exponential.from_log(log_rate=jnp.array(0.0, dtype=jnp.float32)).rate.dtype
        == jnp.float32
    )
    assert np.isinf(Exponential.from_log(log_rate=1000.0).rate)


def test_parameter_derivatives(x64_enabled):
    """Spec 7.3: log f gradient=1/r-t, Hessian=-1/r**2; boundaries retain limits."""

    def fn(r):
        return Exponential(rate=r).log_prob(1.5)

    for derivative in (jax.jacfwd(fn), jax.jacrev(fn), jax.jit(jax.grad(fn))):
        np.testing.assert_allclose(derivative(2.0), -1.0, atol=ATOL[x64_enabled])
    np.testing.assert_allclose(jax.hessian(fn)(2.0), -0.25, atol=ATOL[x64_enabled])
    for name, time, expected in (
        ("survival", 0.0, 0.0),
        ("cumulative_hazard", 0.0, 0.0),
        ("survival", jnp.inf, 0.0),
        ("log_cdf", jnp.inf, 0.0),
        ("log_prob", 0.0, 0.5),
        ("log_hazard", 0.0, 0.5),
        ("log_hazard", jnp.inf, 0.5),
        ("hazard", jnp.inf, 1.0),
    ):
        derivative = jax.grad(
            lambda r, name=name, time=time: getattr(Exponential(rate=r), name)(time)
        )
        np.testing.assert_allclose(
            jax.jit(jax.vmap(derivative))(jnp.array([2.0, 2.0])),
            expected,
            atol=ATOL[x64_enabled],
        )


@pytest.mark.parametrize("batched", [False, True])
def test_mixed_infinity_jacobians(batched, x64_enabled):
    """Spec 7.2/7.3: inactive infinity entries cannot poison finite components."""
    r = jnp.array([2.0, 3.0]) if batched else jnp.array(2.0)
    for name, expected in (
        ("log_survival", -1.0),
        ("cumulative_hazard", 1.0),
        ("log_prob", -0.5),
        ("log_hazard", 0.5),
        ("hazard", 1.0),
    ):

        def fn(r, name=name):
            return getattr(Exponential(rate=r), name)(jnp.array([1.0, jnp.inf]))[0]

        target = [expected, 0.0] if batched else expected
        for jacobian in (jax.jacfwd(fn), jax.jacrev(fn)):
            np.testing.assert_allclose(
                jax.jit(jacobian)(r), target, atol=ATOL[x64_enabled]
            )
