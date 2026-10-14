"""Weibull regressions from S=exp(-(t/b)**k), spec 4.1/4.5/7.2/7.3."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

from survjax import Weibull


def test_log_inputs_normalize_before_exp_and_obey_chain_rule(x64_enabled):
    """Spec 3.3/4.1/7.3: d/da = k*d/dk, d/dc = b*d/db for log inputs."""
    atol, rtol = ATOL[x64_enabled], RTOL[x64_enabled]
    logs = jnp.array([0.0, np.log(2)])

    def fn(p):
        return Weibull.from_log(log_shape=p[0], log_scale=p[1]).log_prob(1.5)

    # k=1, b=2, H=0.75: (k*dlogf/dk, b*dlogf/db)=(1+0.25*log(.75), -.25).
    expected = [1 + 0.25 * np.log(0.75), -0.25]
    for derivative in (jax.jacfwd(fn), jax.jacrev(fn)):
        out = jax.jit(derivative)(logs)
        np.testing.assert_allclose(out, expected, atol=atol, rtol=rtol)
    dist = Weibull.from_log(log_shape=np.array([0], np.float32), log_scale=0)
    assert dist.shape.dtype == dist.scale.dtype == jnp.float32
    np.testing.assert_array_equal(jax.tree.leaves(dist), [[1], [1]])
    extreme = Weibull.from_log(log_shape=1000, log_scale=1000)
    assert np.isinf(extreme.shape) and np.isinf(extreme.scale)


@pytest.mark.parametrize("bad", [True, 1j, "1", np.array([1], dtype=object), None])
def test_nonreal_inputs_are_rejected_before_exp(bad):
    """Spec 3.3: either natural/log parameter rejects nonreal input kinds."""
    for values in ((bad, 1), (1, bad)):
        with pytest.raises(TypeError):
            Weibull(shape=values[0], scale=values[1])
        with pytest.raises(TypeError):
            Weibull.from_log(log_shape=values[0], log_scale=values[1])


@pytest.mark.parametrize("time", [0.0, jnp.inf])
def test_constant_boundary_parameter_derivatives(time, x64_enabled):
    """Spec 4.5: these limits are constant in both parameters, even at extreme scale."""
    names = (
        ("survival", "log_survival", "cumulative_hazard")
        if time == 0
        else ("survival", "log_cdf")
    )
    params = jnp.array([[0.5, 1e-30], [1, 2], [2, 1e30]])
    for name in names:

        def fn(p, name=name):
            return getattr(Weibull(shape=p[0], scale=p[1]), name)(time)

        for derivative in (jax.jacfwd(fn), jax.jacrev(fn)):
            out = jax.jit(jax.vmap(derivative))(params)
            np.testing.assert_allclose(out, 0, atol=ATOL[x64_enabled])


@pytest.mark.parametrize("time", [0.0, jnp.inf])
def test_shape_one_boundary_hazard_scale_derivative(time, x64_enabled):
    """Spec 4.5: h=1/b, so dh/db=-1/b**2=-1/4 at zero/infinity, b=2."""

    def fn(b):
        return Weibull(shape=1, scale=b).hazard(time)

    for derivative in (jax.jacfwd(fn), jax.jacrev(fn)):
        out = jax.jit(derivative)(2.0)
        np.testing.assert_allclose(out, -0.25, atol=ATOL[x64_enabled])


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("k", [0.7, 1.0, 2.0])
def test_mixed_boundary_full_parameter_derivatives(batched, k, x64_enabled):
    """Spec 7.2/7.3: shared/per-entry parameters retain interior Jacobians/Hessians."""
    atol, rtol = ATOL[x64_enabled], RTOL[x64_enabled]
    # At b=2, t=1: H=2**(-k), u=-log(2). Derivatives of H and log h:
    u, h = -np.log(2), 0.5**k
    gh = np.array([h * u, -k * h / 2])
    cross = -h * (1 + k * u) / 2
    hh = np.array([[h * u**2, cross], [cross, k * (k + 1) * h / 4]])
    gl = np.array([1 / k + u, -k / 2])
    hl = np.array([[-1 / k**2, -0.5], [-0.5, k / 4]])
    for name, g, second in (
        ("log_survival", -gh, -hh),
        ("cumulative_hazard", gh, hh),
        ("log_prob", gl - gh, hl - hh),
        ("log_hazard", gl, hl),
        ("hazard", k * h * gl, k * h * (hl + np.outer(gl, gl))),
    ):

        def fn(p, name=name):
            k, b = (p[:, 0], p[:, 1]) if batched else (p[0], p[1])
            dist = Weibull(shape=k, scale=b)
            return getattr(dist, name)(jnp.array([0, 1, jnp.inf]))[1]

        p = jnp.array([[k, 2.0]] * 3 if batched else [k, 2.0])
        gradient, hessian = np.zeros(p.shape), np.zeros(p.shape + p.shape)
        if batched:
            gradient[1], hessian[1, :, 1, :] = g, second
        else:
            gradient, hessian = g, second
        for derivative, expected in (
            (jax.jacfwd(fn), gradient),
            (jax.jacrev(fn), gradient),
            (jax.hessian(fn), hessian),
            (jax.jacrev(jax.jacfwd(fn)), hessian),
        ):
            out = jax.jit(jax.vmap(derivative))(jnp.stack([p, p]))[0]
            np.testing.assert_allclose(out, expected, atol=atol, rtol=rtol)


def test_nan_parameters_are_isolated_and_raw_methods_broadcast(x64_enabled):
    """Spec 4.5/5: NaN leaves remain NaN; raw formulas broadcast scalar times."""
    dist = Weibull(shape=[2, jnp.nan, 2], scale=[2, 2, jnp.nan])
    times = jnp.array([[0], [1], [jnp.inf]])
    names = "log_survival survival log_cdf cumulative_hazard log_prob log_hazard hazard"
    for name in names.split():
        out = jax.jit(lambda d, name=name: getattr(d, name)(times))(dist)
        assert np.isnan(out[:, 1:]).all() and not np.isnan(out[:, 0]).any()
    # At t=b=2, H=1 and h=k/b=1 for k=2, so log f=-1 and log h=0.
    valid = Weibull(shape=[2, 2], scale=2)
    for name, expected in (
        ("log_survival", -1),
        ("cumulative_hazard", 1),
        ("log_prob", -1),
        ("log_hazard", 0),
    ):
        raw = getattr(Weibull, name)._survjax_raw
        for time, shape in ((2.0, (2,)), (jnp.full((2, 1), 2.0), (2, 2))):
            out = jax.jit(lambda t, raw=raw: raw(valid, t))(time)
            assert out.shape == shape
            np.testing.assert_allclose(out, expected, atol=ATOL[x64_enabled])


def test_extreme_ratios_and_reference_scales(x64_enabled):
    """Spec 4.2/7.2: (1e-60)**.25=1e-15, (1e60)**.25=1e15 without ratio loss."""
    dist = Weibull(shape=[0.25, 0.25], scale=[1e30, 1e-30])
    np.testing.assert_allclose(
        jax.jit(dist.cumulative_hazard)(jnp.array([1e-30, 1e30])),
        [1e-15, 1e15],
        atol=0,
        rtol=RTOL[x64_enabled],
    )
    ref = jax.jit(lambda d: d.reference_time())(dist)
    assert ref.shape == dist.batch_shape and ref.dtype == dist.scale.dtype
    assert np.isfinite(ref).all() and (ref > 0).all()
    # At its scale every valid shape has cumulative hazard exactly one.
    np.testing.assert_allclose(dist.cumulative_hazard(ref), 1, atol=ATOL[x64_enabled])


def test_log_cdf_retains_tail_when_cumulative_hazard_underflows(x64_enabled):
    """Spec 4.2/7.1: log F = k*log(t/b) when H is below dtype precision."""
    time = jnp.asarray(1e-5)
    actual = Weibull(shape=10.0, scale=1.0).log_cdf(time)
    expected = 10.0 * np.log(1e-5)
    np.testing.assert_allclose(
        actual, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    if not x64_enabled:
        assert Weibull(shape=10.0, scale=1.0).cumulative_hazard(time) == 0


def test_log_cdf_switch_is_continuous_with_finite_gradients(x64_enabled):
    """Spec 4.3/7.2: both sides of the eps switch retain finite shape gradients."""
    dtype = jnp.float64 if x64_enabled else jnp.float32
    eps = np.finfo(np.float64 if x64_enabled else np.float32).eps
    log_eps = jnp.log(jnp.asarray(eps, dtype=dtype))
    offset = 4 * np.sqrt(eps)
    times = jnp.asarray(
        np.exp(np.asarray([log_eps - offset, log_eps + offset])), dtype=dtype
    )
    log_h = jnp.log(times)  # shape=scale=1 gives log H=log(t).
    assert bool(log_h[0] < log_eps) and bool(log_h[1] > log_eps)

    def log_cdf(shape, time):
        return Weibull(shape=shape, scale=1.0).log_cdf(time)

    with jax.debug_nans(True):
        values = jax.jit(lambda t: Weibull(shape=1.0, scale=1.0).log_cdf(t))(times)
        gradients = jax.jit(jax.vmap(jax.grad(log_cdf, argnums=0), in_axes=(None, 0)))(
            1.0, times
        )
    np.testing.assert_allclose(values, log_h, atol=eps, rtol=0)
    assert np.isfinite(np.asarray(gradients)).all()
