"""Independent staged Weibull RMST derivative evidence (spec 8.7/12.1)."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from impl_tolerances import (
    RMST_CUTOFF_ATOL,
    RMST_PARAMETER_ATOL,
    RTOL,
    TAIL_ATOL,
    TAIL_RTOL,
)

from survjax import Weibull


def _value(p):
    return Weibull(shape=p[0], scale=p[1]).rmst(p[2])


def _points():
    cases = [
        (k, b, r * b)
        for k in (0.2, 0.5, 1.0, 2.0, 10.0, 50.0)
        for r in (0.01, 0.1, 0.5, 1.0, 10.0, 100.0, 1e5)
        for b in (1e-4, 1.0, 1e4)
    ]
    # Straddle each switch, including the steep k=50 transition.
    cases += [
        (k, 1.0, (x * factor) ** (1 / k))
        for k in (0.2, 1.0, 2.0, 50.0)
        for x in (0.1, 64 + 4 / k)
        for factor in (0.999, 1.001)
    ]
    return jnp.array(cases)


@pytest.fixture(scope="module")
def quadrature_reference():
    points = _points()
    values, gradients = [], []
    with mp.workdps(60):
        # Reference the actual rounded inputs, not their pre-conversion values.
        for point in np.asarray(points):
            k, b, tau = (mp.mpf(float(x)) for x in point)
            r = tau / b
            cuts = [0] + [mp.mpf(10) ** n for n in range(-8, 11) if 10**n < r] + [r]
            m = mp.quad(lambda u, k=k: mp.exp(-(u**k)), cuts)
            # Differentiate survival under the integral, independently of gamma.
            dk = -b * mp.quad(lambda u, k=k: mp.exp(-(u**k)) * u**k * mp.log(u), cuts)
            # dS/db=(k/b)*u**k*S. Integrate directly to avoid m-r*S cancellation.
            db = mp.quad(lambda u, k=k: k * u**k * mp.exp(-(u**k)), cuts)
            survival = mp.exp(-(r**k))
            values.append(float(b * m))
            gradients.append([float(dk), float(db), float(survival)])
    return points, np.asarray(values), np.asarray(gradients)


def test_weibull_rmst_first_derivatives_full_range(quadrature_reference, x64_enabled):
    """Spec 8.7/12.1: both modes, scales, tails and both sides of switches."""
    points, values, expected = quadrature_reference
    # Natural Jacobians are rescaled to log-parameter units before comparison:
    # |k*error_k|/M, |b*error_b|/M; cutoff error uses absolute scale 1.
    units = np.column_stack(
        (np.asarray(points[:, 0]) / values, np.asarray(points[:, 1]) / values)
    )
    for transform in (jax.jacfwd, jax.jacrev):
        derivative = transform(_value)
        for actual in (
            jax.vmap(derivative)(points),
            jax.jit(jax.vmap(derivative))(points),
            jax.vmap(jax.jit(derivative))(points),
        ):
            np.testing.assert_allclose(
                (np.asarray(actual[:, :2]) - expected[:, :2]) * units,
                0,
                atol=RMST_PARAMETER_ATOL[x64_enabled],
                rtol=0,
            )
            np.testing.assert_allclose(
                actual[:, 2], expected[:, 2], atol=RMST_CUTOFF_ATOL[x64_enabled], rtol=0
            )
        # Unbatched eager evaluation must agree too, spanning all three branches.
        for i in (0, 63, 125):
            actual = np.asarray(derivative(points[i]))
            np.testing.assert_allclose(
                (actual[:2] - expected[i, :2]) * units[i],
                0,
                atol=RMST_PARAMETER_ATOL[x64_enabled],
                rtol=0,
            )
            np.testing.assert_allclose(
                actual[2],
                expected[i, 2],
                atol=RMST_CUTOFF_ATOL[x64_enabled],
                rtol=0,
            )


def test_weibull_rmst_cutoff_derivative_equals_survival(x64_enabled):
    """Spec 8.7: the fundamental theorem holds at absolute cutoff scale 1."""
    points = _points()
    actual = jax.jit(jax.vmap(jax.grad(_value)))(points)[:, 2]
    expected = Weibull(shape=points[:, 0], scale=points[:, 1]).survival(points[:, 2])
    np.testing.assert_allclose(
        actual, expected, atol=RMST_CUTOFF_ATOL[x64_enabled], rtol=0
    )


def test_weibull_rmst_supported_hessians_are_finite():
    """Spec 8.7/12.1: scale/cutoff Hessians remain finite across the range."""

    def supported(p, k):
        return Weibull(shape=k, scale=p[0]).rmst(p[1])

    points = _points()
    for transform in (jax.jacfwd, jax.jacrev):
        hessian = transform(transform(supported))
        with jax.debug_nans(True):
            actual = jax.jit(jax.vmap(hessian))(points[:, 1:], points[:, 0])
        assert np.isfinite(actual).all()


@pytest.mark.parametrize("k,tau", [(2.0, 0.1), (0.5, 0.75), (50.0, 1.0)])
def test_weibull_rmst_supported_hessian_matches_calculus(k, tau, x64_enabled):
    """Spec 8.7: nonshape Hessians agree with independent survival calculus."""
    b = 1.25
    with mp.workdps(60):
        r = mp.mpf(tau) / b
        s = mp.exp(-(r**k))
        # Differentiate dM/db=m-r*S and dM/dtau=S:
        # M_bb=-k*r**(k+1)*S/b; M_bt=k*r**k*S/b; M_tt=-k*r**(k-1)*S/b.
        bb, bt, tt = [
            float(v)
            for v in (
                -k * r ** (k + 1) * s / b,
                k * r**k * s / b,
                -k * r ** (k - 1) * s / b,
            )
        ]
        expected = [[bb, bt], [bt, tt]]

    def fn(p):
        return Weibull(shape=k, scale=p[0]).rmst(p[1])

    for outer in (jax.jacfwd, jax.jacrev):
        for inner in (jax.jacfwd, jax.jacrev):
            actual = jax.jit(outer(inner(fn)))(jnp.array([b, tau]))
            np.testing.assert_allclose(
                actual,
                expected,
                atol=TAIL_ATOL[x64_enabled],
                rtol=TAIL_RTOL[x64_enabled],
            )


def test_weibull_rmst_shape_hessian_raises_in_both_modes():
    """Spec 8.7: native igamma shape gradients have no higher-order rule."""

    def fn(k):
        return Weibull(shape=k, scale=1.25).rmst(0.75)

    for outer in (jax.jacfwd, jax.jacrev):
        for inner in (jax.jacfwd, jax.jacrev):
            with pytest.raises(NotImplementedError):
                jax.jit(outer(inner(fn)))(2.0)


@pytest.mark.parametrize("field", ["shape", "scale", "cutoff"])
def test_weibull_rmst_shared_parameters_isolate_valid_entries(field, x64_enabled):
    """Spec 7.2/8.7: invalid rows cannot poison shared valid derivatives."""
    bad = jnp.array([1.0, 1.0, 0.0, -1.0, jnp.nan, jnp.inf, -jnp.inf])
    cutoff = jnp.array([0.5, 0.75, -1.0, -1.0, jnp.nan, jnp.inf, -jnp.inf])

    def mixed(k, b):
        k = jnp.where(jnp.arange(7) < 2, k, bad) if field == "shape" else k
        b = jnp.where(jnp.arange(7) < 2, b, bad) if field == "scale" else b
        t = cutoff if field == "cutoff" else jnp.array([0.5, 0.75, 1, 1, 1, 1, 1])
        return Weibull(shape=k, scale=b).rmst(t)[:2]

    def isolated(k, b):
        return Weibull(shape=k, scale=b).rmst(jnp.array([0.5, 0.75]))

    for transform in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            jax.jit(transform(mixed, argnums=(0, 1)))(2.0, 1.25),
            transform(isolated, argnums=(0, 1))(2.0, 1.25),
            rtol=RTOL[x64_enabled],
        )
        # Keep shape outside the differentiated argument: packing [k,b] can
        # materialize a zero shape tangent and invoke unsupported igamma_grad_a AD.
        hessian = transform(transform(mixed, argnums=1), argnums=1)
        reference = transform(transform(isolated, argnums=1), argnums=1)
        np.testing.assert_allclose(
            jax.jit(hessian)(2.0, 1.25), reference(2.0, 1.25), rtol=RTOL[x64_enabled]
        )
