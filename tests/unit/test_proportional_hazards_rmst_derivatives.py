"""PH RMST range, chain-rule and supported Hessian evidence (spec 4.4/8.7)."""

from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from impl_tolerances import (
    ATOL,
    DERIVATIVE_ATOL,
    DERIVATIVE_RTOL,
    LIKELIHOOD_GRAD_EPS,
    RMST_CUTOFF_ATOL,
    RMST_PARAMETER_ATOL,
    RTOL,
)
from jax.test_util import check_grads

from survjax import Exponential, Weibull
from survjax.distributions.wrappers import ProportionalHazards


def _points(family):
    rows = []
    shapes = (1.0,) if family == "exponential" else (0.2, 0.5, 1, 2, 10, 50)
    for k, ratio, b, eta in product(
        shapes, (0.01, 0.1, 0.5, 1, 10, 100, 1e5), (1e-4, 1.0, 1e4), (-0.4, 0.3)
    ):
        tau = ratio * b * np.exp(-eta / k)
        rows.append([1 / b, eta, tau] if family == "exponential" else [k, b, eta, tau])
    if family == "weibull":
        for k, x, factor in product((0.2, 1, 2, 50), (0.1, None), (0.999, 1.001)):
            x = 64 + 4 / k if x is None else x
            rows.append([k, 1, 0.3, (x * factor) ** (1 / k) * np.exp(-0.3 / k)])
    return jnp.array(rows)


def _value(q, family):
    base = (
        Exponential(rate=q[0])
        if family == "exponential"
        else Weibull(shape=q[0], scale=q[1])
    )
    return ProportionalHazards(base, log_hazard_ratio=q[-2]).rmst(q[-1])


@pytest.fixture(scope="module", params=["exponential", "weibull"])
def reference(request):
    family = request.param
    points, values, gradients = _points(family), [], []
    with mp.workdps(60):
        for row in np.asarray(points):
            q = [mp.mpf(float(x)) for x in row]
            k, b = (mp.mpf(1), 1 / q[0]) if family == "exponential" else q[:2]
            eta, tau = q[-2:]
            a, limit = mp.exp(eta), tau / b
            unit = mp.exp(-eta / k)
            cuts = (
                [0]
                + [
                    unit * mp.mpf(10) ** n
                    for n in range(-8, 11)
                    if unit * mp.mpf(10) ** n < limit
                ]
                + [limit]
            )

            def survival(u, a=a, k=k):
                return mp.exp(-a * u**k)

            m = b * mp.quad(survival, cuts)
            # Differentiate physical PH survival under the integral; no gamma,
            # closure RMST or library output supplies expected numbers.
            ih = mp.quad(lambda u, a=a, k=k, s=survival: a * u**k * s(u), cuts)
            de = -b * ih
            if family == "exponential":
                gradient = [de / q[0], de]
            else:
                dk = -b * mp.quad(
                    lambda u, a=a, k=k, s=survival: a * u**k * mp.log(u) * s(u), cuts
                )
                gradient = [dk, k * ih, de]
            values.append(float(m))
            gradients.append([float(x) for x in (*gradient, survival(limit))])
    return family, points, np.asarray(values), np.asarray(gradients)


def test_full_range_values_and_first_parameter_derivatives(reference, x64_enabled):
    """Spec 8.7/12.1: full required range, scale changes, eta signs and switches."""
    family, points, values, expected = reference
    fn = partial(_value, family=family)
    compare = partial(np.testing.assert_allclose, equal_nan=False)
    for evaluate in (jax.vmap(fn), jax.jit(jax.vmap(fn))):
        compare(evaluate(points), values, atol=0, rtol=RTOL[x64_enabled])
    units = np.asarray(points[:, :-1]).copy()
    units[:, -1] = 1  # eta is already a log coordinate; others use r/k/b units.
    for transform in (jax.jacfwd, jax.jacrev):
        derivative = transform(fn)
        for evaluate in (jax.vmap(derivative), jax.jit(jax.vmap(derivative))):
            error = np.asarray(evaluate(points)) - expected
            compare(
                error[:, :-1] * units / values[:, None],
                0,
                atol=RMST_PARAMETER_ATOL[x64_enabled],
                rtol=0,
            )
            compare(error[:, -1], 0, atol=RMST_CUTOFF_ATOL[x64_enabled], rtol=0)
        for i in (0, len(points) // 2, len(points) - 1):
            error = np.asarray(derivative(points[i])) - expected[i]
            compare(
                error[:-1] * units[i] / values[i],
                0,
                atol=RMST_PARAMETER_ATOL[x64_enabled],
                rtol=0,
            )
            compare(error[-1], 0, atol=RMST_CUTOFF_ATOL[x64_enabled], rtol=0)


@pytest.mark.parametrize("family", ["exponential", "weibull"])
def test_broadcast_closure_identity_in_every_first_derivative(family, x64_enabled):
    """Spec 4.4/5: (S,1) baseline x (S,2) eta, paired and explicit cutoff axes."""
    b = jnp.array([[0.7], [1.3], [2.0]])
    eta = jnp.array([[0.2, -0.3], [0.4, 0.1], [-0.2, 0.3]])
    parameters = (
        (b, eta)
        if family == "exponential"
        else (jnp.array([[0.7], [1], [1.7]]), b, eta)
    )
    for cutoff in (jnp.array(0.75), jnp.array([0.25, 1.25, 2.25])[:, None, None]):

        def actual(q, cutoff=cutoff):
            base = (
                Exponential(rate=q[0])
                if family == "exponential"
                else Weibull(shape=q[0], scale=q[1])
            )
            return ProportionalHazards(base, log_hazard_ratio=q[-1]).rmst(cutoff)

        def equivalent(q, cutoff=cutoff):
            # Analytical laws r_PH=r*exp(eta), b_PH=b*exp(-eta/k).
            dist = (
                Exponential(rate=q[0] * jnp.exp(q[1]))
                if family == "exponential"
                else Weibull(shape=q[0], scale=q[1] * jnp.exp(-q[2] / q[0]))
            )
            return dist.rmst(cutoff)

        draws = jax.tree.map(lambda x: jnp.stack([x, x * 1.1]), parameters)
        for transform in (jax.jacfwd, jax.jacrev):
            expected = transform(equivalent)(parameters)
            for evaluate in (transform(actual), jax.jit(transform(actual))):
                for value, target in zip(
                    jax.tree.leaves(evaluate(parameters)),
                    jax.tree.leaves(expected),
                    strict=True,
                ):
                    np.testing.assert_allclose(
                        value,
                        target,
                        atol=ATOL[x64_enabled],
                        rtol=RTOL[x64_enabled],
                        equal_nan=False,
                    )
            actual_draws = jax.jit(jax.vmap(transform(actual)))(draws)
            expected_draws = jax.vmap(transform(equivalent))(draws)
            for value, target in zip(
                jax.tree.leaves(actual_draws),
                jax.tree.leaves(expected_draws),
                strict=True,
            ):
                np.testing.assert_allclose(
                    value,
                    target,
                    atol=ATOL[x64_enabled],
                    rtol=RTOL[x64_enabled],
                    equal_nan=False,
                )


def _supported(q, k, family):
    # k stays outside the differentiated argument, including under vmap.
    base = (
        Exponential(rate=q[0])
        if family == "exponential"
        else Weibull(shape=k, scale=q[0])
    )
    return ProportionalHazards(base, log_hazard_ratio=q[1]).rmst(q[2])


@pytest.mark.parametrize("family", ["exponential", "weibull"])
def test_supported_hessians_are_finite_across_full_range(family):
    """Spec 8.7: all four AD orders, including lower/gamma/mean branches."""
    points = _points(family)
    parameters = points if family == "exponential" else points[:, 1:]
    shapes = jnp.ones(len(points)) if family == "exponential" else points[:, 0]
    fn = partial(_supported, family=family)
    for outer, inner in product((jax.jacfwd, jax.jacrev), repeat=2):
        values = jax.jit(jax.vmap(outer(inner(fn))))(parameters, shapes)
        assert np.isfinite(values).all()


@pytest.mark.parametrize("family", ["exponential", "weibull"])
def test_supported_hessians_match_independent_integrals(family, x64_enabled):
    """Spec 7.3/8.7: mixed baseline/eta/tau Hessians, independent physical law."""
    k = float(jnp.asarray(1.4))
    p = jnp.array([1.3, 0.2, 0.75] if family == "exponential" else [2.0, 0.2, 0.75])
    fn = partial(_supported, k=k, family=family)
    with mp.workdps(60):
        q = tuple(mp.mpf(float(x)) for x in np.asarray(p))

        def integral(*x):
            if family == "exponential":
                return mp.quad(lambda t: mp.exp(-mp.exp(x[1]) * x[0] * t), [0, x[2]])
            return mp.quad(
                lambda t: mp.exp(-mp.exp(x[1]) * (t / x[0]) ** mp.mpf(k)), [0, x[2]]
            )

        orders, expected = np.eye(3, dtype=int), []
        for row in orders:
            expected.append(
                [float(mp.diff(integral, q, tuple(row + column))) for column in orders]
            )
    # O(1) entries: AD/quadrature rounding, using the unchanged ordinary bounds.
    for outer, inner in product((jax.jacfwd, jax.jacrev), repeat=2):
        for evaluate in (outer(inner(fn)), jax.jit(outer(inner(fn)))):
            np.testing.assert_allclose(
                evaluate(p),
                expected,
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
                equal_nan=False,
            )
    check_grads(
        fn,
        (p,),
        order=2,
        modes=("fwd", "rev"),
        eps=LIKELIHOOD_GRAD_EPS[x64_enabled],
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_log_space_closure_fallback_keeps_representable_rmst(sign, x64_enabled):
    """Spec 4.4/8.7: an unrepresentable multiplier need not lose a finite scale."""
    eta = sign * (800.0 if x64_enabled else 100.0)
    b = (
        (1e300 if x64_enabled else 1e30)
        if sign > 0
        else (1e-300 if x64_enabled else 1e-30)
    )
    p = jnp.array([1.0, b, eta, 0.5 * np.exp(np.log(b) - eta)])
    with mp.workdps(60):
        _, scale, e, tau = (mp.mpf(float(x)) for x in np.asarray(p))
        # At k=1, the PH survival integral is exponential with rate exp(eta)/b.
        rate = mp.exp(e) / scale
        expected = float(-mp.expm1(-rate * tau) / rate)
    fn = partial(_value, family="weibull")
    for evaluate in (fn, jax.jit(fn)):
        np.testing.assert_allclose(
            evaluate(p), expected, atol=0, rtol=RTOL[x64_enabled], equal_nan=False
        )
