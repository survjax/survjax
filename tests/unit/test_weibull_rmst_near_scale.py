"""Cancellation regressions for native and PH Weibull RMST (spec 8.7)."""

from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from impl_tolerances import (
    ATOL,
    RMST_CUTOFF_ATOL,
    RMST_PARAMETER_ATOL,
    RTOL,
)

from survjax import Weibull
from survjax.distributions.wrappers import ProportionalHazards


def _value(q, wrapped):
    k, b, eta, tau = q
    baseline = Weibull(shape=k, scale=b)
    dist = ProportionalHazards(baseline, log_hazard_ratio=eta) if wrapped else baseline
    return dist.rmst(tau)


def _supported(q, k, wrapped):
    # Keeping shape outside differentiated arguments avoids unsupported igamma AD.
    return _value((k, *q), wrapped)


@pytest.fixture(scope="module", params=[False, True], ids=["native", "ph"])
def reference(request):
    wrapped = request.param
    rows = []
    for k, b, eta, steps in product(
        (0.2, 2.0, 50.0), (1e-4, 1e4), (0.0, 0.3) if wrapped else (0.0,), (-6, 0, 6)
    ):
        tau = jnp.asarray(b * np.exp(-eta / k))
        for _ in range(abs(steps)):
            tau = jnp.nextafter(tau, -jnp.inf if steps < 0 else jnp.inf)
        rows.append((k, b, eta, float(tau)))
    # Straddle the equivalent-expression switch at |log(tau/b)|=1/2.
    rows += [
        (k, 1e4, 0.0, 1e4 * np.exp(sign * 0.5 * factor))
        for k, sign, factor in product((0.2, 2.0, 50.0), (-1, 1), (0.9999, 1.0001))
    ]
    points = jnp.array(rows)
    values, gradients, hessians = [], [], []
    with mp.workdps(60):
        for row in np.asarray(points):
            k, b, eta, tau = [mp.mpf(float(v)) for v in row]
            a, r = mp.exp(eta), tau / b

            # Integrate the physical law, independent of gamma or closure outputs.
            def survival(u, a=a, k=k):
                return mp.exp(-a * u**k)

            m = b * mp.quad(survival, [0, r / 2, r])
            ih = mp.quad(lambda u, a=a, k=k, s=survival: a * u**k * s(u), [0, r / 2, r])
            dk = -b * mp.quad(
                lambda u, a=a, k=k, s=survival: a * u**k * mp.log(u) * s(u),
                [0, r / 2, r],
            )
            x, s = a * r**k, survival(r)
            values.append(float(m))
            gradients.append(
                [float(dk), float(k * ih), float(-b * ih) if wrapped else 0, float(s)]
            )
            # Differentiate M_b=k*I_H, M_eta=-b*I_H, M_tau=S(tau).
            # Integration by parts gives dI_H/deta=(r*x*S-I_H)/k.
            be = r * x * s - ih if wrapped else 0
            ee = -b * be / k if wrapped else 0
            et = -x * s if wrapped else 0
            bt = k * x * s / b
            hessians.append(
                [
                    [float(-r * bt), float(be), float(bt)],
                    [float(be), float(ee), float(et)],
                    [float(bt), float(et), float(-k * x * s / tau)],
                ]
            )
    return wrapped, points, np.array(values), np.array(gradients), np.array(hessians)


def test_cutoff_gradient_preserves_near_scale_precision(reference, x64_enabled):
    """Spec 8.7/12.1: independent S(tau), including six ULPs above b=10000."""
    wrapped, points, _, expected, _ = reference
    fn = partial(_value, wrapped=wrapped)
    for mode in (jax.jacfwd, jax.jacrev):
        derivative = mode(fn)
        for actual in (
            jax.vmap(derivative)(points),
            jax.jit(jax.vmap(derivative))(points),
        ):
            np.testing.assert_allclose(
                actual[:, -1],
                expected[:, -1],
                atol=RMST_CUTOFF_ATOL[x64_enabled],
                rtol=0,
                equal_nan=False,
            )
        # Explicit unbatched eager and jit checks at the reported failing tuple.
        i = next(
            i
            for i, q in enumerate(np.asarray(points))
            if q[0] == 50 and q[1] == 1e4 and q[2] == 0 and q[3] > q[1]
        )
        for actual in (derivative(points[i]), jax.jit(derivative)(points[i])):
            np.testing.assert_allclose(
                actual[-1],
                expected[i, -1],
                atol=RMST_CUTOFF_ATOL[x64_enabled],
                rtol=0,
                equal_nan=False,
            )


def test_values_and_parameter_gradients_near_scale(reference, x64_enabled):
    """Spec 8.7/12.1: the alternate expression preserves every first derivative."""
    wrapped, points, values, expected, _ = reference
    fn = partial(_value, wrapped=wrapped)
    for actual in (jax.vmap(fn)(points), jax.jit(jax.vmap(fn))(points)):
        np.testing.assert_allclose(
            actual, values, atol=0, rtol=RTOL[x64_enabled], equal_nan=False
        )
    # Spec 12.1: k/b errors have log-parameter units; eta is already logarithmic.
    units = (
        np.column_stack((np.asarray(points[:, :2]), np.ones(len(points))))
        / values[:, None]
    )
    for mode in (jax.jacfwd, jax.jacrev):
        actual = jax.jit(jax.vmap(mode(fn)))(points)
        np.testing.assert_allclose(
            (np.asarray(actual[:, :3]) - expected[:, :3]) * units,
            0,
            atol=RMST_PARAMETER_ATOL[x64_enabled],
            rtol=0,
            equal_nan=False,
        )


def test_supported_hessians_near_scale_and_expression_switch(reference, x64_enabled):
    """Spec 8.7: all nonshape second-AD compositions match survival calculus."""
    wrapped, points, values, _, expected = reference
    fn = partial(_supported, wrapped=wrapped)
    # Express Hessians in dimensionless b/eta/tau units, normalized by M.
    units = np.column_stack(
        (np.asarray(points[:, 1]), np.ones(len(points)), np.asarray(points[:, 3]))
    )
    scale = units[:, :, None] * units[:, None, :] / values[:, None, None]
    for outer, inner in product((jax.jacfwd, jax.jacrev), repeat=2):
        derivative = outer(inner(fn))
        for actual in (
            jax.vmap(derivative)(points[:, 1:], points[:, 0]),
            jax.jit(jax.vmap(derivative))(points[:, 1:], points[:, 0]),
        ):
            np.testing.assert_allclose(
                np.asarray(actual) * scale,
                expected * scale,
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
                equal_nan=False,
            )


@pytest.mark.parametrize("wrapped", [False, True], ids=["native", "ph"])
def test_separated_scales_keep_inactive_log1p_safe(wrapped, x64_enabled):
    """Spec 7.2/8.7: discarded subtraction/division/log1p cannot poison AD."""
    b = jnp.array([1e30, 1e30, 1e-30])
    tau = jnp.array([0.0, 1e-30, 1e30])

    def fn(t, scale):
        return _value((2.0, scale, 0.0, t), wrapped)

    # At ratio 1e-60, |M-tau|/tau < (tau/b)^2/3 < 1e-120.
    # At ratio 1e60, M differs from b*Gamma(3/2) by an exponentially tiny tail.
    with mp.workdps(60):
        expected = [
            0,
            float(tau[1]),
            float(mp.mpf(float(b[2])) * mp.gamma(mp.mpf("1.5"))),
        ]
    with jax.debug_nans(True):
        actual = jax.jit(jax.vmap(fn))(tau, b)
    np.testing.assert_allclose(
        actual, expected, atol=0, rtol=RTOL[x64_enabled], equal_nan=False
    )
    for mode in (jax.jacfwd, jax.jacrev):
        derivative = mode(fn, argnums=0)
        with jax.debug_nans(True):
            actual = jax.jit(jax.vmap(derivative))(tau, b)
        # S(0)=1; both positive-cutoff survival limits round exactly to 1/0.
        np.testing.assert_array_equal(actual, [1, 1, 0])
