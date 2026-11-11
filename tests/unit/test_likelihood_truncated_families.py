"""Independent truncated-family parameter derivatives (spec 2, 7.3, 12.1)."""

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
    RTOL,
)
from jax.test_util import check_grads

import survjax

WINDOWS = {"left": (0.5, np.inf), "right": (0.0, 3.0), "two-sided": (0.5, 3.0)}
POINTS = {
    "lognormal": [[0.25, 1.25], [-0.375, 0.875]],
    "loglogistic": [[0.75, 1.5], [2.5, 2.0]],
}
# Exact, right, left, overlap at L, overlap at U, and interior interval.
# Dyadic inputs avoid a dtype-dependent reference dataset.
ROWS = (
    (True, 1.5, 1.5),
    (False, 1.5, np.inf),
    (False, 0.0, 1.5),
    (False, 0.25, 1.5),
    (False, 2.0, 5.0),
    (False, 1.0, 2.0),
)


def _distribution(family, point):
    if family == "lognormal":
        return survjax.LogNormal(meanlog=point[0], sdlog=point[1])
    return survjax.LogLogistic(shape=point[0], scale=point[1])


def _data(window):
    exact, lower, upper = zip(*ROWS, strict=True)
    left, right = WINDOWS[window]
    return survjax.SurvivalData.from_bounds(
        lower, upper, exact=exact, left_trunc=left, right_trunc=right
    )


def _reference(family, point, window):
    """Differentiate the section-2 ratio using only mpmath, at 80 digits."""
    left, right = WINDOWS[window]
    values, gradients, hessians = [], [], []
    with mp.workdps(80):
        parameters = tuple(mp.mpf(float(p)) for p in point)
        for exact, lower, upper in ROWS:

            def contribution(p, q, exact=exact, lower=lower, upper=upper):
                def survival(time):
                    if time == 0:
                        return mp.mpf(1)
                    if time == np.inf:
                        return mp.mpf(0)
                    t = mp.mpf(time)
                    if family == "lognormal":
                        return mp.erfc((mp.log(t) - p) / (q * mp.sqrt(2))) / 2
                    return 1 / (1 + (t / q) ** p)

                normalizer = survival(left) - survival(right)
                if exact:
                    t = mp.mpf(lower)
                    if family == "lognormal":
                        z = (mp.log(t) - p) / q
                        density = mp.exp(-(z**2) / 2) / (t * q * mp.sqrt(2 * mp.pi))
                    else:
                        power = (t / q) ** p
                        density = p * power / (t * (1 + power) ** 2)
                    return mp.log(density / normalizer)
                mass = survival(max(lower, left)) - survival(min(upper, right))
                return mp.log(mass / normalizer)

            values.append(float(contribution(*parameters)))
            gradients.append(
                [float(mp.diff(contribution, parameters, d)) for d in ((1, 0), (0, 1))]
            )
            hessians.append(
                [
                    [float(mp.diff(contribution, parameters, d)) for d in row]
                    for row in (((2, 0), (1, 1)), ((1, 1), (0, 2)))
                ]
            )
    return np.asarray(values), np.asarray(gradients), np.asarray(hessians)


@pytest.mark.parametrize("family", POINTS)
@pytest.mark.parametrize("window", WINDOWS)
def test_truncated_mixed_values_match_independent_ratios(family, window, x64_enabled):
    """Spec 2/7.2: all kinds, clipped sets and inactive pieces are safe."""
    data = _data(window)
    points = jnp.asarray(POINTS[family])
    expected = np.stack([_reference(family, p, window)[0] for p in points])

    def rows(p):
        return _distribution(family, p).log_likelihood(data)

    assert data.issues.ok
    with jax.debug_nans(True):
        eager = jnp.stack([rows(p) for p in points])
    for actual in (eager, jax.jit(jax.vmap(rows))(points)):
        np.testing.assert_allclose(
            actual, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


@pytest.mark.parametrize("family", POINTS)
@pytest.mark.parametrize("window", WINDOWS)
def test_truncated_row_jacobians_and_hessians_match_mpmath(family, window, x64_enabled):
    """Spec 7.3: per-row derivatives, all four AD compositions and jit/vmap."""
    data = _data(window)
    points = jnp.asarray(POINTS[family])
    references = [_reference(family, p, window) for p in points]
    gradient = np.stack([r[1] for r in references])
    hessian = np.stack([r[2] for r in references])
    tol = {"atol": ATOL[x64_enabled], "rtol": RTOL[x64_enabled]}

    def rows(p):
        return _distribution(family, p).log_likelihood(data)

    for first in (jax.jacfwd, jax.jacrev):
        derivative = first(rows)
        np.testing.assert_allclose(derivative(points[0]), gradient[0], **tol)
        np.testing.assert_allclose(
            jax.jit(jax.vmap(derivative))(points), gradient, **tol
        )
        for second in (jax.jacfwd, jax.jacrev):
            derivative2 = second(derivative)
            np.testing.assert_allclose(derivative2(points[0]), hessian[0], **tol)
            np.testing.assert_allclose(
                jax.jit(jax.vmap(derivative2))(points), hessian, **tol
            )


@pytest.mark.parametrize("family", POINTS)
@pytest.mark.parametrize("window", WINDOWS)
def test_truncated_interior_order_two_finite_differences(family, window, x64_enabled):
    """Spec 12.1: numerical derivative check on a compiled interior objective."""
    data = _data(window)

    def objective(p):
        return _distribution(family, p).log_likelihood(data).mean()

    check_grads(
        jax.jit(objective),
        (jnp.asarray(POINTS[family][0]),),
        order=2,
        modes=("fwd", "rev"),
        eps=LIKELIHOOD_GRAD_EPS[x64_enabled],
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
    )


@pytest.mark.parametrize("family", POINTS)
def test_whole_windows_have_exact_zero_parameter_derivatives(family):
    """Spec 2/7.1/7.3: P(window|window)=1, so log value and derivatives are zero."""
    data = survjax.SurvivalData.from_bounds(
        [0.0, 0.25, 0.5],
        [5.0, 5.0, np.inf],
        left_trunc=[0.0, 0.5, 0.5],
        right_trunc=[3.0, 3.0, np.inf],
    )

    def rows(p):
        return _distribution(family, p).log_likelihood(data)

    points = jnp.asarray(POINTS[family])
    assert data.issues.ok
    np.testing.assert_array_equal(data.issues.uninformative, True)
    np.testing.assert_array_equal(jax.jit(jax.vmap(rows))(points), 0)
    for first in (jax.jacfwd, jax.jacrev):
        np.testing.assert_array_equal(jax.jit(jax.vmap(first(rows)))(points), 0)
        for second in (jax.jacfwd, jax.jacrev):
            np.testing.assert_array_equal(
                jax.jit(jax.vmap(second(first(rows))))(points), 0
            )
