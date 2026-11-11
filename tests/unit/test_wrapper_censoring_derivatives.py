"""Independent PH/AFT likelihood mathematics (spec 2, 4.4, 7.3, 12.1)."""

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

POINTS = {
    "exponential": [[0.75, 0.25], [1.25, -0.375]],
    "weibull": [[0.75, 1.5, 0.25], [1.75, 2.0, -0.375]],
    "lognormal": [[0.25, 1.25, 0.25], [-0.375, 0.875, -0.375]],
    "loglogistic": [[0.75, 1.5, 0.25], [2.5, 2.0, -0.375]],
}
WINDOWS = ((0.0, np.inf), (0.5, np.inf), (0.0, 3.0), (0.5, 3.0))
KINDS = ((True, 1.5, 1.5), (False, 1.5, np.inf), (False, 0.0, 1.5), (False, 1.0, 2.0))
# Each tuple is (exact, lower, upper, L, U). The final three rows cover windows.
ROWS = tuple(
    (ex, a, b, left, right) for left, right in WINDOWS for ex, a, b in KINDS
) + (
    (False, 0.25, 1.5, 0.5, 3.0),  # partial overlap at entry
    (False, 2.0, 5.0, 0.5, 3.0),  # partial overlap at upper truncation
    (False, 0.0, 5.0, 0.0, 3.0),
    (False, 0.25, 5.0, 0.5, 3.0),
    (False, 0.5, np.inf, 0.5, np.inf),
)


def _data():
    exact, lower, upper, left, right = zip(*ROWS, strict=True)
    return survjax.SurvivalData.from_bounds(
        lower, upper, exact=exact, left_trunc=left, right_trunc=right
    )


def _wrapped(wrapper, family, p):
    if family == "exponential":
        baseline = survjax.Exponential(rate=p[0])
    elif family == "weibull":
        baseline = survjax.Weibull(shape=p[0], scale=p[1])
    elif family == "lognormal":
        baseline = survjax.LogNormal(meanlog=p[0], sdlog=p[1])
    else:
        baseline = survjax.LogLogistic(shape=p[0], scale=p[1])
    if wrapper == "ph":
        return survjax.ProportionalHazards(baseline=baseline, log_hazard_ratio=p[-1])
    return survjax.AcceleratedFailureTime(baseline=baseline, log_time_ratio=p[-1])


def _survival(family, t, p):
    if t == 0:
        return mp.mpf(1)
    if t == mp.inf:
        return mp.mpf(0)
    if family == "exponential":
        return mp.exp(-p[0] * t)
    if family == "weibull":
        return mp.exp(-((t / p[1]) ** p[0]))
    if family == "lognormal":
        return mp.erfc((mp.log(t) - p[0]) / (p[1] * mp.sqrt(2))) / 2
    return 1 / (1 + (t / p[1]) ** p[0])


def _density(family, t, p):
    if family == "exponential":
        return p[0] * mp.exp(-p[0] * t)
    if family == "weibull":
        power = (t / p[1]) ** p[0]
        return p[0] * power * mp.exp(-power) / t
    if family == "lognormal":
        z = (mp.log(t) - p[0]) / p[1]
        return mp.exp(-(z**2) / 2) / (t * p[1] * mp.sqrt(2 * mp.pi))
    power = (t / p[1]) ** p[0]
    return p[0] * power / (t * (1 + power) ** 2)


def _reference(wrapper, family, point):
    """mpmath derivatives of the defining wrapper law and conditional ratios."""
    values, gradients, hessians = [], [], []
    with mp.workdps(80):
        parameters = tuple(mp.mpf(float(p)) for p in point)
        size = len(parameters)
        basis = [tuple(int(i == j) for j in range(size)) for i in range(size)]
        for row in ROWS:

            def contribution(*p, row=row):
                exact, lower, upper, left, right = row
                multiplier = mp.exp(p[-1])

                def survival(time):
                    t = mp.mpf(time)
                    if wrapper == "ph":
                        return _survival(family, t, p) ** multiplier
                    return _survival(family, t / multiplier, p)

                normalizer = survival(left) - survival(right)
                if exact:
                    t = mp.mpf(lower)
                    if wrapper == "ph":
                        # f_PH = exp(eta)*f0*S0**(exp(eta)-1), spec 4.4.
                        density = multiplier * _density(family, t, p)
                        density *= _survival(family, t, p) ** (multiplier - 1)
                    else:
                        # The density has the time-rescaling Jacobian, spec 4.4.
                        density = _density(family, t / multiplier, p) / multiplier
                    return mp.log(density / normalizer)
                mass = survival(max(lower, left)) - survival(min(upper, right))
                return mp.log(mass / normalizer)

            values.append(float(contribution(*parameters)))
            gradients.append(
                [float(mp.diff(contribution, parameters, d)) for d in basis]
            )
            hessians.append(
                [
                    [
                        float(
                            mp.diff(
                                contribution,
                                parameters,
                                tuple(a + b for a, b in zip(i, j, strict=True)),
                            )
                        )
                        for j in basis
                    ]
                    for i in basis
                ]
            )
    return np.asarray(values), np.asarray(gradients), np.asarray(hessians)


@pytest.mark.parametrize("wrapper", ("ph", "aft"))
@pytest.mark.parametrize("family", POINTS)
def test_mixed_wrapper_likelihood_and_joint_derivatives(wrapper, family, x64_enabled):
    """Spec 2/4.4/7.3: per-row baseline/eta Jacobians and all mixed Hessians."""
    data = _data()
    points = jnp.asarray(POINTS[family])
    reference = [_reference(wrapper, family, point) for point in points]
    expected = [np.stack([r[order] for r in reference]) for order in range(3)]
    tol = {"atol": ATOL[x64_enabled], "rtol": RTOL[x64_enabled]}

    def rows(p):
        return _wrapped(wrapper, family, p).log_likelihood(data)

    assert data.issues.ok
    np.testing.assert_array_equal(data.issues.uninformative[-3:], True)
    with jax.debug_nans(True):
        np.testing.assert_allclose(rows(points[0]), expected[0][0], **tol)
    transforms = [(lambda f: f, 0), (jax.jacfwd, 1), (jax.jacrev, 1)]
    for outer in (jax.jacfwd, jax.jacrev):
        for inner in (jax.jacfwd, jax.jacrev):
            transforms.append((lambda f, outer=outer, inner=inner: outer(inner(f)), 2))
    for transform, order in transforms:
        evaluate = transform(rows)
        eager = evaluate(points[0])
        mapped = jax.jit(jax.vmap(evaluate))(points)
        np.testing.assert_allclose(eager, expected[order][0], **tol)
        np.testing.assert_allclose(mapped, expected[order], **tol)
        # P(window|window)=1 independently of every baseline parameter and eta.
        np.testing.assert_array_equal(eager[-3:], 0)
        np.testing.assert_array_equal(mapped[:, -3:], 0)


@pytest.mark.parametrize("wrapper", ("ph", "aft"))
@pytest.mark.parametrize("family", POINTS)
def test_wrapper_censoring_order_two_finite_differences(wrapper, family, x64_enabled):
    """Spec 12.1: numerical parameter derivative check at a smooth interior point."""
    data = _data()

    def objective(p):
        return _wrapped(wrapper, family, p).log_likelihood(data).mean()

    check_grads(
        jax.jit(objective),
        (jnp.asarray(POINTS[family][0]),),
        order=2,
        modes=("fwd", "rev"),
        eps=LIKELIHOOD_GRAD_EPS[x64_enabled],
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
    )
