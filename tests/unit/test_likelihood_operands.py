"""A6.4a: safe operands for selected direct-method contributions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

import survjax


def selected_contributions(parameters, kind, middle_exact):
    if kind == "rate":
        dist = survjax.Exponential(rate=parameters)
    elif kind == "shape":
        dist = survjax.Weibull(shape=parameters, scale=2.0)
    else:
        dist = survjax.Weibull(shape=2.0, scale=parameters)
    time = jnp.array([1.0, 1.0, 3.0])
    exact = jnp.array([True, middle_exact, False])
    reference = jax.lax.stop_gradient(dist.reference_time())
    density = dist.log_prob(jnp.where(exact, time, reference))
    survival = dist.log_survival(jnp.where(exact, jnp.zeros_like(time), time))
    values = jnp.where(exact, density, survival)
    return jnp.where(dist.parameter_issues().invalid_entries, jnp.nan, values)


@pytest.mark.parametrize("kind", ["rate", "shape", "scale"])
@pytest.mark.parametrize(
    "middle_exact", [True, False], ids=["active-density", "inactive-density"]
)
@pytest.mark.parametrize("derivative", [jax.jacfwd, jax.jacrev], ids=["fwd", "rev"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_zero_parameter_keeps_other_rows_jacobian_finite(
    kind, middle_exact, derivative, compiled, x64_enabled
):
    """Spec 7.1–7.3: valid rows retain their derivatives beside a zero parameter."""
    parameters = jnp.array([0.5, 0.0, 2.0] if kind == "rate" else [2.0, 0.0, 2.0])

    def evaluate(parameters):
        return selected_contributions(parameters, kind, middle_exact)

    jacobian = derivative(evaluate)
    if compiled:
        jacobian = jax.jit(jacobian)
    actual = np.asarray(jacobian(parameters))[[0, 2]]
    # Exact exponential: 1/r-t; right: -t. Weibull with z=t/b:
    # d_k log f=1/k+(1-z**k)*log z, d_k log S=-z**k*log z;
    # d_b log f=k*(z**k-1)/b, d_b log S=k*z**k/b.
    diagonals = {
        "rate": (1.0, -3.0),
        "shape": (0.5 - 0.75 * np.log(2), -2.25 * np.log(1.5)),
        "scale": (-0.75, 2.25),
    }
    first, last = diagonals[kind]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual,
        [[first, 0, 0], [0, 0, last]],
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    # These entries must be exactly zero, including the invalid parameter column.
    np.testing.assert_array_equal(actual[:, 1], [0.0, 0.0])


def test_exponential_zero_rate_preserves_direct_formula_values():
    """Spec 4.5: log f=log 0-0*t=-inf; log S=-0*t=0, even at zero time."""
    dist = survjax.Exponential(rate=0.0)
    for time in (0.0, 1.0):
        assert dist.log_prob(time) == -jnp.inf
        assert dist.log_survival(time) == 0


@pytest.mark.parametrize(
    "shape,scale,log_survival,log_hazard",
    [
        (0.0, 2.0, -1.0, -np.inf),
        (2.0, 0.0, -np.inf, np.inf),
        (1.0, 0.0, -np.inf, np.nan),
        (0.5, 0.0, -np.inf, np.nan),
        (-1.0, 0.0, 0.0, np.nan),
        (0.0, 0.0, np.nan, np.nan),
    ],
)
def test_weibull_zero_parameters_preserve_interior_formula_values(
    shape, scale, log_survival, log_hazard
):
    """Spec 4.5: retain exp(k*(log t-log b)) and log-hazard extended values."""
    # At t=1, b=0 gives log(t/b)=inf: H is inf/0/NaN for k>0/<0/=0.
    # log h is inf for k>1, NaN for k<=1 (log k or inf cancellation).
    # With k=0,b=2, H=1 and log h=-inf. Thus log f=log h-H below.
    dist = survjax.Weibull(shape=shape, scale=scale)
    np.testing.assert_array_equal(dist.log_survival(1.0), log_survival)
    np.testing.assert_array_equal(dist.log_hazard(1.0), log_hazard)
    expected_density = -np.inf if shape == 0.0 and scale == 2.0 else np.nan
    np.testing.assert_array_equal(dist.log_prob(1.0), expected_density)


@pytest.mark.parametrize(
    "shape,expected", [(0.0, np.inf), (0.5, np.inf), (1.0, np.inf), (2.0, -np.inf)]
)
def test_zero_scale_retains_family_zero_time_density_limits(shape, expected):
    """Spec 4.5: at t=0 the family supplies hazard limits, not interior formulas."""
    dist = survjax.Weibull(shape=shape, scale=0.0)
    np.testing.assert_array_equal(dist.log_prob(0.0), expected)
    assert dist.log_survival(0.0) == 0
    assert dist.log_prob(jnp.inf) == -jnp.inf


def test_valid_density_derivatives_and_transformations(x64_enabled):
    """Spec 7.3: retain analytical gradients/Hessians on valid smooth interiors."""

    def exponential(rate):
        return survjax.Exponential(rate=rate).log_prob(1.0)

    def weibull(parameters):
        return survjax.Weibull(shape=parameters[0], scale=parameters[1]).log_prob(1.0)

    # log f=log r-r: derivatives 1/r-1 and -1/r**2 at r=2.
    np.testing.assert_allclose(
        jax.grad(exponential)(2.0), -0.5, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    np.testing.assert_allclose(
        jax.hessian(exponential)(2.0),
        -0.25,
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    point = jnp.array([2.0, 2.0])
    log2 = np.log(2)
    # Differentiate log k-log b+(k-1)*log(t/b)-(t/b)**k at t=1,k=b=2.
    expected_gradient = [0.5 - 0.75 * log2, -0.75]
    expected_hessian = [
        [-0.25 - 0.25 * log2**2, -0.375 - 0.25 * log2],
        [-0.375 - 0.25 * log2, 0.125],
    ]
    for transform in (lambda f: f, jax.jit):
        np.testing.assert_allclose(
            transform(jax.grad(weibull))(point),
            expected_gradient,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )
        np.testing.assert_allclose(
            transform(jax.hessian(weibull))(point),
            expected_hessian,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )
    with jax.debug_nans(True):
        eager = weibull(point)
    np.testing.assert_allclose(
        eager, -log2 - 0.25, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    np.testing.assert_array_equal(jax.jit(weibull)(point), eager)
    np.testing.assert_array_equal(
        jax.vmap(weibull)(jnp.stack([point, point])), [eager, eager]
    )
