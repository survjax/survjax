"""D-022 log-CDF tails and transformations (spec 4.2/4.5/7/12.1)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from impl_tolerances import DERIVATIVE_ATOL, DERIVATIVE_RTOL, TAIL_ATOL, TAIL_RTOL
from jax.test_util import check_grads
from jax.typing import ArrayLike

import survjax


@survjax.register_family
@dataclass(frozen=True)
class Generic(survjax.SurvivalDistribution):
    rate: jax.Array = survjax.param()

    def log_survival(self, t: ArrayLike) -> jax.Array:
        return -self.rate * jnp.asarray(t)

    def reference_time(self) -> jax.Array:
        return jnp.reciprocal(self.rate)


@pytest.fixture(params=[Generic, survjax.Exponential, survjax.Weibull])
def family(request):
    return request.param


def _parameters(family):
    return jnp.array([1.5, 2.0] if family is survjax.Weibull else [1.0])


def _distribution(family, parameters):
    if family is survjax.Weibull:
        return family(shape=parameters[0], scale=parameters[1])
    return family(rate=parameters[0])


def _mp_log_cdf(family, time, *parameters):
    # Independent defining identities: H=r*t or (t/b)**k; F=1-exp(-H).
    time = mp.mpf(float(time))
    if family is survjax.Weibull:
        hazard = (time / parameters[1]) ** parameters[0]
    else:
        hazard = parameters[0] * time
    return mp.log1p(-mp.exp(-hazard))


def _mp_derivatives(family, time, parameters):
    with mp.workdps(100):
        point = tuple(mp.mpf(float(p)) for p in parameters)
        identity = np.eye(len(point), dtype=int)

        def reference(*p):
            return _mp_log_cdf(family, time, *p)

        gradient = [float(mp.diff(reference, point, tuple(row))) for row in identity]
        hessian = [
            [float(mp.diff(reference, point, tuple(a + b))) for b in identity]
            for a in identity
        ]
        return float(reference(*point)), np.array(gradient), np.array(hessian)


def test_upper_tail_values_match_mpmath(family, x64_enabled):
    """Spec 4.2/7.1/12.1: small negative log F must not round to zero."""
    parameters = _parameters(family)
    hazards = np.array([1.0, 20.1, 33.1, 54.6, 600.0 if x64_enabled else 70.0])
    times = jnp.asarray(
        2 * hazards ** (1 / 1.5) if family is survjax.Weibull else hazards
    )
    with mp.workdps(100):
        p = [mp.mpf(float(v)) for v in parameters]
        expected = [float(_mp_log_cdf(family, t, *p)) for t in np.asarray(times)]
    method = _distribution(family, parameters).log_cdf
    for actual in (method(times), jax.jit(method)(times), jax.vmap(method)(times)):
        assert actual.dtype == parameters.dtype
        assert np.all(np.asarray(actual) < 0)
        np.testing.assert_allclose(
            actual, expected, atol=TAIL_ATOL[x64_enabled], rtol=TAIL_RTOL[x64_enabled]
        )


@pytest.mark.parametrize(
    "hazard", [np.log(2) * 0.9999, np.log(2), np.log(2) * 1.0001, 33.1]
)
def test_switch_and_upper_tail_parameter_derivatives(family, hazard, x64_enabled):
    """Spec 7.3: both AD modes/Hessians match 100-digit mpmath derivatives."""
    parameters = _parameters(family)
    time = float(
        jnp.asarray(2 * hazard ** (1 / 1.5) if family is survjax.Weibull else hazard)
    )

    def value(p):
        return _distribution(family, p).log_cdf(time)

    expected_value, gradient, hessian = _mp_derivatives(family, time, parameters)
    for function, expected in (
        (value, expected_value),
        (jax.jacfwd(value), gradient),
        (jax.jacrev(value), gradient),
        (jax.hessian(value), hessian),
    ):
        for actual in (
            function(parameters),
            jax.jit(jax.vmap(function))(parameters[None, :])[0],
        ):
            np.testing.assert_allclose(
                actual,
                expected,
                atol=TAIL_ATOL[x64_enabled],
                rtol=TAIL_RTOL[x64_enabled],
            )


def test_interior_second_order_finite_differences(family, x64_enabled):
    """Spec 7.3/12.1: smooth interior parameter derivatives agree numerically."""
    check_grads(
        lambda p: _distribution(family, p).log_cdf(1.0),
        (_parameters(family),),
        order=2,
        modes=("fwd", "rev"),
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
    )


def test_mixed_boundaries_do_not_contaminate_derivatives(family, x64_enabled):
    """Spec 4.5/7.2: finite mixed entries retain their actual derivatives."""
    parameters = _parameters(family)
    times = jnp.array([0.0, 1.0, 20.0, jnp.inf])

    def values(p):
        return _distribution(family, p).log_cdf(times)

    references = [_mp_derivatives(family, float(t), parameters) for t in times[1:3]]
    with jax.debug_nans(True):
        actual = values(parameters)
        assert actual[0] == -jnp.inf and actual[-1] == 0
        # At t=0, log F=-inf has no derivative guarantee. At +inf it is constant.
        for transform in (jax.jacfwd, jax.jacrev):
            jacobian = transform(values)(parameters)
            np.testing.assert_array_equal(jacobian[-1], 0)
            for i in (1, 2):
                np.testing.assert_allclose(
                    jacobian[i],
                    references[i - 1][1],
                    atol=TAIL_ATOL[x64_enabled],
                    rtol=TAIL_RTOL[x64_enabled],
                )
        actual_hessian = jax.hessian(lambda p: values(p)[1:].sum())(parameters)
        expected_hessian = references[0][2] + references[1][2]
        np.testing.assert_allclose(
            actual_hessian,
            expected_hessian,
            atol=TAIL_ATOL[x64_enabled],
            rtol=TAIL_RTOL[x64_enabled],
        )


def test_nan_times_and_parameters_propagate(family):
    """Spec 4.5: NaN inputs stay NaN without value-based exceptions."""
    parameters = _parameters(family)
    assert np.isnan(_distribution(family, parameters).log_cdf(jnp.nan))
    invalid = parameters.at[0].set(jnp.nan)
    assert np.isnan(_distribution(family, invalid).log_cdf(20.0))
    if family is Generic:
        assert np.isnan(Generic(rate=-1.0).log_cdf(1.0))  # Positive log S is invalid.
        assert Generic(rate=jnp.inf).log_cdf(1.0) == 0  # log S=-inf.


def test_equal_length_grid_retains_upper_tails(family, x64_enabled):
    """Spec 5/12.1: K=N gives all pairs, with mpmath references per cell."""
    parameters = jnp.stack([_parameters(family), _parameters(family) * 1.1], axis=1)
    times = jnp.array([20.0, 30.0])
    with mp.workdps(100):
        expected = [
            [
                float(_mp_log_cdf(family, t, *(mp.mpf(float(p)) for p in column)))
                for column in parameters.T
            ]
            for t in np.asarray(times)
        ]
    actual = jax.jit(_distribution(family, parameters).log_cdf_grid)(times)
    assert actual.shape == (2, 2)
    np.testing.assert_allclose(
        actual, expected, atol=TAIL_ATOL[x64_enabled], rtol=TAIL_RTOL[x64_enabled]
    )
