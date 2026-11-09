"""Independent log-normal values, transformations and diagnostics (spec 4/7/12.1)."""

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
    TAIL_ATOL,
    TAIL_RTOL,
)
from jax.test_util import check_grads

import survjax

METHODS = (
    "log_survival",
    "log_cdf",
    "survival",
    "cumulative_hazard",
    "log_prob",
    "log_hazard",
    "hazard",
)


def _reference(name, time, mu, sigma):
    # Defining normal integral and density, independent of JAX special functions.
    z = (mp.log(time) - mu) / sigma
    survival = mp.erfc(z / mp.sqrt(2)) / 2
    density = mp.exp(-z * z / 2) / (time * sigma * mp.sqrt(2 * mp.pi))
    return {
        "log_survival": lambda: mp.log(survival),
        "log_cdf": lambda: mp.log(mp.erfc(-z / mp.sqrt(2)) / 2),
        "survival": lambda: survival,
        "cumulative_hazard": lambda: -mp.log(survival),
        "log_prob": lambda: mp.log(density),
        "log_hazard": lambda: mp.log(density / survival),
        "hazard": lambda: density / survival,
    }[name]()


@pytest.mark.parametrize("name", METHODS)
def test_tails_and_transformations(name, x64_enabled):
    """Spec 4.1/12.1: relative tail accuracy over standardized x in [-10,10]/[-7,7]."""
    limit = 10 if x64_enabled else 7
    times = jnp.exp(jnp.linspace(-limit, limit, 41))
    dist = survjax.LogNormal(meanlog=0, sdlog=1)
    with mp.workdps(100):
        expected = [float(_reference(name, float(t), 0, 1)) for t in times]
    method = getattr(dist, name)
    for actual in (method(times), jax.jit(method)(times), jax.vmap(method)(times)):
        np.testing.assert_allclose(
            actual, expected, atol=TAIL_ATOL[x64_enabled], rtol=TAIL_RTOL[x64_enabled]
        )


@pytest.mark.parametrize("name", METHODS)
def test_parameter_derivatives(name, x64_enabled):
    """Spec 7.3: mpmath first/second derivatives in both tails and at the switch."""
    p = jnp.array([0.2, 1.3])
    for z in (-6.0, -1e-6, 0.0, 1e-6, 2 - 1e-6, 2 + 1e-6, 4.0, 6.0):
        time = float(jnp.exp(p[0] + p[1] * z))
        with mp.workdps(100):
            point = tuple(mp.mpf(float(v)) for v in p)

            def reference(mu, sigma, time=time):
                return _reference(name, time, mu, sigma)

            gradient = [float(mp.diff(reference, point, d)) for d in ((1, 0), (0, 1))]
            orders = ((1, 0), (0, 1))
            hessian = [
                [
                    float(
                        mp.diff(
                            reference,
                            point,
                            tuple(a + b for a, b in zip(i, j, strict=True)),
                        )
                    )
                    for j in orders
                ]
                for i in orders
            ]

        def value(p, time=time):
            return getattr(survjax.LogNormal(meanlog=p[0], sdlog=p[1]), name)(time)

        for function, expected in (
            (jax.jacfwd(value), gradient),
            (jax.jacrev(value), gradient),
            (jax.hessian(value), hessian),
        ):
            np.testing.assert_allclose(
                jax.jit(jax.vmap(function))(p[None])[0],
                expected,
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )
    check_grads(
        lambda p: getattr(survjax.LogNormal(meanlog=p[0], sdlog=p[1]), name)(1.7),
        (p,),
        order=2,
        modes=("fwd", "rev"),
        eps=LIKELIHOOD_GRAD_EPS[x64_enabled],
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
    )


def test_boundaries_and_zero_gradients():
    """Spec 4.5/7.2: analytic limits and mixed boundary rows have safe gradients."""
    expected = (
        [0, 0, -np.inf],
        [-np.inf, -np.inf, 0],
        [1, 1, 0],
        [0, 0, np.inf],
        [-np.inf] * 3,
        [-np.inf] * 3,
        [0] * 3,
    )
    times = jnp.array([-1.0, 0.0, jnp.inf])
    for name, limits in zip(METHODS, expected, strict=True):

        def values(p, name=name):
            return getattr(survjax.LogNormal(meanlog=p[0], sdlog=p[1]), name)(times)

        p = jnp.array([0.2, 1.3])
        np.testing.assert_array_equal(jax.jit(values)(p), limits)
        np.testing.assert_array_equal(jax.jacrev(values)(p)[np.isfinite(limits)], 0)
        assert np.isnan(getattr(survjax.LogNormal(meanlog=0, sdlog=1), name)(jnp.nan))

    def mixed(mu):
        return (
            survjax.LogNormal(meanlog=mu, sdlog=1.3)
            .hazard(jnp.array([0.0, 1.0, jnp.inf]))
            .sum()
        )

    with jax.debug_nans(True):
        assert np.isfinite(jax.hessian(mixed)(0.2))

        def extreme(mu):
            return survjax.LogNormal(meanlog=mu, sdlog=1e-20).log_hazard(2.0)

        assert np.isfinite(jax.grad(extreme)(0.0))
    assert np.all(
        np.isfinite(survjax.LogNormal(meanlog=[-1e30, 1e30], sdlog=1).reference_time())
    )


def test_registration_grid_and_log_constructor(x64_enabled):
    """Spec 3.3/4.1/5: natural leaves, broadcasting, indexing, vmap and time-first grids."""
    dist = survjax.LogNormal.from_log(meanlog=[0.0, 1.0], log_sdlog=0.0)
    assert dist.batch_shape == (2,)
    np.testing.assert_array_equal(dist.sdlog, [1, 1])
    assert all(leaf.dtype == dist.meanlog.dtype for leaf in jax.tree.leaves(dist))
    assert dist[0].batch_shape == ()
    for name in METHODS:
        grid = getattr(dist, name + "_grid")(jnp.array([1.0, 2.0]))
        expected = jax.vmap(
            lambda d, name=name: getattr(d, name)(jnp.array([1.0, 2.0]))
        )(dist).T
        assert grid.shape == (2, 2)
        np.testing.assert_allclose(
            grid, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )

    def value(log_sigma, logged):
        d = (
            survjax.LogNormal.from_log(meanlog=0.2, log_sdlog=log_sigma)
            if logged
            else survjax.LogNormal(meanlog=0.2, sdlog=jnp.exp(log_sigma))
        )
        return d.log_prob(2.0)

    assert jax.grad(lambda s: value(s, True))(0.3) == jax.grad(
        lambda s: value(s, False)
    )(0.3)
    with pytest.raises(TypeError):
        survjax.LogNormal(meanlog=True, sdlog=1)
    with pytest.raises(TypeError):
        dist.log_survival(jnp.ones(3))


def test_diagnostics_and_likelihood(x64_enabled):
    """Spec 4.7/7.1: constraints select NaN likelihood rows; valid mixed rows are safe."""
    dist = survjax.LogNormal(meanlog=[0, jnp.nan, 0, 0], sdlog=[1, 1, -1, jnp.inf])
    np.testing.assert_array_equal(
        dist.parameter_issues().invalid_entries, [False, True, True, True]
    )
    assert np.isinf(survjax.LogNormal.from_log(meanlog=0, log_sdlog=1000).sdlog)
    result = dist.log_likelihood(time=jnp.ones(4), event=True)
    assert np.isnan(result[1:]).all()
    valid = survjax.LogNormal(meanlog=0.2, sdlog=1.3)
    with jax.debug_nans(True):
        result = valid.log_likelihood(
            time=jnp.array([1.0, 2.0]), event=jnp.array([True, False])
        )
    with mp.workdps(100):
        expected = [
            float(_reference(n, t, float(valid.meanlog), float(valid.sdlog)))
            for n, t in (("log_prob", 1), ("log_survival", 2))
        ]
    np.testing.assert_allclose(
        result, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
