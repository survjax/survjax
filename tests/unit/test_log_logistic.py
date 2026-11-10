"""Analytical and high-precision log-logistic checks, spec 4/5/7/12.1."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from impl_tolerances import ATOL, DERIVATIVE_ATOL, DERIVATIVE_RTOL, RTOL, TAIL_RTOL
from jax.test_util import check_grads

from survjax import LogLogistic
from survjax.distributions import LogLogistic as ExportedFamily

METHODS = (
    "log_survival",
    "survival",
    "log_cdf",
    "cumulative_hazard",
    "log_prob",
    "log_hazard",
    "hazard",
)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("log_survival", -np.log(5)),
        ("survival", 0.2),
        ("log_cdf", np.log(0.8)),
        ("cumulative_hazard", np.log(5)),
        ("log_prob", np.log(4 / 75)),
        ("log_hazard", np.log(4 / 15)),
        ("hazard", 4 / 15),
    ],
)
def test_values_grids_and_transforms(name, expected, x64_enabled):
    """Spec 4.1/5: beta=2, alpha=3, t=6 gives S=1/5, h=4/15, f=4/75."""
    tol = {"atol": ATOL[x64_enabled], "rtol": RTOL[x64_enabled]}
    dist = LogLogistic(shape=[[2.0], [2.0]], scale=[3.0, 3.0])
    assert ExportedFamily is LogLogistic and dist.batch_shape == (2, 2)

    def fn(d):
        return getattr(d, name)(6.0)

    for out in (
        fn(dist),
        jax.jit(fn)(dist),
        jax.vmap(fn)(dist),
        getattr(dist, name + "_grid")(jnp.array([6.0, 6.0])),
    ):
        np.testing.assert_allclose(out, expected, **tol)
    assert getattr(dist, name + "_grid")(jnp.array([6.0, 6.0, 6.0])).shape == (3, 2, 2)
    assert dist[0].batch_shape == (2,)
    assert jax.tree.map(lambda p: p[0], dist).batch_shape == (2,)


@pytest.mark.parametrize("shape", [0.5, 1.0, 2.0])
def test_boundaries(shape, x64_enabled):
    """Spec 4.5: analytic zero density/hazard; proper infinity and negative times."""
    tol = {"atol": ATOL[x64_enabled], "rtol": RTOL[x64_enabled]}
    dist = LogLogistic(shape=shape, scale=3.0)
    limit = np.inf if shape < 1 else -np.log(3) if shape == 1 else -np.inf
    expected = (
        [0, 0, -np.inf],
        [1, 1, 0],
        [-np.inf, -np.inf, 0],
        [0, 0, np.inf],
        [-np.inf, limit, -np.inf],
        [-np.inf, limit, -np.inf],
        [0, np.exp(limit), 0],
    )
    for name, values in zip(METHODS, expected, strict=True):
        out = jax.jit(getattr(dist, name))(jnp.array([-1.0, 0.0, jnp.inf, jnp.nan]))
        np.testing.assert_allclose(out[:3], values, **tol)
        assert np.isnan(out[3])
    for name, t, target in (
        ("survival", 0.0, 0.0),
        ("survival", jnp.inf, 0.0),
        ("log_cdf", jnp.inf, 0.0),
        ("hazard", jnp.inf, 0.0),
        ("log_prob", 0.0, -1 / 3),
        ("log_hazard", 0.0, -1 / 3),
    ):
        if t == 0 and name in ("log_prob", "log_hazard") and shape != 1:
            continue
        grad = jax.grad(
            lambda a, name=name, t=t: getattr(LogLogistic(shape=shape, scale=a), name)(
                t
            )
        )
        np.testing.assert_allclose(jax.jit(grad)(3.0), target, atol=ATOL[x64_enabled])


@pytest.mark.parametrize("name", ["log_survival", "log_cdf", "log_prob", "log_hazard"])
def test_tails_against_mpmath(name, x64_enabled):
    """Spec 4.1/7.2: mpmath log-space formulas, including unrepresentable probabilities."""
    times = [1e-30, 0.03, 1.0, 100.0, 1e30]
    expected = []
    with mp.workdps(160):
        for t in times:
            t, b, a = mp.mpf(t), mp.mpf(12), mp.mpf(3)
            z = b * mp.log(t / a)
            s, f = -mp.log1p(mp.exp(z)), -mp.log1p(mp.exp(-z))
            h = mp.log(b) - mp.log(t) + f
            values = dict(
                zip(
                    ("log_survival", "log_cdf", "log_hazard", "log_prob"),
                    (s, f, h, h + s),
                    strict=True,
                )
            )
            expected.append(float(values[name]))
    out = jax.jit(getattr(LogLogistic(shape=12.0, scale=3.0), name))(jnp.array(times))
    np.testing.assert_allclose(out, expected, atol=0, rtol=TAIL_RTOL[x64_enabled])


@pytest.mark.parametrize("name", METHODS)
def test_second_parameter_derivatives(name, x64_enabled):
    """Spec 7.3: second-order forward/reverse finite differences, both sides and at t=alpha."""
    for time in (0.8, 1.3, 2.0):

        def fn(p, time=time):
            return getattr(LogLogistic(shape=p[0], scale=p[1]), name)(time)

        check_grads(
            fn,
            (jnp.array([1.6, 1.3]),),
            order=2,
            modes=("fwd", "rev"),
            eps=1e-4 if x64_enabled else 1e-3,
            atol=DERIVATIVE_ATOL[x64_enabled],
            rtol=DERIVATIVE_RTOL[x64_enabled],
        )


def test_log_constructor_and_diagnostics(x64_enabled):
    """Spec 3.3/4.1/4.7: natural leaves, overflow diagnostics and identical chain rule."""
    tol = {"atol": ATOL[x64_enabled], "rtol": RTOL[x64_enabled]}
    logs = jnp.array([0.2, 0.7])

    d = LogLogistic.from_log(log_shape=logs[0], log_scale=logs[1])
    np.testing.assert_allclose(
        [d.shape, d.scale], np.exp(np.asarray(logs)), rtol=RTOL[x64_enabled]
    )
    assert d.shape.dtype == jnp.asarray(1.0).dtype
    assert d.reference_time() == d.scale

    def logged(p):
        return LogLogistic.from_log(log_shape=p[0], log_scale=p[1]).log_prob(2.0)

    def natural(p):
        return LogLogistic(shape=jnp.exp(p[0]), scale=jnp.exp(p[1])).log_prob(2.0)

    np.testing.assert_allclose(
        jax.hessian(logged)(logs),
        jax.hessian(natural)(logs),
        **tol,
    )
    invalid = LogLogistic(shape=[0.0, -1.0, jnp.nan, jnp.inf, 1.0], scale=2.0)
    np.testing.assert_array_equal(
        invalid.parameter_issues().invalid_entries, [1, 1, 1, 1, 0]
    )
    overflow = LogLogistic.from_log(log_shape=1000.0, log_scale=0.0)
    assert overflow.parameter_issues().invalid_entries
    assert np.isnan(LogLogistic(shape=-1.0, scale=2.0).log_prob(1.0))


def test_mixed_likelihood_debug_nans(x64_enabled):
    """Spec 7.2/12.1: exact/right rows; f(3)=1/6 and S(3)=1/2 at beta=2, alpha=3."""
    tol = {"atol": ATOL[x64_enabled], "rtol": RTOL[x64_enabled]}

    def rows(p):
        return LogLogistic(shape=p[0], scale=p[1]).log_likelihood(
            time=jnp.array([0.0, 3.0, 3.0]), event=jnp.array([False, True, False])
        )

    with jax.debug_nans(True):
        out = rows(jnp.array([2.0, 3.0]))
        gradient = jax.grad(lambda p: rows(p).sum())(jnp.array([2.0, 3.0]))
    np.testing.assert_allclose(
        out,
        [0.0, -np.log(6), -np.log(2)],
        **tol,
    )
    # log f at t=alpha: d/db=1/b, d/da=0; log S: d/db=0, d/da=b/(2a).
    np.testing.assert_allclose(gradient, [0.5, 1 / 3], **tol)
