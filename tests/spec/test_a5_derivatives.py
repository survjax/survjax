"""A5: derivatives and transformations of the built-in families (spec 4.1, 7.3, 12.1).

Hand-derived parameter derivatives on interior cases (``z = t/b``):

Exponential: ``log f = log r - r t``, ``d/dr = 1/r - t``, ``d2/dr2 = -1/r**2``;
``log S = -r t``, ``d/dr = -t``.

Weibull: ``log f = log k - log b + (k-1) log z - z**k``;
``d/dk = 1/k + log z - z**k log z``; ``d/db = (k/b)(z**k - 1)``;
``d2/dk2 = -1/k**2 - z**k (log z)**2``;
``d2/db2 = -(k/b**2)((k+1) z**k - 1)``;
``log S = -z**k``: ``d/dk = -z**k log z``, ``d/db = k z**k / b``.
"""

import jax
import numpy as np
import pytest
from helpers import X64, assert_close, f
from jax.test_util import check_grads

import survjax

requires_x64 = pytest.mark.skipif(
    not X64, reason="second-order finite differences need float64"
)

DIRECT_METHODS = (
    "log_survival",
    "survival",
    "log_cdf",
    "cumulative_hazard",
    "log_prob",
    "log_hazard",
    "hazard",
)

# --------------------------------------------------------------- eager / jit / vmap


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_weibull_methods_agree_under_jit(method):
    """spec 12.1, rev 21: eager and `jit` agree for every direct method."""
    dist = survjax.Weibull(shape=f([0.5, 1.0, 2.5]), scale=f([1.0, 2.0, 3.0]))
    t = f([[0.3], [1.0], [4.0]])
    jitted = jax.jit(lambda d, s: getattr(d, method)(s))(dist, t)
    assert_close(jitted, getattr(dist, method)(t))


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_exponential_methods_agree_under_jit(method):
    """spec 12.1, rev 21: eager and `jit` agree for every direct method."""
    dist = survjax.Exponential(rate=f([0.5, 2.0]))
    t = f([[0.3], [1.0], [4.0]])
    jitted = jax.jit(lambda d, s: getattr(d, method)(s))(dist, t)
    assert_close(jitted, getattr(dist, method)(t))


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_weibull_methods_agree_under_vmap_over_distribution_and_time(method):
    """spec 4.1/12.1, rev 21: `vmap` over paired distribution entries and times agrees with the paired batched call."""
    dist = survjax.Weibull(shape=f([0.5, 1.0, 2.5]), scale=f([1.0, 2.0, 3.0]))
    t = f([0.3, 1.0, 4.0])
    mapped = jax.vmap(lambda d, s: getattr(d, method)(s))(dist, t)
    assert_close(mapped, getattr(dist, method)(t))


# --------------------------------------------------------------- first derivatives


@pytest.mark.step("A5")
def test_exponential_log_prob_gradient():
    """spec 7.3, rev 21: `d/dr log f = 1/r - t`."""
    r, t = np.array([0.5, 1.0, 2.0]), np.array([2.0, 0.5, 1.5])
    g = jax.grad(lambda rr: survjax.Exponential(rate=rr).log_prob(f(t)).sum())(f(r))
    assert_close(g, 1.0 / r - t)


@pytest.mark.step("A5")
def test_exponential_log_survival_gradient():
    """spec 7.3, rev 21: `d/dr log S = -t`."""
    r, t = np.array([0.5, 1.0, 2.0]), np.array([2.0, 0.5, 1.5])
    g = jax.grad(lambda rr: survjax.Exponential(rate=rr).log_survival(f(t)).sum())(f(r))
    assert_close(g, -t)


@pytest.mark.step("A5")
def test_weibull_log_prob_gradient():
    """spec 7.3, rev 21: `d/dk log f = 1/k + log z - z**k log z`, `d/db log f = (k/b)(z**k - 1)`."""
    k, b, t = (
        np.array([0.7, 1.0, 2.5]),
        np.array([1.5, 2.0, 3.0]),
        np.array([0.5, 3.0, 2.0]),
    )
    gk, gb = jax.grad(
        lambda kk, bb: survjax.Weibull(shape=kk, scale=bb).log_prob(f(t)).sum(),
        argnums=(0, 1),
    )(f(k), f(b))
    z = t / b
    assert_close(gk, 1.0 / k + np.log(z) - z**k * np.log(z))
    assert_close(gb, (k / b) * (z**k - 1.0))


@pytest.mark.step("A5")
def test_weibull_log_survival_gradient():
    """spec 7.3, rev 21: `d/dk log S = -z**k log z`, `d/db log S = k z**k / b`."""
    k, b, t = (
        np.array([0.7, 1.0, 2.5]),
        np.array([1.5, 2.0, 3.0]),
        np.array([0.5, 3.0, 2.0]),
    )
    gk, gb = jax.grad(
        lambda kk, bb: survjax.Weibull(shape=kk, scale=bb).log_survival(f(t)).sum(),
        argnums=(0, 1),
    )(f(k), f(b))
    z = t / b
    assert_close(gk, -(z**k) * np.log(z))
    assert_close(gb, k * z**k / b)


@pytest.mark.step("A5")
def test_weibull_forward_and_reverse_gradients_agree():
    """spec 7.3, rev 21: first parameter derivatives "in forward and reverse modes" agree."""
    t = f([0.5, 3.0, 2.0])

    def fn(k):
        return survjax.Weibull(shape=k, scale=f(2.0)).log_prob(t).sum()

    k = f([0.7, 1.0, 2.5])
    assert_close(jax.jacfwd(fn)(k), jax.jacrev(fn)(k))


@pytest.mark.step("A5")
def test_weibull_gradient_under_jit_and_vmap():
    """spec 7.3, rev 21: parameter gradients compose with `jit` and `vmap`; `d/db log f = (k/b)(z**k - 1)`."""
    k, t = 2.0, 1.5
    b = np.array([1.0, 2.0, 3.0])
    g = jax.jit(
        jax.vmap(
            jax.grad(lambda bb: survjax.Weibull(shape=f(k), scale=bb).log_prob(f(t)))
        )
    )(f(b))
    z = t / b
    assert_close(g, (k / b) * (z**k - 1.0))


# --------------------------------------------------------------- second derivatives


@pytest.mark.step("A5")
def test_exponential_log_prob_hessian():
    """spec 7.3, rev 21: second parameter derivatives; `d2/dr2 log f = -1/r**2`."""
    h = jax.hessian(lambda r: survjax.Exponential(rate=r).log_prob(f(1.5)))(f(0.8))
    assert_close(h, -1.0 / 0.8**2)


@pytest.mark.step("A5")
def test_weibull_log_prob_hessian_in_shape():
    """spec 7.3, rev 21: `d2/dk2 log f = -1/k**2 - z**k (log z)**2` (`k=1.5`, `b=2`, `t=3`)."""
    k, b, t = 1.5, 2.0, 3.0
    h = jax.hessian(lambda kk: survjax.Weibull(shape=kk, scale=f(b)).log_prob(f(t)))(
        f(k)
    )
    z = t / b
    assert_close(h, -1.0 / k**2 - z**k * np.log(z) ** 2)


@pytest.mark.step("A5")
def test_weibull_log_prob_hessian_in_scale():
    """spec 7.3, rev 21: `d2/db2 log f = -(k/b**2)((k+1) z**k - 1)` (`k=1.5`, `b=2`, `t=3`)."""
    k, b, t = 1.5, 2.0, 3.0
    h = jax.hessian(lambda bb: survjax.Weibull(shape=f(k), scale=bb).log_prob(f(t)))(
        f(b)
    )
    z = t / b
    assert_close(h, -(k / b**2) * ((k + 1.0) * z**k - 1.0))


@pytest.mark.step("A5")
@requires_x64
@pytest.mark.parametrize(
    "method", ["log_prob", "log_survival", "log_cdf", "log_hazard"]
)
def test_check_grads_order_two_exponential(method):
    """spec 7.3/12.1, rev 21: `check_grads(order=2, modes=("fwd","rev"))` on an interior case."""
    t = f([0.5, 2.0])
    check_grads(
        lambda r: getattr(survjax.Exponential(rate=r), method)(t),
        (f([0.7, 1.3]),),
        order=2,
        modes=("fwd", "rev"),
    )


@pytest.mark.step("A5")
@requires_x64
@pytest.mark.parametrize(
    "method", ["log_prob", "log_survival", "log_cdf", "log_hazard"]
)
def test_check_grads_order_two_weibull(method):
    """spec 7.3/12.1, rev 21: `check_grads(order=2, modes=("fwd","rev"))` on interior Weibull cases (`k<1`, `k>1`)."""
    t = f([0.5, 2.0, 4.0])
    check_grads(
        lambda k, b: getattr(survjax.Weibull(shape=k, scale=b), method)(t),
        (f([0.7, 1.5, 3.0]), f([1.5, 2.0, 2.5])),
        order=2,
        modes=("fwd", "rev"),
    )


@pytest.mark.step("A5")
@requires_x64
def test_check_grads_order_two_weibull_at_shape_one():
    """spec 7.3/12.1, rev 21: `k=1` is an interior point of the Weibull parameter space; derivatives are smooth there."""
    t = f([0.5, 2.0])
    check_grads(
        lambda k, b: survjax.Weibull(shape=k, scale=b).log_prob(t),
        (f([1.0, 1.0]), f([1.5, 2.0])),
        order=2,
        modes=("fwd", "rev"),
    )
