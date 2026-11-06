"""A11: ordinary-range PH parameter derivatives (spec 7.3, 12.1, 12.2; rev 41).

7.3: first and second parameter derivatives of the PH wrapper over built-in
baselines "on finite, valid, smooth interior cases, composed with `jit` and
`vmap`, in forward and reverse modes, including Hessians", subject to the
initial-release extreme-scaling exception; 12.1: "Ordinary-range
baseline/eta gradients and Hessians (including mixed terms), forward/reverse
and `jit`/`vmap` checks ... remain completion requirements". Every case here
is ordinary range (``|eta| <= 1.5``, times within a few scales); nothing in
the 7.3 extreme regime is differentiated.

Hand derivation, PH-Weibull, ``z = t/b0``, ``L = log z``, ``w = exp(eta) z**k``
(so ``d w/d b0 = -k w/b0``, ``d w/d k = w L``, ``d w/d eta = w``):

- ``log S = -w``: gradient ``(-w L, k w/b0, -w)`` in ``(k, b0, eta)``;
  Hessian ``kk = -w L**2``, ``kb = w (1 + k L)/b0``, ``ke = -w L``,
  ``bb = -k (k+1) w/b0**2``, ``be = k w/b0``, ``ee = -w``.
- ``log f = log k - log b0 + (k-1) L + eta - w``: gradient
  ``(1/k + L - w L, (k/b0)(w - 1), 1 - w)``; Hessian
  ``kk = -1/k**2 - w L**2``, ``kb = (w - 1)/b0 + k w L/b0``, ``ke = -w L``,
  ``bb = -(k/b0**2)((k+1) w - 1)``, ``be = k w/b0``, ``ee = -w``.

PH-Exponential, ``w = r exp(eta) t``: ``log S = -w`` has gradient
``(-w/r, -w)`` in ``(r, eta)`` and Hessian ``rr = 0``, ``re = -w/r``,
``ee = -w``; ``log f = log r + eta - w`` has gradient ``(1/r - w/r, 1 - w)``
and Hessian ``rr = -1/r**2``, ``re = -w/r``, ``ee = -w``.

Derivatives here are O(1) to O(10), so the 12.1 ordinary tolerances apply.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import X64, assert_close, b, f, ph
from jax.test_util import check_grads

import survjax

requires_x64 = pytest.mark.skipif(not X64, reason="second-order finite differences need float64")


def weibull_reference(kind, k, b0, eta, t):
    """Hand gradient (3,) and Hessian (3, 3) in ``(k, b0, eta)``; see the module docstring."""
    z = t / b0
    lz = np.log(z)
    w = np.exp(eta) * z**k
    if kind == "log_survival":
        g = np.array([-w * lz, k * w / b0, -w])
        kk, kb, ke = -w * lz**2, w * (1.0 + k * lz) / b0, -w * lz
        bb, be, ee = -k * (k + 1.0) * w / b0**2, k * w / b0, -w
    else:
        g = np.array([1.0 / k + lz - w * lz, (k / b0) * (w - 1.0), 1.0 - w])
        kk, kb, ke = -1.0 / k**2 - w * lz**2, (w - 1.0) / b0 + k * w * lz / b0, -w * lz
        bb, be, ee = -(k / b0**2) * ((k + 1.0) * w - 1.0), k * w / b0, -w
    h = np.array([[kk, kb, ke], [kb, bb, be], [ke, be, ee]])
    return g, h


def exponential_reference(kind, r, eta, t):
    """Hand gradient (2,) and Hessian (2, 2) in ``(r, eta)``."""
    w = r * np.exp(eta) * t
    if kind == "log_survival":
        g = np.array([-w / r, -w])
        h = np.array([[0.0, -w / r], [-w / r, -w]])
    else:
        g = np.array([1.0 / r - w / r, 1.0 - w])
        h = np.array([[-1.0 / r**2, -w / r], [-w / r, -w]])
    return g, h


def _weibull_fn(kind, t):
    def fn(x):
        return getattr(ph(survjax.Weibull(shape=x[0], scale=x[1]), x[2]), kind)(f(t))

    return fn


def _exponential_fn(kind, t):
    def fn(x):
        return getattr(ph(survjax.Exponential(rate=x[0]), x[1]), kind)(f(t))

    return fn


HESSIANS = {
    "rev-over-rev": lambda fn: jax.jacrev(jax.jacrev(fn)),
    "fwd-over-rev": lambda fn: jax.jacfwd(jax.jacrev(fn)),
    "fwd-over-fwd": lambda fn: jax.jacfwd(jax.jacfwd(fn)),
    "rev-over-fwd": lambda fn: jax.jacrev(jax.jacfwd(fn)),
}
GRADIENTS = {"jacrev": jax.jacrev, "jacfwd": jax.jacfwd}

WEIBULL_POINTS = [(0.7, 1.5, -0.5, 2.0), (1.0, 2.0, 0.0, 1.0), (2.5, 3.0, 0.8, 4.0), (1.5, 0.5, -1.5, 0.3)]
EXPONENTIAL_POINTS = [(0.5, -0.5, 2.0), (2.0, 0.8, 0.4), (1.0, 0.0, 1.0)]
KINDS = ("log_prob", "log_survival")


# --------------------------------------------------------------- direct methods


@pytest.mark.step("A11")
@pytest.mark.parametrize("mode", list(GRADIENTS))
@pytest.mark.parametrize("point", WEIBULL_POINTS, ids=str)
@pytest.mark.parametrize("kind", KINDS)
def test_ph_weibull_gradient_is_hand_derived(kind, point, mode):
    """spec 7.3, rev 41: PH-Weibull gradient in `(k, b0, eta)`, forward and reverse."""
    k, b0, eta, t = point
    got = GRADIENTS[mode](_weibull_fn(kind, t))(f([k, b0, eta]))
    assert_close(got, weibull_reference(kind, k, b0, eta, t)[0])


@pytest.mark.step("A11")
@pytest.mark.parametrize("mode", list(HESSIANS))
@pytest.mark.parametrize("point", WEIBULL_POINTS, ids=str)
@pytest.mark.parametrize("kind", KINDS)
def test_ph_weibull_hessian_with_mixed_terms_is_hand_derived(kind, point, mode):
    """spec 7.3/12.1, rev 41: full PH-Weibull Hessian including mixed baseline/`eta` terms, all four mode compositions."""
    k, b0, eta, t = point
    got = HESSIANS[mode](_weibull_fn(kind, t))(f([k, b0, eta]))
    assert_close(got, weibull_reference(kind, k, b0, eta, t)[1])


@pytest.mark.step("A11")
@pytest.mark.parametrize("mode", list(GRADIENTS))
@pytest.mark.parametrize("point", EXPONENTIAL_POINTS, ids=str)
@pytest.mark.parametrize("kind", KINDS)
def test_ph_exponential_gradient_is_hand_derived(kind, point, mode):
    """spec 7.3, rev 41: PH-Exponential gradient in `(r, eta)`, forward and reverse."""
    r, eta, t = point
    got = GRADIENTS[mode](_exponential_fn(kind, t))(f([r, eta]))
    assert_close(got, exponential_reference(kind, r, eta, t)[0])


@pytest.mark.step("A11")
@pytest.mark.parametrize("mode", list(HESSIANS))
@pytest.mark.parametrize("point", EXPONENTIAL_POINTS, ids=str)
@pytest.mark.parametrize("kind", KINDS)
def test_ph_exponential_hessian_with_mixed_terms_is_hand_derived(kind, point, mode):
    """spec 7.3/12.1, rev 41: PH-Exponential Hessian including the mixed `r`/`eta` term."""
    r, eta, t = point
    got = HESSIANS[mode](_exponential_fn(kind, t))(f([r, eta]))
    assert_close(got, exponential_reference(kind, r, eta, t)[1])


# --------------------------------------------------------------- likelihood


def _likelihood_data():
    # Exact at t=2 and right-censored at t=1 (untruncated, Slice A modes).
    return survjax.SurvivalData(f([2.0, 1.0]), b([True, False]))


@pytest.mark.step("A11")
@pytest.mark.parametrize("mode", list(HESSIANS))
def test_ph_weibull_likelihood_hessian_is_hand_derived(mode):
    """spec 7.3, rev 41: summed likelihood of an exact (`t=2`) and a right-censored (`t=1`) row, shared `(k, b0, eta)`.

    The Hessian is the exact row's ``log f`` Hessian plus the censored row's ``log S`` Hessian.
    """
    k, b0, eta = 1.5, 2.0, 0.3
    data = _likelihood_data()

    def loss(x):
        return jnp.sum(ph(survjax.Weibull(shape=x[0], scale=x[1]), x[2]).log_likelihood(data))

    got = HESSIANS[mode](loss)(f([k, b0, eta]))
    want = weibull_reference("log_prob", k, b0, eta, 2.0)[1] + weibull_reference("log_survival", k, b0, eta, 1.0)[1]
    assert_close(got, want)


@pytest.mark.step("A11")
@pytest.mark.parametrize("mode", list(GRADIENTS))
def test_ph_weibull_likelihood_gradient_is_hand_derived(mode):
    """spec 7.3, rev 41: the same likelihood's gradient, forward and reverse."""
    k, b0, eta = 1.5, 2.0, 0.3
    data = _likelihood_data()

    def loss(x):
        return jnp.sum(ph(survjax.Weibull(shape=x[0], scale=x[1]), x[2]).log_likelihood(data))

    got = GRADIENTS[mode](loss)(f([k, b0, eta]))
    want = weibull_reference("log_prob", k, b0, eta, 2.0)[0] + weibull_reference("log_survival", k, b0, eta, 1.0)[0]
    assert_close(got, want)


@pytest.mark.step("A11")
@pytest.mark.parametrize("mode", list(HESSIANS))
def test_ph_exponential_likelihood_hessian_is_hand_derived(mode):
    """spec 7.3, rev 41: PH-Exponential summed likelihood Hessian in `(r, eta)`."""
    r, eta = 0.8, -0.4
    data = _likelihood_data()

    def loss(x):
        return jnp.sum(ph(survjax.Exponential(rate=x[0]), x[1]).log_likelihood(data))

    got = HESSIANS[mode](loss)(f([r, eta]))
    want = exponential_reference("log_prob", r, eta, 2.0)[1] + exponential_reference("log_survival", r, eta, 1.0)[1]
    assert_close(got, want)


@pytest.mark.step("A11")
def test_per_row_predictor_hessian_is_diagonal():
    """spec 7.3, rev 41: with shared baseline and per-row `eta`, `d2/deta_i deta_j` is diagonal: `-w_i` on each row.

    Exact row ``log f`` and censored row ``log S`` both have ``ee = -w``.
    """
    k, b0 = 1.5, 2.0
    eta = np.array([0.3, -0.7])
    t = np.array([2.0, 1.0])
    data = _likelihood_data()

    def loss(e):
        return jnp.sum(ph(survjax.Weibull(shape=f(k), scale=f(b0)), e).log_likelihood(data))

    got = jax.hessian(loss)(f(eta))
    w = np.exp(eta) * (t / b0) ** k
    assert_close(got, np.diag(-w))


# --------------------------------------------------------------- transformations


@pytest.mark.step("A11")
def test_hessian_under_jit_matches_hand_value():
    """spec 7.3, rev 41: Hessians compose with `jit`."""
    k, b0, eta, t = 2.5, 3.0, 0.8, 4.0
    got = jax.jit(jax.hessian(_weibull_fn("log_prob", t)))(f([k, b0, eta]))
    assert_close(got, weibull_reference("log_prob", k, b0, eta, t)[1])


@pytest.mark.step("A11")
def test_hessian_under_vmap_over_parameter_draws():
    """spec 7.3, rev 41: Hessians compose with `vmap` over parameter draws."""
    points = np.array([p[:3] for p in WEIBULL_POINTS])
    t = 1.7
    got = jax.vmap(jax.hessian(_weibull_fn("log_survival", t)))(f(points))
    want = np.stack([weibull_reference("log_survival", *p, t)[1] for p in points])
    assert_close(got, want)


@pytest.mark.step("A11")
def test_gradient_under_jit_of_vmap_over_predictor():
    """spec 7.3, rev 41: `jit(vmap(grad))` over `eta` draws: `d/deta log f = 1 - w`."""
    k, b0, t = 1.5, 2.0, 3.0
    eta = np.array([-1.0, 0.0, 0.5, 1.2])

    def lp(e):
        return ph(survjax.Weibull(shape=f(k), scale=f(b0)), e).log_prob(f(t))

    got = jax.jit(jax.vmap(jax.grad(lp)))(f(eta))
    assert_close(got, 1.0 - np.exp(eta) * (t / b0) ** k)


@pytest.mark.step("A11")
def test_gradient_with_draw_by_arm_broadcasting_is_hand_derived():
    """spec 4.4/7.3, rev 41: baseline `(S, 1)`, `eta` `(S, 2)`: the shared `b0` of a draw collects both arms.

    ``d/db0 sum_j log S_j = sum_j k w_j / b0`` (log S gradient above).
    """
    k = np.array([[1.5], [2.0]])
    b0 = np.array([[1.0], [2.0]])
    eta = np.array([[0.0, 0.5], [-0.5, 0.2]])
    t = 1.3

    def total(bb):
        return jnp.sum(ph(survjax.Weibull(shape=f(k), scale=bb), f(eta)).log_survival(f(t)))

    got = jax.grad(total)(f(b0))
    w = np.exp(eta) * (t / b0) ** k
    assert np.shape(got) == (2, 1)
    assert_close(got, np.sum(k * w / b0, axis=1, keepdims=True))


# --------------------------------------------------------------- check_grads


@pytest.mark.step("A11")
@requires_x64
@pytest.mark.parametrize("method", ["log_prob", "log_survival", "log_cdf", "log_hazard"])
def test_check_grads_order_two_ph_weibull(method):
    """spec 7.3/12.1, rev 41: `check_grads(order=2, modes=("fwd","rev"))` in `(k, b0, eta)` on ordinary-range interior cases."""
    t = f([0.5, 2.0, 4.0])
    check_grads(
        lambda k, b0, eta: getattr(ph(survjax.Weibull(shape=k, scale=b0), eta), method)(t),
        (f([0.7, 1.5, 3.0]), f([1.5, 2.0, 2.5]), f([-0.5, 0.0, 0.8])),
        order=2,
        modes=("fwd", "rev"),
    )


@pytest.mark.step("A11")
@requires_x64
@pytest.mark.parametrize("method", ["log_prob", "log_survival", "log_cdf", "log_hazard"])
def test_check_grads_order_two_ph_exponential(method):
    """spec 7.3/12.1, rev 41: `check_grads(order=2)` in `(r, eta)`."""
    t = f([0.5, 2.0])
    check_grads(
        lambda r, eta: getattr(ph(survjax.Exponential(rate=r), eta), method)(t),
        (f([0.7, 1.3]), f([0.4, -0.6])),
        order=2,
        modes=("fwd", "rev"),
    )


@pytest.mark.step("A11")
@requires_x64
def test_check_grads_order_two_ph_weibull_likelihood():
    """spec 7.3/12.1, rev 41: `check_grads(order=2)` of PH-Weibull likelihood contributions, per-row `eta`, shared baseline."""
    data = survjax.SurvivalData(f([0.5, 2.0, 3.0]), b([True, False, True]))
    check_grads(
        lambda k, b0, eta: ph(survjax.Weibull(shape=k, scale=b0), eta).log_likelihood(data),
        (f(1.4), f(2.0), f([-0.3, 0.2, 0.9])),
        order=2,
        modes=("fwd", "rev"),
    )


@pytest.mark.step("A11")
@requires_x64
def test_check_grads_order_two_ph_exponential_likelihood():
    """spec 7.3/12.1, rev 41: `check_grads(order=2)` of PH-Exponential likelihood contributions."""
    data = survjax.SurvivalData(f([0.5, 2.0, 3.0]), b([True, False, True]))
    check_grads(
        lambda r, eta: ph(survjax.Exponential(rate=r), eta).log_likelihood(data),
        (f(0.8), f([-0.3, 0.2, 0.9])),
        order=2,
        modes=("fwd", "rev"),
    )
