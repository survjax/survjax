"""A9: first and second parameter derivatives of the likelihood under transformations (spec 7.2, 7.3, 12.1; rev 34).

Spec 7.3: "Built-in likelihoods require first and second parameter
derivatives on finite, valid, smooth interior cases, composed with `jit` and
`vmap`, in forward and reverse modes, including Hessians." 12.1
(Transformations): "eager/`jit`/`vmap` agreement;
`jax.test_util.check_grads(order=2, modes=("fwd","rev"))` on interior cases
for families and wrappers; reference-input gradient isolation". Wrappers are
Slice D.

Hand-derived contributions (spec 2, 4.1; untruncated, ``z = t/b``,
``L = log z``):

Exponential: exact ``log r - r t``: ``d/dr = 1/r - t``, ``d2/dr2 = -1/r**2``;
right ``-r c``: ``d/dr = -c``, ``d2/dr2 = 0``. Summed with a shared rate,
``d/dr = d/r - T`` and ``d2/dr2 = -d/r**2`` (``d`` events, ``T`` total time).

Weibull exact ``log k - log b + (k-1) L - z**k``:
``d/dk = 1/k + L - z**k L``; ``d/db = (k/b)(z**k - 1)``;
``d2/dk2 = -1/k**2 - z**k L**2``;
``d2/dkdb = (z**k - 1)/b + k z**k L / b``;
``d2/db2 = -(k/b**2)((k+1) z**k - 1)``.
Weibull right ``-z**k``: ``d/dk = -z**k L``; ``d/db = k z**k / b``;
``d2/dk2 = -z**k L**2``; ``d2/dkdb = z**k/b + k z**k L / b``;
``d2/db2 = -k(k+1) z**k / b**2``.

``check_grads`` uses second-order finite differences, which need float64; as at
A4/A5, float32 runs the hand-derived first and second derivatives instead.
Hand-derived Hessians use the 12.1 tolerances: the entries here are of order
``1e-2`` to ``10`` and come from a few elementary operations.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import X64, assert_close, b, f
from jax.test_util import check_grads

import survjax

requires_x64 = pytest.mark.skipif(
    not X64, reason="second-order finite differences need float64"
)

TIME = [0.5, 1.2, 2.0, 3.5, 0.8]
EVENT = [True, False, True, False, True]


def _data():
    return survjax.SurvivalData(f(TIME), b(EVENT))


def _exp_sum(rate):
    return jnp.sum(survjax.log_likelihood(survjax.Exponential(rate=rate), _data()))


def _weibull_sum(shape, scale):
    dist = survjax.Weibull(shape=shape, scale=scale)
    return jnp.sum(survjax.log_likelihood(dist, _data()))


def _weibull_hand(k, b_, t, exact):
    """Gradient ``(d/dk, d/db)`` and Hessian of one Weibull contribution, hand-derived (module docstring)."""
    z = t / b_
    big_l = np.log(z)
    zk = z**k
    if exact:
        grad = np.array([1.0 / k + big_l - zk * big_l, (k / b_) * (zk - 1.0)])
        hkk = -1.0 / k**2 - zk * big_l**2
        hkb = (zk - 1.0) / b_ + k * zk * big_l / b_
        hbb = -(k / b_**2) * ((k + 1.0) * zk - 1.0)
    else:
        grad = np.array([-zk * big_l, k * zk / b_])
        hkk = -zk * big_l**2
        hkb = zk / b_ + k * zk * big_l / b_
        hbb = -k * (k + 1.0) * zk / b_**2
    return grad, np.array([[hkk, hkb], [hkb, hbb]])


def _weibull_sum_hand(k, b_):
    grad, hess = np.zeros(2), np.zeros((2, 2))
    for t, e in zip(TIME, EVENT, strict=True):
        g, h = _weibull_hand(k, b_, t, e)
        grad, hess = grad + g, hess + h
    return grad, hess


# --------------------------------------------------- hand-derived second order


@pytest.mark.step("A9")
def test_exponential_shared_rate_hessian_is_hand_derived():
    """spec 7.3, rev 28: Hessian of the summed exponential likelihood, `-d/r**2`.

    ``r = 0.7``, ``d = 3`` events: ``-3/0.49``.
    """
    hess = jax.hessian(_exp_sum)(jnp.asarray(f(0.7)))
    assert_close(np.asarray(hess), -3.0 / 0.49)


@pytest.mark.step("A9")
def test_exponential_per_row_rate_hessian_is_diagonal_and_hand_derived():
    """spec 5 and 7.3, rev 28: per-row rates give a diagonal Hessian `-event_i/r_i**2`; rows do not couple."""
    rates = f([0.5, 0.7, 1.1, 1.6, 2.0])
    hess = jax.hessian(_exp_sum)(jnp.asarray(rates))
    expected = np.diag(-np.asarray(EVENT, dtype=float) / np.asarray(rates, dtype=float) ** 2)
    assert_close(np.asarray(hess), expected)


@pytest.mark.step("A9")
def test_weibull_shared_parameter_gradient_is_hand_derived():
    """spec 7.3, rev 28: gradient of the summed Weibull likelihood in `(k, b)`, mixed exact/right rows."""
    grad = jax.grad(_weibull_sum, argnums=(0, 1))(jnp.asarray(f(1.4)), jnp.asarray(f(2.2)))
    expected, _ = _weibull_sum_hand(float(f(1.4)), float(f(2.2)))
    assert_close(np.asarray(grad), expected)


@pytest.mark.step("A9")
@pytest.mark.parametrize("k", [0.6, 1.0, 2.5])
def test_weibull_shared_parameter_hessian_is_hand_derived(k):
    """spec 7.3, rev 28: full `(k, b)` Hessian including the cross term, for `k<1`, `k=1`, `k>1`."""
    hess = jax.hessian(_weibull_sum, argnums=(0, 1))(jnp.asarray(f(k)), jnp.asarray(f(2.2)))
    actual = np.array([[hess[0][0], hess[0][1]], [hess[1][0], hess[1][1]]], dtype=float)
    _, expected = _weibull_sum_hand(float(f(k)), float(f(2.2)))
    assert_close(actual, expected)


@pytest.mark.step("A9")
def test_weibull_per_row_shape_hessian_is_diagonal_and_hand_derived():
    """spec 5 and 7.3, rev 28: per-row shapes with a shared scale give a diagonal `d2/dk_i2` Hessian."""
    shapes = f([0.6, 0.9, 1.3, 2.0, 3.0])
    hess = jax.hessian(lambda k: _weibull_sum(k, f(2.2)))(jnp.asarray(shapes))
    expected = np.diag(
        [
            _weibull_hand(float(k), float(f(2.2)), t, e)[1][0, 0]
            for k, t, e in zip(shapes, TIME, EVENT, strict=True)
        ]
    )
    assert_close(np.asarray(hess), expected)


@pytest.mark.step("A9")
def test_weibull_from_log_hessian_follows_the_chain_rule():
    """spec 4.1 and 7.3, rev 28: `Weibull.from_log` derivatives in `(log k, log b)` by the chain rule.

    With ``k = exp(u)``, ``b = exp(v)``: ``d/du = k g_k``,
    ``d2/du2 = k g_k + k**2 H_kk``, ``d2/dudv = k b H_kb``,
    ``d2/dv2 = b g_b + b**2 H_bb``.
    """

    def loss(u, v):
        dist = survjax.Weibull.from_log(log_shape=u, log_scale=v)
        return jnp.sum(survjax.log_likelihood(dist, _data()))

    u, v = np.log(1.4), np.log(2.2)
    hess = jax.hessian(loss, argnums=(0, 1))(jnp.asarray(f(u)), jnp.asarray(f(v)))
    actual = np.array([[hess[0][0], hess[0][1]], [hess[1][0], hess[1][1]]], dtype=float)
    k, b_ = np.exp(float(f(u))), np.exp(float(f(v)))
    g, h = _weibull_sum_hand(k, b_)
    expected = np.array(
        [
            [k * g[0] + k**2 * h[0, 0], k * b_ * h[0, 1]],
            [k * b_ * h[0, 1], b_ * g[1] + b_**2 * h[1, 1]],
        ]
    )
    assert_close(actual, expected)


@pytest.mark.step("A9")
@pytest.mark.parametrize("k", [0.5, 1.0, 2.5])
def test_right_censored_at_zero_has_zero_hessian(k):
    """spec 2, 7.2 and 7.3, rev 28: a right-censored row at `0` contributes exactly zero, so its parameter Hessian is zero.

    Its inactive density piece (infinite ``f(0)`` at ``k<1``) must not leak
    into second derivatives; ``reference_time()`` replaces it (7.2).
    """
    data = survjax.SurvivalData(f([0.0]), b([False]))

    def loss(shape, scale):
        dist = survjax.Weibull(shape=shape, scale=scale)
        return jnp.sum(survjax.log_likelihood(dist, data))

    hess = jax.hessian(loss, argnums=(0, 1))(jnp.asarray(f(k)), jnp.asarray(f(2.0)))
    np.testing.assert_array_equal(np.asarray(hess, dtype=float), np.zeros((2, 2)))


# ------------------------------------------------- modes and transformations


@pytest.mark.step("A9")
def test_forward_and_reverse_hessians_agree():
    """spec 7.3, rev 28: forward and reverse modes: `jacfwd(jacrev)`, `jacrev(jacfwd)`, `jacrev(jacrev)` and `jacfwd(jacfwd)` agree."""
    params = jnp.asarray(f([1.4, 2.2]))

    def loss(p):
        return _weibull_sum(p[0], p[1])

    _, expected = _weibull_sum_hand(float(params[0]), float(params[1]))
    for outer in (jax.jacfwd, jax.jacrev):
        for inner in (jax.jacfwd, jax.jacrev):
            actual = np.asarray(outer(inner(loss))(params))
            assert_close(actual, expected, err_msg=f"{outer.__name__}({inner.__name__})")


@pytest.mark.step("A9")
def test_per_row_jacobian_in_both_modes():
    """spec 7.3, rev 28: `jacfwd` and `jacrev` of the contribution vector in per-row rates are `diag(1/r - t)` or `diag(-c)`."""
    rates = jnp.asarray(f([0.5, 0.7, 1.1, 1.6, 2.0]))

    def contributions(r):
        return survjax.log_likelihood(survjax.Exponential(rate=r), _data())

    r = np.asarray(rates, dtype=float)
    t = np.asarray(TIME)
    expected = np.diag(np.where(EVENT, 1.0 / r - t, -t))
    assert_close(np.asarray(jax.jacfwd(contributions)(rates)), expected)
    assert_close(np.asarray(jax.jacrev(contributions)(rates)), expected)


@pytest.mark.step("A9")
def test_hessian_under_jit_matches_eager():
    """spec 7.3 and 12.1, rev 28: the Hessian composes with `jit` and agrees with eager."""
    hess = jax.hessian(_weibull_sum, argnums=(0, 1))
    args = (jnp.asarray(f(1.4)), jnp.asarray(f(2.2)))
    eager = np.asarray(hess(*args), dtype=float)
    assert_close(np.asarray(jax.jit(hess)(*args), dtype=float), eager)


@pytest.mark.step("A9")
def test_hessian_under_vmap_over_parameter_draws():
    """spec 7.3 and 12.1, rev 28: `vmap` of the Hessian over parameter draws equals per-draw hand values."""
    shapes = f([0.6, 1.4, 2.5])
    scales = f([1.5, 2.2, 3.0])
    hess = jax.vmap(jax.hessian(lambda p: _weibull_sum(p[0], p[1])))(
        jnp.stack([jnp.asarray(shapes), jnp.asarray(scales)], axis=-1)
    )
    expected = np.stack(
        [_weibull_sum_hand(float(k), float(s))[1] for k, s in zip(shapes, scales, strict=True)]
    )
    assert_close(np.asarray(hess), expected)


@pytest.mark.step("A9")
def test_gradient_under_vmap_over_datasets():
    """spec 7.3 and 12.1, rev 28: `vmap` over datasets of the summed-likelihood gradient equals per-dataset `d/r - T`."""
    times = f([[0.5, 1.0, 2.0], [1.5, 0.5, 4.0]])
    events = b([[True, False, True], [False, False, True]])
    data = survjax.SurvivalData(times, events)

    def grad_one(d):
        return jax.grad(
            lambda r: jnp.sum(survjax.log_likelihood(survjax.Exponential(rate=r), d))
        )(jnp.asarray(f(0.8)))

    # d/r - T: (2/0.8 - 3.5, 1/0.8 - 6.0)
    expected = [2.0 / 0.8 - 3.5, 1.0 / 0.8 - 6.0]
    assert_close(np.asarray(jax.vmap(grad_one)(data)), expected)
    assert_close(np.asarray(jax.jit(jax.vmap(grad_one))(data)), expected)


@pytest.mark.step("A9")
def test_method_hessian_under_jit_and_vmap_equals_function():
    """spec 7.1 and 7.3, rev 29: `dist.log_likelihood` gives the function's Hessian under `jit` and `vmap`."""

    def method_sum(p):
        return jnp.sum(survjax.Weibull(shape=p[0], scale=p[1]).log_likelihood(_data()))

    params = jnp.asarray(f([[0.8, 1.5], [1.4, 2.2]]))
    actual = np.asarray(jax.jit(jax.vmap(jax.hessian(method_sum)))(params))
    expected = np.stack([_weibull_sum_hand(float(k), float(s))[1] for k, s in params])
    assert_close(actual, expected)


@pytest.mark.step("A9")
def test_raw_form_hessian_equals_data_form():
    """spec 6.1 and 7.3, rev 28: the raw `time=`/`event=` form has the data form's Hessian."""

    def raw_sum(p):
        dist = survjax.Weibull(shape=p[0], scale=p[1])
        return jnp.sum(survjax.log_likelihood(dist, time=f(TIME), event=b(EVENT)))

    params = jnp.asarray(f([1.4, 2.2]))
    _, expected = _weibull_sum_hand(1.4, 2.2)
    assert_close(np.asarray(jax.hessian(raw_sum)(params)), expected)


# ------------------------------------------------------------- check_grads


@pytest.mark.step("A9")
@requires_x64
def test_check_grads_exponential_shared_rate():
    """spec 7.3 and 12.1, rev 28: `check_grads(order=2, modes=("fwd","rev"))`, summed likelihood, shared rate."""
    check_grads(_exp_sum, (f(0.7),), order=2, modes=("fwd", "rev"))


@pytest.mark.step("A9")
@requires_x64
def test_check_grads_exponential_per_row_contributions():
    """spec 7.3 and 12.1, rev 28: `check_grads` on the contribution vector with per-row rates."""

    def contributions(r):
        return survjax.log_likelihood(survjax.Exponential(rate=r), _data())

    check_grads(contributions, (f([0.5, 0.7, 1.1, 1.6, 2.0]),), order=2, modes=("fwd", "rev"))


@pytest.mark.step("A9")
@requires_x64
@pytest.mark.parametrize("k", [0.6, 1.0, 1.4, 3.0])
def test_check_grads_weibull_shared_parameters(k):
    """spec 7.3 and 12.1, rev 28: `check_grads` in `(k, b)`; `k=1` is interior."""
    check_grads(_weibull_sum, (f(k), f(2.2)), order=2, modes=("fwd", "rev"))


@pytest.mark.step("A9")
@requires_x64
def test_check_grads_weibull_per_row_contributions():
    """spec 7.3 and 12.1, rev 28: `check_grads` on the contribution vector with per-row shapes and scales."""

    def contributions(k, s):
        return survjax.log_likelihood(survjax.Weibull(shape=k, scale=s), _data())

    check_grads(
        contributions,
        (f([0.6, 0.9, 1.3, 2.0, 3.0]), f([1.0, 1.5, 2.2, 3.0, 0.7])),
        order=2,
        modes=("fwd", "rev"),
    )


@pytest.mark.step("A9")
@requires_x64
def test_check_grads_weibull_draw_by_individual_contributions():
    """spec 5 and 7.3, rev 28: `check_grads` through the `(D,1)` draw broadcast to `(D,N)` contributions."""

    def contributions(k, s):
        return survjax.log_likelihood(survjax.Weibull(shape=k, scale=s), _data())

    check_grads(
        contributions, (f([[0.8], [1.6]]), f([[1.5], [2.5]])), order=2, modes=("fwd", "rev")
    )


@pytest.mark.step("A9")
@requires_x64
def test_check_grads_weibull_from_log():
    """spec 4.1 and 7.3, rev 28: `check_grads` through `Weibull.from_log`."""

    def loss(u, v):
        dist = survjax.Weibull.from_log(log_shape=u, log_scale=v)
        return jnp.sum(survjax.log_likelihood(dist, _data()))

    check_grads(loss, (f(0.3), f(0.8)), order=2, modes=("fwd", "rev"))


@pytest.mark.step("A9")
@requires_x64
def test_check_grads_method_form():
    """spec 7.1 and 7.3, rev 29: `check_grads` through `dist.log_likelihood`."""

    def loss(k, s):
        return jnp.sum(survjax.Weibull(shape=k, scale=s).log_likelihood(_data()))

    check_grads(loss, (f(1.4), f(2.2)), order=2, modes=("fwd", "rev"))


@pytest.mark.step("A9")
@requires_x64
def test_check_grads_under_jit():
    """spec 7.3 and 12.1, rev 28: `check_grads` of the jitted summed likelihood."""
    check_grads(jax.jit(_weibull_sum), (f(1.4), f(2.2)), order=2, modes=("fwd", "rev"))


@pytest.mark.step("A9")
@requires_x64
def test_check_grads_under_vmap_over_draws():
    """spec 7.3 and 12.1, rev 28: `check_grads` of a `vmap` over parameter draws."""
    mapped = jax.vmap(_weibull_sum)
    check_grads(mapped, (f([0.7, 1.4, 2.5]), f([1.5, 2.2, 3.0])), order=2, modes=("fwd", "rev"))


@pytest.mark.step("A9")
@requires_x64
def test_check_grads_with_traced_construction():
    """spec 6.2 and 7.3, rev 28: `check_grads` when the data are constructed inside the differentiated function."""

    def loss(k, s):
        data = survjax.SurvivalData(jnp.asarray(f(TIME)), jnp.asarray(b(EVENT)))
        return jnp.sum(survjax.log_likelihood(survjax.Weibull(shape=k, scale=s), data))

    check_grads(jax.jit(loss), (f(1.4), f(2.2)), order=2, modes=("fwd", "rev"))


# --------------------------------------------- custom families (generic defaults)


@pytest.mark.step("A9")
@requires_x64
def test_check_grads_custom_family_on_generic_defaults():
    """spec 4.2 and 7.3, rev 28: "The custom-family recipe is verified where its primitives are twice differentiable".

    ``power_hazard`` supplies only ``log_survival``; exact rows use the
    autodiff ``log_prob`` default, which "adds one derivative order".
    """

    def loss(k, s):
        dist = fam.power_hazard()(shape=k, scale=s)
        return jnp.sum(survjax.log_likelihood(dist, _data()))

    check_grads(loss, (f(1.4), f(2.2)), order=2, modes=("fwd", "rev"))


@pytest.mark.step("A9")
def test_custom_family_hessian_matches_built_in():
    """spec 4.2 and 7.3, rev 28: a custom Weibull-like family on generic defaults has the built-in Weibull Hessian."""

    def custom_sum(p):
        dist = fam.power_hazard()(shape=p[0], scale=p[1])
        return jnp.sum(survjax.log_likelihood(dist, _data()))

    params = jnp.asarray(f([1.4, 2.2]))
    _, expected = _weibull_sum_hand(1.4, 2.2)
    assert_close(np.asarray(jax.hessian(custom_sum)(params)), expected)
    assert_close(np.asarray(jax.jit(jax.hessian(custom_sum))(params)), expected)


# --------------------------------------------- reference-input gradient isolation


@functools.cache
def _steep_reference():
    """Exponential-like family whose ``reference_time()`` has an infinite derivative at ``rate = 1``.

    ``S(t) = exp(-rate t)``; ``reference_time() = 1 + sqrt(rate - 1)``: the
    value is ``1`` (positive finite, as 4.2 requires) and
    ``d/drate = 1/(2 sqrt(rate - 1)) = +inf`` at ``rate = 1``.
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class SteepReference(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()

        def log_survival(self, t):
            t = jnp.asarray(t)
            ts = jnp.where(t < 0, 0.0, t)
            return jnp.where(t < 0, 0.0, -self.rate * ts)

        def reference_time(self):
            return 1.0 + jnp.sqrt(self.rate - 1.0)

    return SteepReference


@pytest.mark.step("A9")
def test_reference_input_does_not_reach_right_censored_gradients():
    """spec 7.2 and 12.1, rev 28: "`stop_gradient` is applied to substituted values" (reference-input gradient isolation).

    A right-censored row's inactive density piece is evaluated at
    ``reference_time()``, whose derivative is infinite here. Isolated, the
    contribution ``-rate c`` has ``d/drate = -c`` and ``d2/drate2 = 0`` in
    both modes; a leak would give ``0 * inf = NaN``.
    """
    data = survjax.SurvivalData(f([2.0, 3.0]), b([False, False]))

    def loss(rate):
        return jnp.sum(survjax.log_likelihood(_steep_reference()(rate=rate), data))

    rate = jnp.asarray(f(1.0))
    assert_close(np.asarray(jax.grad(loss)(rate)), -5.0)
    assert_close(np.asarray(jax.jacfwd(loss)(rate)), -5.0)
    assert_close(np.asarray(jax.hessian(loss)(rate)), 0.0)
    assert_close(np.asarray(jax.jit(jax.grad(loss))(rate)), -5.0)


@pytest.mark.step("A9")
def test_reference_input_does_not_reach_mixed_batch_gradients():
    """spec 7.2 and 7.3, rev 28: with exact rows sharing the rate, the summed gradient is `d/r - T` and the Hessian `-d/r**2`.

    Rows: exact ``1``, right ``2``, exact ``0.5``; ``r = 1``:
    ``d/dr = 2 - 3.5 = -1.5``, ``d2/dr2 = -2``.
    """
    data = survjax.SurvivalData(f([1.0, 2.0, 0.5]), b([True, False, True]))

    def loss(rate):
        return jnp.sum(survjax.log_likelihood(_steep_reference()(rate=rate), data))

    rate = jnp.asarray(f(1.0))
    assert_close(np.asarray(jax.grad(loss)(rate)), -1.5)
    assert_close(np.asarray(jax.hessian(loss)(rate)), -2.0)
