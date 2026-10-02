"""A4: the autodiff `log_prob` default, `log(-d/dt log S(t)) + log S(t)` (spec 4.2, 5, 7.3).

Expected values are hand-derived from each test family's survival function
(see ``spec_families.py``):

- ConstHazard: ``log S = -r t``, so ``-d/dt log S = r`` and
  ``log f = log r - r t``.
- PowerHazard (``k = shape``, ``b = scale``, ``z = t/b``): ``log S = -z**k``,
  ``-d/dt log S = (k/b) z**(k-1)``, so
  ``log f = log k - log b + (k-1) log z - z**k``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import FLOAT, MISUSE, X64, assert_close, f
from jax.test_util import check_grads

requires_x64 = pytest.mark.skipif(
    not X64, reason="second-order finite differences need float64"
)


def _const_log_f(r, t):
    r, t = np.asarray(r, np.float64), np.asarray(t, np.float64)
    return np.log(r) - r * t


def _power_log_f(k, b, t):
    k, b, t = (np.asarray(x, np.float64) for x in (k, b, t))
    z = t / b
    return np.log(k) - np.log(b) + (k - 1.0) * np.log(z) - z**k


# --------------------------------------------------------------- values


@pytest.mark.step("A4")
def test_default_log_prob_of_constant_hazard():
    """spec 4.2, rev 11: default `log_prob` is `log(-d/dt log S) + log S`; here `log r - r t`."""
    r, t = [0.5, 1.0, 3.0], [2.0, 0.25, 1.5]
    out = fam.const_hazard()(rate=f(r)).log_prob(f(t))
    assert_close(out, _const_log_f(r, t))


@pytest.mark.step("A4")
def test_default_log_prob_of_power_hazard_decreasing_hazard():
    """spec 4.2, rev 11: default `log_prob` for a Weibull-like family with `k<1`."""
    t = [0.1, 1.0, 5.0]
    out = fam.power_hazard()(shape=f(0.5), scale=f(2.0)).log_prob(f(t))
    assert_close(out, _power_log_f(0.5, 2.0, t))


@pytest.mark.step("A4")
def test_default_log_prob_of_power_hazard_constant_hazard():
    """spec 4.2, rev 11: default `log_prob` for a Weibull-like family with `k=1` (`log(1/b) - t/b`)."""
    t = [0.1, 1.0, 5.0]
    out = fam.power_hazard()(shape=f(1.0), scale=f(2.0)).log_prob(f(t))
    assert_close(out, -np.log(2.0) - np.asarray(t) / 2.0)


@pytest.mark.step("A4")
def test_default_log_prob_of_power_hazard_increasing_hazard():
    """spec 4.2, rev 11: default `log_prob` for a Weibull-like family with `k>1`."""
    t = [0.1, 1.0, 5.0]
    out = fam.power_hazard()(shape=f(3.0), scale=f(2.0)).log_prob(f(t))
    assert_close(out, _power_log_f(3.0, 2.0, t))


@pytest.mark.step("A4")
def test_default_log_prob_over_wide_time_range():
    """spec 4.2, rev 11: the default is accurate "wherever the hazard is representable in linear space".

    ``k=1.5``, ``b=2``, ``t`` from 1e-6 to 1e6: the hazard ``(k/b) z**0.5`` stays
    between about 5e-4 and 5e2, and ``z**k`` stays below 4e8, all representable in
    float32. Relative tolerance 12.1 applies to ``log f``.
    """
    t = [1e-6, 1e-3, 1.0, 1e3, 1e6]
    out = fam.power_hazard()(shape=f(1.5), scale=f(2.0)).log_prob(f(t))
    assert_close(out, _power_log_f(1.5, 2.0, t))


@pytest.mark.step("A4")
def test_default_log_prob_with_component_parameter():
    """spec 4.2, rev 11: the default works for families with component axes.

    ``log S = -r c t`` with ``c = sum(w) = 6``, so ``log f = log(6 r) - 6 r t``.
    """
    dist = fam.component_rate()(rate=f([0.5, 1.0]), w=f([1.0, 2.0, 3.0]))
    out = dist.log_prob(f(0.5))
    assert_close(
        out, np.log(6.0 * np.array([0.5, 1.0])) - 6.0 * np.array([0.5, 1.0]) * 0.5
    )


@pytest.mark.step("A4")
def test_default_log_prob_of_nested_family():
    """spec 4.2, rev 11: the default works through a nested distribution field.

    ``log S = -exp(a) r t``: a constant hazard ``exp(a) r``; with ``r=0.5``, ``a=log 2``,
    ``log f = log 1 - t``.
    """
    dist = fam.scaled_hazard()(
        base=fam.const_hazard()(rate=f(0.5)), log_ratio=f(np.log(2.0))
    )
    assert_close(dist.log_prob(f([0.5, 2.0])), [-0.5, -2.0])


@pytest.mark.step("A4")
def test_log_prob_override_is_used():
    """spec 4.2, rev 11: `log_prob` is an optional override; a family's own method replaces the default."""
    dist = fam.log_prob_override()(rate=f([0.5, 1.0]))
    np.testing.assert_array_equal(
        np.asarray(dist.log_prob(f(1.0))), fam.OVERRIDE_SENTINEL
    )


@pytest.mark.step("A4")
def test_log_prob_returns_a_single_array():
    """spec 4.5, rev 11: direct methods "carry no issue masks": the result is one array, not a tuple."""
    out = fam.const_hazard()(rate=f([0.5, 1.0])).log_prob(f(1.0))
    assert isinstance(out, jax.Array)


@pytest.mark.step("A4")
def test_log_prob_dtype_matches_parameters():
    """spec 3.3, rev 11: calculations use the common promoted dtype (default float here)."""
    out = fam.const_hazard()(rate=f([0.5, 1.0])).log_prob(f(1.0))
    assert np.dtype(out.dtype) == FLOAT


# --------------------------------------------------------------- shapes (spec 5)


@pytest.mark.step("A4")
def test_scalar_time_scalar_distribution_gives_scalar():
    """spec 5, rev 11: scalar time and scalar distribution give `()`."""
    assert jnp.shape(fam.const_hazard()(rate=f(0.5)).log_prob(f(1.0))) == ()


@pytest.mark.step("A4")
def test_scalar_time_batched_distribution_gives_batch_shape():
    """spec 5, rev 11: time scalar, distribution `(N,)` gives `(N,)`."""
    out = fam.const_hazard()(rate=f([0.5, 1.0, 2.0])).log_prob(f(1.0))
    assert jnp.shape(out) == (3,)
    assert_close(out, _const_log_f([0.5, 1.0, 2.0], 1.0))


@pytest.mark.step("A4")
def test_paired_time_and_distribution():
    """spec 5, rev 11: time `(N,)`, distribution `(N,)` pair elementwise."""
    out = fam.const_hazard()(rate=f([0.5, 1.0, 2.0])).log_prob(f([1.0, 2.0, 3.0]))
    assert_close(out, _const_log_f([0.5, 1.0, 2.0], [1.0, 2.0, 3.0]))


@pytest.mark.step("A4")
def test_time_column_against_batch_gives_outer_shape():
    """spec 5, rev 11: trailing-axis broadcasting: time `(K, 1)` against batch `(N,)` gives `(K, N)`."""
    r, t = np.array([0.5, 1.0, 2.0]), np.array([[1.0], [2.0]])
    out = fam.const_hazard()(rate=f(r)).log_prob(f(t))
    assert jnp.shape(out) == (2, 3)
    assert_close(out, _const_log_f(r, t))


@pytest.mark.step("A4")
def test_time_vector_against_scalar_distribution():
    """spec 5, rev 11: time `(K,)` against a scalar distribution gives `(K,)`."""
    out = fam.power_hazard()(shape=f(3.0), scale=f(2.0)).log_prob(
        f([0.5, 1.0, 2.0, 4.0])
    )
    assert_close(out, _power_log_f(3.0, 2.0, [0.5, 1.0, 2.0, 4.0]))


@pytest.mark.step("A4")
def test_component_axes_never_appear_in_outputs():
    """spec 5, rev 11: "Component axes never appear in outputs"."""
    dist = fam.component_rate()(rate=f([0.5, 1.0, 2.0]), w=f(np.ones((3, 4))))
    assert jnp.shape(dist.log_prob(f(1.0))) == (3,)
    assert jnp.shape(dist.log_prob(f([[1.0], [2.0]]))) == (2, 3)


@pytest.mark.step("A4")
def test_mismatched_lengths_do_not_broadcast():
    """spec 4.1/5, rev 16: a batch `(D,)` against time `(N,)` with different lengths, neither 1, raises `TypeError`."""
    dist = fam.const_hazard()(rate=f([0.5, 1.0, 2.0]))
    with pytest.raises(MISUSE):
        dist.log_prob(f([1.0, 2.0]))


@pytest.mark.step("A4")
def test_time_column_not_broadcasting_against_two_dimensional_batch_is_rejected():
    """spec 4.1, rev 16: time `(4, 1)` against batch `(2, 3)` does not broadcast: `TypeError`."""
    dist = fam.const_hazard()(rate=f(np.ones((2, 3))))
    with pytest.raises(MISUSE):
        dist.log_prob(f(np.ones((4, 1))))


@pytest.mark.step("A4")
def test_mismatched_time_is_rejected_by_every_direct_method():
    """spec 4.1, rev 16: the shape rule applies to "a direct method", including `log_survival` and `survival`."""
    dist = fam.const_hazard()(rate=f([0.5, 1.0, 2.0]))
    for name in (
        "log_survival",
        "survival",
        "log_cdf",
        "cumulative_hazard",
        "log_hazard",
        "hazard",
    ):
        with pytest.raises(MISUSE):
            getattr(dist, name)(f([1.0, 2.0]))


@pytest.mark.step("A4")
def test_mismatched_time_is_rejected_for_nested_family():
    """spec 4.1, rev 16: a nested family's direct method also raises `TypeError` on a non-broadcasting time."""
    dist = fam.scaled_hazard()(
        base=fam.const_hazard()(rate=f(0.5)), log_ratio=f([0.0, 0.1, 0.2])
    )
    with pytest.raises(MISUSE):
        dist.log_prob(f([1.0, 2.0]))


@pytest.mark.step("A4")
def test_mismatched_time_is_rejected_under_jit():
    """spec 4.1/6.2, rev 16: static shape misuse raises also when traced."""
    dist = fam.const_hazard()(rate=f([0.5, 1.0, 2.0]))
    with pytest.raises(MISUSE):
        jax.jit(lambda d, s: d.log_prob(s))(dist, f([1.0, 2.0]))


# --------------------------------------------------------------- transformations (spec 7.3, 12.1)


@pytest.mark.step("A4")
def test_default_log_prob_agrees_under_jit():
    """spec 12.1, rev 11: eager and `jit` agree."""
    dist = fam.power_hazard()(shape=f([0.5, 1.0, 3.0]), scale=f(2.0))
    t = f([0.3, 1.0, 4.0])
    assert_close(jax.jit(lambda d, s: d.log_prob(s))(dist, t), dist.log_prob(t))


@pytest.mark.step("A4")
def test_default_log_prob_agrees_under_vmap_over_time():
    """spec 12.1, rev 11: eager and `vmap` (over a time axis) agree."""
    dist = fam.power_hazard()(shape=f(3.0), scale=f(2.0))
    t = f([0.3, 1.0, 4.0])
    assert_close(jax.vmap(dist.log_prob)(t), dist.log_prob(t))


@pytest.mark.step("A4")
def test_gradient_of_default_log_prob_constant_hazard():
    """spec 7.3, rev 11: parameter gradient of the default; `d/dr (log r - r t) = 1/r - t`."""
    grad = jax.grad(lambda r: fam.const_hazard()(rate=r).log_prob(f(3.0)))(f(0.5))
    assert_close(grad, 1.0 / 0.5 - 3.0)


@pytest.mark.step("A4")
def test_gradient_of_default_log_prob_power_hazard():
    """spec 7.3, rev 11: parameter gradients of the default for a Weibull-like family.

    With ``z = t/b`` and ``u = log z``, ``log f = log k - log b + (k-1) u - z**k``:
    ``d/dk = 1/k + u - z**k u`` and ``d/db = k (z**k - 1) / b``.
    """
    k, b, t = 1.5, 2.0, 3.0
    dk, db = jax.grad(
        lambda kk, bb: fam.power_hazard()(shape=kk, scale=bb).log_prob(f(t)),
        argnums=(0, 1),
    )(f(k), f(b))
    z = t / b
    u = np.log(z)
    assert_close(dk, 1.0 / k + u - z**k * u)
    assert_close(db, k * (z**k - 1.0) / b)


@pytest.mark.step("A4")
def test_forward_and_reverse_gradients_agree():
    """spec 7.3, rev 11: first parameter derivatives in forward and reverse mode agree."""

    def value(k):
        return fam.power_hazard()(shape=k, scale=f(2.0)).log_prob(f(3.0))

    assert_close(jax.jacfwd(value)(f(1.5)), jax.jacrev(value)(f(1.5)))


@pytest.mark.step("A4")
def test_hessian_of_default_log_prob_constant_hazard():
    """spec 7.3, rev 11: second parameter derivatives of the default (one extra autodiff order); `d2/dr2 = -1/r**2`."""
    hess = jax.hessian(lambda r: fam.const_hazard()(rate=r).log_prob(f(3.0)))(f(0.5))
    assert_close(hess, -1.0 / 0.25)


@pytest.mark.step("A4")
def test_hessian_of_default_log_prob_power_hazard_in_shape():
    """spec 7.3, rev 11: `d2/dk2 log f = -1/k**2 - z**k u**2` (from `d/dk = 1/k + u - z**k u`)."""
    k, b, t = 1.5, 2.0, 3.0
    hess = jax.hessian(
        lambda kk: fam.power_hazard()(shape=kk, scale=f(b)).log_prob(f(t))
    )(f(k))
    z = t / b
    u = np.log(z)
    assert_close(hess, -1.0 / k**2 - z**k * u**2)


@pytest.mark.step("A4")
@requires_x64
def test_check_grads_order_two_constant_hazard():
    """spec 7.3/12.1, rev 11: `check_grads(order=2, modes=("fwd","rev"))` on an interior case."""
    t = f([0.5, 2.0])
    check_grads(
        lambda r: fam.const_hazard()(rate=r).log_prob(t),
        (f([0.7, 1.3]),),
        order=2,
        modes=("fwd", "rev"),
    )


@pytest.mark.step("A4")
@requires_x64
def test_check_grads_order_two_power_hazard():
    """spec 7.3/12.1, rev 11: `check_grads(order=2, modes=("fwd","rev"))` for the Weibull-like default."""
    t = f([0.5, 2.0, 4.0])
    check_grads(
        lambda k, b: fam.power_hazard()(shape=k, scale=b).log_prob(t),
        (f([0.7, 1.5, 3.0]), f([1.5, 2.0, 2.5])),
        order=2,
        modes=("fwd", "rev"),
    )


@pytest.mark.step("A4")
def test_gradient_of_default_log_prob_under_jit_and_vmap():
    """spec 7.3, rev 11: parameter gradients compose with `jit` and `vmap`; `d/dr = 1/r - t`."""
    r = f([0.5, 1.0, 2.0])
    grad = jax.jit(
        jax.vmap(jax.grad(lambda rr: fam.const_hazard()(rate=rr).log_prob(f(1.5))))
    )(r)
    assert_close(grad, 1.0 / np.array([0.5, 1.0, 2.0]) - 1.5)
