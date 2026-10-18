"""A5: direct-method value policy and boundary values on the built-in families (spec 4.1, 4.5).

The base's enforcement of spec 4.5 is tested on custom families at A4. Here the
rows are checked on ``Exponential`` and ``Weibull`` themselves. Rev 26: the
base supplies ``log S``, ``S``, ``log F`` and ``H`` at ``t=0`` and ``t=+inf``
(the built-ins are proper, so ``log_survival_at_infinity()`` is ``-inf``) and
``log_prob(+inf)``; the built-in ``log_prob`` and ``log_hazard`` overrides
supply the density and hazard limits. Built-in values are unchanged.

Hand-derived boundary values (``z = t/b``):

- ``t=0``: ``log S = 0``, ``S = 1``, ``H = 0``, ``log F = -inf``. Exponential
  ``log f = log h = log r``. Weibull ``log h = log(k/b) + (k-1) log z``:
  ``+inf`` for ``k<1``, ``log(1/b)`` for ``k=1`` (spec 4.1: "`xlogy`-style
  terms"), ``-inf`` for ``k>1``; ``log f`` equals it, since ``log S(0)=0``.
- ``t=+inf``: ``log S = -inf``, ``S = 0``, ``H = +inf``, ``log F = 0``,
  ``log f = -inf`` (spec 4.5). ``log_hazard`` is the analytic limit of the
  built-in override (spec 4.2: "Built-ins override analytically"; 4.5, rev 26:
  "from a `log_hazard` override: the hazard limit"): Exponential ``log r``;
  Weibull ``-inf`` for ``k<1``, ``log(1/b)`` for ``k=1``, ``+inf`` for ``k>1``.
"""

import jax
import numpy as np
import pytest
from helpers import INF, NAN, assert_close, f

import survjax

DIRECT_METHODS = (
    "log_survival",
    "survival",
    "log_cdf",
    "cumulative_hazard",
    "log_prob",
    "log_hazard",
    "hazard",
)

# Spec 4.5, row `t<0`.
NEGATIVE_TIME = {
    "log_survival": 0.0,
    "survival": 1.0,
    "log_cdf": -INF,
    "cumulative_hazard": 0.0,
    "log_prob": -INF,
    "log_hazard": -INF,
    "hazard": 0.0,
}


def _family(name):
    """Built-in cases: Exponential ``r=0.5`` and Weibull ``b=2`` with ``k`` in ``(0.5, 1, 2)``."""
    if name == "exponential":
        return survjax.Exponential(rate=f(0.5))
    k = {"weibull_k0.5": 0.5, "weibull_k1": 1.0, "weibull_k2": 2.0}[name]
    return survjax.Weibull(shape=f(k), scale=f(2.0))


FAMILIES = ("exponential", "weibull_k0.5", "weibull_k1", "weibull_k2")

# log h at t=0 and at t=+inf, per case (see module docstring).
LOG_HAZARD_AT_ZERO = {
    "exponential": np.log(0.5),
    "weibull_k0.5": INF,
    "weibull_k1": np.log(0.5),
    "weibull_k2": -INF,
}
LOG_HAZARD_AT_INF = {
    "exponential": np.log(0.5),
    "weibull_k0.5": -INF,
    "weibull_k1": np.log(0.5),
    "weibull_k2": INF,
}


def _at_zero(name):
    lh = LOG_HAZARD_AT_ZERO[name]
    return {
        "log_survival": 0.0,
        "survival": 1.0,
        "log_cdf": -INF,
        "cumulative_hazard": 0.0,
        "log_prob": lh,
        "log_hazard": lh,
        "hazard": np.exp(lh),
    }


def _at_inf(name):
    lh = LOG_HAZARD_AT_INF[name]
    return {
        "log_survival": -INF,
        "survival": 0.0,
        "log_cdf": 0.0,
        "cumulative_hazard": INF,
        "log_prob": -INF,
        "log_hazard": lh,
        "hazard": np.exp(lh),
    }


# --------------------------------------------------------------- t < 0


@pytest.mark.step("A5")
@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_negative_time_gives_policy_value(name, method):
    """spec 4.5, rev 21: for `t<0`, `log S=0`, `S=1`, `log_prob=-inf`, `log_cdf=-inf`, `H=0`, `log_hazard=-inf`, `h=0`."""
    out = getattr(_family(name), method)(f([-1e-30, -1.0, -INF]))
    np.testing.assert_array_equal(np.asarray(out), [NEGATIVE_TIME[method]] * 3)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_negative_time_policy_values_have_zero_weibull_gradients(method):
    """spec 4.5, rev 21: "With valid parameters the policy values keep zero parameter gradients" (Weibull `k=0.5`, `t=-1`)."""
    gk, gb = jax.grad(
        lambda k, b: getattr(survjax.Weibull(shape=k, scale=b), method)(f(-1.0)),
        argnums=(0, 1),
    )(f(0.5), f(2.0))
    np.testing.assert_array_equal(np.asarray(gk), 0.0)
    np.testing.assert_array_equal(np.asarray(gb), 0.0)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_negative_time_policy_values_have_zero_exponential_gradient(method):
    """spec 4.5, rev 21: zero parameter gradients of the `t<0` policy values (Exponential, `t=-1`)."""
    g = jax.grad(lambda r: getattr(survjax.Exponential(rate=r), method)(f(-1.0)))(
        f(0.5)
    )
    np.testing.assert_array_equal(np.asarray(g), 0.0)


# --------------------------------------------------------------- t = 0


@pytest.mark.step("A5")
@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_zero_time_boundary_value(name, method):
    """spec 4.5, rev 26: `t=0`: `log S=0`, `S=1`, `log_cdf=-inf`, `H=0` from the base; density and hazard limits from the family (Weibull `k<1`: infinite density; `k=1`: `log(1/b)`; `k>1`: `-inf`)."""
    out = getattr(_family(name), method)(f(0.0))
    assert_close(out, _at_zero(name)[method])


@pytest.mark.step("A5")
def test_weibull_shape_one_density_at_zero_is_not_nan():
    """spec 4.1, rev 21: "`(k-1)*log(t/b)` at `t=0`, `k=1` gives `log(1/b)` rather than NaN"; `b=4`."""
    out = survjax.Weibull(shape=f(1.0), scale=f(4.0)).log_prob(f(0.0))
    assert_close(out, np.log(0.25))


# --------------------------------------------------------------- t = +inf


@pytest.mark.step("A5")
@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_infinite_time_boundary_value(name, method):
    """spec 4.5, rev 26: `t=+inf`: `log S = log_survival_at_infinity() = -inf` (proper), `S`, `log F`, `H` from it, `log_prob=-inf`; `log_hazard` is the override's limit."""
    out = getattr(_family(name), method)(f(INF))
    assert_close(out, _at_inf(name)[method])


# --------------------------------------------------------------- NaN t and invalid parameters


@pytest.mark.step("A5")
@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_nan_time_gives_nan(name, method):
    """spec 4.5, rev 21: NaN `t` gives NaN."""
    out = getattr(_family(name), method)(f(NAN))
    assert np.isnan(np.asarray(out))


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_nan_rate_gives_nan_for_every_time(method):
    """spec 4.5, rev 26: NaN parameters give NaN at every `t`, including `t<0`, NaN `t`, `t=0` and `t=+inf`.

    For base-supplied values (``t<0``, NaN ``t``, and ``log S``, ``S``,
    ``log F``, ``H`` at ``0`` and ``+inf``, ``log_prob`` at ``+inf``) the
    family's value at the substituted input is NaN, so the result is NaN
    (rev 26); the density and hazard overrides are NaN at ``t=0`` and the
    ``log_hazard`` override at ``+inf``.
    """
    out = getattr(survjax.Exponential(rate=f(NAN)), method)(
        f([-1.0, 0.0, 1.0, INF, NAN])
    )
    assert np.all(np.isnan(np.asarray(out)))


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_nan_weibull_scale_gives_nan_for_every_time(method):
    """spec 4.5, rev 26: NaN `scale` gives NaN at every `t`, `t=0` and `t=+inf` included, with no exception."""
    out = getattr(survjax.Weibull(shape=f(2.0), scale=f(NAN)), method)(
        f([-1.0, 0.0, 1.0, INF, NAN])
    )
    assert np.all(np.isnan(np.asarray(out)))


@pytest.mark.step("A5")
def test_nan_batch_entry_does_not_affect_other_entries():
    """spec 4.5, rev 21: an invalid batch entry gives NaN only in its own position; `log S = -(t/b)**k`."""
    dist = survjax.Weibull(shape=f([2.0, NAN, 1.0]), scale=f([2.0, 2.0, 4.0]))
    out = np.asarray(dist.log_survival(f(2.0)))
    assert np.isnan(out[1])
    assert_close(out[[0, 2]], [-1.0, -0.5])


# --------------------------------------------------------------- mixed and transformed


@pytest.mark.step("A5")
def test_weibull_policy_applies_elementwise_in_one_call():
    """spec 4.5, rev 21: total over `[-inf, +inf]`, elementwise, without raising.

    ``k=2``, ``b=2`` at ``(-inf, -1, 0, 1, +inf, NaN)``: ``log f`` is
    ``(-inf, -inf, -inf, log f(1), -inf, NaN)``, where
    ``log f(1) = log(k/b) + (k-1) log(t/b) - (t/b)**k = 0 + log(1/2) - 1/4``.
    """
    out = np.asarray(
        survjax.Weibull(shape=f(2.0), scale=f(2.0)).log_prob(
            f([-INF, -1.0, 0.0, 1.0, INF, NAN])
        )
    )
    np.testing.assert_array_equal(out[[0, 1, 2, 4]], [-INF] * 4)
    assert_close(out[3], np.log(0.5) - 0.25)
    assert np.isnan(out[5])


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_boundary_values_agree_under_jit(method):
    """spec 4.5/12.1, rev 21: boundary and policy values are the same eagerly and under `jit`."""
    dist = survjax.Weibull(shape=f([0.5, 1.0, 2.0]), scale=f(2.0))
    t = f([[-1.0], [0.0], [1.0], [INF], [NAN]])
    eager = getattr(dist, method)(t)
    jitted = jax.jit(lambda d, s: getattr(d, method)(s))(dist, t)
    assert_close(jitted, eager)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_boundary_values_agree_under_vmap_over_time(method):
    """spec 4.5/12.1, rev 21: boundary and policy values are the same eagerly and under `vmap` over time."""
    dist = survjax.Exponential(rate=f(0.5))
    t = f([-1.0, 0.0, 1.0, INF, NAN])
    assert_close(jax.vmap(getattr(dist, method))(t), getattr(dist, method)(t))


# --------------------------------------------------------------- gradients at t = 0


@pytest.mark.step("A5")
@pytest.mark.parametrize("k", [0.5, 1.0, 2.0])
def test_gradient_of_weibull_survival_at_zero_in_shape(k):
    """spec 4.5, rev 21: test case "the gradient of `S(0)` with respect to `k`".

    ``S(0) = 1`` for every ``k>0``, so the gradient is ``0`` (a naive
    ``(t/b)**k`` gives ``0 * log 0 = NaN``).
    """
    g = jax.grad(lambda kk: survjax.Weibull(shape=kk, scale=f(2.0)).survival(f(0.0)))(
        f(k)
    )
    np.testing.assert_array_equal(np.asarray(g), 0.0)


@pytest.mark.step("A5")
@pytest.mark.parametrize("k", [0.5, 1.0, 2.0])
def test_gradient_of_weibull_survival_at_zero_in_scale(k):
    """spec 4.5, rev 21: "Boundary values must also avoid NaN parameter gradients where the value is finite"; `dS(0)/db = 0`."""
    g = jax.grad(lambda b: survjax.Weibull(shape=f(k), scale=b).survival(f(0.0)))(
        f(2.0)
    )
    np.testing.assert_array_equal(np.asarray(g), 0.0)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", ["log_survival", "cumulative_hazard"])
@pytest.mark.parametrize("k", [0.5, 1.0, 2.0])
def test_gradient_of_weibull_log_survival_and_cumulative_hazard_at_zero(method, k):
    """spec 4.5, rev 21: `log S(0) = 0` and `H(0) = 0` for every `k`, `b`: both parameter gradients are `0`, not NaN."""
    gk, gb = jax.grad(
        lambda kk, bb: getattr(survjax.Weibull(shape=kk, scale=bb), method)(f(0.0)),
        argnums=(0, 1),
    )(f(k), f(2.0))
    np.testing.assert_array_equal(np.asarray(gk), 0.0)
    np.testing.assert_array_equal(np.asarray(gb), 0.0)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", ["log_prob", "log_hazard"])
def test_gradient_in_scale_of_weibull_shape_one_density_at_zero(method):
    """spec 4.5, rev 22: at `k=1`, `log f(0) = log h(0) = -log b` is finite, so `d/db = -1/b` (`b=4`), not NaN.

    The derivative in ``k`` at this point "is outside the guarantee" (rev 22):
    the value jumps from ``+inf`` to ``-inf`` across ``k=1``.
    """
    g = jax.grad(
        lambda b: getattr(survjax.Weibull(shape=f(1.0), scale=b), method)(f(0.0))
    )(f(4.0))
    assert_close(g, -0.25)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", ["log_prob", "log_hazard"])
def test_gradient_of_exponential_density_at_zero(method):
    """spec 4.5, rev 21: `log f(0) = log h(0) = log r`, so `d/dr = 1/r` (`r=0.5`)."""
    g = jax.grad(lambda r: getattr(survjax.Exponential(rate=r), method)(f(0.0)))(f(0.5))
    assert_close(g, 2.0)


@pytest.mark.step("A5")
def test_gradient_of_weibull_survival_at_zero_under_jit_and_vmap():
    """spec 4.5/7.3, rev 21: the `S(0)` gradient in `k` is `0` under `jit` and `vmap` over shapes."""
    g = jax.jit(
        jax.vmap(
            jax.grad(
                lambda kk: survjax.Weibull(shape=kk, scale=f(2.0)).survival(f(0.0))
            )
        )
    )(f([0.5, 1.0, 2.0]))
    np.testing.assert_array_equal(np.asarray(g), [0.0, 0.0, 0.0])


# --------------------------------------------------------------- gradients at t = +inf (rev 22)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", ["survival", "log_cdf"])
@pytest.mark.parametrize("k", [0.5, 1.0, 2.0])
def test_gradient_of_weibull_finite_values_at_infinity(method, k):
    """spec 4.5, rev 26: base-supplied values have parameter gradients "`0` for ... with the proper default, for `S(+inf)` and `log F(+inf)`".

    ``S(+inf) = 0`` and ``log F(+inf) = 0`` for every ``k``, ``b``, so both
    gradients are ``0`` (a naive ``exp(-z**k)`` gives ``0 * inf = NaN``).
    """
    gk, gb = jax.grad(
        lambda kk, bb: getattr(survjax.Weibull(shape=kk, scale=bb), method)(f(INF)),
        argnums=(0, 1),
    )(f(k), f(2.0))
    np.testing.assert_array_equal(np.asarray(gk), 0.0)
    np.testing.assert_array_equal(np.asarray(gb), 0.0)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", ["survival", "log_cdf"])
def test_gradient_of_exponential_finite_values_at_infinity(method):
    """spec 4.5, rev 26: `S(+inf) = 0` and `log F(+inf) = 0` for every `r` (base-supplied): the gradient is `0`, not NaN."""
    g = jax.grad(lambda r: getattr(survjax.Exponential(rate=r), method)(f(INF)))(f(0.5))
    np.testing.assert_array_equal(np.asarray(g), 0.0)


@pytest.mark.step("A5")
def test_gradient_of_exponential_hazard_at_infinity():
    """spec 4.5, rev 22: the gradient equals "the derivative of the value": `log h(+inf) = log r`, `d/dr = 1/r`; `h(+inf) = r`, `d/dr = 1`."""
    glh = jax.grad(lambda r: survjax.Exponential(rate=r).log_hazard(f(INF)))(f(0.5))
    gh = jax.grad(lambda r: survjax.Exponential(rate=r).hazard(f(INF)))(f(0.5))
    assert_close(glh, 2.0)
    assert_close(gh, 1.0)


@pytest.mark.step("A5")
def test_gradient_of_weibull_shape_one_log_hazard_at_infinity():
    """spec 4.5, rev 22: `k=1`: `log h(+inf) = -log b` (the limit), so `d/db = -1/b` (`b=4`)."""
    g = jax.grad(lambda b: survjax.Weibull(shape=f(1.0), scale=b).log_hazard(f(INF)))(
        f(4.0)
    )
    assert_close(g, -0.25)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", ["survival", "log_cdf"])
def test_gradient_at_infinity_under_jit_and_vmap(method):
    """spec 4.5/7.3, rev 22: the zero gradients at `t=+inf` hold under `jit` and `vmap` over shapes."""
    g = jax.jit(
        jax.vmap(
            jax.grad(
                lambda kk: getattr(survjax.Weibull(shape=kk, scale=f(2.0)), method)(
                    f(INF)
                )
            )
        )
    )(f([0.5, 1.0, 2.0]))
    np.testing.assert_array_equal(np.asarray(g), [0.0, 0.0, 0.0])
