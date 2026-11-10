"""Feature `family.loglogistic`: direct-method value policy, boundaries and input policy (spec 4.2, 4.5, 12.1; rev 55).

4.5: base-supplied `log S`, `S`, `log F`, `H` at `t=0` and `t=+inf` (proper
default), `log_prob(+inf) = -inf`; the family supplies the density and
hazard limits at `t=0` and the hazard limit at `t=+inf` (through its
analytic `log_hazard`, 4.2); `t<0` and NaN rows; NaN parameters; "Boundary
values must also avoid NaN parameter gradients where the value is finite and
its parameter derivative exists": base-supplied values have the derivative of
the value (`0`), and "Family-supplied density and hazard limits of the
built-in families, where finite, have finite parameter gradients equal to the
derivative of the value"; "Where a boundary value changes discontinuously in
a parameter (Weibull `log f(0)` in `k` at `k=1`), that derivative is outside
the guarantee". The `t` dtype rule and float64 conversion.

Hand-derived limits (``log f = log beta - log alpha + (beta-1) log(t/alpha) -
2 softplus(u)``, ``log h = log f + softplus(u)``, ``u = beta log(t/alpha)``):

- ``t -> 0`` (``u -> -inf``, ``softplus -> 0``): ``log f(0) = log h(0)`` is
  ``+inf`` for ``beta < 1``, ``-log alpha`` for ``beta = 1`` and ``-inf`` for
  ``beta > 1``; ``h(0)`` is ``+inf``, ``1/alpha`` and ``0``.
- ``t -> +inf``: ``log h = log beta - log t - log1p(exp(-u)) -> -inf`` for
  every ``beta``: ``h(+inf) = 0``.

Finite family limits and their derivatives: ``beta = 1``: ``log h(0) =
-log alpha`` (``d/dalpha = -1/alpha``), ``h(0) = 1/alpha`` (``-1/alpha**2``);
the ``beta`` derivative there crosses the discontinuity and is not tested.
``beta = 2``: ``h(0) = 0`` near ``beta = 2``, so both derivatives are ``0``.
``h(+inf) = 0`` for all parameters: derivatives ``0``.
"""

import math
import warnings

import family_reference as fr
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import INF, NAN, X64, assert_close, assert_tail_close, f

import survjax

LOGLOGISTIC = pytest.mark.feature("family.loglogistic")
METHODS = fr.DIRECT_METHODS
BASE_METHODS = ("log_survival", "survival", "log_cdf", "cumulative_hazard")
ALPHA = 2.0

NEGATIVE_TIME = {
    "log_survival": 0.0,
    "survival": 1.0,
    "log_cdf": -INF,
    "cumulative_hazard": 0.0,
    "log_prob": -INF,
    "log_hazard": -INF,
    "hazard": 0.0,
}
LOG_H_AT_ZERO = {0.5: INF, 1.0: -math.log(ALPHA), 2.0: -INF}


def _dist(beta=2.0, alpha=ALPHA):
    return survjax.LogLogistic(shape=f(beta), scale=f(alpha))


def _at_zero(beta):
    lh = LOG_H_AT_ZERO[beta]
    return {
        "log_survival": 0.0,
        "survival": 1.0,
        "log_cdf": -INF,
        "cumulative_hazard": 0.0,
        "log_prob": lh,
        "log_hazard": lh,
        "hazard": math.exp(lh) if math.isfinite(lh) else (INF if lh > 0 else 0.0),
    }


AT_INF = {
    "log_survival": -INF,
    "survival": 0.0,
    "log_cdf": 0.0,
    "cumulative_hazard": INF,
    "log_prob": -INF,
    "log_hazard": -INF,
    "hazard": 0.0,
}


# --------------------------------------------------------------- policy rows


@LOGLOGISTIC
@pytest.mark.parametrize("method", METHODS)
def test_negative_time_gives_policy_value(method):
    """spec 4.5, rev 55: for `t<0`, `log S=0`, `S=1`, `log_prob=-inf`, `log_cdf=-inf`, `H=0`, `log_hazard=-inf`, `h=0`."""
    out = getattr(_dist(), method)(f([-1e-30, -1.0, -INF]))
    np.testing.assert_array_equal(np.asarray(out), [NEGATIVE_TIME[method]] * 3)


@LOGLOGISTIC
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_values_at_zero(beta, method):
    """spec 4.5, rev 55: at `t=0` the base supplies `log S=0`, `S=1`, `log F=-inf`, `H=0`; the built-in supplies `log f(0) = log h(0)`: `+inf` (`beta<1`), `-log alpha` (`beta=1`), `-inf` (`beta>1`)."""
    got = np.asarray(getattr(_dist(beta), method)(f(0.0)))
    want = _at_zero(beta)[method]
    if math.isfinite(want) and want != 0.0:
        assert_close(got, want)
    else:
        np.testing.assert_array_equal(got, want)


@LOGLOGISTIC
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_values_at_infinity(beta, method):
    """spec 4.5, rev 55: at `t=+inf`: `log S=-inf`, `S=0`, `log F=0`, `H=+inf`, `log_prob=-inf`; the analytic `log_hazard` gives `-inf` (`h -> 0`) for every `beta`."""
    np.testing.assert_array_equal(np.asarray(getattr(_dist(beta), method)(f(INF))), AT_INF[method])


@LOGLOGISTIC
@pytest.mark.parametrize("method", METHODS)
def test_nan_time_gives_nan(method):
    """spec 4.5, rev 55: NaN `t` gives NaN."""
    assert np.isnan(np.asarray(getattr(_dist(), method)(f(NAN))))


@LOGLOGISTIC
@pytest.mark.parametrize("method", METHODS)
def test_nan_parameters_give_nan_for_every_time(method):
    """spec 4.5, rev 55: NaN parameters give NaN at every time, including base-supplied rows (the family's value at the substituted input is NaN)."""
    for d in (survjax.LogLogistic(shape=f(NAN), scale=f(1.0)), survjax.LogLogistic(shape=f(2.0), scale=f(NAN))):
        out = np.asarray(getattr(d, method)(f([-1.0, 0.0, 1.5, INF])))
        assert np.isnan(out).all(), out


@LOGLOGISTIC
@pytest.mark.parametrize("method", METHODS)
def test_mixed_vector_of_boundary_and_interior_times(method):
    """spec 4.5/5, rev 55: one call over `[-1, 0, 1.5, +inf, NaN]` (`beta = 2`) gives each row's value."""
    out = np.asarray(getattr(_dist(), method)(f([-1.0, 0.0, 1.5, INF, NAN])))
    np.testing.assert_array_equal(out[[0, 1, 3]], [NEGATIVE_TIME[method], _at_zero(2.0)[method], AT_INF[method]])
    want, rtol = fr.direct("loglogistic", (2.0, ALPHA), 1.5, method)
    assert_tail_close(out[2], want, rtol)
    assert np.isnan(out[4])


# --------------------------------------------------------------- boundary gradients


@LOGLOGISTIC
@pytest.mark.parametrize("method", BASE_METHODS)
@pytest.mark.parametrize("t", [-1.0, 0.0, INF], ids=["negative", "zero", "posinf"])
@pytest.mark.parametrize("beta", [0.5, 2.0])
def test_base_supplied_values_have_zero_parameter_gradients(beta, t, method):
    """spec 4.5, rev 55: base-supplied values have gradient `0` in `shape` and `scale` (both modes), including `S(0)`, whose plain `-softplus` form has a NaN `beta` gradient at `t=0` (the 4.5 test case, here on the built-in)."""
    fn = lambda k, a: getattr(survjax.LogLogistic(shape=k, scale=a), method)(f(t))  # noqa: E731
    for g in jax.grad(fn, argnums=(0, 1))(f(beta), f(ALPHA)) + tuple(jax.jacfwd(fn, argnums=(0, 1))(f(beta), f(ALPHA))):
        np.testing.assert_array_equal(np.asarray(g), 0.0)


@LOGLOGISTIC
@pytest.mark.parametrize("method, want", [("log_prob", -1 / ALPHA), ("log_hazard", -1 / ALPHA), ("hazard", -1 / ALPHA**2)])
@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_finite_density_and_hazard_limits_at_zero_have_the_scale_derivative(mode, method, want):
    """spec 4.5, rev 55: `beta = 1`: `log f(0) = log h(0) = -log alpha`, `h(0) = 1/alpha` are finite family limits with "finite parameter gradients equal to the derivative of the value" in `scale` (the `shape` derivative crosses the `beta = 1` discontinuity)."""
    g = (jax.grad if mode == "rev" else jax.jacfwd)(lambda a: getattr(_dist(1.0, a), method)(f(0.0)))(f(ALPHA))
    assert_close(g, want)


@LOGLOGISTIC
@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_zero_hazard_limits_have_zero_parameter_gradients(mode):
    """spec 4.5, rev 55: `h(0) = 0` for `beta = 2` and `h(+inf) = 0` are finite family limits, constant near these parameters: gradients `0` in `shape` and `scale`."""
    grad = jax.grad if mode == "rev" else jax.jacfwd
    for t in (0.0, INF):
        gk, ga = grad(lambda k, a, t=t: survjax.LogLogistic(shape=k, scale=a).hazard(f(t)), argnums=(0, 1))(f(2.0), f(ALPHA))
        np.testing.assert_array_equal(np.asarray(gk), 0.0, err_msg=f"d/dshape at t={t}")
        np.testing.assert_array_equal(np.asarray(ga), 0.0, err_msg=f"d/dscale at t={t}")


@LOGLOGISTIC
@pytest.mark.parametrize("method", ["log_prob", "log_hazard", "hazard"])
def test_negative_time_family_rows_have_zero_parameter_gradients(method):
    """spec 4.5, rev 55: "With valid parameters the policy values keep zero parameter gradients" for the `t<0` density and hazard rows."""
    gk, ga = jax.grad(lambda k, a: getattr(survjax.LogLogistic(shape=k, scale=a), method)(f(-1.0)), argnums=(0, 1))(
        f(0.5), f(ALPHA)
    )
    np.testing.assert_array_equal(np.asarray(gk), 0.0)
    np.testing.assert_array_equal(np.asarray(ga), 0.0)


@LOGLOGISTIC
def test_log_survival_at_infinity_is_the_proper_default():
    """spec 4.2, rev 55: the built-in is proper even for heavy tails (`beta <= 1`): `log_survival_at_infinity()` is `-inf` per batch entry."""
    out = survjax.LogLogistic(shape=f([0.5, 1.0, 2.0]), scale=f(1.0)).log_survival_at_infinity()
    assert jnp.result_type(out) == jnp.result_type(f(0.0))
    np.testing.assert_array_equal(np.asarray(out), [-INF] * 3)


# --------------------------------------------------------------- t argument policy


NON_NUMERIC = pytest.mark.parametrize(
    "bad",
    [np.array(True), np.array(1.0 + 0j), np.array("1.0"), np.array(1.0, dtype=object)],
    ids=["bool", "complex", "string", "object"],
)


@LOGLOGISTIC
@NON_NUMERIC
@pytest.mark.parametrize("method", METHODS)
def test_time_with_a_non_numeric_dtype_is_rejected(method, bad):
    """spec 4.5, rev 55: boolean, complex, string and object `t` raise `TypeError`."""
    with pytest.raises(TypeError):
        getattr(_dist(), method)(bad)


@LOGLOGISTIC
@pytest.mark.parametrize("method", METHODS)
def test_boolean_time_is_rejected_during_tracing(method):
    """spec 4.5, rev 55: the check uses the dtype, so it also applies to a traced `t`."""
    d = _dist()
    with pytest.raises(TypeError):
        jax.jit(lambda t: getattr(d, method)(t))(jnp.asarray(True))


@pytest.mark.skipif(X64, reason="the float64-to-float32 conversion exists only with 64-bit mode disabled")
@LOGLOGISTIC
@pytest.mark.parametrize("method", METHODS)
def test_float64_time_converts_without_warning_in_32_bit_mode(method):
    """spec 4.5, rev 55: a float64 `t` with 64-bit mode disabled converts to float32 without a warning."""
    d = _dist()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = getattr(d, method)(np.asarray([0.5, 2.0], dtype=np.float64))
    assert jnp.result_type(got) == jnp.float32
    assert_close(got, getattr(d, method)(f([0.5, 2.0])))


# --------------------------------------------------------------- transformations


@LOGLOGISTIC
@pytest.mark.parametrize("method", METHODS)
def test_direct_methods_agree_under_jit_and_vmap(method):
    """spec 4.1/12.1, rev 55: eager, `jit` and `vmap` over the batch agree, including boundary rows."""
    d = survjax.LogLogistic(shape=f([0.5, 1.0, 3.0]), scale=f([0.5, 2.0, 1.0]))
    t = f([0.0, 1.7, INF])
    want = np.asarray(getattr(d, method)(t))
    assert_close(jax.jit(lambda dd, tt: getattr(dd, method)(tt))(d, t), want)
    assert_close(jax.vmap(lambda dd, tt: getattr(dd, method)(tt))(d, t), want)


@LOGLOGISTIC
@pytest.mark.parametrize("beta", [0.5, 1.0, 3.0])
def test_direct_methods_under_debug_nans(beta):
    """spec 7.2/12.1, rev 55: valid parameters over `[0, 1e-30, interior, upper tail, +inf]` raise no NaN eagerly under `jax_debug_nans`, in values or interior parameter gradients."""
    t = f([0.0, 1e-30, 1.5, fr.loglogistic_time(beta, ALPHA, 15.0), INF])  # u = 15: S ≈ 3e-7, t <= 2 e**30
    with jax.debug_nans(True):
        for method in METHODS:
            out = np.asarray(getattr(_dist(beta), method)(t))
            assert not np.isnan(out).any(), (method, out)
            g = jax.grad(lambda k, a, method=method: getattr(survjax.LogLogistic(shape=k, scale=a), method)(t[2]), argnums=(0, 1))(
                f(beta), f(ALPHA)
            )
            assert np.isfinite(np.asarray(g)).all(), (method, g)
