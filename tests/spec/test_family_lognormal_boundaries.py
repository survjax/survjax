"""Feature `family.lognormal`: direct-method value policy, boundaries and input policy (spec 4.2, 4.5, 12.1; rev 55).

4.5: the base applies the rows "for every registered family"; at `t=0` it
supplies `log S=0`, `S=1`, `log_cdf=-inf`, `H=0`, and the family supplies
the density and hazard limits; at `t=+inf` it supplies `log S =
log_survival_at_infinity()` (`-inf`, proper), `S`, `log_cdf`, `H` and
`log_prob=-inf`, and the hazard limit comes "from a `log_hazard` override"
(4.2: "Built-ins override analytically"); `t<0` and NaN `t` rows; invalid
parameters give "whatever the family's formula gives: NaN where it is
undefined (NaN parameters ...)"; base-supplied values have parameter
gradients "equal to the derivative of the value": `0` for `S(0)`, `log F(0)`,
`H(0)` and, with the proper default, `S(+inf)` and `log F(+inf)`. The `t`
argument: boolean, complex, string and object `t` raise `TypeError`, also
during tracing; float64 `t` converts to float32 without a warning in 32-bit
mode.

Hand-derived log-normal limits (``z = (log t - mu)/sigma``): as ``t -> 0``,
``z -> -inf`` and ``log f = -log t - log sigma - log(2 pi)/2 - z**2/2 ->
-inf`` (the ``-z**2/2`` term dominates ``-log t = -mu - sigma z``), and ``h =
f/S -> 0`` since ``S -> 1``: ``log f(0) = log h(0) = -inf``, ``h(0) = 0``.
As ``t -> +inf``, ``h ~ z/(sigma t) -> 0`` (Mills ratio ``phi(z)/Phi(-z) ~
z``): ``log h(+inf) = -inf``, ``h(+inf) = 0``.
"""

import warnings

import family_reference as fr
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import INF, NAN, X64, assert_close, assert_tail_close, f

import survjax

LOGNORMAL = pytest.mark.feature("family.lognormal")
METHODS = fr.DIRECT_METHODS
BASE_METHODS = ("log_survival", "survival", "log_cdf", "cumulative_hazard")

NEGATIVE_TIME = {
    "log_survival": 0.0,
    "survival": 1.0,
    "log_cdf": -INF,
    "cumulative_hazard": 0.0,
    "log_prob": -INF,
    "log_hazard": -INF,
    "hazard": 0.0,
}
AT_ZERO = {
    "log_survival": 0.0,
    "survival": 1.0,
    "log_cdf": -INF,
    "cumulative_hazard": 0.0,
    "log_prob": -INF,
    "log_hazard": -INF,
    "hazard": 0.0,
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


def _dist():
    return survjax.LogNormal(meanlog=f(0.3), sdlog=f(0.8))


# --------------------------------------------------------------- policy rows


@LOGNORMAL
@pytest.mark.parametrize("method", METHODS)
def test_negative_time_gives_policy_value(method):
    """spec 4.5, rev 55: for `t<0`, `log S=0`, `S=1`, `log_prob=-inf`, `log_cdf=-inf`, `H=0`, `log_hazard=-inf`, `h=0`."""
    out = getattr(_dist(), method)(f([-1e-30, -1.0, -INF]))
    np.testing.assert_array_equal(np.asarray(out), [NEGATIVE_TIME[method]] * 3)


@LOGNORMAL
@pytest.mark.parametrize("method", METHODS)
def test_values_at_zero(method):
    """spec 4.5, rev 55: at `t=0` the base supplies `log S=0`, `S=1`, `log F=-inf`, `H=0`; the built-in supplies `log f(0) = log h(0) = -inf`, `h(0) = 0` (module docstring)."""
    np.testing.assert_array_equal(np.asarray(getattr(_dist(), method)(f(0.0))), AT_ZERO[method])


@LOGNORMAL
@pytest.mark.parametrize("method", METHODS)
def test_values_at_infinity(method):
    """spec 4.5, rev 55: at `t=+inf`: `log S=-inf`, `S=0`, `log F=0`, `H=+inf`, `log_prob=-inf`; the analytic `log_hazard` override gives the limit `-inf` (`h -> 0`)."""
    np.testing.assert_array_equal(np.asarray(getattr(_dist(), method)(f(INF))), AT_INF[method])


@LOGNORMAL
@pytest.mark.parametrize("method", METHODS)
def test_nan_time_gives_nan(method):
    """spec 4.5, rev 55: NaN `t` gives NaN."""
    assert np.isnan(np.asarray(getattr(_dist(), method)(f(NAN))))


@LOGNORMAL
@pytest.mark.parametrize("method", METHODS)
def test_nan_parameters_give_nan_for_every_time(method):
    """spec 4.5, rev 55: NaN parameters: "NaN where it is undefined (NaN parameters ...)"; base-supplied rows are NaN when the family's value at the substituted input is NaN."""
    for d in (
        survjax.LogNormal(meanlog=f(NAN), sdlog=f(1.0)),
        survjax.LogNormal(meanlog=f(0.0), sdlog=f(NAN)),
    ):
        out = np.asarray(getattr(d, method)(f([-1.0, 0.0, 1.5, INF])))
        assert np.isnan(out).all(), out


@LOGNORMAL
@pytest.mark.parametrize("method", METHODS)
def test_mixed_vector_of_boundary_and_interior_times(method):
    """spec 4.5/5, rev 55: one call over `[-1, 0, 1.5, +inf, NaN]` gives each row's value (interior against mpmath)."""
    out = np.asarray(getattr(_dist(), method)(f([-1.0, 0.0, 1.5, INF, NAN])))
    np.testing.assert_array_equal(out[[0, 1, 3]], [NEGATIVE_TIME[method], AT_ZERO[method], AT_INF[method]])
    want, rtol = fr.direct("lognormal", (0.3, 0.8), 1.5, method)
    assert_tail_close(out[2], want, rtol)
    assert np.isnan(out[4])


# --------------------------------------------------------------- gradients of base-supplied values


@LOGNORMAL
@pytest.mark.parametrize("method", BASE_METHODS)
@pytest.mark.parametrize("t", [-1.0, 0.0, INF], ids=["negative", "zero", "posinf"])
def test_base_supplied_values_have_zero_parameter_gradients(t, method):
    """spec 4.5, rev 55: base-supplied values have gradients "equal to the derivative of the value": `0` at `t<0`, `t=0` and (proper default) `t=+inf`, in `meanlog` and `sdlog`, both modes."""
    fn = lambda m, s: getattr(survjax.LogNormal(meanlog=m, sdlog=s), method)(f(t))  # noqa: E731
    for g in jax.grad(fn, argnums=(0, 1))(f(0.3), f(0.8)) + tuple(jax.jacfwd(fn, argnums=(0, 1))(f(0.3), f(0.8))):
        np.testing.assert_array_equal(np.asarray(g), 0.0)


@LOGNORMAL
@pytest.mark.parametrize("method", ["log_prob", "log_hazard", "hazard"])
def test_negative_time_family_rows_have_zero_parameter_gradients(method):
    """spec 4.5, rev 55: "With valid parameters the policy values keep zero parameter gradients" for the `t<0` density and hazard rows too."""
    gm, gs = jax.grad(lambda m, s: getattr(survjax.LogNormal(meanlog=m, sdlog=s), method)(f(-1.0)), argnums=(0, 1))(
        f(0.3), f(0.8)
    )
    np.testing.assert_array_equal(np.asarray(gm), 0.0)
    np.testing.assert_array_equal(np.asarray(gs), 0.0)


@LOGNORMAL
def test_log_survival_at_infinity_is_the_proper_default():
    """spec 4.2, rev 55: the built-in is proper: `log_survival_at_infinity()` is `-inf` per batch entry, in the parameter dtype."""
    d = survjax.LogNormal(meanlog=f([0.0, 1.0, 2.0]), sdlog=f(1.0))
    out = d.log_survival_at_infinity()
    assert jnp.result_type(out) == jnp.result_type(f(0.0))
    np.testing.assert_array_equal(np.asarray(out), [-INF] * 3)


# --------------------------------------------------------------- t argument policy


NON_NUMERIC = pytest.mark.parametrize(
    "bad",
    [np.array(True), np.array(1.0 + 0j), np.array("1.0"), np.array(1.0, dtype=object)],
    ids=["bool", "complex", "string", "object"],
)


@LOGNORMAL
@NON_NUMERIC
@pytest.mark.parametrize("method", METHODS)
def test_time_with_a_non_numeric_dtype_is_rejected(method, bad):
    """spec 4.5, rev 55: boolean, complex, string and object `t` raise `TypeError`."""
    with pytest.raises(TypeError):
        getattr(_dist(), method)(bad)


@LOGNORMAL
@pytest.mark.parametrize("method", METHODS)
def test_boolean_time_is_rejected_during_tracing(method):
    """spec 4.5, rev 55: the check uses the dtype, so it also applies to a traced `t`."""
    d = _dist()
    with pytest.raises(TypeError):
        jax.jit(lambda t: getattr(d, method)(t))(jnp.asarray(True))


@pytest.mark.skipif(X64, reason="the float64-to-float32 conversion exists only with 64-bit mode disabled")
@LOGNORMAL
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


@LOGNORMAL
@pytest.mark.parametrize("method", METHODS)
def test_direct_methods_agree_under_jit_and_vmap(method):
    """spec 4.1/12.1, rev 55: eager, `jit` and `vmap` over the batch agree, including boundary rows."""
    d = survjax.LogNormal(meanlog=f([0.0, 1.0, -0.5]), sdlog=f([0.5, 1.0, 2.0]))
    t = f([0.0, 1.7, INF])
    want = np.asarray(getattr(d, method)(t))
    assert_close(jax.jit(lambda dd, tt: getattr(dd, method)(tt))(d, t), want)
    assert_close(jax.vmap(lambda dd, tt: getattr(dd, method)(tt))(d, t), want)


@LOGNORMAL
def test_direct_methods_under_debug_nans():
    """spec 7.2/12.1, rev 55: valid parameters over `[0, 1e-30, interior, upper tail, +inf]` raise no NaN eagerly under `jax_debug_nans`, in values or parameter gradients."""
    t = f([0.0, 1e-30, 1.5, fr.lognormal_time(0.3, 0.8, -30.0), INF])
    with jax.debug_nans(True):
        for method in METHODS:
            d = _dist()
            out = np.asarray(getattr(d, method)(t))
            assert not np.isnan(out).any(), (method, out)
            g = jax.grad(lambda m, s, method=method: getattr(survjax.LogNormal(meanlog=m, sdlog=s), method)(t[2]), argnums=(0, 1))(
                f(0.3), f(0.8)
            )
            assert np.isfinite(np.asarray(g)).all(), (method, g)
