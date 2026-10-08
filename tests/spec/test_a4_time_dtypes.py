"""A4: dtype of the direct-method time argument (spec 4.5, rev 20).

Rev 20: "The `t` argument follows JAX conversion and promotion. ... Unlike
observation fields and parameters, a float64 array `t` with 64-bit mode
disabled is converted to float32 without a warning". Rev 21: "Boolean,
complex, string and object `t` raise `TypeError` ..., also during tracing;
the check uses the dtype, not values."

The sentence is about "the `t` argument" of "Distribution methods" (4.5), so
every Slice A direct method is checked. Error message text is not asserted.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import FLOAT, MISUSE, X64, assert_close, f

requires_x32 = pytest.mark.skipif(
    X64, reason="float64 conversion of `t` applies with 64-bit mode disabled"
)
requires_x64 = pytest.mark.skipif(
    not X64, reason="float32 `t` promotes to float64 only with 64-bit mode enabled"
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


def _exp(rate=0.5):
    return fam.const_hazard()(rate=f(rate))


def _not_deprecations(record):
    return [
        w
        for w in record
        if not issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning))
    ]


# --------------------------------------------------------------- boolean and complex t


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_python_bool_time_is_rejected(method):
    """spec 4.5, rev 20: boolean `t` raises `TypeError` (a Python `bool`)."""
    with pytest.raises(MISUSE):
        getattr(_exp(), method)(True)


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_numpy_boolean_array_time_is_rejected(method):
    """spec 4.5, rev 20: boolean `t` raises `TypeError` (a NumPy boolean array)."""
    with pytest.raises(MISUSE):
        getattr(_exp(), method)(np.array([True, False]))


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_jax_boolean_array_time_is_rejected(method):
    """spec 4.5, rev 20: boolean `t` raises `TypeError` (a JAX boolean array)."""
    with pytest.raises(MISUSE):
        getattr(_exp(), method)(jnp.array([True, False]))


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_complex_array_time_is_rejected(method):
    """spec 4.5, rev 20: complex `t` raises `TypeError` (a NumPy complex array with zero imaginary part)."""
    with pytest.raises(MISUSE):
        getattr(_exp(), method)(np.array([1.0 + 0.0j, 2.0 + 0.0j]))


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_python_complex_time_is_rejected(method):
    """spec 4.5, rev 20: complex `t` raises `TypeError` (a Python `complex`)."""
    with pytest.raises(MISUSE):
        getattr(_exp(), method)(1.0 + 0.0j)


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_string_time_is_rejected(method):
    """spec 4.5, rev 21: string `t` raises `TypeError`."""
    with pytest.raises(MISUSE):
        getattr(_exp(), method)(np.array(["1.0", "2.0"]))


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_object_array_time_is_rejected(method):
    """spec 4.5, rev 21: object `t` raises `TypeError` (an object array holding numbers)."""
    with pytest.raises(MISUSE):
        getattr(_exp(), method)(np.array([1.0, 2.0], dtype=object))


@pytest.mark.step("A4")
def test_boolean_time_is_rejected_under_jit():
    """spec 4.5, rev 21: boolean `t` raises `TypeError` "also during tracing"."""
    with pytest.raises(MISUSE):
        jax.jit(lambda d, s: d.log_survival(s))(_exp(), np.array([True, False]))


@pytest.mark.step("A4")
def test_complex_time_is_rejected_under_jit():
    """spec 4.5, rev 21: complex `t` raises `TypeError` "also during tracing"."""
    with pytest.raises(MISUSE):
        jax.jit(lambda d, s: d.log_prob(s))(_exp(), np.array([1.0 + 0.0j]))


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_boolean_time_is_rejected_under_jit_for_every_method(method):
    """spec 4.5, rev 21: the traced check covers every direct method (a boolean array traced by `jit`)."""
    with pytest.raises(MISUSE):
        jax.jit(lambda d, s: getattr(d, method)(s))(_exp(), np.array([True, False]))


@pytest.mark.step("A4")
def test_boolean_time_is_rejected_under_vmap():
    """spec 4.5, rev 21: boolean `t` raises `TypeError` when traced by `vmap`."""
    with pytest.raises(MISUSE):
        jax.vmap(_exp().log_survival)(np.array([True, False]))


# --------------------------------------------------------------- float64 t under 32-bit mode


@pytest.mark.step("A4")
@requires_x32
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_float64_time_converts_without_warning_under_32bit_mode(method):
    """spec 4.5, rev 20: "a float64 array `t` with 64-bit mode disabled is converted to float32 without a warning".

    ``simplefilter("always")`` defeats Python's once-per-location registry,
    so a warning suppressed only by deduplication would still be recorded.
    """
    dist = _exp()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        getattr(dist, method)(np.array([0.5, 1.0, 2.0], dtype=np.float64))
    assert _not_deprecations(record) == []


@pytest.mark.step("A4")
@requires_x32
def test_float64_time_result_is_float32_under_32bit_mode():
    """spec 4.5, rev 20: float64 `t` is "converted to float32"; with float32 parameters the result is float32.

    ``log S(t) = -r t`` with ``r = 0.5``: ``(-0.25, -0.5, -1)``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _exp().log_survival(np.array([0.5, 1.0, 2.0], dtype=np.float64))
    assert np.asarray(out).dtype == np.float32
    assert_close(out, [-0.25, -0.5, -1.0])


@pytest.mark.step("A4")
@requires_x64
def test_float32_time_promotes_to_parameter_dtype_under_64bit_mode():
    """spec 4.5, rev 20: `t` follows JAX promotion; "the output precision is set by the parameters".

    float32 ``t`` with float64 ``rate`` promotes to float64. ``log S(1.5) = -0.75``
    (``1.5`` is exact in float32).
    """
    out = _exp().log_survival(np.array([1.5], dtype=np.float32))
    assert np.asarray(out).dtype == np.float64
    assert_close(out, [-0.75])


# --------------------------------------------------------------- integer t


@pytest.mark.step("A4")
def test_integer_array_time_log_survival_matches_float_time():
    """spec 4.5, rev 20: other numeric `t` "follows JAX conversion and promotion": integer `t` gives the float value.

    Constant hazard ``r = 0.5``: ``log S(t) = -r t`` at ``t = (0, 1, 2)`` is ``(0, -0.5, -1)``.
    """
    out = _exp().log_survival(np.array([0, 1, 2], dtype=np.int32))
    assert np.asarray(out).dtype == FLOAT
    assert_close(out, [0.0, -0.5, -1.0])


@pytest.mark.step("A4")
def test_integer_array_time_log_prob_matches_float_time():
    """spec 4.5, rev 20: integer `t` gives the float value for the autodiff `log_prob` default.

    ``log f(t) = log r - r t`` with ``r = 0.5`` at ``t = (0, 1, 2)``:
    ``(log 0.5, log 0.5 - 0.5, log 0.5 - 1)``. Differentiating with respect
    to an integer ``t`` is not possible in JAX, so this checks conversion
    happens before the derivative.
    """
    out = _exp().log_prob(np.array([0, 1, 2], dtype=np.int32))
    assert np.asarray(out).dtype == FLOAT
    assert_close(out, np.log(0.5) - 0.5 * np.array([0.0, 1.0, 2.0]))


@pytest.mark.step("A4")
def test_integer_array_time_hazard_matches_float_time():
    """spec 4.5, rev 20: integer `t` gives the float value for the generic `hazard` default (``h = r = 0.5``)."""
    out = _exp().hazard(np.array([1, 2], dtype=np.int32))
    assert np.asarray(out).dtype == FLOAT
    assert_close(out, [0.5, 0.5])


@pytest.mark.step("A4")
def test_python_int_time_matches_float_time():
    """spec 4.5, rev 20: a Python `int` `t` gives the float value: ``S(2) = exp(-0.5 * 2) = exp(-1)``."""
    out = _exp().survival(2)
    assert np.asarray(out).dtype == FLOAT
    assert_close(out, np.exp(-1.0))


@pytest.mark.step("A4")
def test_negative_integer_time_gets_policy_value():
    """spec 4.5, rev 20: an integer `t<0` is converted and gets the `t<0` policy (`S=1`, `log_prob=-inf`)."""
    dist = _exp()
    t = np.array([-1, -3], dtype=np.int32)
    np.testing.assert_array_equal(np.asarray(dist.survival(t)), [1.0, 1.0])
    np.testing.assert_array_equal(np.asarray(dist.log_prob(t)), [-np.inf, -np.inf])
