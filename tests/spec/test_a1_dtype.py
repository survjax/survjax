"""A1: dtype policy for observation data (spec 3.3)."""

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import FLOAT, REJECTED, TIME_FIELDS, X64, b, f

import chronox_jax as cx

requires_x32 = pytest.mark.skipif(X64, reason="policy applies with 64-bit mode disabled")
requires_x64 = pytest.mark.skipif(not X64, reason="policy applies with 64-bit mode enabled")

_HERE = Path(__file__).resolve()


def _at_this_file(record):
    """Warnings attributed to a line of this file, i.e. to the caller's call site (Q-004)."""
    return [w for w in record if Path(w.filename).resolve() == _HERE]


def _time_dtypes(data):
    return {name: np.asarray(getattr(data, name)).dtype for name in TIME_FIELDS}


def _assert_time_dtype(data, dtype):
    for name, got in _time_dtypes(data).items():
        assert got == np.dtype(dtype), name


def _conversion_warnings(record):
    """Warnings other than deprecations from any library (category otherwise unconstrained, Q-004)."""
    return [
        w
        for w in record
        if not issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning))
    ]


# ---------------------------------------------------------------- promotion


@pytest.mark.step("A1")
def test_time_and_bound_fields_share_one_floating_dtype():
    """spec 3.3, rev 4: time and bound fields normalize to a common floating dtype."""
    data = cx.SurvivalData(
        f([1.0, 2.0]), b([True, False]), left_trunc=f([0.5, 0.0]), right_trunc=f([4.0, np.inf])
    )
    _assert_time_dtype(data, FLOAT)


@pytest.mark.step("A1")
def test_integer_time_uses_default_float_dtype():
    """spec 3.3, rev 4: integer-only input uses the configured default floating dtype."""
    data = cx.SurvivalData(np.array([1, 2, 3], dtype=np.int32))
    _assert_time_dtype(data, FLOAT)


@pytest.mark.step("A1")
def test_integer_time_and_truncation_use_default_float_dtype():
    """spec 3.3, rev 4: integer-only input (time and truncation) uses the default floating dtype."""
    data = cx.SurvivalData(
        np.array([2, 3], dtype=np.int32),
        b([True, False]),
        left_trunc=np.array([1, 0], dtype=np.int32),
    )
    _assert_time_dtype(data, FLOAT)


@pytest.mark.step("A1")
def test_integer_bounds_use_default_float_dtype():
    """spec 3.3, rev 4: integer-only from_bounds input uses the default floating dtype."""
    data = cx.SurvivalData.from_bounds(
        np.array([1, 0], dtype=np.int32), np.array([2, 3], dtype=np.int32)
    )
    _assert_time_dtype(data, FLOAT)


@pytest.mark.step("A1")
def test_float16_time_is_promoted_to_float32():
    """spec 3.3, rev 4: the common dtype is at least float32."""
    data = cx.SurvivalData(np.array([1.0, 2.0], dtype=np.float16))
    _assert_time_dtype(data, np.float32)


@pytest.mark.step("A1")
def test_bfloat16_time_is_promoted_to_float32():
    """spec 3.3, rev 4: the common dtype is at least float32."""
    data = cx.SurvivalData(np.array([1.0, 2.0], dtype=jnp.bfloat16))
    _assert_time_dtype(data, np.float32)


@pytest.mark.step("A1")
def test_float32_time_with_int32_truncation_is_float32():
    """spec 3.3, rev 4: JAX promotion of float32 with int32 gives float32 (both modes)."""
    data = cx.SurvivalData(
        f([2.0, 3.0], np.float32), left_trunc=np.array([1, 0], dtype=np.int32)
    )
    _assert_time_dtype(data, np.float32)


@pytest.mark.step("A1")
@requires_x64
def test_float32_input_stays_float32_under_64bit_mode():
    """spec 3.3, rev 4: JAX promotion, at least float32: float32-only input stays float32."""
    data = cx.SurvivalData(f([1.0, 2.0], np.float32), b([True, False]))
    _assert_time_dtype(data, np.float32)


@pytest.mark.step("A1")
@requires_x64
def test_float32_time_with_float64_truncation_promotes_to_float64():
    """spec 3.3, rev 4: fields normalize to a common dtype using JAX promotion."""
    data = cx.SurvivalData(
        f([2.0, 3.0], np.float32), left_trunc=np.array([1.0, 0.0], dtype=np.float64)
    )
    _assert_time_dtype(data, np.float64)


@pytest.mark.step("A1")
@requires_x64
def test_float64_input_is_kept_under_64bit_mode():
    """spec 3.3, rev 4: float64 is supported when 64-bit mode is enabled."""
    data = cx.SurvivalData(np.array([1.0, 2.0], dtype=np.float64))
    _assert_time_dtype(data, np.float64)


# ------------------------------------- weak typing: Python scalars and lists


@pytest.mark.step("A1")
def test_python_float_scalar_time_uses_default_float_dtype():
    """spec 3.3, rev 4: scalar-only input uses the configured default floating dtype."""
    _assert_time_dtype(cx.SurvivalData(2.0), FLOAT)


@pytest.mark.step("A1")
def test_python_int_scalar_time_uses_default_float_dtype():
    """spec 3.3, rev 4: scalar-only (integer) input uses the configured default floating dtype."""
    _assert_time_dtype(cx.SurvivalData(2), FLOAT)


@pytest.mark.step("A1")
def test_python_scalar_time_and_truncation_use_default_float_dtype():
    """spec 3.3, rev 4: scalar-only input, truncation included, uses the default floating dtype."""
    _assert_time_dtype(cx.SurvivalData(2.0, left_trunc=1.0, right_trunc=5.0), FLOAT)


@pytest.mark.step("A1")
def test_python_scalar_bounds_use_default_float_dtype():
    """spec 3.3, rev 4: scalar-only from_bounds input uses the default floating dtype."""
    _assert_time_dtype(cx.SurvivalData.from_bounds(1.0, 2.0), FLOAT)


@pytest.mark.step("A1")
def test_python_float_truncation_adopts_float32_time_dtype():
    """spec 3.3, rev 4: Python scalars adopt the dtype of the array inputs (weak typing).

    Under 64-bit mode a strongly typed float64 truncation would promote to
    float64; a weakly typed Python float must not.
    """
    data = cx.SurvivalData(f([2.0, 3.0], np.float32), left_trunc=0.5, right_trunc=5.0)
    _assert_time_dtype(data, np.float32)


@pytest.mark.step("A1")
def test_python_int_truncation_adopts_float32_time_dtype():
    """spec 3.3, rev 4: Python int scalars adopt the dtype of the array inputs."""
    data = cx.SurvivalData(f([2.0, 3.0], np.float32), left_trunc=1)
    _assert_time_dtype(data, np.float32)


@pytest.mark.step("A1")
def test_python_infinite_upper_adopts_float32_lower_dtype():
    """spec 3.3, rev 4: Python scalars adopt the array dtype in from_bounds (`upper=float("inf")`)."""
    data = cx.SurvivalData.from_bounds(f([1.0, 2.0], np.float32), float("inf"))
    _assert_time_dtype(data, np.float32)


@pytest.mark.step("A1")
def test_python_float_truncation_with_integer_time_uses_default_float_dtype():
    """spec 3.3, rev 4: JAX promotion of an int32 array with a weak Python float is the default float."""
    data = cx.SurvivalData(np.array([2, 3], dtype=np.int32), left_trunc=0.5)
    _assert_time_dtype(data, FLOAT)


@pytest.mark.step("A1")
def test_python_float_list_time_uses_default_float_dtype():
    """spec 3.3, rev 4: a Python list behaves as `jnp.asarray` of the list (default float dtype)."""
    _assert_time_dtype(cx.SurvivalData([1.0, 2.0]), FLOAT)


@pytest.mark.step("A1")
def test_python_int_list_time_uses_default_float_dtype():
    """spec 3.3, rev 4: an integer list is integer-only input and uses the default floating dtype."""
    _assert_time_dtype(cx.SurvivalData([1, 2, 3]), FLOAT)


@pytest.mark.step("A1")
def test_python_list_bounds_use_default_float_dtype():
    """spec 3.3, rev 4: lists in from_bounds behave as `jnp.asarray` of the list."""
    _assert_time_dtype(cx.SurvivalData.from_bounds([1.0, 0.0], [2.0, 3.0]), FLOAT)


@pytest.mark.step("A1")
def test_python_list_truncation_promotes_as_jnp_asarray():
    """spec 3.3, rev 4: a list truncation argument promotes exactly as `jnp.asarray(list)` would.

    The oracle is JAX's own promotion of the two inputs, which is what the spec
    defines: float32 in 32-bit mode; in 64-bit mode `jnp.asarray` of a list of
    Python floats is a (non-weak) float64 array, so the result is float64.
    """
    time = f([2.0, 3.0], np.float32)
    trunc = [0.5, 0.0]
    expected = jnp.result_type(time, jnp.asarray(trunc))
    data = cx.SurvivalData(time, left_trunc=trunc)
    _assert_time_dtype(data, expected)


# ------------------------------------------- default truncation arguments


@pytest.mark.step("A1")
def test_passing_none_truncation_matches_omitting_it():
    """spec 3.3, rev 4: omitting a truncation argument and passing its default (None, 3.1) give the same dtype (Q-015)."""
    time = f([2.0, 3.0], np.float32)
    omitted = cx.SurvivalData(time)
    passed = cx.SurvivalData(time, left_trunc=None, right_trunc=None)
    assert _time_dtypes(passed) == _time_dtypes(omitted)


@pytest.mark.step("A1")
def test_passing_zero_left_truncation_matches_omitting_it_for_float32_time():
    """spec 3.3, rev 4: weak typing: Python `left_trunc=0` adopts the time dtype, so it matches omitting the argument."""
    time = f([2.0, 3.0], np.float32)
    assert _time_dtypes(cx.SurvivalData(time, left_trunc=0)) == _time_dtypes(
        cx.SurvivalData(time)
    )


@pytest.mark.step("A1")
def test_passing_infinite_right_truncation_matches_omitting_it_for_float32_time():
    """spec 3.3, rev 4: weak typing: Python `right_trunc=inf` adopts the time dtype, so it matches omitting the argument."""
    time = f([2.0, 3.0], np.float32)
    assert _time_dtypes(cx.SurvivalData(time, right_trunc=float("inf"))) == _time_dtypes(
        cx.SurvivalData(time)
    )


@pytest.mark.step("A1")
def test_passing_default_truncation_matches_omitting_it_for_integer_time():
    """spec 3.3, rev 4: weak typing with integer time: Python `0.0`/`inf` truncation keeps the omitted-argument dtype."""
    time = np.array([2, 3], dtype=np.int32)
    passed = cx.SurvivalData(time, left_trunc=0.0, right_trunc=float("inf"))
    assert _time_dtypes(passed) == _time_dtypes(cx.SurvivalData(time))


@pytest.mark.step("A1")
def test_passing_default_truncation_matches_omitting_it_in_from_bounds():
    """spec 3.3, rev 4: weak typing in from_bounds: Python `0`/`inf` truncation keeps the omitted-argument dtype."""
    lower, upper = f([1.0, 0.0], np.float32), f([2.0, 3.0], np.float32)
    passed = cx.SurvivalData.from_bounds(lower, upper, left_trunc=0, right_trunc=float("inf"))
    assert _time_dtypes(passed) == _time_dtypes(cx.SurvivalData.from_bounds(lower, upper))


# ------------------------------------------------ float64 under 32-bit mode


@pytest.mark.step("A1")
@requires_x32
def test_float64_input_converts_to_float32_under_32bit_mode():
    """spec 3.3, rev 4: NumPy float64 array input with 64-bit mode disabled is converted to float32, not an error."""
    time = np.array([1.0, 2.0], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = cx.SurvivalData(time, np.array([True, False]))
    _assert_time_dtype(data, np.float32)
    np.testing.assert_array_equal(np.asarray(data.lower), time.astype(np.float32))


@pytest.mark.step("A1")
@requires_x32
def test_float64_bounds_convert_to_float32_under_32bit_mode():
    """spec 3.3, rev 4: from_bounds follows the same float64 conversion policy."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = cx.SurvivalData.from_bounds(
            np.array([1.0, 0.0], dtype=np.float64), np.array([2.0, 3.0], dtype=np.float64)
        )
    _assert_time_dtype(data, np.float32)


@pytest.mark.step("A1")
@requires_x32
def test_float64_input_warns_once_per_call():
    """spec 3.3, rev 4: one warning per call site, even when several fields arrive as float64."""
    t64 = np.array([1.0, 2.0], dtype=np.float64)
    lt64 = np.array([0.5, 0.0], dtype=np.float64)
    rt64 = np.array([5.0, np.inf], dtype=np.float64)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        cx.SurvivalData(t64, left_trunc=lt64, right_trunc=rt64)
    assert len(_at_this_file(record)) == 1


@pytest.mark.step("A1")
@requires_x32
def test_repeated_calls_from_one_call_site_warn_once():
    """spec 3.3, rev 4: one warning per call site (Python default warning filter semantics)."""
    t64 = np.array([1.0, 2.0], dtype=np.float64)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default")
        for _ in range(3):
            cx.SurvivalData(t64)
    assert len(_at_this_file(record)) == 1


@pytest.mark.step("A1")
@requires_x32
def test_two_call_sites_warn_once_each():
    """spec 3.3, rev 4: one warning per call site, so two distinct call sites warn twice."""
    t64 = np.array([1.0, 2.0], dtype=np.float64)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default")
        cx.SurvivalData(t64)
        cx.SurvivalData(t64)
    assert len(_at_this_file(record)) == 2


@pytest.mark.step("A1")
@requires_x32
def test_pandas_float64_series_converts_to_float32_with_one_warning():
    """spec 3.3, rev 4: pandas float64 array input converts to float32 with one warning (Q-004; Q-016 resolved)."""
    pd = pytest.importorskip("pandas")
    time = pd.Series([1.0, 2.0], dtype="float64")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        data = cx.SurvivalData(time)
    _assert_time_dtype(data, np.float32)
    assert len(_at_this_file(record)) == 1


@pytest.mark.step("A1")
@requires_x32
def test_python_float_scalar_converts_without_warning():
    """spec 3.3, rev 4: Python scalars carry no chosen precision and convert without a warning."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        data = cx.SurvivalData(1.5, left_trunc=0.5, right_trunc=4.0)
    _assert_time_dtype(data, np.float32)
    assert _conversion_warnings(record) == []


@pytest.mark.step("A1")
@requires_x32
def test_python_float_list_converts_without_warning():
    """spec 3.3, rev 4: Python lists carry no chosen precision and convert without a warning."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        data = cx.SurvivalData([1.0, 2.0], [True, False], left_trunc=[0.5, 0.0])
    _assert_time_dtype(data, np.float32)
    assert _conversion_warnings(record) == []


@pytest.mark.step("A1")
@requires_x32
def test_python_list_bounds_convert_without_warning():
    """spec 3.3, rev 4: lists in from_bounds convert without a warning."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        cx.SurvivalData.from_bounds([1.0, 0.0], [2.0, float("inf")])
    assert _conversion_warnings(record) == []


@pytest.mark.step("A1")
@requires_x32
def test_python_scalar_truncation_with_float32_array_does_not_warn():
    """spec 3.3, rev 4: Python scalar truncation next to float32 arrays converts without a warning."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        cx.SurvivalData(f([2.0, 3.0], np.float32), left_trunc=0.5, right_trunc=float("inf"))
    assert _conversion_warnings(record) == []


# --------------------------------------------------------- explicit dtype


@pytest.mark.step("A1")
@requires_x32
def test_explicit_float64_request_fails_under_32bit_mode():
    """spec 3.3, rev 4: an explicit request for an unavailable dtype fails clearly."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 2.0]), dtype=jnp.float64)


@pytest.mark.step("A1")
@requires_x32
def test_explicit_float64_request_fails_in_from_bounds_under_32bit_mode():
    """spec 3.3, rev 4: an explicit request for an unavailable dtype fails clearly (bounds constructor)."""
    with pytest.raises(REJECTED):
        cx.SurvivalData.from_bounds(f([1.0]), f([2.0]), dtype=jnp.float64)


@pytest.mark.step("A1")
def test_explicit_float32_request_gives_float32_fields():
    """spec 3.3, rev 4: float32 is always available; an explicit request is honoured in both modes."""
    data = cx.SurvivalData(f([1.0, 2.0]), b([True, False]), dtype=jnp.float32)
    _assert_time_dtype(data, np.float32)


@pytest.mark.step("A1")
@requires_x64
def test_explicit_float64_request_gives_float64_fields_under_64bit_mode():
    """spec 3.3, rev 4: an explicit available dtype is honoured, even for float32 input."""
    data = cx.SurvivalData(f([1.0, 2.0], np.float32), dtype=jnp.float64)
    _assert_time_dtype(data, np.float64)


# -------------------------------------------------------- rejected inputs


@pytest.mark.step("A1")
def test_boolean_time_is_rejected():
    """spec 3.3, rev 4: boolean times are rejected."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(b([True, False]))


@pytest.mark.step("A1")
def test_boolean_time2_is_rejected():
    """spec 3.3, rev 4: boolean times are rejected (interval upper endpoint)."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([0.0]), interval_censored=b([True]), time2=b([True]))


@pytest.mark.step("A1")
def test_boolean_lower_bound_is_rejected():
    """spec 3.3, rev 4: boolean times are rejected (bounds constructor)."""
    with pytest.raises(REJECTED):
        cx.SurvivalData.from_bounds(b([False]), f([2.0]))


@pytest.mark.step("A1")
def test_boolean_truncation_is_rejected():
    """spec 3.3, rev 4: boolean values are rejected for bound fields."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([2.0]), left_trunc=b([True]))


@pytest.mark.step("A1")
def test_complex_time_is_rejected():
    """spec 3.3, rev 4: complex values are rejected."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(np.array([1.0 + 0.0j, 2.0 + 0.0j]))


@pytest.mark.step("A1")
def test_complex_truncation_is_rejected():
    """spec 3.3, rev 4: complex values are rejected for bound fields."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([2.0]), right_trunc=np.array([5.0 + 0.0j]))


@pytest.mark.step("A1")
def test_string_time_is_rejected():
    """spec 3.3, rev 4: strings are rejected."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(np.array(["1.0", "2.0"]))


@pytest.mark.step("A1")
def test_object_array_with_none_time_is_rejected():
    """spec 3.3, rev 4: object or mixed-None arrays are rejected."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(np.array([1.0, None], dtype=object))


# ------------------------------------------------------ global JAX config


@pytest.mark.step("A1")
def test_construction_does_not_change_global_64bit_setting():
    """spec 3.3, rev 4: Chronox never changes JAX's global 64-bit setting."""
    before = bool(jax.config.jax_enable_x64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cx.SurvivalData(np.array([1.0, 2.0], dtype=np.float64))
    assert bool(jax.config.jax_enable_x64) == before


@pytest.mark.step("A1")
@requires_x32
def test_failed_float64_request_does_not_enable_64bit_mode():
    """spec 3.3, rev 4: Chronox never changes JAX's global 64-bit setting, even on an explicit request."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0]), dtype=jnp.float64)
    assert not bool(jax.config.jax_enable_x64)
