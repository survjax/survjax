"""Dtype policy cases independent of the implementation."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from chronox_jax import SurvivalData


def test_integer_only_uses_configured_default_float(x64_enabled: bool) -> None:
    """Spec 3.3: integers do not determine a narrow floating dtype."""
    data = SurvivalData(np.array([1, 2], dtype=np.int32), left_trunc=0)
    assert data.lower.dtype == (jnp.float64 if x64_enabled else jnp.float32)


def test_small_float_promotes_to_float32() -> None:
    """Spec 3.3: time fields are at least float32."""
    data = SurvivalData(np.array([1, 2], dtype=np.float16))
    assert all(
        getattr(data, name).dtype == jnp.float32
        for name in ("lower", "upper", "left_trunc", "right_trunc")
    )


def test_float64_warning_and_conversion_in_32bit_mode(x64_enabled: bool) -> None:
    """Spec 3.3: implicit float64 input warns once and converts in 32-bit mode."""
    if x64_enabled:
        pytest.skip("conversion policy applies only with 64-bit mode disabled")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        data = SurvivalData(np.array([1, 2], dtype=np.float64))
    assert data.lower.dtype == jnp.float32
    assert len([item for item in recorded if item.filename == __file__]) == 1


def test_explicit_unavailable_float64_fails(x64_enabled: bool) -> None:
    """Spec 3.3: an unavailable explicit dtype must fail clearly."""
    if x64_enabled:
        pytest.skip("float64 is available with 64-bit mode enabled")
    with pytest.raises(ValueError, match="float64"):
        SurvivalData([1, 2], dtype=jnp.float64)


@pytest.mark.parametrize(
    "value",
    [np.array([True]), np.array([1 + 0j]), np.array(["1"]), np.array([1, None])],
)
def test_nonreal_time_input_is_rejected(value: np.ndarray) -> None:
    """Spec 3.3: bool, complex, string, and object times are invalid."""
    with pytest.raises((TypeError, ValueError)):
        SurvivalData(value)
