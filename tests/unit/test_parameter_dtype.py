"""Shared parameter dtype policy, before public family registration exists."""

import inspect
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from survjax._dtypes import normalize_parameters


def test_component_parameters_share_dtype_and_keep_shapes() -> None:
    """Spec 3.3: promotion includes component leaves without reshaping them."""
    result = normalize_parameters(
        {
            "rate": np.array([1, 2], dtype=np.float32),
            "weights": np.ones((2, 3), dtype=np.float16),
            "scale": 0.5,
        }
    )
    assert {value.dtype for value in result.values()} == {jnp.dtype(jnp.float32)}
    assert {name: value.shape for name, value in result.items()} == {
        "rate": (2,),
        "weights": (2, 3),
        "scale": (),
    }
    np.testing.assert_array_equal(result["weights"], np.ones((2, 3)))


def test_float64_leaf_promotes_family(x64_enabled: bool) -> None:
    """Spec 3.3: chosen float64 precision promotes all own leaves in x64."""
    if not x64_enabled:
        pytest.skip("float64 promotion requires 64-bit mode")
    result = normalize_parameters(
        {"rate": np.ones(2, dtype=np.float32), "scale": np.float64(2)}
    )
    assert all(value.dtype == jnp.float64 for value in result.values())


@pytest.mark.parametrize("value", [1, 1.5, [1, 2], [1.5, 2.0], jnp.asarray(1.5)])
def test_weak_or_integer_inputs_use_default_float(value, x64_enabled: bool) -> None:
    """Spec 3.3: scalar-only/integer-only inputs use the configured default."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = normalize_parameters({"rate": value})
    assert result["rate"].dtype == (jnp.float64 if x64_enabled else jnp.float32)
    assert recorded == []


def test_float16_alone_promotes_to_float32() -> None:
    """Spec 3.3: even a single component leaf has a float32 minimum."""
    result = normalize_parameters({"weights": np.ones((2, 3), dtype=np.float16)})
    assert result["weights"].dtype == jnp.float32


@pytest.mark.parametrize(
    "value",
    [
        True,
        [True],
        1j,
        np.array([1j]),
        "1",
        ["1"],
        np.array([1], dtype=object),
        [1, None],
    ],
)
def test_nonreal_parameters_raise_type_error(value) -> None:
    """Spec 3.3: input-kind misuse fails before numerical calculation."""
    with pytest.raises(TypeError, match="rate must contain real numeric parameters"):
        normalize_parameters({"rate": value})


def test_normalization_preserves_parameter_values() -> None:
    """Spec 4.1: dtype normalization performs no domain validation."""
    result = normalize_parameters({"rate": [-1.0, np.nan, np.inf]})
    np.testing.assert_array_equal(result["rate"], [-1.0, np.nan, np.inf])


def test_traced_weak_leaf_adopts_strong_dtype() -> None:
    """Spec 3.3: jit/vmap normalization never inspects traced values on host."""

    def normalize(scale):
        return normalize_parameters(
            {"rate": jnp.ones((2, 3), dtype=jnp.float32), "scale": scale}
        )

    eager = normalize(0.5)
    compiled = jax.jit(normalize)(0.5)
    mapped = jax.vmap(normalize)(jnp.array([0.5, 1.0], dtype=jnp.float32))
    assert compiled["scale"].dtype == jnp.float32
    np.testing.assert_array_equal(compiled["rate"], eager["rate"])
    np.testing.assert_array_equal(compiled["scale"], 0.5)
    np.testing.assert_array_equal(mapped["scale"], [0.5, 1.0])
    assert mapped["rate"].shape == (2, 2, 3)


def test_x32_warning_is_once_at_user_call_site(x64_enabled: bool) -> None:
    """Spec 3.3: default warning filtering deduplicates each user source line."""
    if x64_enabled:
        pytest.skip("conversion warning requires 64-bit mode disabled")
    values = {"rate": np.ones(2, dtype=np.float64), "scale": np.float64(2)}
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("default")
        for _ in range(2):
            call_line = inspect.currentframe().f_lineno + 1
            result = normalize_parameters(values)
    assert len(recorded) == 1
    assert recorded[0].filename == __file__
    assert recorded[0].lineno == call_line
    assert recorded[0].category is RuntimeWarning
    assert all(value.dtype == jnp.float32 for value in result.values())
