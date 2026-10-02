"""A4: parameter handling: batch shape, broadcasting, component axes, dtype policy, traced construction (spec 3.3, 4.1, 4.2)."""

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import FLOAT, MISUSE, X64, assert_close, f, leaves

requires_x32 = pytest.mark.skipif(
    X64, reason="policy applies with 64-bit mode disabled"
)
requires_x64 = pytest.mark.skipif(
    not X64, reason="policy applies with 64-bit mode enabled"
)

_HERE = Path(__file__).resolve()


def _at_this_file(record):
    """Warnings attributed to a line of this file, i.e. to the caller's call site (Q-004)."""
    return [w for w in record if Path(w.filename).resolve() == _HERE]


def _dtypes(dist):
    return {np.asarray(leaf).dtype for leaf in leaves(dist)}


# --------------------------------------------------------------- batch shape


@pytest.mark.step("A4")
def test_scalar_parameters_give_empty_batch_shape():
    """spec 4.2, rev 16: `batch_shape` from parameter fields; scalars give `()`."""
    dist = fam.power_hazard()(shape=f(1.5), scale=f(2.0))
    assert tuple(dist.batch_shape) == ()


@pytest.mark.step("A4")
def test_batch_shape_is_the_parameter_shape():
    """spec 4.2, rev 16: `batch_shape` from parameter fields."""
    dist = fam.power_hazard()(shape=f([1.5, 2.5, 3.5]), scale=f([2.0, 3.0, 4.0]))
    assert tuple(dist.batch_shape) == (3,)


@pytest.mark.step("A4")
def test_batch_shape_excludes_component_axes():
    """spec 4.1/4.2, rev 11: `batch_shape` excludes component axes (here `w` has `component_ndim=1`)."""
    dist = fam.component_rate()(rate=f([1.0, 2.0, 3.0]), w=f(np.ones((3, 4))))
    assert tuple(dist.batch_shape) == (3,)


@pytest.mark.step("A4")
def test_multidimensional_batch_shape():
    """spec 4.1, rev 11: batch dimensions may be multidimensional, e.g. `(D, N)`."""
    dist = fam.const_hazard()(rate=f(np.ones((2, 3))))
    assert tuple(dist.batch_shape) == (2, 3)


@pytest.mark.step("A4")
def test_batch_shape_is_a_tuple_of_python_ints():
    """spec 4.1, rev 16: `batch_shape` is "a property returning a tuple of Python ints"."""
    shape = fam.component_rate()(
        rate=f([1.0, 2.0, 3.0]), w=f(np.ones((3, 4)))
    ).batch_shape
    assert isinstance(shape, tuple)
    assert all(type(n) is int for n in shape)
    assert shape == (3,)


@pytest.mark.step("A4")
def test_batch_shape_is_static_under_jit():
    """spec 4.1, rev 16: `batch_shape` is Python ints, so it can size arrays inside `jit`."""
    out = jax.jit(lambda d: jnp.zeros(d.batch_shape))(
        fam.const_hazard()(rate=f(np.ones((2, 3))))
    )
    assert out.shape == (2, 3)


@pytest.mark.step("A4")
def test_component_parameter_with_incompatible_batch_is_rejected():
    """spec 4.1, rev 16: leading (batch) axes of a component parameter that do not broadcast raise `TypeError`."""
    with pytest.raises(MISUSE):
        fam.component_rate()(rate=f([1.0, 2.0, 3.0]), w=f(np.ones((2, 4))))


# --------------------------------------------------------------- broadcasting


@pytest.mark.step("A4")
def test_batch_dimensions_are_broadcast_at_construction():
    """spec 4.1, rev 11: "every parameter leaf shares the leading batch shape" after construction."""
    dist = fam.power_hazard()(shape=f([[1.5], [2.5]]), scale=f([2.0, 3.0, 4.0]))
    assert np.shape(dist.shape) == (2, 3)
    assert np.shape(dist.scale) == (2, 3)
    assert tuple(dist.batch_shape) == (2, 3)


@pytest.mark.step("A4")
def test_broadcasting_keeps_parameter_values():
    """spec 4.1, rev 11: broadcasting replicates values; it does not transform them."""
    dist = fam.power_hazard()(shape=f([[1.5], [2.5]]), scale=f([2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(
        np.asarray(dist.shape), [[1.5, 1.5, 1.5], [2.5, 2.5, 2.5]]
    )
    np.testing.assert_array_equal(
        np.asarray(dist.scale), [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]
    )


@pytest.mark.step("A4")
def test_scalar_parameter_is_broadcast_to_batch_shape():
    """spec 4.1, rev 11: a scalar parameter leaf gains the batch shape of the others."""
    dist = fam.power_hazard()(shape=f(1.5), scale=f([2.0, 3.0, 4.0]))
    assert np.shape(dist.shape) == (3,)


@pytest.mark.step("A4")
def test_component_axis_stays_trailing_when_batch_is_broadcast():
    """spec 4.1, rev 11: "component axes such as a coefficient vector stay trailing"."""
    dist = fam.component_rate()(rate=f([1.0, 2.0, 3.0]), w=f([1.0, 2.0, 3.0, 4.0]))
    assert np.shape(dist.rate) == (3,)
    assert np.shape(dist.w) == (3, 4)


@pytest.mark.step("A4")
def test_component_axis_is_not_broadcast_against_batch():
    """spec 4.1/4.2, rev 11: a component axis whose length equals the batch length is still a component axis."""
    dist = fam.component_rate()(rate=f([1.0, 2.0, 3.0, 4.0]), w=f([1.0, 2.0, 3.0, 4.0]))
    assert tuple(dist.batch_shape) == (4,)
    assert np.shape(dist.w) == (4, 4)


@pytest.mark.step("A4")
def test_batch_shape_can_come_from_a_component_parameter():
    """spec 4.1/4.2, rev 11: leading axes of a component parameter are batch axes and broadcast the others."""
    dist = fam.component_rate()(rate=f(2.0), w=f(np.ones((5, 4))))
    assert tuple(dist.batch_shape) == (5,)
    assert np.shape(dist.rate) == (5,)
    assert np.shape(dist.w) == (5, 4)


@pytest.mark.step("A4")
def test_incompatible_batch_shapes_are_rejected():
    """spec 4.1, rev 16: incompatible parameter batch shapes at construction are static misuse (`TypeError`)."""
    with pytest.raises(MISUSE):
        fam.power_hazard()(shape=f([1.5, 2.5]), scale=f([2.0, 3.0, 4.0]))


# --------------------------------------------------------------- .shape


@pytest.mark.step("A4")
def test_weibull_style_shape_keyword_is_the_parameter():
    """spec 4.1, rev 17: `.shape` is never the array shape; a `shape` keyword is a parameter leaf."""
    dist = fam.power_hazard()(shape=f([1.5, 2.5, 3.5]), scale=f(2.0))
    np.testing.assert_array_equal(np.asarray(dist.shape), [1.5, 2.5, 3.5])
    assert tuple(dist.batch_shape) == (3,)


@pytest.mark.step("A4")
def test_distribution_without_shape_field_has_no_shape_attribute():
    """spec 4.1, rev 17: "The base class defines no `shape` attribute"."""
    dist = fam.const_hazard()(rate=f([1.0, 2.0]))
    assert not hasattr(dist, "shape")


@pytest.mark.step("A4")
def test_missing_shape_attribute_raises_attribute_error():
    """spec 4.1, rev 17: accessing `.shape` on a family without that field raises `AttributeError`, so `getattr` defaults work."""
    dist = fam.const_hazard()(rate=f([1.0, 2.0]))
    with pytest.raises(AttributeError):
        _ = dist.shape
    assert getattr(dist, "shape", None) is None


# --------------------------------------------------------------- parameter values are not checked


@pytest.mark.step("A4")
def test_constructor_accepts_negative_parameter_eagerly():
    """spec 4.1, rev 11: "Constructors never check parameter values"; the value is stored unchanged."""
    dist = fam.const_hazard()(rate=f(-1.0))
    np.testing.assert_array_equal(np.asarray(dist.rate), -1.0)


@pytest.mark.step("A4")
def test_constructor_accepts_nan_and_infinite_parameters_eagerly():
    """spec 4.1, rev 11: constructors never check parameter values (NaN, inf are stored)."""
    dist = fam.power_hazard()(shape=f(np.nan), scale=f(np.inf))
    assert np.isnan(np.asarray(dist.shape))
    assert np.isposinf(np.asarray(dist.scale))


@pytest.mark.step("A4")
def test_constructor_accepts_invalid_parameters_under_jit():
    """spec 4.1, rev 11: eager and jit construction behave the same for invalid parameter values."""
    rate = jax.jit(lambda r: fam.const_hazard()(rate=r).rate)(f(-1.0))
    np.testing.assert_array_equal(np.asarray(rate), -1.0)


# --------------------------------------------------------------- traced construction


@pytest.mark.step("A4")
def test_construction_with_traced_parameters_inside_jit():
    """spec 4.1, rev 11: construction with traced parameters inside a model must work.

    ``log f(t) = log r - r t``; with ``r=0.5``, ``t=2``: ``log 0.5 - 1``.
    """
    out = jax.jit(lambda r: fam.const_hazard()(rate=r).log_prob(f(2.0)))(f(0.5))
    assert_close(out, np.log(0.5) - 1.0)


@pytest.mark.step("A4")
def test_traced_construction_broadcasts_like_eager():
    """spec 4.1, rev 11: batch broadcasting at construction also applies to traced parameters."""
    k, b = f([[1.5], [2.5]]), f([2.0, 3.0, 4.0])

    def build(k, b):
        dist = fam.power_hazard()(shape=k, scale=b)
        return dist.shape, dist.scale

    traced_k, traced_b = jax.jit(build)(k, b)
    eager_k, eager_b = build(k, b)
    np.testing.assert_array_equal(np.asarray(traced_k), np.asarray(eager_k))
    np.testing.assert_array_equal(np.asarray(traced_b), np.asarray(eager_b))


@pytest.mark.step("A4")
def test_construction_inside_vmap():
    """spec 4.1, rev 11: construction from batched traced parameters inside `vmap` works.

    ``log f(t) = log r - r t`` at ``t=2`` for each rate.
    """
    rates = f([0.5, 1.0, 2.0])
    out = jax.vmap(lambda r: fam.const_hazard()(rate=r).log_prob(f(2.0)))(rates)
    assert_close(out, np.log([0.5, 1.0, 2.0]) - 2.0 * np.array([0.5, 1.0, 2.0]))


@pytest.mark.step("A4")
def test_gradient_through_construction():
    """spec 4.1, rev 11: traced construction inside an objective is differentiable.

    ``d/dr (log r - r t) = 1/r - t``; with ``r=0.5``, ``t=1``: ``2 - 1 = 1``.
    """
    grad = jax.grad(lambda r: fam.const_hazard()(rate=r).log_prob(f(1.0)))(f(0.5))
    assert_close(grad, 1.0)


# --------------------------------------------------------------- dtype policy


@pytest.mark.step("A4")
def test_integer_parameters_use_default_float_dtype():
    """spec 3.3, rev 11: parameters follow the data dtype policy; integer-only input uses the default float dtype."""
    dist = fam.power_hazard()(
        shape=np.array([1, 2], dtype=np.int32), scale=np.array([3, 4], dtype=np.int32)
    )
    assert _dtypes(dist) == {FLOAT}


@pytest.mark.step("A4")
def test_python_scalar_parameters_use_default_float_dtype():
    """spec 3.3, rev 11: scalar-only input uses the configured default floating dtype."""
    dist = fam.power_hazard()(shape=1.5, scale=2)
    assert _dtypes(dist) == {FLOAT}


@pytest.mark.step("A4")
def test_python_list_parameter_behaves_as_jnp_asarray():
    """spec 3.3, rev 11: Python lists behave as `jnp.asarray` of the list (floats: default float dtype)."""
    dist = fam.const_hazard()(rate=[0.5, 1.5])
    assert _dtypes(dist) == {FLOAT}
    assert np.shape(dist.rate) == (2,)


@pytest.mark.step("A4")
def test_float16_parameter_is_promoted_to_float32():
    """spec 3.3, rev 11: the result is at least float32."""
    dist = fam.const_hazard()(rate=np.array([0.5, 1.5], dtype=np.float16))
    assert _dtypes(dist) == {np.dtype(np.float32)}


@pytest.mark.step("A4")
def test_python_scalar_adopts_float32_array_dtype():
    """spec 3.3, rev 11: weak typing: a Python scalar parameter adopts the dtype of an array parameter."""
    dist = fam.power_hazard()(shape=np.array([1.5, 2.5], dtype=np.float32), scale=2.0)
    assert _dtypes(dist) == {np.dtype(np.float32)}


@pytest.mark.step("A4")
@requires_x64
def test_mixed_float_parameters_share_promoted_dtype():
    """spec 3.3, rev 11: parameters normalize to a common floating dtype using JAX promotion."""
    dist = fam.power_hazard()(
        shape=np.array([1.5, 2.5], dtype=np.float32),
        scale=np.array([2.0, 3.0], dtype=np.float64),
    )
    assert _dtypes(dist) == {np.dtype(np.float64)}


@pytest.mark.step("A4")
def test_component_parameter_shares_the_common_dtype():
    """spec 3.3, rev 11: the dtype policy keeps structured (component) dimensions and applies to them too."""
    dist = fam.component_rate()(
        rate=np.array([1.0, 2.0], dtype=np.float32),
        w=np.array([1, 2, 3], dtype=np.int32),
    )
    assert _dtypes(dist) == {np.dtype(np.float32)}
    assert np.shape(dist.w) == (2, 3)


@pytest.mark.step("A4")
@requires_x32
def test_float64_parameter_converts_to_float32_under_32bit_mode():
    """spec 3.3, rev 11: float64 array input with 64-bit mode disabled converts to float32; not an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dist = fam.const_hazard()(rate=np.array([0.5, 1.5], dtype=np.float64))
    assert _dtypes(dist) == {np.dtype(np.float32)}


@pytest.mark.step("A4")
@requires_x32
def test_float64_parameters_warn_once_per_call():
    """spec 3.3, rev 11: one warning per call site, even when several parameters arrive as float64 (Q-004)."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        fam.power_hazard()(
            shape=np.array([1.5], dtype=np.float64),
            scale=np.array([2.0], dtype=np.float64),
        )
    assert len(_at_this_file(record)) == 1


@pytest.mark.step("A4")
def test_python_scalar_parameters_convert_without_warning():
    """spec 3.3, rev 11: Python scalars and lists carry no chosen precision and convert without a warning."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        fam.power_hazard()(shape=1.5, scale=[2.0, 3.0])
    relevant = [
        w
        for w in record
        if not issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning))
    ]
    assert relevant == []


@pytest.mark.step("A4")
def test_boolean_parameter_is_rejected():
    """spec 3.3, rev 15: boolean values are argument misuse (`TypeError`) for parameters as for times."""
    with pytest.raises(MISUSE):
        fam.const_hazard()(rate=np.array([True, False]))


@pytest.mark.step("A4")
def test_complex_parameter_is_rejected():
    """spec 3.3, rev 15: complex values are argument misuse (`TypeError`)."""
    with pytest.raises(MISUSE):
        fam.const_hazard()(rate=np.array([1.0 + 1.0j]))


@pytest.mark.step("A4")
def test_string_parameter_is_rejected():
    """spec 3.3, rev 15: strings are argument misuse (`TypeError`)."""
    with pytest.raises(MISUSE):
        fam.const_hazard()(rate="0.5")


@pytest.mark.step("A4")
def test_static_field_is_not_converted():
    """spec 4.2, rev 11: dtype normalization applies to parameters; static configuration stays as given."""
    dist = fam.component_rate()(rate=f(1.0), w=f([1.0, 2.0]), method="max")
    assert dist.method == "max"
    assert isinstance(dist.method, str)


@pytest.mark.step("A4")
def test_traced_parameters_keep_their_dtype():
    """spec 3.3, rev 11: traced parameters follow the same policy (float32 traced input stays float32)."""
    dtype = jax.jit(lambda r: fam.const_hazard()(rate=r).rate)(
        jnp.asarray([0.5, 1.5], dtype=jnp.float32)
    ).dtype
    assert np.dtype(dtype) == np.dtype(np.float32)


@pytest.mark.step("A4")
def test_object_parameter_is_rejected():
    """spec 3.3, rev 15: object inputs raise `TypeError`."""
    with pytest.raises(MISUSE):
        fam.const_hazard()(rate=np.array([0.5, None], dtype=object))


@pytest.mark.step("A4")
def test_constructor_takes_no_dtype_keyword():
    """spec 3.3, rev 15: "Constructors take no `dtype` keyword; callers cast inputs themselves"."""
    with pytest.raises(TypeError):
        fam.const_hazard()(rate=f(0.5), dtype=np.float32)


@pytest.mark.step("A4")
@requires_x64
def test_nested_distribution_keeps_its_own_dtype():
    """spec 3.3, rev 15: "Nested distribution fields keep their own dtypes"; the outer family's leaves are normalized on their own."""
    base = fam.const_hazard()(rate=np.array([0.5, 1.0], dtype=np.float32))
    outer = fam.scaled_hazard()(
        base=base, log_ratio=np.array([0.1, 0.2], dtype=np.float64)
    )
    assert np.asarray(outer.base.rate).dtype == np.dtype(np.float32)
    assert np.asarray(outer.log_ratio).dtype == np.dtype(np.float64)
