"""A5: constructing the built-in Exponential and Weibull families (spec 4.1, 4.2, 10.2).

The generic base behaviour (registration, broadcasting, dtype policy) is tested
on custom families at A4. These tests check that the two Slice A built-ins are
registered families with the constructor keywords of spec 4.1, and the
built-in-only items: the ``from_log`` classmethods and ``reference_time``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import FLOAT, INF, MISUSE, NAN, X64, assert_close, f, leaves

import survjax

# --------------------------------------------------------------- exports and class


@pytest.mark.step("A5")
def test_exponential_is_exported_at_top_level():
    """spec 10.2, rev 21: distributions are re-exported at top level (`survjax.Exponential`)."""
    assert hasattr(survjax, "Exponential")


@pytest.mark.step("A5")
def test_weibull_is_exported_at_top_level():
    """spec 10.2, rev 21: distributions are re-exported at top level (`survjax.Weibull`)."""
    assert hasattr(survjax, "Weibull")


@pytest.mark.step("A5")
def test_exponential_is_a_survival_distribution():
    """spec 4.1, rev 22: "Built-in families are registered subclasses of `SurvivalDistribution`"."""
    assert isinstance(survjax.Exponential(rate=f(0.5)), survjax.SurvivalDistribution)


@pytest.mark.step("A5")
def test_weibull_is_a_survival_distribution():
    """spec 4.1, rev 22: "Built-in families are registered subclasses of `SurvivalDistribution`"."""
    dist = survjax.Weibull(shape=f(2.0), scale=f(3.0))
    assert isinstance(dist, survjax.SurvivalDistribution)


# --------------------------------------------------------------- keywords


@pytest.mark.step("A5")
def test_exponential_rejects_positional_rate():
    """spec 4.2, rev 21: registered families, "built-in and custom", are constructed with keywords only."""
    with pytest.raises(MISUSE):
        survjax.Exponential(f(0.5))


@pytest.mark.step("A5")
def test_weibull_rejects_positional_parameters():
    """spec 4.2, rev 21: registered families, "built-in and custom", are constructed with keywords only."""
    with pytest.raises(MISUSE):
        survjax.Weibull(f(2.0), f(3.0))


@pytest.mark.step("A5")
def test_weibull_requires_both_parameters():
    """spec 4.2, rev 21: "Parameter fields have no default and are required"."""
    with pytest.raises(TypeError):
        survjax.Weibull(shape=f(2.0))
    with pytest.raises(TypeError):
        survjax.Weibull(scale=f(3.0))


@pytest.mark.step("A5")
def test_exponential_requires_rate():
    """spec 4.2, rev 21: "Parameter fields have no default and are required"."""
    with pytest.raises(TypeError):
        survjax.Exponential()


# --------------------------------------------------------------- natural parameters as leaves


@pytest.mark.step("A5")
def test_exponential_stores_rate_as_its_only_leaf():
    """spec 4.1, rev 21: "Natural parameters are stored as the leaves, under the names the constructor uses"."""
    dist = survjax.Exponential(rate=f([0.5, 2.0]))
    np.testing.assert_array_equal(np.asarray(dist.rate), f([0.5, 2.0]))
    assert len(leaves(dist)) == 1


@pytest.mark.step("A5")
def test_weibull_stores_shape_and_scale_as_its_only_leaves():
    """spec 4.1, rev 21: Weibull leaves are the natural `shape` and `scale`, under those names."""
    dist = survjax.Weibull(shape=f([0.5, 2.0]), scale=f([3.0, 4.0]))
    np.testing.assert_array_equal(np.asarray(dist.shape), f([0.5, 2.0]))
    np.testing.assert_array_equal(np.asarray(dist.scale), f([3.0, 4.0]))
    assert len(leaves(dist)) == 2


@pytest.mark.step("A5")
def test_weibull_shape_attribute_is_the_parameter_not_the_batch_shape():
    """spec 4.1, rev 21: "The Weibull keyword stays `shape`"; `.shape` is the parameter, `batch_shape` the array shape."""
    dist = survjax.Weibull(shape=f([1.5, 2.5, 3.5]), scale=f(2.0))
    assert dist.batch_shape == (3,)
    np.testing.assert_array_equal(np.asarray(dist.shape), f([1.5, 2.5, 3.5]))


@pytest.mark.step("A5")
def test_exponential_has_no_shape_attribute():
    """spec 4.1, rev 21: on a family without a `shape` field, accessing `.shape` raises `AttributeError`."""
    dist = survjax.Exponential(rate=f([0.5, 2.0]))
    with pytest.raises(AttributeError):
        _ = dist.shape


# --------------------------------------------------------------- batch shape


@pytest.mark.step("A5")
def test_scalar_exponential_has_empty_batch_shape():
    """spec 4.1, rev 21: `batch_shape` is a tuple of Python ints from the parameter leaves."""
    assert survjax.Exponential(rate=f(0.5)).batch_shape == ()


@pytest.mark.step("A5")
def test_weibull_parameters_are_broadcast_at_construction():
    """spec 4.1, rev 21: "Batch dimensions are broadcast at construction, so every parameter leaf shares the leading batch shape"."""
    dist = survjax.Weibull(shape=f([1.0, 2.0, 3.0]), scale=f([[1.0], [10.0]]))
    assert dist.batch_shape == (2, 3)
    assert jnp.shape(dist.shape) == (2, 3)
    assert jnp.shape(dist.scale) == (2, 3)
    np.testing.assert_array_equal(np.asarray(dist.shape)[1], f([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(np.asarray(dist.scale)[:, 2], f([1.0, 10.0]))


@pytest.mark.step("A5")
def test_weibull_incompatible_parameter_batches_are_rejected():
    """spec 4.1/5, rev 21: incompatible parameter batch shapes raise `TypeError` (the `(D,)` against `(N,)` pitfall)."""
    with pytest.raises(MISUSE):
        survjax.Weibull(shape=f([1.0, 2.0]), scale=f([1.0, 2.0, 3.0]))


@pytest.mark.step("A5")
def test_weibull_posterior_style_construction_gives_draw_by_individual_batch():
    """spec 5, rev 21: `Weibull(shape=k[:, None], scale=...)` with `D=2` draws and `N=3` scales has batch `(D,N)`."""
    k = f([1.5, 2.0])
    scale = f([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dist = survjax.Weibull(shape=k[:, None], scale=scale)
    assert dist.batch_shape == (2, 3)


# --------------------------------------------------------------- no value checks


@pytest.mark.step("A5")
def test_built_in_constructors_accept_invalid_values_eagerly():
    """spec 4.1, rev 21: "Constructors never check parameter values"."""
    survjax.Exponential(rate=f([-1.0, 0.0, NAN, INF]))
    survjax.Weibull(shape=f([-1.0, 0.0, NAN]), scale=f([0.0, -2.0, INF]))


@pytest.mark.step("A5")
def test_built_in_constructors_accept_invalid_values_under_jit():
    """spec 4.1, rev 21: no value checks, so eager and `jit` construction behave alike."""

    def build(r, k, b):
        return survjax.Exponential(rate=r), survjax.Weibull(shape=k, scale=b)

    jax.jit(build)(f(-1.0), f(0.0), f(-2.0))


# --------------------------------------------------------------- traced construction


@pytest.mark.step("A5")
def test_weibull_construction_with_traced_parameters_matches_eager():
    """spec 4.1, rev 21: "Construction with traced parameters inside a model or objective must work".

    ``log S = -(t/b)**k``; ``k=2``, ``b=4``, ``t=2`` gives ``-0.25``.
    """
    out = jax.jit(lambda k, b: survjax.Weibull(shape=k, scale=b).log_survival(f(2.0)))(
        f(2.0), f(4.0)
    )
    assert_close(out, -0.25)


@pytest.mark.step("A5")
def test_exponential_construction_inside_vmap():
    """spec 4.1, rev 21: construction inside `vmap`; `log S = -r t` per mapped rate."""
    out = jax.vmap(lambda r: survjax.Exponential(rate=r).log_survival(f(2.0)))(
        f([0.5, 1.0, 3.0])
    )
    assert_close(out, [-1.0, -2.0, -6.0])


# --------------------------------------------------------------- indexing and slicing


@pytest.mark.step("A5")
def test_weibull_indexing_selects_batch_entries():
    """spec 4.1, rev 21: `dist[idx]` indexes batch dimensions."""
    dist = survjax.Weibull(shape=f([0.5, 1.0, 2.0]), scale=f([1.0, 2.0, 3.0]))
    sub = dist[1]
    assert isinstance(sub, survjax.Weibull)
    assert sub.batch_shape == ()
    assert_close(sub.shape, 1.0)
    assert_close(sub.scale, 2.0)


@pytest.mark.step("A5")
def test_exponential_tree_map_slicing():
    """spec 4.1, rev 21: `jax.tree.map` slicing works and gives a distribution of the sliced batch."""
    dist = survjax.Exponential(rate=f([0.5, 1.0, 2.0]))
    sub = jax.tree.map(lambda x: x[:2], dist)
    assert isinstance(sub, survjax.Exponential)
    assert sub.batch_shape == (2,)
    assert_close(sub.log_survival(f(1.0)), [-0.5, -1.0])


@pytest.mark.step("A5")
def test_vmap_over_weibull_matches_batched_call():
    """spec 4.1/12.1, rev 21: `vmap` over a distribution agrees with the batched call."""
    dist = survjax.Weibull(shape=f([0.5, 1.0, 2.0]), scale=f([1.0, 2.0, 3.0]))
    t = f(1.5)
    assert_close(jax.vmap(lambda d: d.log_prob(t))(dist), dist.log_prob(t))


# --------------------------------------------------------------- dtype


@pytest.mark.step("A5")
def test_integer_rate_uses_default_float_dtype():
    """spec 3.3, rev 21: integer-only parameters use the configured default floating dtype."""
    dist = survjax.Exponential(rate=np.asarray([1, 2], dtype=np.int32))
    assert np.dtype(dist.rate.dtype) == FLOAT


@pytest.mark.step("A5")
def test_weibull_python_scalar_adopts_float32_array_dtype():
    """spec 3.3, rev 21: weak typing: a Python scalar `scale` adopts the float32 dtype of `shape`."""
    dist = survjax.Weibull(shape=np.asarray([1.5, 2.0], dtype=np.float32), scale=2.0)
    assert np.dtype(dist.shape.dtype) == np.float32
    assert np.dtype(dist.scale.dtype) == np.float32


@pytest.mark.step("A5")
@pytest.mark.skipif(not X64, reason="float64 parameters exist only with 64-bit mode")
def test_weibull_mixed_float_parameters_share_promoted_dtype():
    """spec 3.3, rev 21: all leaves share one floating dtype from JAX promotion (float32 with float64 gives float64)."""
    dist = survjax.Weibull(
        shape=np.asarray(1.5, dtype=np.float32), scale=np.asarray(2.0, dtype=np.float64)
    )
    assert np.dtype(dist.shape.dtype) == np.float64
    assert np.dtype(dist.scale.dtype) == np.float64


# --------------------------------------------------------------- reference_time


@pytest.mark.step("A5")
def test_exponential_reference_time_is_positive_finite_per_batch_entry():
    """spec 4.2, rev 22: `reference_time()` is "an array of shape `batch_shape` in the parameter dtype, positive and finite for valid parameters"."""
    ref = survjax.Exponential(rate=f([1e-3, 0.5, 2.0, 1e3])).reference_time()
    assert np.dtype(ref.dtype) == FLOAT
    rt = np.asarray(ref)
    assert rt.shape == (4,)
    assert np.all(np.isfinite(rt))
    assert np.all(rt > 0)


@pytest.mark.step("A5")
def test_weibull_reference_time_is_positive_finite_per_batch_entry():
    """spec 4.2, rev 22: `reference_time()` is "an array of shape `batch_shape` in the parameter dtype, positive and finite for valid parameters".

    Shapes from 0.2 to 5 and scales from 1e-3 to 1e3, all valid.
    """
    dist = survjax.Weibull(shape=f([[0.2], [1.0], [5.0]]), scale=f([1e-3, 1.0, 1e3]))
    ref = dist.reference_time()
    assert np.dtype(ref.dtype) == FLOAT
    rt = np.asarray(ref)
    assert rt.shape == (3, 3)
    assert np.all(np.isfinite(rt))
    assert np.all(rt > 0)


@pytest.mark.step("A5")
def test_weibull_reference_time_under_jit_matches_eager():
    """spec 4.1/12.1, rev 21: methods are pure, so `reference_time()` is the same eagerly and under `jit`."""
    dist = survjax.Weibull(shape=f([0.5, 2.0]), scale=f([1.0, 3.0]))
    assert_close(jax.jit(lambda d: d.reference_time())(dist), dist.reference_time())
