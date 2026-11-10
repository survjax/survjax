"""Feature `family.loglogistic`: constructing `LogLogistic` (spec 3.3, 4.1, 4.2, 5, 10.2, 12.2; rev 55).

12.2 (rev 55): the log-logistic family is enabled "under the full contract of
sections 4.1, 4.2, 4.5, 5 and 7: construction and parameter constraints,
closed-form direct methods with their tail forms, grids, the likelihood modes
implemented at the time, diagnostics and derivatives".

4.1: `LogLogistic(shape=beta, scale=alpha)`, `S(t) = 1/(1+(t/alpha)**beta)`;
registered `SurvivalDistribution` subclass; natural parameters stored as the
leaves "under the names the constructor uses"; "The base class defines no
`shape` attribute"; a `shape` field is the parameter (as for Weibull, whose
"keyword stays `shape`"), and the array shape is `batch_shape`; batch
broadcasting at construction; no value checks; `from_log` "equivalents for
the other families" (keywords: Q-069). 4.2: keyword-only construction,
required parameter fields, `reference_time()`. 3.3: parameter dtype
normalization. 10.2: top-level export.

Hand value used below: at ``t = alpha``, ``S = 1/2`` and ``log S = -log 2``
for every ``beta``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import FLOAT, INF, MISUSE, NAN, X64, assert_close, f, leaves, loglogistic_from_log

import survjax

LOGLOGISTIC = pytest.mark.feature("family.loglogistic")
LOG_HALF = -np.log(2.0)

# --------------------------------------------------------------- exports and class


@LOGLOGISTIC
def test_loglogistic_is_exported_at_top_level():
    """spec 10.2, rev 55: distributions are re-exported at the top level (`survjax.LogLogistic`)."""
    assert hasattr(survjax, "LogLogistic")


@LOGLOGISTIC
def test_loglogistic_is_a_survival_distribution():
    """spec 4.1, rev 55: "Built-in families are registered subclasses of `SurvivalDistribution`"."""
    assert isinstance(survjax.LogLogistic(shape=f(2.0), scale=f(1.0)), survjax.SurvivalDistribution)


# --------------------------------------------------------------- keywords


@LOGLOGISTIC
def test_loglogistic_rejects_positional_parameters():
    """spec 4.2, rev 55: registered families, "built-in and custom", are constructed with keyword arguments only."""
    with pytest.raises(MISUSE):
        survjax.LogLogistic(f(2.0), f(1.0))


@LOGLOGISTIC
def test_loglogistic_requires_both_parameters():
    """spec 4.2, rev 55: "Parameter fields have no default and are required"."""
    with pytest.raises(TypeError):
        survjax.LogLogistic(shape=f(2.0))
    with pytest.raises(TypeError):
        survjax.LogLogistic(scale=f(1.0))


# --------------------------------------------------------------- natural parameters as leaves


@LOGLOGISTIC
def test_loglogistic_stores_shape_and_scale_as_its_only_leaves():
    """spec 4.1, rev 55: the leaves are the natural `shape` and `scale`, under those names."""
    d = survjax.LogLogistic(shape=f([0.5, 3.0]), scale=f([2.0, 0.4]))
    np.testing.assert_array_equal(np.asarray(d.shape), f([0.5, 3.0]))
    np.testing.assert_array_equal(np.asarray(d.scale), f([2.0, 0.4]))
    assert len(leaves(d)) == 2


@LOGLOGISTIC
def test_shape_attribute_is_the_parameter_not_the_batch_shape():
    """spec 4.1, rev 55: the `shape` field is the parameter `beta`; the array shape is `batch_shape`."""
    d = survjax.LogLogistic(shape=f([1.5, 2.5, 3.5]), scale=f(2.0))
    assert d.batch_shape == (3,)
    np.testing.assert_array_equal(np.asarray(d.shape), f([1.5, 2.5, 3.5]))


# --------------------------------------------------------------- batch shape


@LOGLOGISTIC
def test_scalar_loglogistic_has_empty_batch_shape():
    """spec 4.1, rev 55: `batch_shape` is a tuple of Python ints from the parameter leaves."""
    assert survjax.LogLogistic(shape=f(2.0), scale=f(1.0)).batch_shape == ()


@LOGLOGISTIC
def test_loglogistic_parameters_are_broadcast_at_construction():
    """spec 4.1, rev 55: "Batch dimensions are broadcast at construction"."""
    d = survjax.LogLogistic(shape=f([0.5, 1.0, 2.0]), scale=f([[1.0], [10.0]]))
    assert d.batch_shape == (2, 3)
    assert jnp.shape(d.shape) == (2, 3)
    assert jnp.shape(d.scale) == (2, 3)
    np.testing.assert_array_equal(np.asarray(d.shape)[1], f([0.5, 1.0, 2.0]))
    np.testing.assert_array_equal(np.asarray(d.scale)[:, 2], f([1.0, 10.0]))


@LOGLOGISTIC
def test_loglogistic_incompatible_parameter_batches_are_rejected():
    """spec 4.1/5, rev 55: incompatible parameter batch shapes raise `TypeError`."""
    with pytest.raises(MISUSE):
        survjax.LogLogistic(shape=f([1.0, 2.0]), scale=f([1.0, 2.0, 3.0]))


@LOGLOGISTIC
def test_loglogistic_posterior_style_construction_gives_draw_by_individual_batch():
    """spec 5, rev 55: `LogLogistic(shape=beta[:, None], scale=...)` with `D = 2` draws and `N = 3` scales has batch `(D, N)`."""
    d = survjax.LogLogistic(shape=f([1.5, 2.0])[:, None], scale=f([1.0, 2.0, 3.0]))
    assert d.batch_shape == (2, 3)


@LOGLOGISTIC
def test_time_that_does_not_broadcast_is_rejected():
    """spec 4.1, rev 55: a time argument that does not broadcast against `batch_shape` raises `TypeError`."""
    with pytest.raises(MISUSE):
        survjax.LogLogistic(shape=f([1.0, 2.0, 3.0]), scale=f(1.0)).log_survival(f([1.0, 2.0]))


# --------------------------------------------------------------- no value checks


@LOGLOGISTIC
def test_loglogistic_constructor_accepts_invalid_values_eagerly():
    """spec 4.1, rev 55: "Constructors never check parameter values"."""
    survjax.LogLogistic(shape=f([-1.0, 0.0, NAN, 1.0]), scale=f([1.0, 1.0, 1.0, INF]))


@LOGLOGISTIC
def test_loglogistic_constructor_accepts_invalid_values_under_jit():
    """spec 4.1, rev 55: no value checks, so eager and `jit` construction behave alike."""
    jax.jit(lambda k, a: survjax.LogLogistic(shape=k, scale=a))(f(-1.0), f(0.0))


# --------------------------------------------------------------- traced construction, indexing, vmap


@LOGLOGISTIC
def test_loglogistic_construction_with_traced_parameters_matches_hand_value():
    """spec 4.1, rev 55: construction with traced parameters must work; `log S(alpha) = -log 2`."""
    out = jax.jit(lambda k, a: survjax.LogLogistic(shape=k, scale=a).log_survival(a))(f(1.7), f(2.5))
    assert_close(out, LOG_HALF)


@LOGLOGISTIC
def test_loglogistic_construction_inside_vmap():
    """spec 4.1, rev 55: construction inside `vmap`; `log S(alpha) = -log 2` per mapped `scale`."""
    out = jax.vmap(lambda a: survjax.LogLogistic(shape=f(0.8), scale=a).log_survival(a))(f([0.1, 1.0, 30.0]))
    assert_close(out, [LOG_HALF] * 3)


@LOGLOGISTIC
def test_loglogistic_indexing_selects_batch_entries():
    """spec 4.1, rev 55: `dist[idx]` indexes batch dimensions and returns a `LogLogistic`."""
    d = survjax.LogLogistic(shape=f([0.5, 1.0, 2.0]), scale=f([1.0, 2.0, 3.0]))
    sub = d[1]
    assert isinstance(sub, survjax.LogLogistic)
    assert sub.batch_shape == ()
    assert_close(sub.shape, 1.0)
    assert_close(sub.scale, 2.0)


@LOGLOGISTIC
def test_loglogistic_tree_map_slicing():
    """spec 4.1, rev 55: `jax.tree.map` slicing gives a `LogLogistic` of the sliced batch."""
    d = survjax.LogLogistic(shape=f([0.5, 1.0, 2.0]), scale=f([1.0, 2.0, 3.0]))
    sub = jax.tree.map(lambda x: x[:2], d)
    assert isinstance(sub, survjax.LogLogistic)
    assert sub.batch_shape == (2,)
    assert_close(sub.log_survival(f([1.0, 2.0])), [LOG_HALF, LOG_HALF])


@LOGLOGISTIC
def test_vmap_over_loglogistic_matches_batched_call():
    """spec 4.1/12.1, rev 55: `vmap` over a distribution agrees with the batched call."""
    d = survjax.LogLogistic(shape=f([0.5, 1.0, 2.0]), scale=f([1.0, 2.0, 3.0]))
    t = f(1.5)
    assert_close(jax.vmap(lambda dd: dd.log_prob(t))(d), d.log_prob(t))


# --------------------------------------------------------------- dtype


@LOGLOGISTIC
def test_integer_parameters_use_default_float_dtype():
    """spec 3.3, rev 55: integer-only parameters use the configured default floating dtype."""
    d = survjax.LogLogistic(shape=np.asarray([1, 2], dtype=np.int32), scale=np.asarray(3, dtype=np.int32))
    assert np.dtype(d.shape.dtype) == FLOAT
    assert np.dtype(d.scale.dtype) == FLOAT


@LOGLOGISTIC
def test_python_scalar_adopts_float32_array_dtype():
    """spec 3.3, rev 55: weak typing: a Python scalar `scale` adopts the float32 dtype of `shape`."""
    d = survjax.LogLogistic(shape=np.asarray([1.5, 2.0], dtype=np.float32), scale=2.0)
    assert np.dtype(d.shape.dtype) == np.float32
    assert np.dtype(d.scale.dtype) == np.float32


@pytest.mark.skipif(not X64, reason="float64 parameters exist only with 64-bit mode")
@LOGLOGISTIC
def test_mixed_float_parameters_share_promoted_dtype():
    """spec 3.3, rev 55: all leaves share one floating dtype from JAX promotion (float32 with float64 gives float64)."""
    d = survjax.LogLogistic(shape=np.asarray(1.5, dtype=np.float32), scale=np.asarray(2.0, dtype=np.float64))
    assert np.dtype(d.shape.dtype) == np.float64
    assert np.dtype(d.scale.dtype) == np.float64


@LOGLOGISTIC
def test_boolean_parameter_is_rejected():
    """spec 3.3, rev 55: boolean parameter inputs raise `TypeError` (a dtype check)."""
    with pytest.raises(TypeError):
        survjax.LogLogistic(shape=np.asarray(True), scale=f(1.0))


# --------------------------------------------------------------- reference_time


@LOGLOGISTIC
def test_loglogistic_reference_time_is_positive_finite_per_batch_entry():
    """spec 4.2, rev 55: `reference_time()` is "an array of shape `batch_shape` in the parameter dtype, positive and finite for valid parameters" (shapes 0.2 to 20, scales 1e-3 to 1e3)."""
    d = survjax.LogLogistic(shape=f([[0.2], [1.0], [20.0]]), scale=f([1e-3, 1.0, 1e3]))
    ref = d.reference_time()
    assert np.dtype(ref.dtype) == FLOAT
    rt = np.asarray(ref)
    assert rt.shape == (3, 3)
    assert np.all(np.isfinite(rt))
    assert np.all(rt > 0)


@LOGLOGISTIC
def test_loglogistic_reference_time_under_jit_matches_eager():
    """spec 4.1/12.1, rev 55: `reference_time()` is the same eagerly and under `jit`."""
    d = survjax.LogLogistic(shape=f([0.5, 2.0]), scale=f([1.0, 3.0]))
    assert_close(jax.jit(lambda dd: dd.reference_time())(d), d.reference_time())


# --------------------------------------------------------------- from_log (Q-069)


@LOGLOGISTIC
def test_loglogistic_from_log_returns_loglogistic():
    """spec 4.1, rev 55: the log-parameter constructor returns the same class (Q-069)."""
    assert type(loglogistic_from_log(f(0.0), f(0.0))) is survjax.LogLogistic


@LOGLOGISTIC
def test_loglogistic_from_log_stores_natural_parameters():
    """spec 4.1, rev 55: `shape = exp(log_shape)`, `scale = exp(log_scale)` are the leaves (Q-069)."""
    ls, la = [-0.7, 0.0, 1.1], [2.0, -1.0, 0.5]
    d = loglogistic_from_log(f(ls), f(la))
    assert_close(d.shape, np.exp(ls))
    assert_close(d.scale, np.exp(la))


@LOGLOGISTIC
def test_loglogistic_from_log_values_match_natural_constructor():
    """spec 4.1, rev 55: values equal those of the ordinary constructor wherever `exp` is finite (Q-069)."""
    ls, la, t = f([0.2, -0.4]), f([0.3, 1.0]), f([0.7, 3.0])
    want = survjax.LogLogistic(shape=jnp.exp(ls), scale=jnp.exp(la)).log_prob(t)
    assert_close(loglogistic_from_log(ls, la).log_prob(t), want)
