"""Feature `family.lognormal`: constructing `LogNormal` (spec 3.3, 4.1, 4.2, 5, 10.2, 12.2; rev 55).

12.2 (rev 55): "The log-normal and log-logistic families of section 4.1 are
enabled ahead of the rest of Slice D, each by its own step under the full
contract of sections 4.1, 4.2, 4.5, 5 and 7: construction and parameter
constraints, closed-form direct methods with their tail forms, grids, the
likelihood modes implemented at the time, diagnostics and derivatives."

4.1: `LogNormal(meanlog=mu, sdlog=sigma)`, `S(t) = Phi(-(log t - mu)/sigma)`;
built-ins "are registered subclasses of `SurvivalDistribution`"; "Natural
parameters are stored as the leaves, under the names the constructor uses";
batch broadcasting at construction; constructors "never check parameter
values"; log-parameter constructors exist for "the other families" (keyword
names: Q-069). 4.2: keyword-only construction, required parameter fields,
`reference_time()`. 3.3: parameter dtype normalization. 10.2: distributions
are re-exported at the top level.

Hand value used below: at ``t = exp(mu)`` the argument is ``0``, so ``S =
Phi(0) = 1/2`` and ``log S = -log 2``, for every ``sigma``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import FLOAT, INF, MISUSE, NAN, X64, assert_close, f, leaves, lognormal_from_log

import survjax

LOGNORMAL = pytest.mark.feature("family.lognormal")
LOG_HALF = -np.log(2.0)

# --------------------------------------------------------------- exports and class


@LOGNORMAL
def test_lognormal_is_exported_at_top_level():
    """spec 10.2, rev 55: distributions are re-exported at the top level (`survjax.LogNormal`)."""
    assert hasattr(survjax, "LogNormal")


@LOGNORMAL
def test_lognormal_is_a_survival_distribution():
    """spec 4.1, rev 55: "Built-in families are registered subclasses of `SurvivalDistribution`"."""
    assert isinstance(survjax.LogNormal(meanlog=f(0.0), sdlog=f(1.0)), survjax.SurvivalDistribution)


# --------------------------------------------------------------- keywords


@LOGNORMAL
def test_lognormal_rejects_positional_parameters():
    """spec 4.2, rev 55: registered families, "built-in and custom", are constructed with keyword arguments only."""
    with pytest.raises(MISUSE):
        survjax.LogNormal(f(0.0), f(1.0))


@LOGNORMAL
def test_lognormal_requires_both_parameters():
    """spec 4.2, rev 55: "Parameter fields have no default and are required"."""
    with pytest.raises(TypeError):
        survjax.LogNormal(meanlog=f(0.0))
    with pytest.raises(TypeError):
        survjax.LogNormal(sdlog=f(1.0))


# --------------------------------------------------------------- natural parameters as leaves


@LOGNORMAL
def test_lognormal_stores_meanlog_and_sdlog_as_its_only_leaves():
    """spec 4.1, rev 55: the leaves are the natural `meanlog` and `sdlog`, under those names."""
    d = survjax.LogNormal(meanlog=f([-0.5, 2.0]), sdlog=f([0.3, 1.5]))
    np.testing.assert_array_equal(np.asarray(d.meanlog), f([-0.5, 2.0]))
    np.testing.assert_array_equal(np.asarray(d.sdlog), f([0.3, 1.5]))
    assert len(leaves(d)) == 2


@LOGNORMAL
def test_lognormal_has_no_shape_attribute():
    """spec 4.1, rev 55: on a family without a `shape` field, accessing `.shape` raises `AttributeError`."""
    d = survjax.LogNormal(meanlog=f([0.0, 1.0]), sdlog=f(1.0))
    with pytest.raises(AttributeError):
        _ = d.shape


# --------------------------------------------------------------- batch shape


@LOGNORMAL
def test_scalar_lognormal_has_empty_batch_shape():
    """spec 4.1, rev 55: `batch_shape` is a tuple of Python ints from the parameter leaves."""
    assert survjax.LogNormal(meanlog=f(0.0), sdlog=f(1.0)).batch_shape == ()


@LOGNORMAL
def test_lognormal_parameters_are_broadcast_at_construction():
    """spec 4.1, rev 55: "Batch dimensions are broadcast at construction, so every parameter leaf shares the leading batch shape"."""
    d = survjax.LogNormal(meanlog=f([0.0, 1.0, 2.0]), sdlog=f([[0.5], [1.5]]))
    assert d.batch_shape == (2, 3)
    assert jnp.shape(d.meanlog) == (2, 3)
    assert jnp.shape(d.sdlog) == (2, 3)
    np.testing.assert_array_equal(np.asarray(d.meanlog)[1], f([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(np.asarray(d.sdlog)[:, 2], f([0.5, 1.5]))


@LOGNORMAL
def test_lognormal_incompatible_parameter_batches_are_rejected():
    """spec 4.1/5, rev 55: incompatible parameter batch shapes raise `TypeError`."""
    with pytest.raises(MISUSE):
        survjax.LogNormal(meanlog=f([0.0, 1.0]), sdlog=f([1.0, 2.0, 3.0]))


@LOGNORMAL
def test_lognormal_draw_by_individual_batch():
    """spec 5, rev 55: `meanlog` of shape `(N,)` with `sdlog[:, None]` of `D = 2` draws gives batch `(D, N)`."""
    d = survjax.LogNormal(meanlog=f([0.0, 1.0, 2.0]), sdlog=f([0.5, 1.5])[:, None])
    assert d.batch_shape == (2, 3)


@LOGNORMAL
def test_time_that_does_not_broadcast_is_rejected():
    """spec 4.1, rev 55: "a time argument that does not broadcast against `batch_shape` in a direct method" raises `TypeError`."""
    d = survjax.LogNormal(meanlog=f([0.0, 1.0, 2.0]), sdlog=f(1.0))
    with pytest.raises(MISUSE):
        d.log_survival(f([1.0, 2.0]))


# --------------------------------------------------------------- no value checks


@LOGNORMAL
def test_lognormal_constructor_accepts_invalid_values_eagerly():
    """spec 4.1, rev 55: "Constructors never check parameter values"."""
    survjax.LogNormal(meanlog=f([NAN, INF, 0.0, 0.0]), sdlog=f([1.0, 1.0, 0.0, -1.0]))


@LOGNORMAL
def test_lognormal_constructor_accepts_invalid_values_under_jit():
    """spec 4.1, rev 55: no value checks, so eager and `jit` construction behave alike."""
    jax.jit(lambda m, s: survjax.LogNormal(meanlog=m, sdlog=s))(f(NAN), f(-1.0))


# --------------------------------------------------------------- traced construction, indexing, vmap


@LOGNORMAL
def test_lognormal_construction_with_traced_parameters_matches_hand_value():
    """spec 4.1, rev 55: construction with traced parameters must work; `log S(exp(mu)) = -log 2`."""
    out = jax.jit(lambda m, s: survjax.LogNormal(meanlog=m, sdlog=s).log_survival(jnp.exp(m)))(f(0.7), f(1.3))
    assert_close(out, LOG_HALF)


@LOGNORMAL
def test_lognormal_construction_inside_vmap():
    """spec 4.1, rev 55: construction inside `vmap`; `log S(exp(mu)) = -log 2` per mapped `meanlog`."""
    out = jax.vmap(lambda m: survjax.LogNormal(meanlog=m, sdlog=f(0.8)).log_survival(jnp.exp(m)))(f([-1.0, 0.0, 2.0]))
    assert_close(out, [LOG_HALF] * 3)


@LOGNORMAL
def test_lognormal_indexing_selects_batch_entries():
    """spec 4.1, rev 55: `dist[idx]` indexes batch dimensions and returns a `LogNormal`."""
    d = survjax.LogNormal(meanlog=f([0.0, 1.0, 2.0]), sdlog=f([0.5, 1.0, 1.5]))
    sub = d[1]
    assert isinstance(sub, survjax.LogNormal)
    assert sub.batch_shape == ()
    assert_close(sub.meanlog, 1.0)
    assert_close(sub.sdlog, 1.0)


@LOGNORMAL
def test_lognormal_tree_map_slicing():
    """spec 4.1, rev 55: `jax.tree.map` slicing gives a `LogNormal` of the sliced batch."""
    d = survjax.LogNormal(meanlog=f([0.0, 1.0, 2.0]), sdlog=f([0.5, 1.0, 1.5]))
    sub = jax.tree.map(lambda x: x[:2], d)
    assert isinstance(sub, survjax.LogNormal)
    assert sub.batch_shape == (2,)
    assert_close(sub.log_survival(jnp.exp(f([0.0, 1.0]))), [LOG_HALF, LOG_HALF])


@LOGNORMAL
def test_vmap_over_lognormal_matches_batched_call():
    """spec 4.1/12.1, rev 55: `vmap` over a distribution agrees with the batched call."""
    d = survjax.LogNormal(meanlog=f([0.0, 1.0, 2.0]), sdlog=f([0.5, 1.0, 1.5]))
    t = f(1.5)
    assert_close(jax.vmap(lambda dd: dd.log_prob(t))(d), d.log_prob(t))


# --------------------------------------------------------------- dtype


@LOGNORMAL
def test_integer_parameters_use_default_float_dtype():
    """spec 3.3, rev 55: integer-only parameters use the configured default floating dtype."""
    d = survjax.LogNormal(meanlog=np.asarray([0, 1], dtype=np.int32), sdlog=np.asarray(2, dtype=np.int32))
    assert np.dtype(d.meanlog.dtype) == FLOAT
    assert np.dtype(d.sdlog.dtype) == FLOAT


@LOGNORMAL
def test_python_scalar_adopts_float32_array_dtype():
    """spec 3.3, rev 55: weak typing: a Python scalar `sdlog` adopts the float32 dtype of `meanlog`."""
    d = survjax.LogNormal(meanlog=np.asarray([0.0, 1.0], dtype=np.float32), sdlog=1.5)
    assert np.dtype(d.meanlog.dtype) == np.float32
    assert np.dtype(d.sdlog.dtype) == np.float32


@pytest.mark.skipif(not X64, reason="float64 parameters exist only with 64-bit mode")
@LOGNORMAL
def test_mixed_float_parameters_share_promoted_dtype():
    """spec 3.3, rev 55: all leaves share one floating dtype from JAX promotion (float32 with float64 gives float64)."""
    d = survjax.LogNormal(meanlog=np.asarray(0.5, dtype=np.float32), sdlog=np.asarray(1.5, dtype=np.float64))
    assert np.dtype(d.meanlog.dtype) == np.float64
    assert np.dtype(d.sdlog.dtype) == np.float64


@LOGNORMAL
def test_boolean_parameter_is_rejected():
    """spec 3.3, rev 55: boolean parameter inputs raise `TypeError` (a dtype check)."""
    with pytest.raises(TypeError):
        survjax.LogNormal(meanlog=np.asarray(True), sdlog=f(1.0))


# --------------------------------------------------------------- reference_time


@LOGNORMAL
def test_lognormal_reference_time_is_positive_finite_per_batch_entry():
    """spec 4.2, rev 55: `reference_time()` is "an array of shape `batch_shape` in the parameter dtype, positive and finite for valid parameters" (`meanlog` -5 to 5, `sdlog` 0.1 to 3)."""
    d = survjax.LogNormal(meanlog=f([[-5.0], [0.0], [5.0]]), sdlog=f([0.1, 1.0, 3.0]))
    ref = d.reference_time()
    assert np.dtype(ref.dtype) == FLOAT
    rt = np.asarray(ref)
    assert rt.shape == (3, 3)
    assert np.all(np.isfinite(rt))
    assert np.all(rt > 0)


@LOGNORMAL
def test_lognormal_reference_time_under_jit_matches_eager():
    """spec 4.1/12.1, rev 55: methods are pure, so `reference_time()` is the same eagerly and under `jit`."""
    d = survjax.LogNormal(meanlog=f([0.0, 2.0]), sdlog=f([0.5, 1.5]))
    assert_close(jax.jit(lambda dd: dd.reference_time())(d), d.reference_time())


# --------------------------------------------------------------- from_log (Q-069)


@LOGNORMAL
def test_lognormal_from_log_returns_lognormal():
    """spec 4.1, rev 55: the log-parameter constructor "equivalents for the other families" return the same class (Q-069)."""
    assert type(lognormal_from_log(f(0.0), f(0.0))) is survjax.LogNormal


@LOGNORMAL
def test_lognormal_from_log_stores_natural_parameters():
    """spec 4.1, rev 55: natural parameters are the leaves: `sdlog = exp(log_sdlog)`, `meanlog` unchanged (Q-069)."""
    m, ls = [-1.0, 0.0, 2.0], [-0.7, 0.0, 1.1]
    d = lognormal_from_log(f(m), f(ls))
    assert_close(d.meanlog, m)
    assert_close(d.sdlog, np.exp(ls))


@LOGNORMAL
def test_lognormal_from_log_values_match_natural_constructor():
    """spec 4.1, rev 55: "Values and parameter gradients equal those of the ordinary constructor wherever `exp` is finite" (Q-069)."""
    m, ls, t = f([0.3, -1.0]), f([0.2, -0.4]), f([0.7, 3.0])
    want = survjax.LogNormal(meanlog=m, sdlog=jnp.exp(ls)).log_prob(t)
    assert_close(lognormal_from_log(m, ls).log_prob(t), want)
