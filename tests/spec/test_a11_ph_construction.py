"""A11: constructing `ProportionalHazards` (spec 3.3, 4.1, 4.2, 4.4, 5; rev 41).

4.4: "Wrappers take any baseline distribution and a caller-computed linear
predictor ... The wrapper's batch shape is the broadcast of the baseline batch
shape and the linear predictor's shape; both are broadcast at construction".
4.2: a ``survjax.distribution()`` field (the baseline) is a nested
distribution whose leaves are leaves of the outer PyTree; it keeps its own
dtype. 4.4 (rev 42): the wrappers are registered families with fields
``baseline`` and ``log_hazard_ratio``, keyword-only. Construction goes through
``helpers.ph``, except where a warning's call site matters.
"""

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import FLOAT, INF, MISUSE, NAN, X64, assert_close, f, leaves, ph

import survjax

requires_x32 = pytest.mark.skipif(X64, reason="float64 conversion applies in 32-bit mode")
requires_x64 = pytest.mark.skipif(not X64, reason="needs a float64 leaf beside a float32 one")


def _weibull(k=2.0, b=3.0):
    return survjax.Weibull(shape=f(k), scale=f(b))


def _at_this_file(record):
    here = Path(__file__).resolve()
    return [w for w in record if Path(w.filename).resolve() == here]


# --------------------------------------------------------------- class and fields


@pytest.mark.step("A11")
def test_proportional_hazards_is_exported_at_top_level():
    """spec 10.2, rev 41: wrappers are re-exported at top level (`survjax.ProportionalHazards`)."""
    assert hasattr(survjax, "ProportionalHazards")


@pytest.mark.step("A11")
def test_proportional_hazards_is_a_survival_distribution():
    """spec 4.2/4.4, rev 41: the wrapper is a registered distribution (it has `parameter_issues()` and `log_likelihood`)."""
    d = ph(survjax.Exponential(rate=f(0.5)), f(0.3))
    assert isinstance(d, survjax.SurvivalDistribution)


@pytest.mark.step("A11")
def test_baseline_and_linear_predictor_are_stored_under_constructor_names():
    """spec 4.1/4.4, rev 42: fields `baseline` and `log_hazard_ratio`; "Natural parameters are stored as the leaves, under the names the constructor uses"."""
    d = ph(_weibull(2.0, 3.0), f(0.4))
    assert isinstance(d.baseline, survjax.Weibull)
    assert_close(d.log_hazard_ratio, 0.4)
    assert_close(d.baseline.shape, 2.0)
    assert_close(d.baseline.scale, 3.0)


@pytest.mark.step("A11")
def test_leaves_are_the_baseline_parameters_and_the_linear_predictor():
    """spec 4.1/4.2, rev 41: parameter arrays are the dynamic leaves; nested leaves are leaves of the outer PyTree."""
    d = ph(_weibull(2.0, 3.0), f(0.4))
    values = sorted(float(np.asarray(x)) for x in leaves(d))
    assert values == pytest.approx([0.4, 2.0, 3.0], rel=1e-6)


@pytest.mark.step("A11")
def test_flatten_unflatten_round_trip():
    """spec 4.1/4.2, rev 41: the wrapper is a PyTree; rebuilding it from its leaves gives the same law."""
    d = ph(_weibull(2.0, 3.0), f([0.4, -0.2]))
    flat, treedef = jax.tree_util.tree_flatten(d)
    rebuilt = jax.tree_util.tree_unflatten(treedef, flat)
    assert type(rebuilt) is type(d)
    assert type(rebuilt.baseline) is survjax.Weibull
    t = f(1.5)
    assert_close(rebuilt.log_survival(t), d.log_survival(t))


@pytest.mark.step("A11")
def test_tree_map_keeps_wrapper_and_baseline_types():
    """spec 4.1, rev 41: `jax.tree.map` over a wrapper returns a wrapper over the same baseline family."""
    d = ph(survjax.Exponential(rate=f([0.5, 1.0])), f(0.1))
    mapped = jax.tree.map(lambda x: x * 1, d)
    assert type(mapped) is type(d)
    assert type(mapped.baseline) is survjax.Exponential


@pytest.mark.step("A11")
def test_positional_baseline_is_rejected():
    """spec 4.4, rev 42: "Like every registered family they are constructed with keyword arguments only; a positional baseline raises `TypeError`"."""
    with pytest.raises(TypeError):
        survjax.ProportionalHazards(_weibull(), log_hazard_ratio=f(0.1))


# --------------------------------------------------------------- batch shape and broadcasting


@pytest.mark.step("A11")
def test_scalar_baseline_with_matrix_predictor_has_the_predictor_batch():
    """spec 5, rev 41: "PH/AFT | baseline `()`, linear predictor `(D,N)` | distribution `(D,N)`"."""
    d = ph(_weibull(), f(np.zeros((2, 3))))
    assert d.batch_shape == (2, 3)
    assert all(type(n) is int for n in d.batch_shape)


@pytest.mark.step("A11")
def test_vector_baseline_with_scalar_predictor_has_the_baseline_batch():
    """spec 4.4, rev 41: the batch shape is the broadcast of the baseline batch and the predictor shape."""
    d = ph(survjax.Exponential(rate=f([0.5, 1.0, 2.0])), f(0.3))
    assert d.batch_shape == (3,)
    assert np.shape(d.log_hazard_ratio) == (3,)


@pytest.mark.step("A11")
def test_draw_baseline_broadcasts_against_draw_by_arm_predictor():
    """spec 4.4, rev 41: baseline of batch `(S, 1)` with `eta` of shape `(S, 2)`; "both are broadcast at construction"."""
    s = 4
    base = survjax.Weibull(shape=f(np.full((s, 1), 1.5)), scale=f(np.arange(1.0, s + 1)[:, None]))
    d = ph(base, f(np.zeros((s, 2))))
    assert d.batch_shape == (s, 2)
    assert d.baseline.batch_shape == (s, 2)
    assert np.shape(d.baseline.shape) == (s, 2)
    assert np.shape(d.baseline.scale) == (s, 2)
    assert np.shape(d.log_hazard_ratio) == (s, 2)
    np.testing.assert_array_equal(np.asarray(d.baseline.scale)[:, 1], np.arange(1.0, s + 1))


@pytest.mark.step("A11")
def test_incompatible_baseline_and_predictor_batches_are_rejected():
    """spec 4.1, rev 41: incompatible batch shapes at construction are static misuse (`TypeError`)."""
    with pytest.raises(MISUSE):
        ph(survjax.Exponential(rate=f([0.5, 1.0, 2.0])), f([0.1, 0.2]))


@pytest.mark.step("A11")
def test_wrapper_has_no_shape_attribute():
    """spec 4.1, rev 41: "on a family without a `shape` field, accessing `.shape` raises `AttributeError`"."""
    d = ph(_weibull(), f([0.1, 0.2]))
    with pytest.raises(AttributeError):
        d.shape  # noqa: B018
    assert_close(d.baseline.shape, [2.0, 2.0])


# --------------------------------------------------------------- no value checks, traced construction


@pytest.mark.step("A11")
def test_constructor_does_not_check_parameter_values():
    """spec 4.1, rev 41: "Constructors never check parameter values"; NaN, infinite `eta` and a negative scale construct."""
    d = ph(survjax.Weibull(shape=f(2.0), scale=f([-1.0, 1.0, 1.0])), f([0.0, NAN, INF]))
    assert d.batch_shape == (3,)


@pytest.mark.step("A11")
def test_construction_with_traced_parameters_under_jit():
    """spec 4.1, rev 41: construction with traced parameters must work; `log S = -exp(eta) (t/b0)**k`."""
    k, b0, eta, t = 1.5, 2.0, 0.4, 3.0

    @jax.jit
    def fn(kk, bb, ee):
        return ph(survjax.Weibull(shape=kk, scale=bb), ee).log_survival(f(t))

    assert_close(fn(f(k), f(b0), f(eta)), -np.exp(eta) * (t / b0) ** k)


@pytest.mark.step("A11")
def test_construction_under_vmap_over_the_predictor():
    """spec 4.1, rev 41: construction inside `vmap`; Exponential PH `log S = -r exp(eta) t`."""
    r, t = 0.5, 2.0
    eta = np.array([-1.0, 0.0, 1.0])
    out = jax.vmap(lambda e: ph(survjax.Exponential(rate=f(r)), e).log_survival(f(t)))(f(eta))
    assert_close(out, -r * np.exp(eta) * t)


# --------------------------------------------------------------- dtype policy


@pytest.mark.step("A11")
def test_integer_predictor_uses_the_default_float_dtype():
    """spec 3.3, rev 41: integer-only parameter input "uses the configured default floating dtype"."""
    d = ph(_weibull(), np.array([0, 1], dtype=np.int32))
    assert np.dtype(d.log_hazard_ratio.dtype) == FLOAT


@pytest.mark.step("A11")
def test_python_scalar_predictor_uses_the_default_float_dtype():
    """spec 3.3, rev 41: scalar-only parameter input uses the configured default floating dtype."""
    d = ph(_weibull(), 0.5)
    assert np.dtype(jnp.asarray(d.log_hazard_ratio).dtype) == FLOAT


@pytest.mark.step("A11")
def test_boolean_predictor_is_rejected():
    """spec 3.3, rev 41: boolean parameter input raises `TypeError` (a dtype check)."""
    with pytest.raises(TypeError):
        ph(_weibull(), np.array([True, False]))


@pytest.mark.step("A11")
def test_complex_predictor_is_rejected():
    """spec 3.3, rev 41: complex parameter input raises `TypeError`."""
    with pytest.raises(TypeError):
        ph(_weibull(), np.array([0.1 + 1j]))


@pytest.mark.step("A11")
def test_constructor_takes_no_dtype_keyword():
    """spec 3.3, rev 41: "Constructors take no `dtype` keyword"."""
    with pytest.raises(TypeError):
        survjax.ProportionalHazards(
            baseline=_weibull(), log_hazard_ratio=f(0.1), dtype=FLOAT
        )


@pytest.mark.step("A11")
@requires_x32
def test_float64_predictor_in_32_bit_mode_converts_with_one_warning():
    """spec 3.3, rev 41: float64 parameter arrays with 64-bit mode disabled convert to float32 "with one warning per call site"."""
    base = _weibull()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        d = survjax.ProportionalHazards(
            baseline=base, log_hazard_ratio=np.array([0.1, 0.2], dtype=np.float64)
        )
    assert np.dtype(d.log_hazard_ratio.dtype) == np.dtype(np.float32)
    assert len(_at_this_file(record)) == 1


@pytest.mark.step("A11")
@requires_x64
def test_nested_baseline_keeps_its_own_dtype():
    """spec 3.3/4.2, rev 41: "Nested distribution fields keep their own dtypes"; calculations use the promoted dtype."""
    base = survjax.Weibull(
        shape=np.array(2.0, dtype=np.float32), scale=np.array(3.0, dtype=np.float32)
    )
    d = ph(base, np.array([0.1, 0.2], dtype=np.float64))
    assert np.dtype(d.baseline.shape.dtype) == np.dtype(np.float32)
    assert np.dtype(d.baseline.scale.dtype) == np.dtype(np.float32)
    assert np.dtype(d.log_hazard_ratio.dtype) == np.dtype(np.float64)
    assert np.dtype(d.log_survival(1.0).dtype) == np.dtype(np.float64)


# --------------------------------------------------------------- nested-field validation


@pytest.mark.step("A11")
def test_non_distribution_baseline_is_rejected_naming_the_field():
    """spec 4.2/4.4, rev 42: the `distribution()` field `baseline` holding anything but a registered instance raises `TypeError` naming it."""
    with pytest.raises(TypeError, match="baseline"):
        ph(f(1.0), f(0.1))


# --------------------------------------------------------------- indexing and transformations


def _draws_by_arm():
    base = survjax.Weibull(shape=f([[1.5], [2.0], [0.7]]), scale=f([[1.0], [2.0], [3.0]]))
    return ph(base, f([[0.0, 0.5], [-0.5, 1.0], [0.2, -1.0]]))


@pytest.mark.step("A11")
def test_indexing_selects_baseline_and_predictor_entries():
    """spec 4.1/4.2, rev 41: `dist[idx]` indexes batch dimensions, nested leaves included."""
    d = _draws_by_arm()
    row = d[1]
    assert row.batch_shape == (2,)
    assert_close(row.baseline.scale, [2.0, 2.0])
    assert_close(row.log_hazard_ratio, [-0.5, 1.0])
    col = d[:, 0]
    assert col.batch_shape == (3,)
    t = f(1.3)
    assert_close(col.log_survival(t), np.asarray(d.log_survival(t))[:, 0])


@pytest.mark.step("A11")
def test_tree_map_slicing_applies_to_nested_leaves():
    """spec 4.1/4.2, rev 41: `jax.tree.map` slicing works and reaches the broadcast baseline leaves."""
    d = _draws_by_arm()
    sliced = jax.tree.map(lambda x: x[1:], d)
    assert sliced.batch_shape == (2, 2)
    t = f(0.8)
    assert_close(sliced.survival(t), np.asarray(d.survival(t))[1:])


@pytest.mark.step("A11")
def test_vmap_over_the_wrapper():
    """spec 4.1, rev 41: `vmap` over a distribution works; per-entry batch shape under `vmap`."""
    d = _draws_by_arm()
    t = f(1.7)
    out = jax.vmap(lambda dd: dd.log_survival(t))(d)
    assert_close(out, d.log_survival(t))
    shapes = jax.vmap(lambda dd: jnp.zeros(dd.batch_shape))(d)
    assert np.shape(shapes) == (3, 2)


@pytest.mark.step("A11")
def test_wrapper_passes_through_jit_as_an_argument():
    """spec 4.1/12.1, rev 41: eager and `jit` agree with the wrapper as a traced PyTree argument."""
    d = _draws_by_arm()
    t = f(2.2)
    assert_close(jax.jit(lambda dd, tt: dd.survival(tt))(d, t), d.survival(t))
