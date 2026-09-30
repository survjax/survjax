"""A3: pure construction and traced input classification (spec 3.2, 6.2, rev 9)."""

import chronox_jax as cx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import (
    INF,
    MASK_FIELDS,
    NAN,
    X64,
    assert_same_fields,
    b,
    f,
    mask,
    raises_dtype_request,
    raises_misuse,
    treedef,
)


def _same_diagnostics(left, right):
    for name in MASK_FIELDS:
        np.testing.assert_array_equal(mask(left.issues, name), mask(right.issues, name), err_msg=name)
    np.testing.assert_array_equal(np.asarray(left.issues.ok), np.asarray(right.issues.ok))


def _agrees_under_jit(build, *inputs):
    eager = build(*inputs)
    compiled = jax.jit(build)(*inputs)
    assert_same_fields(compiled, eager)
    _same_diagnostics(compiled, eager)


@pytest.mark.step("A3")
def test_concrete_numpy_inputs_inside_jit_keep_value_based_metadata():
    """spec 3.2/6.2, rev 9: concrete NumPy inputs inside jit construct with value-based flags."""
    eager = cx.SurvivalData(f([1.0, 2.0]), b([True, False]))
    inside = jax.jit(lambda: cx.SurvivalData(f([1.0, 2.0]), b([True, False])))()
    assert_same_fields(inside, eager)
    _same_diagnostics(inside, eager)
    assert treedef(inside) == treedef(eager)


@pytest.mark.step("A3")
def test_concrete_invalid_flag_inputs_inside_jit_return_issues():
    """spec 6.2, rev 9: concrete invalid observations inside jit return data and issue bits."""
    data = jax.jit(lambda: cx.SurvivalData(f([1.0, 0.0])))()
    np.testing.assert_array_equal(mask(data.issues, "invalid_time"), [False, True])


@pytest.mark.step("A3")
def test_concrete_invalid_bounds_inputs_inside_jit_return_issues():
    """spec 6.2, rev 9: concrete invalid bounds inside jit return data and issue bits."""
    data = jax.jit(lambda: cx.SurvivalData.from_bounds(f([3.0]), f([2.0])))()
    np.testing.assert_array_equal(mask(data.issues, "invalid_interval"), [True])


@pytest.mark.step("A3")
def test_other_traced_argument_does_not_change_concrete_input_validation():
    """spec 6.2, rev 9: tracing unrelated values does not reject invalid concrete observations."""
    data = jax.jit(lambda scale: cx.SurvivalData(f([1.0, -1.0])))(f(2.0))
    np.testing.assert_array_equal(mask(data.issues, "invalid_time"), [False, True])


@pytest.mark.step("A3")
def test_traced_time_matches_eager():
    """spec 6.2, rev 9: traced flag-form time has the same normalized fields and issues."""
    _agrees_under_jit(lambda t: cx.SurvivalData(t), f([1.0, 0.0]))


@pytest.mark.step("A3")
def test_traced_event_matches_eager():
    """spec 6.2, rev 9: traced event codes retain invalid-indicator evidence."""
    _agrees_under_jit(lambda e: cx.SurvivalData(f([1.0, 2.0]), e), np.array([1, 2], dtype=np.int32))


@pytest.mark.step("A3")
def test_traced_right_flag_matches_eager():
    """spec 6.2, rev 9: traced right-censoring flags are accepted."""
    _agrees_under_jit(lambda r: cx.SurvivalData(f([1.0, 2.0]), right_censored=r), b([False, True]))


@pytest.mark.step("A3")
def test_traced_left_flag_matches_eager():
    """spec 6.2, rev 9: traced left-censoring flags are accepted."""
    _agrees_under_jit(lambda c: cx.SurvivalData(f([1.0, 2.0]), left_censored=c), b([False, True]))


@pytest.mark.step("A3")
def test_traced_interval_flag_matches_eager():
    """spec 6.2, rev 9: traced interval-censoring flags are accepted."""
    _agrees_under_jit(lambda i: cx.SurvivalData(f([1.0, 2.0]), interval_censored=i, time2=f([NAN, 3.0])), b([False, True]))


@pytest.mark.step("A3")
def test_traced_time2_matches_eager():
    """spec 6.2, rev 9: traced time2 is accepted and checked on interval rows."""
    _agrees_under_jit(lambda t2: cx.SurvivalData(f([1.0, 2.0]), interval_censored=b([False, True]), time2=t2), f([NAN, 2.0]))


@pytest.mark.step("A3")
def test_traced_left_truncation_matches_eager():
    """spec 6.2, rev 9: traced left truncation is accepted and checked."""
    _agrees_under_jit(lambda lt: cx.SurvivalData(f([1.0, 2.0]), left_trunc=lt), f([0.0, 2.0]))


@pytest.mark.step("A3")
def test_traced_right_truncation_matches_eager():
    """spec 6.2, rev 9: traced right truncation is accepted and checked."""
    _agrees_under_jit(lambda rt: cx.SurvivalData(f([1.0, 2.0]), right_trunc=rt), f([INF, 1.0]))


@pytest.mark.step("A3")
def test_nested_list_with_tracer_is_accepted():
    """spec 6.2, rev 9: a tracer nested in a Python list is an accepted observation input."""
    def build(x):
        return cx.SurvivalData([[1.0, x], [3.0, 4.0]])
    _agrees_under_jit(build, f(2.0))


@pytest.mark.step("A3")
def test_tuple_with_tracer_is_accepted():
    """spec 6.2, rev 9: a tracer in a Python tuple is an accepted observation input."""
    def build(x):
        return cx.SurvivalData(f([1.0, 2.0]), left_trunc=(0.0, x))
    _agrees_under_jit(build, f(0.5))


@pytest.mark.step("A3")
def test_jnp_array_created_inside_jit_is_accepted():
    """spec 6.2, rev 9: a JAX array created inside jit is accepted as traced input."""
    data = jax.jit(lambda: cx.SurvivalData(jnp.asarray([1.0, 2.0])))()
    np.testing.assert_array_equal(np.asarray(data.lower), [1.0, 2.0])


@pytest.mark.step("A3")
def test_traced_bounds_lower_matches_eager():
    """spec 6.2, rev 9: traced bounds lower is accepted and validated."""
    _agrees_under_jit(lambda lo: cx.SurvivalData.from_bounds(lo, f([2.0, 3.0])), f([1.0, -1.0]))


@pytest.mark.step("A3")
def test_traced_bounds_upper_matches_eager():
    """spec 6.2, rev 9: traced bounds upper is accepted and validated."""
    _agrees_under_jit(lambda hi: cx.SurvivalData.from_bounds(f([1.0, 0.0]), hi), f([2.0, NAN]))


@pytest.mark.step("A3")
def test_traced_bounds_exact_matches_eager():
    """spec 6.2, rev 9: traced bounds exact indicator is accepted and validated."""
    _agrees_under_jit(lambda e: cx.SurvivalData.from_bounds(f([1.0, 1.0]), f([1.0, 2.0]), exact=e), np.array([1, 2], dtype=np.int32))


@pytest.mark.step("A3")
def test_traced_bounds_left_truncation_matches_eager():
    """spec 6.2, rev 9: traced bounds left truncation is accepted."""
    _agrees_under_jit(lambda lt: cx.SurvivalData.from_bounds(f([1.0]), f([2.0]), left_trunc=lt), f([0.5]))


@pytest.mark.step("A3")
def test_traced_bounds_right_truncation_matches_eager():
    """spec 6.2, rev 9: traced bounds right truncation is accepted."""
    _agrees_under_jit(lambda rt: cx.SurvivalData.from_bounds(f([1.0]), f([2.0]), right_trunc=rt), f([3.0]))


@pytest.mark.step("A3")
def test_bounds_nested_tuple_with_tracer_is_accepted():
    """spec 6.2, rev 9: bounds inputs may contain a tracer inside nested tuples."""
    def build(x):
        return cx.SurvivalData.from_bounds(((0.0, x),), ((2.0, 3.0),))
    _agrees_under_jit(build, f(1.0))


@pytest.mark.step("A3")
def test_bounds_vmap_agrees_with_eager_for_valid_and_invalid_rows():
    """spec 6.2/12.1, rev 9: bounds construction agrees under vmap for valid and invalid rows."""
    lower = f([[1.0, 2.0], [1.0, 3.0]])
    upper = f([[1.0, INF], [1.0, 2.0]])
    exact = b([[True, False], [True, False]])
    batched = jax.vmap(lambda lo, hi, ex: cx.SurvivalData.from_bounds(lo, hi, exact=ex))(
        lower, upper, exact
    )
    for i in range(2):
        eager = cx.SurvivalData.from_bounds(lower[i], upper[i], exact=exact[i])
        for name in ("exact", "lower", "upper", "left_trunc", "right_trunc"):
            np.testing.assert_array_equal(np.asarray(getattr(batched, name))[i], np.asarray(getattr(eager, name)))
        for name in MASK_FIELDS:
            np.testing.assert_array_equal(mask(batched.issues, name)[i], mask(eager.issues, name))
        # spec 6.2: `ok` of `data[idx]` reduces over the indexed rows only.
        assert bool(np.asarray(batched[i].issues.ok)) == bool(np.asarray(eager.issues.ok))


@pytest.mark.step("A3")
def test_ok_of_vmap_returned_data_reduces_over_all_rows():
    """spec 6.2, rev 9: `ok` of data returned from vmap is one scalar over every row of the object."""
    lower = f([[1.0, 2.0], [1.0, 3.0]])
    upper = f([[1.0, INF], [1.0, INF]])
    exact = b([[True, False], [True, False]])
    def build(lo, hi, ex):
        return cx.SurvivalData.from_bounds(lo, hi, exact=ex)
    valid = jax.vmap(build)(lower, upper, exact)
    assert np.shape(valid.issues.ok) == ()
    assert bool(np.asarray(valid.issues.ok))
    # Second dataset gets an invalid interval (3 >= 2); the first stays valid.
    mixed = jax.vmap(build)(lower, f([[1.0, INF], [1.0, 2.0]]), exact)
    assert np.shape(mixed.issues.ok) == ()
    assert not bool(np.asarray(mixed.issues.ok))
    assert bool(np.asarray(mixed[0].issues.ok))


@pytest.mark.step("A3")
def test_flag_misuse_still_raises_when_inputs_are_traced():
    """spec 6.2, rev 9: event combined with a censoring flag raises TypeError under jit."""
    with raises_misuse():
        jax.jit(lambda t, e, r: cx.SurvivalData(t, e, right_censored=r))(f([1.0]), b([True]), b([False]))


@pytest.mark.step("A3")
def test_bounds_shape_misuse_still_raises_when_inputs_are_traced():
    """spec 6.2, rev 9: incompatible bounds shapes raise TypeError under jit."""
    with raises_misuse():
        jax.jit(lambda lo, hi: cx.SurvivalData.from_bounds(lo, hi))(f([1.0, 2.0, 3.0]), f([[4.0], [5.0], [6.0]]))


@pytest.mark.step("A3")
def test_nonnumeric_field_misuse_raises_with_other_traced_input():
    """spec 6.2, rev 9: a nonnumeric field raises TypeError even when another input is traced."""
    with raises_misuse():
        jax.jit(lambda t: cx.SurvivalData(t, left_trunc=np.array(["0.0"])))(f([1.0]))


@pytest.mark.step("A3")
@pytest.mark.skipif(X64, reason="float64 requests are available in 64-bit mode")
def test_flag_unavailable_explicit_dtype_raises_under_tracing():
    """spec 3.3/6.2, rev 9: a traced flag call cannot request unavailable float64."""
    with raises_dtype_request():
        jax.jit(lambda t: cx.SurvivalData(t, dtype=np.float64))(f([1.0]))


@pytest.mark.step("A3")
@pytest.mark.skipif(X64, reason="float64 requests are available in 64-bit mode")
def test_bounds_unavailable_explicit_dtype_raises_under_tracing():
    """spec 3.3/6.2, rev 9: a traced bounds call cannot request unavailable float64."""
    with raises_dtype_request():
        jax.jit(lambda lo: cx.SurvivalData.from_bounds(lo, f([2.0]), dtype=np.float64))(f([1.0]))


@pytest.mark.step("A3")
def test_compiled_ok_controls_branch_selection():
    """spec 6.2, rev 9: data.issues.ok is a scalar bool usable in compiled control flow."""
    def choose(time):
        ok = cx.SurvivalData(time).issues.ok
        return jax.lax.cond(ok, lambda _: 1, lambda _: 0, None)

    assert int(jax.jit(choose)(f([1.0]))) == 1
    assert int(jax.jit(choose)(f([0.0]))) == 0


@pytest.mark.step("A3")
def test_traced_diagnostics_cannot_raise_at_host_checkpoint():
    """spec 6.2, rev 9: calling raise_if_invalid on traced diagnostics fails explicitly."""
    with pytest.raises(Exception):  # noqa: B017 - spec leaves the exception type open
        jax.jit(lambda t: cx.SurvivalData(t).issues.raise_if_invalid())(f([1.0]))


@pytest.mark.step("A3")
def test_compiled_invalid_diagnostics_raise_at_explicit_host_checkpoint():
    """spec 6.2, rev 9: concrete diagnostics returned from jit raise only at the host checkpoint."""
    data = jax.jit(lambda t: cx.SurvivalData(t))(f([1.0, 0.0]))
    with pytest.raises(cx.InvalidDataError):
        data.issues.raise_if_invalid()
