"""A3: which observation inputs count as traced, and ordinary construction inside transformations (spec 6.2, rev 6).

Ordinary constructors run eagerly whenever no observation input is traced, even
while a transformed function is being traced; they raise at trace time, naming
their pure counterpart, as soon as any observation input is traced. That error
is ``chronox_jax.TracedInputError`` (a ``TypeError`` subclass), and the tests
also check that its message names the pure counterpart.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import (
    INF,
    INVALID_DATA,
    MISUSE,
    NAN,
    TRACED_INPUT,
    assert_fields,
    assert_same_fields,
    b,
    f,
    raises_misuse,
    with_issues,
)

import chronox_jax as cx

FLAG_COUNTERPART = "with_issues"
BOUNDS_COUNTERPART = "from_bounds_with_issues"


def _flag_rejects_traced(fn, *traced_args):
    with pytest.raises(TRACED_INPUT, match=FLAG_COUNTERPART):
        jax.jit(fn)(*traced_args)


def _bounds_rejects_traced(fn, *traced_args):
    with pytest.raises(TRACED_INPUT, match=BOUNDS_COUNTERPART):
        jax.jit(fn)(*traced_args)


# ------------------------------------------- non-traced inputs inside jit


@pytest.mark.step("A3")
def test_ordinary_constructor_accepts_numpy_inputs_inside_jit():
    """spec 6.2, rev 6: NumPy arrays are non-traced; the ordinary constructor runs eagerly during tracing."""

    def fn(scale):
        return cx.SurvivalData(f([1.0, 2.0]), b([True, False])).lower * scale

    np.testing.assert_array_equal(np.asarray(jax.jit(fn)(f(2.0))), np.asarray(f([2.0, 4.0])))


@pytest.mark.step("A3")
def test_ordinary_constructor_inside_jit_matches_outside():
    """spec 6.2, rev 6: with non-traced inputs, construction inside jit gives the same data as outside."""
    kw = {"right_censored": b([False, True]), "left_trunc": f([0.5, 0.0]), "right_trunc": f([5.0, INF])}
    inside = jax.jit(lambda: cx.SurvivalData(f([1.0, 2.0]), **kw))()
    assert_same_fields(inside, cx.SurvivalData(f([1.0, 2.0]), **kw))


@pytest.mark.step("A3")
def test_ordinary_constructor_accepts_python_scalars_inside_jit():
    """spec 6.2, rev 6: Python scalars are non-traced inside a transformed function."""
    data = jax.jit(lambda: cx.SurvivalData(2.0, left_trunc=1.0))()
    assert_fields(data, exact=True, lower=2.0, upper=2.0, left_trunc=1.0)


@pytest.mark.step("A3")
def test_ordinary_constructor_accepts_python_lists_inside_jit():
    """spec 6.2, rev 6: lists and tuples of Python scalars are non-traced inside a transformed function."""
    data = jax.jit(lambda: cx.SurvivalData([1.0, 2.0], right_censored=(False, True)))()
    assert_fields(data, exact=[True, False], lower=[1.0, 2.0], upper=[1.0, INF])


@pytest.mark.step("A3")
def test_ordinary_constructor_accepts_concrete_jax_array_from_outside_jit():
    """spec 6.2, rev 6: concrete JAX arrays created outside the function are non-traced."""
    time = jnp.asarray(f([1.0, 2.0]))
    data = jax.jit(lambda: cx.SurvivalData(time))()
    assert_fields(data, exact=[True, True], lower=[1.0, 2.0])


@pytest.mark.step("A3")
def test_bounds_constructor_accepts_numpy_inputs_inside_jit():
    """spec 6.2, rev 6: the bounds form also runs eagerly inside jit when no observation input is traced."""
    inside = jax.jit(lambda: cx.SurvivalData.from_bounds(f([1.0, 0.0]), f([2.0, 3.0])))()
    assert_same_fields(inside, cx.SurvivalData.from_bounds(f([1.0, 0.0]), f([2.0, 3.0])))


@pytest.mark.step("A3")
def test_ordinary_constructor_raises_on_invalid_numpy_rows_inside_jit():
    """spec 6.2, rev 7: invalid non-traced observations inside jit raise the same errors as outside a transformation."""
    with pytest.raises(INVALID_DATA):
        jax.jit(lambda: cx.SurvivalData(f([1.0, 0.0])))()


@pytest.mark.step("A3")
def test_bounds_constructor_raises_on_invalid_numpy_rows_inside_jit():
    """spec 6.2, rev 7: invalid non-traced bounds inside jit raise the same errors as outside a transformation."""
    with pytest.raises(INVALID_DATA):
        jax.jit(lambda: cx.SurvivalData.from_bounds(f([3.0]), f([2.0])))()


@pytest.mark.step("A3")
def test_ordinary_constructor_raises_on_invalid_rows_inside_jit_with_other_traced_args():
    """spec 6.2, rev 7: traced arguments that are not observation inputs do not change eager validation."""
    with pytest.raises(INVALID_DATA):
        jax.jit(lambda scale: cx.SurvivalData(f([1.0, -1.0])).lower * scale)(f(2.0))


# ------------------------------------------- traced flag-form inputs


@pytest.mark.step("A3")
def test_traced_time_is_rejected_naming_with_issues():
    """spec 6.2, rev 7: a traced `time` makes the flag constructor raise at trace time, naming `with_issues`."""
    _flag_rejects_traced(lambda t: cx.SurvivalData(t).lower, f([1.0, 2.0]))


@pytest.mark.step("A3")
def test_traced_event_is_rejected_naming_with_issues():
    """spec 6.2, rev 7: `event` is an observation input."""
    _flag_rejects_traced(lambda e: cx.SurvivalData(f([1.0, 2.0]), e).lower, b([True, False]))


@pytest.mark.step("A3")
def test_traced_right_censored_flag_is_rejected_naming_with_issues():
    """spec 6.2, rev 7: the censoring flags are observation inputs (right)."""
    _flag_rejects_traced(lambda r: cx.SurvivalData(f([1.0, 2.0]), right_censored=r).lower, b([False, True]))


@pytest.mark.step("A3")
def test_traced_left_censored_flag_is_rejected_naming_with_issues():
    """spec 6.2, rev 7: the censoring flags are observation inputs (left)."""
    _flag_rejects_traced(lambda c: cx.SurvivalData(f([1.0, 2.0]), left_censored=c).lower, b([False, True]))


@pytest.mark.step("A3")
def test_traced_interval_censored_flag_is_rejected_naming_with_issues():
    """spec 6.2, rev 7: the censoring flags are observation inputs (interval)."""
    _flag_rejects_traced(
        lambda i: cx.SurvivalData(f([1.0, 2.0]), interval_censored=i, time2=f([NAN, 3.0])).lower,
        b([False, True]),
    )


@pytest.mark.step("A3")
def test_traced_time2_is_rejected_naming_with_issues():
    """spec 6.2, rev 7: `time2` is an observation input."""
    _flag_rejects_traced(
        lambda t2: cx.SurvivalData(f([1.0, 2.0]), interval_censored=b([False, True]), time2=t2).lower,
        f([NAN, 3.0]),
    )


@pytest.mark.step("A3")
def test_traced_left_trunc_is_rejected_naming_with_issues():
    """spec 6.2, rev 7: `left_trunc` is an observation input."""
    _flag_rejects_traced(lambda lt: cx.SurvivalData(f([1.0, 2.0]), left_trunc=lt).lower, f([0.5, 0.0]))


@pytest.mark.step("A3")
def test_traced_right_trunc_is_rejected_naming_with_issues():
    """spec 6.2, rev 7: `right_trunc` is an observation input."""
    _flag_rejects_traced(lambda rt: cx.SurvivalData(f([1.0, 2.0]), right_trunc=rt).lower, f([5.0, INF]))


@pytest.mark.step("A3")
def test_list_containing_a_tracer_is_rejected():
    """spec 6.2, rev 7: a Python list containing a tracer is traced."""
    _flag_rejects_traced(lambda x: cx.SurvivalData([x, 2.0]).lower, f(1.0))


@pytest.mark.step("A3")
def test_tuple_containing_a_tracer_is_rejected():
    """spec 6.2, rev 7: a Python tuple containing a tracer is traced."""
    _flag_rejects_traced(lambda x: cx.SurvivalData(f([1.0, 2.0]), left_trunc=(0.0, x)).lower, f(0.5))


@pytest.mark.step("A3")
def test_nested_list_containing_a_tracer_is_rejected():
    """spec 6.2, rev 7: a list is traced if it contains a tracer at any nesting depth."""
    _flag_rejects_traced(lambda x: cx.SurvivalData([[1.0, x], [3.0, 4.0]]).lower, f(2.0))


@pytest.mark.step("A3")
def test_jnp_array_created_inside_jit_is_rejected():
    """spec 6.2, rev 7: arrays created with `jax.numpy` inside the function are traced, even with constant values."""
    # Depends on JAX staging constant jnp ops inside jit as tracers (true in
    # current JAX); if this fails, check the JAX version first (TESTMAP.md).
    with pytest.raises(TRACED_INPUT, match=FLAG_COUNTERPART):
        jax.jit(lambda: cx.SurvivalData(jnp.asarray([1.0, 2.0])).lower)()


# ------------------------------------------- traced bounds-form inputs


@pytest.mark.step("A3")
def test_traced_lower_is_rejected_naming_from_bounds_with_issues():
    """spec 6.2, rev 7: a traced `lower` makes `from_bounds` raise at trace time, naming `from_bounds_with_issues`."""
    _bounds_rejects_traced(lambda lo: cx.SurvivalData.from_bounds(lo, f([2.0, 3.0])).lower, f([1.0, 0.0]))


@pytest.mark.step("A3")
def test_traced_upper_is_rejected_naming_from_bounds_with_issues():
    """spec 6.2, rev 7: `upper` is an observation input."""
    _bounds_rejects_traced(lambda hi: cx.SurvivalData.from_bounds(f([1.0, 0.0]), hi).lower, f([2.0, 3.0]))


@pytest.mark.step("A3")
def test_traced_exact_mask_is_rejected_naming_from_bounds_with_issues():
    """spec 6.2, rev 7: `exact` is an observation input."""
    _bounds_rejects_traced(
        lambda e: cx.SurvivalData.from_bounds(f([1.0, 1.0]), f([1.0, 2.0]), exact=e).lower, b([True, False])
    )


@pytest.mark.step("A3")
def test_traced_bounds_left_trunc_is_rejected_naming_from_bounds_with_issues():
    """spec 6.2, rev 7: `left_trunc` is an observation input of the bounds form."""
    _bounds_rejects_traced(
        lambda lt: cx.SurvivalData.from_bounds(f([1.0, 0.0]), f([2.0, 3.0]), left_trunc=lt).lower, f([0.5, 0.0])
    )


@pytest.mark.step("A3")
def test_traced_bounds_right_trunc_is_rejected_naming_from_bounds_with_issues():
    """spec 6.2, rev 7: `right_trunc` is an observation input of the bounds form."""
    _bounds_rejects_traced(
        lambda rt: cx.SurvivalData.from_bounds(f([1.0, 0.0]), f([2.0, 3.0]), right_trunc=rt).lower, f([5.0, 5.0])
    )


# ------------------------------------------- argument misuse in pure constructors


@pytest.mark.step("A3")
def test_with_issues_raises_on_event_with_censoring_flag():
    """spec 6.2, rev 7: `event` combined with any non-None censoring flag raises in the pure constructor too."""
    with pytest.raises(MISUSE):
        with_issues(f([1.0, 2.0]), b([True, False]), right_censored=b([False, False]))


@pytest.mark.step("A3")
def test_with_issues_raises_on_event_with_censoring_flag_under_jit():
    """spec 6.2, rev 7: argument misuse raises independent of tracing."""
    with pytest.raises(MISUSE):
        jax.jit(lambda t, e, r: with_issues(t, e, right_censored=r)[1].ok)(
            f([1.0, 2.0]), b([True, False]), b([False, False])
        )


@pytest.mark.step("A3")
def test_ordinary_constructor_raises_misuse_error_before_traced_input_error():
    """spec 6.2, rev 7: misuse is checked before the traced-input check (`event` with a flag, all traced)."""
    with raises_misuse():
        jax.jit(lambda t, e, r: cx.SurvivalData(t, e, right_censored=r).lower)(
            f([1.0, 2.0]), b([True, False]), b([False, False])
        )


@pytest.mark.step("A3")
def test_bounds_constructor_raises_misuse_error_before_traced_input_error():
    """spec 6.2, rev 7: misuse is checked before the traced-input check (incompatible shapes, all traced)."""
    with raises_misuse():
        jax.jit(lambda lo, hi: cx.SurvivalData.from_bounds(lo, hi).lower)(
            f([1.0, 2.0, 3.0]), f([[4.0], [5.0], [6.0]])
        )
