"""A6: the `survjax.constraints` vocabulary (spec 4.7, 10.2, 12.1; rev 28, 30).

Each constraint is checked inside its domain, outside it, on its boundary and
at NaN and ``±inf``, eagerly, under ``jit`` and under ``vmap``; the suite runs
once per dtype mode. Boundary neighbours are ``nextafter`` of the bound in the
active dtype, so they are representable exactly.

Revision 30: comparisons are JAX's ordinary operators in the parameter dtype,
and where the backend flushes subnormals (XLA on CPU) "a subnormal value or
bound ... may be classified either way". No test uses a subnormal value or
bound: the neighbour of the bound ``0`` is the smallest normal number
``finfo.tiny``, not ``nextafter(0)``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import FLOAT, INF, NAN, f

import survjax

def _c():
    """Looked up per test, so a missing module fails only the tests that use it."""
    return survjax.constraints


def _up(x):
    return np.nextafter(FLOAT.type(x), FLOAT.type(INF))


def _down(x):
    return np.nextafter(FLOAT.type(x), FLOAT.type(-INF))


def _tiny():
    """Smallest positive normal number of the active dtype (never flushed)."""
    return float(np.finfo(FLOAT).tiny)


def _event_dim(c):
    return getattr(c, "event_dim", 0)


def _check(c, values, expected):
    """``c(values)`` equals ``expected`` eagerly, under jit and under vmap over the leading axis."""
    x = f(values)
    expected = np.asarray(expected, dtype=bool)
    eager = np.asarray(c(x))
    assert eager.dtype == np.bool_
    assert eager.shape == expected.shape
    np.testing.assert_array_equal(eager, expected, err_msg="eager")
    np.testing.assert_array_equal(
        np.asarray(jax.jit(c)(x)), expected, err_msg="jit"
    )
    np.testing.assert_array_equal(
        np.asarray(jax.vmap(c)(x)), expected, err_msg="vmap"
    )


# ------------------------------------------------------------- module and names


@pytest.mark.step("A6")
def test_constraints_module_is_exported_at_top_level():
    """spec 10.2, rev 28: the `constraints` module is re-exported as `survjax.constraints`."""
    import survjax.constraints as module

    assert survjax.constraints is module


@pytest.mark.step("A6")
def test_vocabulary_names_exist():
    """spec 4.7, rev 28: `survjax.constraints` provides the nine constraints of the vocabulary table."""
    for name in (
        "real",
        "positive",
        "nonnegative",
        "greater_than",
        "interval",
        "closed_interval",
        "unit_interval",
        "ordered",
        "simplex",
    ):
        assert hasattr(_c(), name), name


@pytest.mark.step("A6")
def test_scalar_constraints_have_event_dim_zero():
    """spec 4.7, rev 28: `event_dim` is 0 for every scalar constraint (absent means 0)."""
    c = _c()
    for con in (
        c.real,
        c.positive,
        c.nonnegative,
        c.greater_than(1.0),
        c.interval(0.0, 1.0),
        c.closed_interval(0.0, 1.0),
        c.unit_interval,
    ):
        assert callable(con)
        assert _event_dim(con) == 0


@pytest.mark.step("A6")
def test_vector_constraints_have_event_dim_one():
    """spec 4.7, rev 28: `ordered` and `simplex` reduce one trailing axis (`event_dim=1`)."""
    assert _c().ordered.event_dim == 1
    assert _c().simplex.event_dim == 1


# ------------------------------------------------------------------------ real


@pytest.mark.step("A6")
def test_real_accepts_finite_values():
    """spec 4.7, rev 28: `real` is "finite"."""
    big = float(np.finfo(FLOAT).max)
    _check(_c().real, [-big, -1.0, 0.0, 1.0, big], [True] * 5)


@pytest.mark.step("A6")
def test_real_rejects_nan_and_infinities():
    """spec 4.7, rev 28: NaN is outside every constraint; `real` excludes infinite values."""
    _check(_c().real, [NAN, INF, -INF], [False] * 3)


# -------------------------------------------------------------------- positive


@pytest.mark.step("A6")
def test_positive_accepts_finite_positive_values():
    """spec 4.7, rev 28: `positive` is "finite, `>0`"; the smallest normal number is inside."""
    tiny = float(np.finfo(FLOAT).tiny)
    _check(_c().positive, [tiny, 1.0, 1e30], [True] * 3)


@pytest.mark.step("A6")
def test_positive_rejects_zero():
    """spec 4.7, rev 28: `positive` excludes its boundary `0`."""
    _check(_c().positive, [0.0], [False])


@pytest.mark.step("A6")
def test_positive_rejects_negative_values():
    """spec 4.7, rev 28: `positive` is `>0`."""
    _check(_c().positive, [-1e-30, -1.0], [False, False])


@pytest.mark.step("A6")
def test_positive_rejects_smallest_negative_normal():
    """spec 4.7, rev 30: `-finfo.tiny` is normal, so the ordinary comparison `x > 0` rejects it."""
    _check(_c().positive, [-_tiny()], [False])


@pytest.mark.step("A6")
def test_positive_rejects_negative_zero():
    """spec 4.7, rev 30: with JAX's ordinary comparison, `-0.0 > 0` is false."""
    _check(_c().positive, [-0.0], [False])


@pytest.mark.step("A6")
def test_positive_rejects_nan_and_infinities():
    """spec 4.7, rev 28: `positive` is finite; NaN is outside every constraint."""
    _check(_c().positive, [NAN, INF, -INF], [False] * 3)


# ----------------------------------------------------------------- nonnegative


@pytest.mark.step("A6")
def test_nonnegative_accepts_zero_and_positive_values():
    """spec 4.7, rev 28: `nonnegative` is "finite, `>=0`", so its boundary `0` is inside."""
    _check(_c().nonnegative, [0.0, 1e-30, 5.0], [True] * 3)


@pytest.mark.step("A6")
def test_nonnegative_rejects_negative_values():
    """spec 4.7, rev 28: `nonnegative` is `>=0`."""
    _check(_c().nonnegative, [-1e-30, -1.0], [False, False])


@pytest.mark.step("A6")
def test_nonnegative_accepts_smallest_positive_normal():
    """spec 4.7, rev 30: `finfo.tiny` is normal and `>=0`, so it is inside."""
    _check(_c().nonnegative, [_tiny()], [True])


@pytest.mark.step("A6")
def test_nonnegative_rejects_smallest_negative_normal():
    """spec 4.7, rev 30: `-finfo.tiny` is normal, so the ordinary comparison `x >= 0` rejects it."""
    _check(_c().nonnegative, [-_tiny()], [False])


@pytest.mark.step("A6")
def test_nonnegative_accepts_negative_zero():
    """spec 4.7, rev 30: comparisons are JAX's ordinary operators, and `-0.0 >= 0` is true; no sign test applies."""
    _check(_c().nonnegative, [-0.0], [True])


@pytest.mark.step("A6")
def test_nonnegative_rejects_nan_and_infinities():
    """spec 4.7, rev 28: `nonnegative` is finite; NaN is outside every constraint."""
    _check(_c().nonnegative, [NAN, INF, -INF], [False] * 3)


# ---------------------------------------------------------------- greater_than


@pytest.mark.step("A6")
def test_greater_than_is_strict_at_its_bound():
    """spec 4.7, rev 28: `greater_than(a)` is "finite, `>a`": `a` itself is outside, its successor inside."""
    _check(
        _c().greater_than(2.0),
        [_down(2.0), 2.0, _up(2.0), 100.0],
        [False, False, True, True],
    )


@pytest.mark.step("A6")
def test_greater_than_with_negative_bound():
    """spec 4.7, rev 28: `greater_than(a)` with `a<0` admits zero and negative values above `a`."""
    _check(_c().greater_than(-1.5), [-2.0, -1.5, -1.0, 0.0], [False, False, True, True])


@pytest.mark.step("A6")
def test_greater_than_accepts_python_int_bound():
    """spec 4.7, rev 28: bounds are finite Python numbers, so an `int` is accepted."""
    _check(_c().greater_than(0), [0.0, 1.0], [False, True])


@pytest.mark.step("A6")
def test_greater_than_rejects_nan_and_infinities():
    """spec 4.7, rev 28: `greater_than(a)` is finite, so `+inf` is outside even though `inf > a`."""
    _check(_c().greater_than(2.0), [NAN, INF, -INF], [False] * 3)


# -------------------------------------------------------------------- interval


@pytest.mark.step("A6")
def test_interval_is_open_at_both_ends():
    """spec 4.7, rev 28: `interval(a, b)` is `a<x<b`; both endpoints are outside."""
    _check(
        _c().interval(-1.5, 2.5),
        [_down(-1.5), -1.5, _up(-1.5), 0.0, _down(2.5), 2.5, _up(2.5)],
        [False, False, True, True, True, False, False],
    )


@pytest.mark.step("A6")
def test_interval_rejects_nan_and_infinities():
    """spec 4.7, rev 28: NaN and `±inf` lie outside `interval(a, b)`."""
    _check(_c().interval(-1.5, 2.5), [NAN, INF, -INF], [False] * 3)


# ------------------------------------------------------------- closed_interval


@pytest.mark.step("A6")
def test_closed_interval_includes_both_ends():
    """spec 4.7, rev 28: `closed_interval(a, b)` is `a<=x<=b`; both endpoints are inside."""
    _check(
        _c().closed_interval(-1.5, 2.5),
        [_down(-1.5), -1.5, 0.0, 2.5, _up(2.5)],
        [False, True, True, True, False],
    )


@pytest.mark.step("A6")
def test_closed_interval_rejects_nan_and_infinities():
    """spec 4.7, rev 28: NaN and `±inf` lie outside `closed_interval(a, b)`."""
    _check(_c().closed_interval(-1.5, 2.5), [NAN, INF, -INF], [False] * 3)


# --------------------------------------------------------------- unit_interval


@pytest.mark.step("A6")
def test_unit_interval_includes_zero_and_one():
    """spec 4.7, rev 30: `unit_interval` is `0<=x<=1`; below 0 is checked at `-finfo.tiny`, not a subnormal."""
    _check(
        _c().unit_interval,
        [-_tiny(), 0.0, 0.5, 1.0, _up(1.0)],
        [False, True, True, True, False],
    )


@pytest.mark.step("A6")
def test_unit_interval_rejects_nan_and_infinities():
    """spec 4.7, rev 28: NaN is outside every constraint; `±inf` lie outside the bounded `unit_interval`."""
    _check(_c().unit_interval, [NAN, INF, -INF], [False] * 3)


# ---------------------------------------------------------------- invalid bounds


@pytest.mark.step("A6")
@pytest.mark.parametrize("a", [NAN, INF, -INF])
def test_greater_than_rejects_non_finite_bound(a):
    """spec 4.7, rev 28: bounds are finite Python numbers; anything else raises `TypeError`."""
    with pytest.raises(TypeError):
        _c().greater_than(a)


@pytest.mark.step("A6")
def test_greater_than_rejects_array_bound():
    """spec 4.7, rev 28: a JAX array is not a Python number, so it raises `TypeError`."""
    with pytest.raises(TypeError):
        _c().greater_than(jnp.asarray(1.0))


@pytest.mark.step("A6")
def test_greater_than_rejects_string_bound():
    """spec 4.7, rev 28: a string is not a Python number, so it raises `TypeError`."""
    with pytest.raises(TypeError):
        _c().greater_than("1")


@pytest.mark.step("A6")
@pytest.mark.parametrize("make", ["interval", "closed_interval"])
@pytest.mark.parametrize(
    ("a", "b"),
    [(2.0, 1.0), (1.0, 1.0), (0.0, INF), (-INF, 0.0), (NAN, 1.0), (0.0, NAN)],
    ids=["reversed", "equal", "inf-upper", "inf-lower", "nan-lower", "nan-upper"],
)
def test_interval_constraints_reject_invalid_bounds(make, a, b):
    """spec 4.7, rev 28: bounds must be finite with `a<b`; anything else raises `TypeError`."""
    with pytest.raises(TypeError):
        getattr(_c(), make)(a, b)


@pytest.mark.step("A6")
@pytest.mark.parametrize("make", ["interval", "closed_interval"])
def test_interval_constraints_reject_array_bounds(make):
    """spec 4.7, rev 28: JAX array bounds are not Python numbers, so they raise `TypeError`."""
    with pytest.raises(TypeError):
        getattr(_c(), make)(jnp.asarray(0.0), jnp.asarray(1.0))


# --------------------------------------------------------------------- ordered


@pytest.mark.step("A6")
def test_ordered_accepts_strictly_increasing_vectors():
    """spec 4.7, rev 28: `ordered` is "finite, strictly increasing along the last axis"."""
    _check(
        _c().ordered,
        [[1.0, 2.0, 3.0], [-1e30, 0.0, 1e30]],
        [True, True],
    )


@pytest.mark.step("A6")
def test_ordered_rejects_ties():
    """spec 4.7, rev 28 and 12.1: "`ordered` fails ties"."""
    _check(_c().ordered, [[1.0, 1.0, 2.0], [1.0, 2.0, 2.0]], [False, False])


@pytest.mark.step("A6")
def test_ordered_rejects_decreasing_steps():
    """spec 4.7, rev 28: one decreasing step anywhere along the last axis is outside `ordered`."""
    _check(_c().ordered, [[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]], [False, False])


@pytest.mark.step("A6")
def test_ordered_rejects_infinite_entries():
    """spec 4.7, rev 28: `ordered` is finite, so increasing vectors with `±inf` entries are outside."""
    _check(_c().ordered, [[-INF, 0.0, 1.0], [0.0, 1.0, INF]], [False, False])


@pytest.mark.step("A6")
def test_ordered_rejects_nan_entries():
    """spec 4.7, rev 28: NaN is outside every constraint."""
    _check(_c().ordered, [[NAN, 1.0, 2.0], [0.0, NAN, 2.0]], [False, False])


@pytest.mark.step("A6")
def test_ordered_reduces_only_the_last_axis():
    """spec 4.7, rev 28: the result has the shape of `x` without its last `event_dim=1` axis."""
    x = f(np.arange(24.0).reshape(2, 3, 4))
    x[1, 2, 3] = 0.0
    out = np.asarray(_c().ordered(x))
    assert out.shape == (2, 3)
    expected = np.ones((2, 3), dtype=bool)
    expected[1, 2] = False
    np.testing.assert_array_equal(out, expected)


# --------------------------------------------------------------------- simplex


@pytest.mark.step("A6")
@pytest.mark.parametrize("n", [2, 10, 100, 1000])
def test_simplex_accepts_softmax_output(n):
    """spec 4.7 and 12.1, rev 28: `simplex` passes `softmax` output of length up to 1000 in both dtypes."""
    logits = np.random.default_rng(n).normal(scale=3.0, size=(3, n))
    w = np.asarray(jax.nn.softmax(jnp.asarray(f(logits)), axis=-1))
    assert w.dtype == FLOAT
    _check(_c().simplex, w, [True, True, True])


@pytest.mark.step("A6")
def test_simplex_rejects_sum_off_by_1e_minus_3():
    """spec 12.1, rev 28: `simplex` fails a length-10 vector summing to `1+1e-3`.

    The tolerance is `4*n*eps`: `4.4e-15` (float64) or `4.8e-6` (float32) for
    `n=10`, both far below `1e-3`.
    """
    w = np.full(10, 0.1)
    w[0] += 1e-3
    _check(_c().simplex, [w], [False])


@pytest.mark.step("A6")
def test_simplex_accepts_zero_entries():
    """spec 4.7, rev 28: "every entry `>=0`"; "Zero entries are allowed"."""
    _check(
        _c().simplex,
        [[0.0, 0.25, 0.75], [0.0, 0.0, 1.0]],
        [True, True],
    )


@pytest.mark.step("A6")
def test_simplex_rejects_negative_entries_even_when_sum_is_one():
    """spec 4.7, rev 28: every entry must be `>=0`."""
    _check(_c().simplex, [[-0.25, 0.5, 0.75]], [False])


@pytest.mark.step("A6")
def test_simplex_rejects_nan_and_infinite_entries():
    """spec 4.7, rev 28: NaN is outside every constraint; an infinite entry cannot sum to 1."""
    _check(
        _c().simplex,
        [[NAN, 0.5, 0.5], [INF, 0.0, 0.0], [-INF, 1.0, 1.0]],
        [False, False, False],
    )


@pytest.mark.step("A6")
def test_simplex_reduces_only_the_last_axis():
    """spec 4.7, rev 28: the result has the shape of `x` without its last axis."""
    x = np.full((2, 3, 4), 0.25)
    x[0, 1, 0] = 0.5
    out = np.asarray(_c().simplex(f(x)))
    assert out.shape == (2, 3)
    expected = np.ones((2, 3), dtype=bool)
    expected[0, 1] = False
    np.testing.assert_array_equal(out, expected)
