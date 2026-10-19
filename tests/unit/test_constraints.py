"""Domain masks from the inequalities in spec 4.7, not family formulas."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from survjax import constraints as c


@pytest.mark.parametrize(
    ("constraint", "expected"),
    [
        (c.real, [False, False, True, True, True, False]),
        (c.positive, [False, False, False, False, True, False]),
        (c.nonnegative, [False, False, False, True, True, False]),
        (c.unit_interval, [False, False, False, True, True, False]),
        (c.greater_than(-1), [False, False, False, True, True, False]),
        (c.interval(-1, 1), [False, False, False, True, False, False]),
        (c.closed_interval(-1, 1), [False, False, True, True, True, False]),
    ],
)
def test_scalar_domains_eager_jit_vmap(constraint, expected):
    """Spec 4.7: bounds, interior and nonfinite values have stated polarity."""
    x = jnp.array([jnp.nan, -jnp.inf, -1, 0, 1, jnp.inf])
    for evaluate in (constraint, jax.jit(constraint), jax.vmap(constraint)):
        np.testing.assert_array_equal(evaluate(x), expected)


@pytest.mark.parametrize(
    "make",
    [c.greater_than, lambda a: c.interval(a, 2), lambda a: c.closed_interval(a, 2)],
)
@pytest.mark.parametrize("bound", [jnp.nan, jnp.inf, -jnp.inf, "0", jnp.array(0.0)])
def test_non_python_or_nonfinite_bounds_raise(make, bound):
    """Spec 4.7: malformed bounds raise TypeError."""
    with pytest.raises(TypeError):
        make(bound)


def test_interval_bounds_require_strict_order():
    """Spec 4.7: neither equal nor reversed bounds define a constraint."""
    for make in (c.interval, c.closed_interval):
        for bounds in ((1, 1), (2, 1), (0, jnp.inf)):
            with pytest.raises(TypeError):
                make(*bounds)


@pytest.mark.parametrize(
    "make",
    [
        c.greater_than,
        lambda a: c.interval(a, 1),
        lambda b: c.interval(0, b),
        lambda a: c.closed_interval(a, 1),
        lambda b: c.closed_interval(0, b),
    ],
)
def test_overflowing_bound_check_raises_type_error(make):
    """Spec 4.7: bound checks report TypeError, including finite-check overflow."""
    with pytest.raises(TypeError, match="finite Python numbers"):
        make(10**400)


def test_zero_boundaries_distinguish_subnormals_from_signed_zero():
    """Spec 4.7: signed zeros are zero; nextafter neighbours have strict signs."""
    dtype = jnp.asarray(0.0).dtype
    tiny = np.nextafter(dtype.type(0), dtype.type(1))
    x = jnp.asarray([-tiny, -0.0, 0.0, tiny])
    for con, expected in (
        (c.positive, [False, False, False, True]),
        (c.nonnegative, [False, True, True, True]),
        (c.unit_interval, [False, True, True, True]),
        (c.interval(0, 1), [False, False, False, True]),
        (c.closed_interval(0, 1), [False, True, True, True]),
    ):
        for evaluate in (con, jax.jit(con), jax.vmap(con)):
            np.testing.assert_array_equal(evaluate(x), expected)


def test_ordered_reduces_last_axis_and_rejects_ties():
    """Spec 4.7: finite strict order; hand-enumerated vectors."""
    x = jnp.array([[[0.0, 1.0], [1.0, 1.0]], [[2.0, 1.0], [0.0, jnp.inf]]])
    for evaluate in (c.ordered, jax.jit(c.ordered), jax.vmap(c.ordered)):
        np.testing.assert_array_equal(evaluate(x), [[True, False], [False, False]])
    assert c.ordered.event_dim == 1


def test_simplex_domains_and_softmax_summation_tolerance():
    """Spec 4.7/12.1: nonnegative unit sums, including softmax length 1000."""
    for n in (2, 10, 100, 1000):
        weights = jax.nn.softmax(
            jnp.asarray(np.random.default_rng(n).normal(size=(2, n)))
        )
        for evaluate in (c.simplex, jax.jit(c.simplex), jax.vmap(c.simplex)):
            np.testing.assert_array_equal(evaluate(weights), [True, True])
    x = jnp.array([[0.0, 1.0], [-1.0, 2.0], [jnp.nan, 1.0], [jnp.inf, 0.0]])
    np.testing.assert_array_equal(c.simplex(x), [True, False, False, False])
    assert not bool(c.simplex(jnp.array([0.101] + [0.1] * 9)))  # sum=1.001
    assert c.simplex.event_dim == 1


def test_simplex_tolerance_includes_four_n_eps_and_excludes_next_value(x64_enabled):
    """Spec 4.7: the inclusive threshold is exactly 4*n*eps of the input dtype."""
    dtype = np.float64 if x64_enabled else np.float32
    eps = np.finfo(dtype).eps
    vectors = np.zeros((4, 10), dtype=dtype)
    # Each sum is exactly 1+k*eps: k<=4*10 passes, the next representable k fails.
    vectors[:, 0] = 1 + np.array([30, 40, 41, 50], dtype=dtype) * eps
    x = jnp.asarray(vectors)
    for evaluate in (c.simplex, jax.jit(c.simplex), jax.vmap(c.simplex)):
        np.testing.assert_array_equal(evaluate(x), [True, True, False, False])
