"""A2a: direct checks of the private per-row issue bitfield."""

import jax
import jax.numpy as jnp
import numpy as np

from survjax._data_validation import row_issues


def test_row_issues_preserves_each_raw_input_mask() -> None:
    """Spec 6.2: the five caller-supplied faults occupy distinct row bits."""
    masks = jnp.eye(5, dtype=bool)
    false = jnp.zeros(5, dtype=bool)
    bits = row_issues(
        false,
        jnp.ones(5),
        jnp.full(5, 2.0),
        jnp.zeros(5),
        jnp.full(5, jnp.inf),
        invalid_indicator=masks[0],
        conflicting_indicators=masks[1],
        missing_time2=masks[2],
        invalid_time=masks[3],
        invalid_time2=masks[4],
        check_interval=false,
    )
    # The five independent input rules use bits 0 through 4.
    np.testing.assert_array_equal(bits, [1, 2, 4, 8, 16])


def test_row_issues_checks_interval_order_only_after_endpoint_domains() -> None:
    """Spec 6.2 / D-004–005: bad endpoints suppress interval-order faults."""
    false = jnp.zeros(5, dtype=bool)
    bits = row_issues(
        false,
        jnp.array([2, 3, 3, 3, 3]),
        jnp.full(5, 2.0),
        jnp.zeros(5),
        jnp.full(5, jnp.inf),
        invalid_indicator=false,
        conflicting_indicators=false,
        missing_time2=jnp.array([False, False, True, False, False]),
        invalid_time=jnp.array([False, False, False, True, False]),
        invalid_time2=jnp.array([False, False, False, False, True]),
        check_interval=jnp.ones(5, dtype=bool),
    )
    # Equal and reversed bounds are invalid intervals; malformed endpoints
    # retain only their own fault even though lower >= upper.
    np.testing.assert_array_equal(bits, [32, 32, 4, 8, 16])


def test_row_issues_rejects_invalid_truncation_windows() -> None:
    """Spec 3.3/6.2: L must be finite/nonnegative and L < U; U cannot be NaN."""
    false = jnp.zeros(5, dtype=bool)
    bits = row_issues(
        jnp.ones(5, dtype=bool),
        jnp.full(5, 2.0),
        jnp.full(5, 2.0),
        jnp.array([-1.0, jnp.inf, 0.0, 1.0, 2.0]),
        jnp.array([jnp.inf, jnp.inf, jnp.nan, 1.0, 1.0]),
        invalid_indicator=false,
        conflicting_indicators=false,
        missing_time2=false,
        invalid_time=false,
        invalid_time2=false,
        check_interval=false,
    )
    np.testing.assert_array_equal(bits, [64, 64, 64, 64, 64])


def test_row_issues_exact_membership_has_open_left_closed_right() -> None:
    """Spec 3.3: with (L,U]=(1,3], t=1 and t=4 are outside."""
    false = jnp.zeros(4, dtype=bool)
    times = jnp.array([1.0, 2.0, 3.0, 4.0])
    bits = row_issues(
        jnp.ones(4, dtype=bool),
        times,
        times,
        jnp.ones(4),
        jnp.full(4, 3.0),
        invalid_indicator=false,
        conflicting_indicators=false,
        missing_time2=false,
        invalid_time=false,
        invalid_time2=false,
        check_interval=false,
    )
    np.testing.assert_array_equal(bits, [128, 0, 0, 128])


def test_row_issues_right_censoring_before_but_not_at_entry() -> None:
    """Spec 6.2 / D-003: c<L faults; c=L and L<c<U remain valid."""
    false = jnp.zeros(3, dtype=bool)
    bits = row_issues(
        false,
        jnp.array([0.5, 1.0, 2.0]),
        jnp.full(3, jnp.inf),
        jnp.ones(3),
        jnp.full(3, 3.0),
        invalid_indicator=false,
        conflicting_indicators=false,
        missing_time2=false,
        invalid_time=false,
        invalid_time2=false,
        check_interval=false,
    )
    np.testing.assert_array_equal(bits, [256, 0, 0])


def test_row_issues_marks_empty_censored_intersections() -> None:
    """Spec 2/6.2: upper<=L or lower>=U makes the intersection empty."""
    false = jnp.zeros(4, dtype=bool)
    bits = row_issues(
        false,
        jnp.array([0.0, 3.0, 0.0, 1.0]),
        jnp.array([1.0, jnp.inf, 3.0, 3.0]),
        jnp.ones(4),
        jnp.full(4, 3.0),
        invalid_indicator=false,
        conflicting_indicators=false,
        missing_time2=false,
        invalid_time=false,
        invalid_time2=false,
        check_interval=jnp.ones(4, dtype=bool),
    )
    # The last two sets contain the full window (1,3].
    np.testing.assert_array_equal(bits, [512, 512, 0, 0])


def test_row_issues_keeps_independent_faults_and_observation_shape() -> None:
    """Spec 3.2/6.2: faults overlap without losing the int32 row layout."""
    false = jnp.zeros((2, 2), dtype=bool)
    bits = row_issues(
        false,
        jnp.ones((2, 2)),
        jnp.full((2, 2), 2.0),
        jnp.array([[-1.0, 0.0], [0.0, 0.0]]),
        jnp.full((2, 2), jnp.inf),
        invalid_indicator=jnp.array([[True, False], [False, False]]),
        conflicting_indicators=jnp.array([[True, False], [False, False]]),
        missing_time2=jnp.array([[False, True], [False, False]]),
        invalid_time=false,
        invalid_time2=false,
        check_interval=false,
    )
    assert bits.shape == (2, 2)
    assert bits.dtype == jnp.int32
    np.testing.assert_array_equal(bits, [[67, 4], [0, 0]])


def test_row_issues_agrees_under_jit_and_vmap() -> None:
    """Spec 6.2: array-only validation gives the same right-censoring bits."""

    def one(lower: jax.Array) -> jax.Array:
        false = jnp.asarray(False)
        return row_issues(
            false,
            lower,
            jnp.asarray(jnp.inf),
            jnp.asarray(1.0),
            jnp.asarray(3.0),
            invalid_indicator=false,
            conflicting_indicators=false,
            missing_time2=false,
            invalid_time=false,
            invalid_time2=false,
            check_interval=false,
        )

    times = jnp.array([0.5, 1.0, 3.0])
    eager = jax.vmap(one)(times)
    np.testing.assert_array_equal(eager, [256, 0, 512])
    np.testing.assert_array_equal(jax.jit(jax.vmap(one))(times), eager)
