"""Traced constructor inputs and the public diagnostic property (spec 6.2)."""

import jax
import jax.numpy as jnp
import numpy as np

from survjax import SurvivalData


def test_only_ordinary_constructors_are_public() -> None:
    """Spec 6.2/10.2: issue-returning constructor methods are removed."""
    assert not hasattr(SurvivalData, "with_issues")
    assert not hasattr(SurvivalData, "from_bounds_with_issues")


def test_each_flag_argument_can_be_traced() -> None:
    """Spec 6.2: each supplied observation argument accepts a tracer."""

    def build(
        time: jax.Array,
        event: jax.Array,
        right: jax.Array,
        left: jax.Array,
        interval: jax.Array,
        time2: jax.Array,
        entry: jax.Array,
        exit: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        event_data = SurvivalData(time, event, left_trunc=entry, right_trunc=exit)
        flag_data = SurvivalData(
            time,
            right_censored=right,
            left_censored=left,
            interval_censored=interval,
            time2=time2,
            left_trunc=entry,
            right_trunc=exit,
        )
        return event_data._issue_bits, flag_data._issue_bits

    event_bits, flag_bits = jax.jit(build)(
        jnp.array([1.0, 2.0]),
        jnp.array([1, 0]),
        jnp.array([False, False]),
        jnp.array([False, False]),
        jnp.array([False, True]),
        jnp.array([0.0, 3.0]),
        jnp.array([0.0, 0.0]),
        jnp.array([4.0, 4.0]),
    )
    np.testing.assert_array_equal(event_bits, [0, 0])
    np.testing.assert_array_equal(flag_bits, [0, 0])


def test_each_bounds_argument_can_be_traced() -> None:
    """Spec 6.2: bounds, exactness and truncation accept tracers."""
    data = jax.jit(SurvivalData.from_bounds)(
        jnp.array([1.0, 2.0]),
        jnp.array([1.0, 3.0]),
        exact=jnp.array([True, False]),
        left_trunc=jnp.array([0.0, 0.0]),
        right_trunc=jnp.array([4.0, 4.0]),
    )
    np.testing.assert_array_equal(data._issue_bits, [0, 0])


def test_nested_traced_list_and_concrete_invalid_inside_jit() -> None:
    """Spec 6.2: nested tracers and captured invalid values return data."""
    nested = jax.jit(lambda value: SurvivalData([[value]]).issues.invalid_time)(0.0)
    np.testing.assert_array_equal(nested, [[True]])
    captured = jax.jit(lambda: SurvivalData([0.0]).issues.invalid_time)()
    np.testing.assert_array_equal(captured, [True])


def test_issues_survive_tree_reconstruction_and_slicing() -> None:
    """Spec 3.2/6.2: stored issue evidence is decoded after tree operations."""
    data = SurvivalData([0.0, 2.0])
    leaves, structure = jax.tree.flatten(data)
    rebuilt = jax.tree.unflatten(structure, leaves)
    sliced = jax.tree.map(lambda leaf: leaf[1:], rebuilt)
    np.testing.assert_array_equal(rebuilt.issues.invalid_time, [True, False])
    np.testing.assert_array_equal(sliced.issues.invalid_time, [False])
    assert bool(np.asarray(sliced.issues.ok))
