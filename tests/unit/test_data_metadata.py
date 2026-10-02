"""Concrete metadata, tracer classification and reuse (spec 3.2, 6.2)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from survjax import InvalidDataError, SurvivalData


def test_normalized_kinds_and_truncation_flags() -> None:
    """Spec 3.2: flags derive from every normalized row, not keyword choices."""
    data = SurvivalData(
        [1.0, 2.0, 3.0, 1.0],
        right_censored=[0, 1, 0, 0],
        left_censored=[0, 0, 1, 0],
        interval_censored=[0, 0, 0, 1],
        time2=[0.0, 0.0, 0.0, 4.0],
        left_trunc=0.5,
        right_trunc=10.0,
    )
    bounds = SurvivalData.from_bounds(
        [1.0, 2.0, 0.0, 1.0],
        [1.0, np.inf, 3.0, 4.0],
        exact=[1, 0, 0, 0],
        left_trunc=0.5,
        right_trunc=10.0,
    )
    assert data.tree_flatten()[1] == (True,) * 6
    assert jax.tree.structure(data) == jax.tree.structure(bounds)
    # The infinite upper normalizes to right censoring, despite the interval flag.
    invalid = SurvivalData([2.0], interval_censored=True, time2=np.inf, right_trunc=2.0)
    assert not bool(invalid.issues.ok)
    assert invalid.tree_flatten()[1] == (False, True, False, False, False, True)
    assert data[:1].tree_flatten()[1] == data.tree_flatten()[1]
    assert data[:0].tree_flatten()[1] == data.tree_flatten()[1]


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, (True, False, False, False, False, False)),
        (
            {"left_trunc": 0.0, "right_trunc": np.inf},
            (True, False, False, False, False, False),
        ),
        ({"left_trunc": 0.5}, (True, False, False, False, True, False)),
        ({"right_trunc": 10.0}, (True, False, False, False, False, True)),
        ({"right_censored": True}, (False, True, False, False, False, False)),
        ({"left_censored": True}, (False, False, True, False, False, False)),
        (
            {"interval_censored": True, "time2": 3.0},
            (False, False, False, True, False, False),
        ),
    ],
)
def test_independent_presence_flags(kwargs: dict, expected: tuple[bool, ...]) -> None:
    """Spec 3.2: each normalized kind and truncation has a separate bool."""
    data = SurvivalData([1.0, 2.0], **kwargs)
    assert data.tree_flatten()[1] == expected
    assert jax.tree.structure(data) == jax.tree.structure(SurvivalData([1.0], **kwargs))


@pytest.mark.parametrize(
    "time",
    [
        [1.0, 2.0],
        (1.0, 2.0),
        np.array([1.0, 2.0], dtype=np.float32),
        jnp.array([1.0, 2.0]),
    ],
)
def test_concrete_inputs_inside_jit_keep_eager_flags(time: object) -> None:
    """Spec 6.2: input classification precedes operations; unrelated tracers do not matter."""
    eager = SurvivalData(time)
    result = jax.jit(lambda scale: (SurvivalData(time), scale + 1))(jnp.array(2.0))[0]
    assert jax.tree.structure(result) == jax.tree.structure(eager)
    bounds = jax.jit(lambda: SurvivalData.from_bounds(time, time, exact=True))()
    assert jax.tree.structure(bounds) == jax.tree.structure(eager)
    for result_data in (result, bounds):
        for actual, expected in zip(
            jax.tree.leaves(result_data), jax.tree.leaves(eager), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)


def test_unbounded_zero_lower_is_right_censored_only() -> None:
    """Spec 3.3: (0, +inf] is right censoring at entry, not left censoring."""
    data = SurvivalData.from_bounds(0.0, np.inf)
    assert data.tree_flatten()[1] == (False, True, False, False, False, False)
    assert bool(data.issues.uninformative)


def test_nan_truncation_is_invalid_despite_false_presence_flags() -> None:
    """Spec 3.2/6.2: NaN comparisons are false; issue bits still report invalidity."""
    data = SurvivalData([1.0], left_trunc=np.nan, right_trunc=np.nan)
    assert data.tree_flatten()[1] == (True, False, False, False, False, False)
    np.testing.assert_array_equal(data.issues.invalid_truncation, [True])


def test_negative_entry_is_invalid_despite_false_presence_flags() -> None:
    """Spec 3.2/6.2: negative entry is not positive truncation, but is invalid."""
    data = SurvivalData([1.0], left_trunc=-1.0)
    assert data.tree_flatten()[1] == (True, False, False, False, False, False)
    np.testing.assert_array_equal(data.issues.invalid_truncation, [True])


@pytest.mark.parametrize("bounds", [False, True], ids=["flags", "bounds"])
def test_concrete_construction_inside_jit_has_no_traced_leaves(bounds: bool) -> None:
    """Spec 3.2/6.2: concrete normalization and raw issue bits match eager results."""

    def build() -> SurvivalData:
        if bounds:
            return SurvivalData.from_bounds([1.0, 2.0], [1.0, 2.0], exact=[1, 2])
        return SurvivalData([1.0, 2.0], [1, 2])

    def inside() -> SurvivalData:
        data = build()
        assert not any(
            isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(data)
        )
        return data

    result = jax.jit(inside)()
    for actual, expected in zip(
        jax.tree.leaves(result), jax.tree.leaves(build()), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bounds", [False, True], ids=["flags", "bounds"])
def test_concrete_data_checkpoint_inside_jit_remains_host_only(bounds: bool) -> None:
    """Spec 6.2: diagnostics decoded inside jit cannot use the host checkpoint."""

    def inside() -> None:
        data = (
            SurvivalData.from_bounds([0.0], [0.0], exact=True)
            if bounds
            else SurvivalData([0.0])
        )
        data.issues.raise_if_invalid()

    with pytest.raises(ValueError, match="host-only") as error:
        jax.jit(inside)()
    assert not isinstance(error.value, InvalidDataError)


@pytest.mark.parametrize("bounds", [False, True], ids=["flags", "bounds"])
def test_vmap_construction_has_all_presence_flags(bounds: bool) -> None:
    """Spec 6.2/12.1: observation inputs passed through vmap are traced."""
    times = jnp.ones((2, 2))
    build = (
        (lambda t: SurvivalData.from_bounds(t, t, exact=True))
        if bounds
        else SurvivalData
    )
    data = jax.vmap(build)(times)
    assert data.tree_flatten()[1] == (True,) * 6
    np.testing.assert_array_equal(data.lower, [[1, 1], [1, 1]])


def test_observation_tracers_force_all_flags_without_changing_leaves() -> None:
    """Spec 3.2/6.2: nested and single-argument tracers conservatively set every flag."""
    builders = (
        lambda x: SurvivalData([[1.0, x]]),
        lambda x: SurvivalData.from_bounds(((1.0, x),), ((1.0, 2.0),), exact=True),
        lambda x: SurvivalData([1.0, 2.0], left_trunc=x),
    )
    for build, value in zip(builders, (2.0, 2.0, 0.0), strict=True):
        eager = build(value)
        traced = jax.jit(build)(jnp.array(value))
        assert traced.tree_flatten()[1] == (True,) * 6
        for actual, expected in zip(
            jax.tree.leaves(traced), jax.tree.leaves(eager), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
    created = jax.jit(lambda: SurvivalData(jnp.asarray([1.0, 2.0])))()
    assert created.tree_flatten()[1] == (True,) * 6


def test_static_flags_control_compilation_reuse() -> None:
    """Spec 3.2/12.1: value changes reuse traces; different flags require a new trace."""
    traces = []

    @jax.jit
    def total(data: SurvivalData) -> jax.Array:
        traces.append(None)
        return data.lower.sum()

    total(SurvivalData([1.0, 2.0]))
    total(SurvivalData([3.0, 4.0]))
    assert len(traces) == 1
    total(SurvivalData([1.0, 2.0], [1, 0]))
    assert len(traces) == 2
    parent = SurvivalData([1.0, 2.0, 3.0, 4.0], [1, 1, 0, 0])
    total(parent[:2])
    total(parent[2:])
    assert len(traces) == 2
    # Select all six flags from a concrete cohort containing every kind/window.
    full = SurvivalData.from_bounds(
        [1.0, 2.0, 0.0, 1.0],
        [1.0, np.inf, 3.0, 4.0],
        exact=[1, 0, 0, 0],
        left_trunc=0.5,
        right_trunc=10.0,
    )[:2]
    total(full)
    total(jax.jit(SurvivalData)(jnp.asarray([1.0, 2.0])))
    assert len(traces) == 3
