"""Shared checked observation normalization (spec 3.2, 6.1–6.2)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from survjax import SurvivalData
from survjax._observation_inputs import _resolve_data


@pytest.mark.parametrize("mode", ["eager", "jit", "vmap"])
def test_shared_raw_form_preserves_invalid_indicator_evidence(mode):
    """Spec 6.1–6.2: invalid runtime codes return data with construction issues."""
    time = jnp.array([1.0, 2.0, 3.0])
    event = jnp.array([1, 2, 0])

    def resolve(t, e):
        return _resolve_data(None, t, e)

    if mode == "vmap":
        data = jax.vmap(resolve)(time[None, :], event[None, :])[0]
    else:
        data = (jax.jit(resolve) if mode == "jit" else resolve)(time, event)
    # Event code 2 is invalid even though normalized observation fields are finite.
    np.testing.assert_array_equal(data.lower, [1, 2, 3])
    np.testing.assert_array_equal(data.issues.invalid_indicator, [False, True, False])
    np.testing.assert_array_equal(data.issues.invalid_rows, [False, True, False])
    assert not bool(data.issues.ok)


def test_prepared_rich_observations_keep_every_leaf():
    """Spec 3.2/6.1: the prepared form retains modes, truncation and raw issues."""
    data = SurvivalData(
        [1.0, 2.0, 3.0],
        left_censored=[0, 1, 0],
        interval_censored=[0, 0, 2],
        time2=[0.0, 0.0, 4.0],
        left_trunc=0.5,
        right_trunc=5.0,
    )
    for resolve in (
        lambda d: _resolve_data(d, None, None),
        jax.jit(lambda d: _resolve_data(d, None, None)),
    ):
        actual = resolve(data)
        assert jax.tree.structure(actual) == jax.tree.structure(data)
        for observed, original in zip(
            jax.tree.leaves(actual), jax.tree.leaves(data), strict=True
        ):
            np.testing.assert_array_equal(observed, original)
            assert observed.dtype == original.dtype
    np.testing.assert_array_equal(data.issues.invalid_indicator, [False, False, True])


@pytest.mark.parametrize(
    ("data", "time", "event"),
    [
        (None, None, None),
        (None, [1.0], None),
        (None, None, [1]),
        ([1.0], None, None),
        (SurvivalData([1.0]), [1.0], None),
        (SurvivalData([1.0]), None, [1]),
    ],
    ids=[
        "missing",
        "no-event",
        "no-time",
        "positional-array",
        "mixed-time",
        "mixed-event",
    ],
)
def test_shared_normalizer_rejects_static_input_misuse(data, time, event):
    """Spec 6.1: missing/mixed forms and positional raw arrays are TypeErrors."""
    with pytest.raises(TypeError):
        _resolve_data(data, time, event)
