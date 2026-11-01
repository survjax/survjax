"""Offline use-case 03 Precision, Outcome generator, and Splits and keys."""

import sys
from pathlib import Path

import impl_tolerances as tol
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.survival_mnist import data as mnist  # noqa: E402


def test_oracle_parameters_follow_digit_rule():
    """03: log b_d = log 10 - d/5, computed in float32."""
    assert mnist.TRUE_LOG_SCALES.dtype == np.float32
    assert mnist.TRUE_SHAPE.dtype == np.float32
    expected = np.log(10) - np.arange(10) / 5
    np.testing.assert_allclose(
        mnist.TRUE_LOG_SCALES, expected, atol=tol.ATOL[False], rtol=tol.RTOL[False]
    )
    assert mnist.TRUE_SHAPE == 1.38


def test_outcome_keys_match_explicit_uniform_algebra():
    """03/00-common: per-global-index key, then split U/C, not a batch key."""
    labels = (
        jnp.zeros(70000, dtype=int)
        .at[jnp.array([0, 59999, 69999])]
        .set(jnp.array([0, 4, 9]))
    )
    actual = mnist.generate_outcomes(labels)
    base = jax.random.fold_in(jax.random.fold_in(jax.random.key(20261025), 3), 0)
    for _row, (digit, index) in enumerate(
        zip((0, 4, 9), (0, 59999, 69999), strict=True)
    ):
        u_key, c_key = jax.random.split(jax.random.fold_in(base, index))
        u = jax.random.uniform(u_key, (), dtype=jnp.float32)
        censor = 7 * jax.random.uniform(c_key, (), dtype=jnp.float32)
        # Independent inverse-CDF algebra; allow float32 exp/log rounding.
        event = 10 * np.exp(-digit / 5) * float(-jnp.log(u)) ** (1 / 1.38)
        np.testing.assert_allclose(
            actual.event_time[index], event, atol=tol.ATOL[False], rtol=tol.RTOL[False]
        )
        assert actual.censor_time[index] == censor
        assert actual.event[index] == (event <= float(censor))
        np.testing.assert_allclose(
            actual.time[index],
            min(event, float(censor)),
            atol=tol.ATOL[False],
            rtol=tol.RTOL[False],
        )
    assert all(
        value.dtype == jnp.float32
        for value in (actual.time, actual.event_time, actual.censor_time)
    )
    assert actual.event.dtype == jnp.bool_


def test_outcomes_are_deterministic():
    """03: repeated generation preserves official-order outcomes."""
    labels = jnp.arange(70000) % 10
    first, second = mnist.generate_outcomes(labels), mnist.generate_outcomes(labels)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
