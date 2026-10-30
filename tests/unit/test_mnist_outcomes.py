"""Offline use-case 03 Precision, Outcome generator, and Splits and keys."""

import sys
from pathlib import Path

import impl_tolerances as tol
import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

import survjax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.survival_mnist import model  # noqa: E402


def test_oracle_parameters_follow_digit_rule():
    """03: log b_d = log 10 - d/5, computed in float64."""
    assert model.TRUE_LOG_SCALES.dtype == np.float64
    assert model.TRUE_SHAPE.dtype == np.float64
    with mp.workdps(40):
        expected = [float(mp.log(10) - mp.mpf(d) / 5) for d in range(10)]
    np.testing.assert_allclose(
        model.TRUE_LOG_SCALES, expected, atol=tol.ATOL[True], rtol=tol.RTOL[True]
    )
    assert model.TRUE_SHAPE == 1.38


def test_calibration_matches_independent_weighted_integral():
    """03: synthetic 3:2:1 digit frequencies, independent mpmath survival integral."""
    labels = np.repeat(np.array([0, 4, 9]), [30000, 20000, 10000])
    c = model.calibrate_mnist(labels)
    with mp.workdps(40):
        integrals = [
            mp.quad(
                lambda t, d=d: mp.exp(
                    -((t / (10 * mp.exp(-mp.mpf(d) / 5))) ** mp.mpf("1.38"))
                ),
                [0, c],
            )
            / c
            for d in (0, 4, 9)
        ]
        fraction = (
            sum(
                weight * value
                for weight, value in zip((3, 2, 1), integrals, strict=True)
            )
            / 6
        )
    np.testing.assert_allclose(
        float(fraction), 0.5, atol=tol.ATOL[True], rtol=tol.RTOL[True]
    )


@pytest.mark.parametrize(
    "labels",
    [
        np.zeros(70000, dtype=int),
        np.zeros(5000, dtype=int),
        np.zeros(60000),
        np.full(60000, 10),
        np.full(60000, -1),
    ],
)
def test_calibration_rejects_wrong_source_cohorts_or_digits(labels):
    """03: never calibrate on test-inclusive rows, smoke subsets or bad labels."""
    with pytest.raises(ValueError, match="training-source"):
        model.calibrate_mnist(labels)


def test_outcome_keys_match_explicit_uniform_algebra():
    """03/00-common: per-global-index key, then split U/C, not a batch key."""
    labels = jnp.array([0, 4, 9])
    indices = jnp.array([0, 59999, 69999])
    actual = model.generate_outcomes(labels, 7.0, global_indices=indices)
    base = jax.random.fold_in(jax.random.fold_in(jax.random.key(20261025), 3), 0)
    for row, (digit, index) in enumerate(
        zip((0, 4, 9), (0, 59999, 69999), strict=True)
    ):
        u_key, c_key = jax.random.split(jax.random.fold_in(base, index))
        u = jax.random.uniform(u_key, (), dtype=jnp.float32)
        censor = 7 * jax.random.uniform(c_key, (), dtype=jnp.float32)
        # Independent inverse-CDF algebra; allow float32 exp/log rounding.
        event = 10 * np.exp(-digit / 5) * float(-jnp.log(u)) ** (1 / 1.38)
        np.testing.assert_allclose(
            actual.event_time[row], event, atol=tol.ATOL[False], rtol=tol.RTOL[False]
        )
        assert actual.censor_time[row] == censor
        assert actual.event[row] == (event <= float(censor))
        np.testing.assert_allclose(
            actual.time[row],
            min(event, float(censor)),
            atol=tol.ATOL[False],
            rtol=tol.RTOL[False],
        )
    assert all(
        value.dtype == jnp.float32
        for value in (actual.time, actual.event_time, actual.censor_time)
    )
    assert actual.event.dtype == jnp.bool_


def test_outcomes_are_stable_under_subsets_jit_and_vmap():
    """03: full/smoke/reordered rows use identical outcomes; float32 even with x64."""
    labels = jnp.arange(20) % 10
    full = model.generate_outcomes(labels, 7.0)
    indices = jnp.array([19, 0, 12, 5])
    subset = model.generate_outcomes(labels[indices], 7.0, global_indices=indices)
    compiled = jax.jit(model.generate_outcomes)(labels, 7.0)
    grouped = jax.vmap(
        lambda digits, rows: model.generate_outcomes(digits, 7.0, global_indices=rows)
    )(labels.reshape(4, 5), jnp.arange(20).reshape(4, 5))
    for original, selected, transformed, batched in zip(
        full, subset, compiled, grouped, strict=True
    ):
        np.testing.assert_array_equal(selected, original[indices])
        np.testing.assert_allclose(
            transformed, original, atol=tol.ATOL[False], rtol=tol.RTOL[False]
        )
        np.testing.assert_allclose(
            batched.ravel(), original, atol=tol.ATOL[False], rtol=tol.RTOL[False]
        )
    data = survjax.SurvivalData(full.time, full.event)
    data.issues.raise_if_invalid()
    oracle = survjax.Weibull.from_log(
        log_shape=jnp.log(jnp.float32(1.38)),
        log_scale=jnp.asarray(model.TRUE_LOG_SCALES, dtype=jnp.float32)[labels],
    )
    assert jnp.all(jnp.isfinite(oracle.log_likelihood(data)))


def test_outcome_shapes_must_align():
    """03: one scalar c and one label per global index."""
    with pytest.raises(ValueError, match="matching vectors"):
        model.generate_outcomes([0, 1], 7.0, global_indices=[0])
    with pytest.raises(ValueError, match="matching vectors"):
        model.generate_outcomes([[0, 1]], 7.0)
    with pytest.raises(ValueError, match="scalar"):
        model.generate_outcomes([0, 1], [7.0, 8.0])
