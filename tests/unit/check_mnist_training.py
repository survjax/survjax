"""Explicit examples-group checks for use-case 03 epoch training."""

import sys
from pathlib import Path

import equinox as eqx
import impl_tolerances as tol
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import survjax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.survival_mnist import model  # noqa: E402


def cohorts():
    markers = jnp.arange(257, dtype=jnp.float32)[:, None, None, None] / 256
    images = jnp.broadcast_to(markers, (257, 1, 28, 28))
    data = survjax.SurvivalData(
        jnp.arange(1, 258, dtype=jnp.float32), jnp.arange(257) < 100
    )
    validation = survjax.SurvivalData(jnp.array([999, 999], dtype=jnp.float32), [0, 1])
    return images, data, images[:2], validation


def test_validation_includes_tail_with_observation_weights():
    """03: 129 censored exponential rows give (128*1+501)/129, not batch mean."""
    cnn = model.CNN(np.log(2.0), key=jax.random.key(0))
    cnn = eqx.tree_at(lambda m: m.dense2.weight, cnn, jnp.zeros_like(cnn.dense2.weight))
    images, _, _, _ = cohorts()
    images = images[:129]
    times = jnp.concatenate(
        (jnp.full(128, 2, dtype=jnp.float32), jnp.array([1002], dtype=jnp.float32))
    )
    data = survjax.SurvivalData(times, jnp.zeros(129, dtype=bool))
    actual = model._validation_nll(
        (cnn, jnp.zeros((), dtype=jnp.float32)), images, data
    )
    np.testing.assert_allclose(
        actual, 629 / 129, atol=tol.ATOL[False], rtol=tol.RTOL[False]
    )


def test_ten_epoch_schedule_shuffling_history_and_earliest_best(monkeypatch):
    """Maintainer initialization: fitted intercept/key, drop one row, 10 epochs, validation tie keeps epoch2."""
    images, data, validation_images, validation = cohorts()
    original_cnn = model.CNN
    captured, rows = {}, []

    def initialize(log_scale, *, key):
        captured["log_scale"], captured["key"] = float(log_scale), key
        return original_cnn(log_scale, key=key)

    def step(params, state, pixels, observations):
        if not rows:
            assert (
                params[1].shape == ()
                and params[1] == jnp.float32(0.2)
                and params[1].dtype == jnp.float32
            )
            np.testing.assert_array_equal(
                params[0].dense2.bias, jnp.array([1.5], dtype=jnp.float32)
            )
        np.testing.assert_array_equal(pixels[:, 0, 0, 0] * 256 + 1, observations.lower)
        rows.append(np.asarray(observations.lower))
        return (params[0], params[1] + jnp.float32(0.01)), state, jnp.float32(len(rows))

    values = [3.0, 2.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    evaluations = []

    def evaluate(params, pixels, observations):
        assert len(observations) == 2 and jnp.all(observations.lower == 999)
        np.testing.assert_array_equal(pixels, validation_images)
        evaluations.append(params)
        return values[len(evaluations) - 1]

    monkeypatch.setattr(model, "CNN", initialize)
    monkeypatch.setattr(model, "_train_step", step)
    monkeypatch.setattr(model, "_validation_nll", evaluate)
    result = model.train(
        images,
        data,
        validation_images,
        validation,
        initial_intercept=np.array([0.2, 1.5], dtype=np.float64),
        config="full",
    )
    uc = jax.random.fold_in(jax.random.key(20261025), 3)
    # Supplied fitted log scale is distinct from either cohort's observed mean.
    assert captured["log_scale"] == 1.5
    np.testing.assert_array_equal(
        jax.random.key_data(captured["key"]),
        jax.random.key_data(jax.random.fold_in(uc, 2)),
    )
    assert len(rows) == 20
    for epoch in range(10):
        actual = np.concatenate(rows[2 * epoch : 2 * epoch + 2])
        key = jax.random.fold_in(jax.random.fold_in(uc, 3), epoch)
        expected = np.asarray(jax.random.permutation(key, 257))[:256] + 1
        np.testing.assert_array_equal(actual, expected)
    # Each epoch has two equal minibatches: mean(2e+1,2e+2)=2e+1.5.
    assert result.training_nll == tuple(2 * epoch + 1.5 for epoch in range(10))
    assert result.validation_nll == tuple(values) and result.selected_epoch == 2
    assert result.params is evaluations[1]  # Immutable earlier checkpoint, not latest.


@pytest.mark.parametrize("stage", ["at initialization", "after step 1"])
def test_nonfinite_gradients_fail_at_explicit_checkpoints(monkeypatch, stage):
    """03: failure is loud at initialization and immediately after update1."""
    images, data, validation_images, validation = cohorts()
    calls = []

    checks = []

    def gradient(*args):
        checks.append(1)
        bad = stage == "at initialization" or len(checks) == 2
        return jnp.float32(1), jnp.float32(jnp.nan if bad else 0)

    def step(params, state, pixels, observations):
        calls.append(1)
        return params, state, jnp.float32(1)

    monkeypatch.setattr(model, "_value_grad", gradient)
    monkeypatch.setattr(model, "_train_step", step)
    with pytest.raises(FloatingPointError, match=stage):
        model.train(
            images,
            data,
            validation_images,
            validation,
            initial_intercept=np.array([0.2, 1.5], dtype=np.float64),
        )
    assert len(calls) == (0 if stage == "at initialization" else 1)


def test_compiled_adam_first_step_matches_exponential_head_algebra():
    """03: first Adam bias/log-k increments are +.001 for negative gradients."""
    cnn = model.CNN(np.log(2.0), key=jax.random.key(0))
    cnn = eqx.tree_at(lambda m: m.dense2.weight, cnn, jnp.zeros_like(cnn.dense2.weight))
    params = cnn, jnp.zeros((), dtype=jnp.float32)
    images = jnp.zeros((3, 1, 28, 28), dtype=jnp.float32)
    data = survjax.SurvivalData(jnp.array([1, 2, 3], dtype=jnp.float32), [1, 0, 1])
    state = optax.adam(1e-3).init(eqx.filter(params, eqx.is_inexact_array))
    updated, _, value = model._train_step(params, state, images, data)
    # NLL=1+2log(2)/3; d/dbias=-1/3, d/dlog_k=(log(3)/2-2)/3 < 0.
    # First bias-corrected Adam update is -.001*g/(abs(g)+1e-8).
    expected = [
        np.log(2) + 0.001 / (1 + 3e-8),
        0.001 / (1 + 1e-8 / abs((np.log(3) / 2 - 2) / 3)),
    ]
    actual = jnp.array([updated[0].dense2.bias[0], updated[1]])
    np.testing.assert_allclose(
        actual, expected, atol=tol.ATOL[False], rtol=tol.RTOL[False]
    )
    np.testing.assert_allclose(
        value, 1 + 2 * np.log(2) / 3, atol=tol.ATOL[False], rtol=tol.RTOL[False]
    )
    np.testing.assert_array_equal(updated[0].conv1.weight, cnn.conv1.weight)


def test_training_rejects_wrong_precision_shapes_and_config():
    """03 Precision: no integer pixels, missing full minibatch or bad config."""
    images, data, validation_images, validation = cohorts()
    with pytest.raises(ValueError, match="float32"):
        model.train(
            images.astype(jnp.int32),
            data,
            validation_images,
            validation,
            initial_intercept=np.array([0.2, 1.5], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="aligned"):
        model.train(
            images[:127],
            data[:127],
            validation_images,
            validation,
            initial_intercept=np.array([0.2, 1.5], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="config"):
        model.train(
            images,
            data,
            validation_images,
            validation,
            initial_intercept=np.array([0.2, 1.5], dtype=np.float64),
            config="unknown",
        )


def test_float64_training_or_validation_data_are_rejected(x64_enabled):
    """03 Precision/MNIST.3 N-3: float64 observations must not promote the loss."""
    if not x64_enabled:
        pytest.skip("float64 observation storage requires x64 mode")
    images, data, validation_images, validation = cohorts()
    for training_dtype, validation_dtype in (
        (jnp.float64, jnp.float32),
        (jnp.float32, jnp.float64),
    ):
        training = survjax.SurvivalData(data.lower.astype(training_dtype), data.exact)
        held_out = survjax.SurvivalData(
            validation.lower.astype(validation_dtype), validation.exact
        )
        with pytest.raises(ValueError, match="float32"):
            model.train(
                images,
                training,
                validation_images,
                held_out,
                initial_intercept=np.array([0.2, 1.5], dtype=np.float64),
            )


@pytest.mark.parametrize("failure", ["training", "validation"])
def test_nonfinite_epoch_nll_never_returns_a_checkpoint(monkeypatch, failure):
    """03: nonfinite histories abort rather than silently leaving epoch0 selected."""
    images, data, validation_images, validation = cohorts()
    training = jnp.float32(jnp.nan if failure == "training" else 1)
    held_out = float("nan") if failure == "validation" else 1.0
    monkeypatch.setattr(model, "_train_step", lambda p, s, x, d: (p, s, training))
    monkeypatch.setattr(model, "_validation_nll", lambda p, x, d: held_out)
    with pytest.raises(FloatingPointError, match="epoch"):
        model.train(
            images,
            data,
            validation_images,
            validation,
            initial_intercept=np.array([0.2, 1.5], dtype=np.float64),
        )


@pytest.mark.parametrize("intercept", [[0], [0, float("nan")], [float("inf"), 1]])
def test_training_rejects_invalid_initial_intercept(intercept):
    """Maintainer initialization: reject malformed/nonfinite fitted log parameters."""
    images, data, validation_images, validation = cohorts()
    with pytest.raises(ValueError, match="initial_intercept"):
        model.train(
            images, data, validation_images, validation, initial_intercept=intercept
        )
