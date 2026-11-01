"""Acceptance checks on the real smoke training batch."""

import sys
from pathlib import Path

import equinox as eqx
import impl_tolerances as tol
import jax
import jax.numpy as jnp
import numpy as np

import survjax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.survival_mnist import data, model  # noqa: E402


def test_gradients_are_finite_at_initialization_and_after_step_one():
    """03 acceptance: differentiate the actual first smoke minibatch twice."""
    pixels, labels = data.load_mnist()
    outcomes = data.generate_outcomes(labels)
    train = data.split_indices("smoke")["train"]
    observations = survjax.SurvivalData(outcomes.time, outcomes.event)[train]
    intercept = model.fit_intercept(outcomes.time[train], outcomes.event[train])
    uc = jax.random.fold_in(jax.random.key(20261025), 3)
    params = model.CNN(intercept[1], key=jax.random.fold_in(uc, 2)), intercept[0]
    key = jax.random.fold_in(jax.random.fold_in(uc, 3), 0)
    idx = jax.random.permutation(key, len(train))[:128]
    images, batch = jnp.asarray(pixels[train])[idx], observations[idx]
    value_grad = eqx.filter_jit(eqx.filter_value_and_grad(model.loss))
    state = model._optimizer.init(eqx.filter(params, eqx.is_inexact_array))
    for step in range(2):
        value, grads = value_grad(params, images, batch)
        assert jnp.isfinite(value)
        assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads))
        if step == 0:
            params, state, _ = model._train_step(params, state, images, batch)


def test_training_keeps_earliest_best_and_drops_partial_batches(monkeypatch):
    """03 Training: fixed shuffles, epoch means, and the earlier tied checkpoint."""
    markers = jnp.arange(257, dtype=jnp.float32)
    images = jnp.broadcast_to(markers[:, None, None, None], (257, 1, 28, 28))
    observations = survjax.SurvivalData(markers + 1, markers < 100)
    validation_images = images[:3]
    validation = survjax.SurvivalData(jnp.ones(3), jnp.zeros(3, dtype=bool))
    original_cnn = model.CNN
    initial_keys, batches, checkpoints, evaluated = [], [], [], []
    validation_values = [3.0, 2.0, 2.0]

    def initialize(log_scale, *, key):
        assert log_scale == jnp.float32(1.5)
        initial_keys.append(key)
        return original_cnn(log_scale, key=key)

    def step(params, state, pixels, batch):
        assert params[1] == 0
        np.testing.assert_array_equal(pixels[:, 0, 0, 0] + 1, batch.lower)
        batches.append(np.asarray(batch.lower))
        cnn = eqx.tree_at(
            lambda m: m.dense2.bias,
            params[0],
            jnp.array([len(batches)], dtype=jnp.float32),
        )
        params = cnn, params[1]
        checkpoints.append(params)
        return params, state, jnp.float32(len(batches))

    def predict(cnn, pixels):
        np.testing.assert_array_equal(pixels, validation_images)
        evaluated.append(cnn)
        # Censored exponential at t=1 has NLL=1/b; choose b=1/target_NLL.
        return jnp.full(3, -jnp.log(jnp.float32(validation_values[len(evaluated) - 1])))

    monkeypatch.setattr(model, "CNN", initialize)
    monkeypatch.setattr(model, "_train_step", step)
    monkeypatch.setattr(model, "predict", predict)
    result = model.train(
        images,
        observations,
        validation_images,
        validation,
        initial_intercept=jnp.array([0, 1.5], dtype=jnp.float32),
        epochs=3,
    )
    uc = jax.random.fold_in(jax.random.key(20261025), 3)
    np.testing.assert_array_equal(
        jax.random.key_data(initial_keys[0]),
        jax.random.key_data(jax.random.fold_in(uc, 2)),
    )
    assert len(batches) == 6
    for epoch in range(3):
        key = jax.random.fold_in(jax.random.fold_in(uc, 3), epoch)
        expected = np.asarray(jax.random.permutation(key, 257))[:256] + 1
        np.testing.assert_array_equal(
            np.concatenate(batches[2 * epoch : 2 * epoch + 2]), expected
        )
        assert evaluated[epoch] is checkpoints[2 * epoch + 1][0]
    # Two counter losses per epoch: mean(2e+1, 2e+2) = 2e+1.5.
    assert result.training_nll == (1.5, 3.5, 5.5)
    np.testing.assert_allclose(
        result.validation_nll,
        validation_values,
        atol=tol.ATOL[False],
        rtol=tol.RTOL[False],
    )
    assert result.selected_epoch == 2
    assert result.params is checkpoints[3]
