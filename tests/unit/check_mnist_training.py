"""Acceptance checks on the real smoke training batch."""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

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
