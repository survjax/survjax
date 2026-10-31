"""Explicit optional-group checks: use-case 03 Network, Precision, Training."""

import sys
from pathlib import Path

import equinox as eqx
import impl_tolerances as tol
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.test_util import check_grads

import survjax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.survival_mnist.model import CNN, loss  # noqa: E402


def setup():
    key = jax.random.fold_in(jax.random.fold_in(jax.random.key(20261025), 3), 2)
    cnn = CNN(np.log(np.float64(2)), key=key)
    images = jax.random.uniform(jax.random.key(19), (3, 1, 28, 28), dtype=jnp.float32)
    data = survjax.SurvivalData(jnp.array([1, 2, 3], dtype=jnp.float32), [1, 0, 1])
    return cnn, images, data


def test_cnn_geometry_bias_precision_and_batching():
    """03: 26→13→11→5, 800→64→1, float32 even when x64 is enabled."""
    cnn, images, _ = setup()
    assert cnn.conv1.weight.shape == (16, 1, 3, 3)
    assert cnn.conv2.weight.shape == (32, 16, 3, 3)
    assert cnn.dense1.weight.shape == (64, 800) and cnn.dense2.weight.shape == (1, 64)
    assert cnn.pool(cnn.conv1(images[0])).shape == (16, 13, 13)
    arrays = jax.tree.leaves(eqx.filter(cnn, eqx.is_inexact_array))
    assert all(leaf.dtype == jnp.float32 for leaf in arrays)
    np.testing.assert_allclose(
        cnn.dense2.bias, np.log(2), atol=tol.ATOL[False], rtol=tol.RTOL[False]
    )
    batched = jax.vmap(cnn)(images)
    assert batched.shape == (3,)
    np.testing.assert_allclose(
        batched,
        jnp.stack([cnn(x) for x in images]),
        atol=tol.ATOL[False],
        rtol=tol.RTOL[False],
    )


def test_forward_matches_numpy_cross_correlation_relu_and_max_pool():
    """03 Network: independent NumPy operations pin every activation and pool."""
    cnn, images, _ = setup()
    hidden = np.asarray(images[0])
    for layer in (cnn.conv1, cnn.conv2):
        windows = np.lib.stride_tricks.sliding_window_view(hidden, (3, 3), axis=(1, 2))
        hidden = np.einsum("chwkl,ockl->ohw", windows, np.asarray(layer.weight))
        hidden = np.maximum(hidden + np.asarray(layer.bias), 0)
        channels, height, width = hidden.shape
        height, width = height // 2 * 2, width // 2 * 2
        hidden = (
            hidden[:, :height, :width]
            .reshape(channels, height // 2, 2, width // 2, 2)
            .max(axis=(2, 4))
        )
    features = np.asarray(cnn.dense1.weight) @ hidden.ravel() + np.asarray(
        cnn.dense1.bias
    )
    expected = (
        np.asarray(cnn.dense2.weight) @ np.maximum(features, 0)
        + np.asarray(cnn.dense2.bias)
    )[0]
    np.testing.assert_allclose(
        cnn(images[0]), expected, atol=tol.ATOL[False], rtol=tol.RTOL[False]
    )


def test_loss_matches_exponential_algebra_and_second_derivatives():
    """03 loss/spec 7.3: constant b=2,k=1 yields NLL=1+2*log(2)/3."""
    cnn, images, data = setup()
    cnn = eqx.tree_at(lambda m: m.dense2.weight, cnn, jnp.zeros_like(cnn.dense2.weight))

    def objective(theta):
        head = eqx.tree_at(lambda m: m.dense2.bias, cnn, theta[:1])
        return loss((head, theta[1]), images, data)

    theta = jnp.array([jnp.log(jnp.float32(2)), 0], dtype=jnp.float32)
    np.testing.assert_allclose(
        objective(theta),
        1 + 2 * np.log(2) / 3,
        atol=tol.ATOL[False],
        rtol=tol.RTOL[False],
    )
    np.testing.assert_allclose(
        jax.grad(objective)(theta)[0],
        -1 / 3,
        atol=tol.ATOL[False],
        rtol=tol.RTOL[False],
    )
    np.testing.assert_allclose(
        jax.hessian(objective)(theta)[0, 0],
        1,
        atol=tol.ATOL[False],
        rtol=tol.RTOL[False],
    )
    check_grads(
        objective,
        (theta,),
        order=2,
        modes=("fwd", "rev"),
        eps=tol.LIKELIHOOD_GRAD_EPS[False],
        atol=tol.DERIVATIVE_ATOL[False],
        rtol=tol.DERIVATIVE_RTOL[False],
    )


def test_filtered_gradients_are_finite_before_and_after_adam_step():
    """03 Training: real CNN leaves and scalar log k participate in one update."""
    cnn, images, data = setup()
    params = cnn, jnp.zeros((), dtype=jnp.float32)
    indices = jnp.array([2, 0])
    value_grad = eqx.filter_value_and_grad(loss)
    compiled = eqx.filter_jit(value_grad)
    value, grads = compiled(params, images[indices], data[indices])
    eager, eager_grads = value_grad(params, images[indices], data[indices])
    np.testing.assert_allclose(value, eager, atol=tol.ATOL[False], rtol=tol.RTOL[False])
    for a, b in zip(jax.tree.leaves(grads), jax.tree.leaves(eager_grads), strict=True):
        np.testing.assert_allclose(a, b, atol=tol.ATOL[False], rtol=tol.RTOL[False])
    adam = optax.adam(1e-3)
    updates, _ = adam.update(
        grads, adam.init(eqx.filter(params, eqx.is_inexact_array)), params
    )
    updated = eqx.apply_updates(params, updates)
    next_value, next_grads = compiled(updated, images[indices], data[indices])
    assert value.dtype == next_value.dtype == jnp.float32
    assert updated[1] != params[1]
    assert any(
        jnp.any(a != b)
        for a, b in zip(jax.tree.leaves(cnn), jax.tree.leaves(updated[0]), strict=True)
    )
    assert all(
        jnp.all(jnp.isfinite(x))
        for tree in (value, grads, updated, next_value, next_grads)
        for x in jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))
    )
