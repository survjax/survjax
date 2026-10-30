"""Float32 neural survival model (use-case 03 Network)."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import survjax

# Preserve the preparation names introduced in MNIST.2; data.py needs no Equinox.
from examples.survival_mnist.data import TRUE_LOG_SCALES as TRUE_LOG_SCALES
from examples.survival_mnist.data import TRUE_SHAPE as TRUE_SHAPE
from examples.survival_mnist.data import calibrate_mnist as calibrate_mnist
from examples.survival_mnist.data import generate_outcomes as generate_outcomes


class CNN(eqx.Module):
    """Map one (1,28,28) float32 image to scalar Weibull log scale."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    dense1: eqx.nn.Linear
    dense2: eqx.nn.Linear
    pool: eqx.nn.MaxPool2d

    def __init__(self, mean_training_time: ArrayLike, *, key: jax.Array) -> None:
        """Initialize from a positive finite training mean and explicit key.

        Args:
            mean_training_time: Scalar mean observed time from training rows only.
            key: Initialization key; use fold_in(fold_in(key(20261025),3),2).
        """
        keys = jax.random.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(1, 16, 3, dtype=jnp.float32, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(16, 32, 3, dtype=jnp.float32, key=keys[1])
        self.dense1 = eqx.nn.Linear(800, 64, dtype=jnp.float32, key=keys[2])
        head = eqx.nn.Linear(64, 1, dtype=jnp.float32, key=keys[3])
        bias = jnp.log(jnp.asarray(mean_training_time, dtype=jnp.float32)).reshape(1)
        self.dense2 = eqx.tree_at(lambda layer: layer.bias, head, bias)
        # 03 Network dimensions require stride 2; Equinox defaults to stride 1.
        self.pool = eqx.nn.MaxPool2d(2, stride=2)

    def __call__(self, image: jax.Array) -> jax.Array:
        """Return scalar log scale for one float32 (1,28,28) image."""
        hidden = self.pool(jax.nn.relu(self.conv1(image)))
        hidden = self.pool(jax.nn.relu(self.conv2(hidden)))
        hidden = jax.nn.relu(self.dense1(hidden.reshape(800)))
        return self.dense2(hidden).squeeze(-1)


def loss(
    params: tuple[CNN, jax.Array], images: jax.Array, data: survjax.SurvivalData
) -> jax.Array:
    """Return mean NLL for aligned float32 images (N,1,28,28) and data (N,).

    Args:
        params: CNN and learned scalar log shape (initialize to float32 zero).
        images: Pixels only; labels never enter the network.
        data: Exact/right observations, typically sliced with data[idx].
    Returns:
        Scalar negative mean survjax Weibull log likelihood.
    """
    cnn, log_k = params
    dist = survjax.Weibull.from_log(log_shape=log_k, log_scale=jax.vmap(cnn)(images))
    return -dist.log_likelihood(data).mean()
