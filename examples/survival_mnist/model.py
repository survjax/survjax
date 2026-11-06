"""Float32 neural survival model (use-case 03 Network)."""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from jax.typing import ArrayLike

import survjax
from examples.survival_mnist.data import UC_KEY


class CNN(eqx.Module):
    """Map one (1,28,28) float32 image to scalar Weibull log scale."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    dense1: eqx.nn.Linear
    dense2: eqx.nn.Linear
    pool: eqx.nn.MaxPool2d

    def __init__(self, initial_log_scale: ArrayLike, *, key: jax.Array) -> None:
        """Initialize from a fitted intercept log scale and explicit key."""
        keys = jax.random.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(1, 16, 3, dtype=jnp.float32, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(16, 32, 3, dtype=jnp.float32, key=keys[1])
        self.dense1 = eqx.nn.Linear(800, 64, dtype=jnp.float32, key=keys[2])
        head = eqx.nn.Linear(64, 1, dtype=jnp.float32, key=keys[3])
        bias = jnp.asarray(initial_log_scale, dtype=jnp.float32).reshape(1)
        self.dense2 = eqx.tree_at(lambda layer: layer.bias, head, bias)
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
    """Return mean NLL for aligned float32 images (N,1,28,28) and data (N,)."""
    cnn, log_k = params
    dist = survjax.Weibull.from_log(log_shape=log_k, log_scale=jax.vmap(cnn)(images))
    return -dist.log_likelihood(data).mean()


class TrainingResult(NamedTuple):
    """Best parameters and epoch training and validation losses."""

    params: tuple[CNN, jax.Array]
    training_nll: tuple[float, ...]
    validation_nll: tuple[float, ...]
    selected_epoch: int


_optimizer = optax.adam(1e-3)


@eqx.filter_jit
def _train_step(params, state, images, data):
    value, grads = eqx.filter_value_and_grad(loss)(params, images, data)
    updates, state = _optimizer.update(grads, state, params)
    return eqx.apply_updates(params, updates), state, value


def predict(cnn: CNN, images: jax.Array) -> jax.Array:
    """Predict log scales in chunks of 1000 images."""
    batch = eqx.filter_jit(eqx.filter_vmap(cnn))
    return jnp.concatenate(
        [batch(images[i : i + 1000]) for i in range(0, len(images), 1000)]
    )


def train(
    images,
    data,
    validation_images,
    validation_data,
    *,
    initial_intercept,
    epochs: int = 1,
) -> TrainingResult:
    """Train the CNN and keep the epoch with the lowest validation NLL."""
    images = jnp.asarray(images)
    params = (
        CNN(initial_intercept[1], key=jax.random.fold_in(UC_KEY, 2)),
        initial_intercept[0],
    )
    state = _optimizer.init(eqx.filter(params, eqx.is_inexact_array))
    best_params, best_nll, selected_epoch = params, float("inf"), 0
    training_history, validation_history = [], []
    for epoch in range(epochs):
        key = jax.random.fold_in(jax.random.fold_in(UC_KEY, 3), epoch)
        permutation = jax.random.permutation(key, len(data))
        losses = []
        for start in range(0, len(data) - 127, 128):
            idx = permutation[start : start + 128]
            params, state, value = _train_step(params, state, images[idx], data[idx])
            losses.append(value)
        cnn, log_k = params
        validation_nll = float(
            -survjax.Weibull.from_log(
                log_shape=log_k, log_scale=predict(cnn, validation_images)
            )
            .log_likelihood(validation_data)
            .mean()
        )
        training_history.append(float(jnp.mean(jnp.stack(losses))))
        validation_history.append(validation_nll)
        if validation_nll < best_nll:
            best_params, best_nll, selected_epoch = params, validation_nll, epoch + 1
    return TrainingResult(
        best_params, tuple(training_history), tuple(validation_history), selected_epoch
    )


def _intercept_nll(theta, data):
    dist = survjax.Weibull.from_log(log_shape=theta[0], log_scale=theta[1])
    return -dist.log_likelihood(data).mean()


def fit_intercept(data: survjax.SurvivalData) -> tuple[jax.Array, jax.Array]:
    """Fit an intercept-only Weibull to float32 training data; return (log k, log b)."""
    # Exponential MLE of the scale (total time / events) as the BFGS start.
    initial_scale = data.lower.sum() / data.exact.sum(dtype=data.lower.dtype)
    initial = jnp.stack((jnp.zeros_like(initial_scale), jnp.log(initial_scale)))
    fitted = optx.minimise(
        _intercept_nll,
        optx.BFGS(rtol=3e-4, atol=3e-4),
        initial,
        args=data,
        max_steps=256,
    ).value
    return fitted[0], fitted[1]


# Compiled once; evaluate() and the runner's timing call the same function.
concordance_index = jax.jit(survjax.concordance_index)


def evaluate(
    log_shape: ArrayLike, log_scale: ArrayLike, data: survjax.SurvivalData
) -> dict[str, float]:
    """Return held-out mean log likelihood, shape k and Harrell's C.

    log_scale is a scalar or one value per row of data; with a shared shape,
    -log_scale is a valid risk score for Harrell's C (use-case 03).
    """
    log_k = jnp.asarray(log_shape, dtype=jnp.float32)
    log_b = jnp.broadcast_to(jnp.asarray(log_scale, dtype=jnp.float32), data.shape)
    dist = survjax.Weibull.from_log(log_shape=log_k, log_scale=log_b)
    return {
        "mean_log_likelihood": float(dist.log_likelihood(data).mean()),
        "shape": float(jnp.exp(log_k)),
        "concordance": float(concordance_index(-log_b, data)),
    }
