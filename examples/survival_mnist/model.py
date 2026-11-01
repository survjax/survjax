"""Float32 neural survival model (use-case 03 Network)."""

from time import perf_counter
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import optimistix as optx
from jax.typing import ArrayLike

import survjax


class CNN(eqx.Module):
    """Map one (1,28,28) float32 image to scalar Weibull log scale."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    dense1: eqx.nn.Linear
    dense2: eqx.nn.Linear
    pool: eqx.nn.MaxPool2d

    def __init__(self, initial_log_scale: ArrayLike, *, key: jax.Array) -> None:
        """Initialize from a fitted intercept log scale and explicit key.

        Args:
            initial_log_scale: Scalar fitted log scale from training rows only.
            key: Initialization key; use fold_in(fold_in(key(20261025),3),2).
        """
        keys = jax.random.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(1, 16, 3, dtype=jnp.float32, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(16, 32, 3, dtype=jnp.float32, key=keys[1])
        self.dense1 = eqx.nn.Linear(800, 64, dtype=jnp.float32, key=keys[2])
        head = eqx.nn.Linear(64, 1, dtype=jnp.float32, key=keys[3])
        bias = jnp.asarray(initial_log_scale, dtype=jnp.float32).reshape(1)
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
        params: CNN and learned scalar log shape, initialized from the intercept fit.
        images: Pixels only; labels never enter the network.
        data: Exact/right observations, typically sliced with data[idx].
    Returns:
        Scalar negative mean survjax Weibull log likelihood.
    """
    cnn, log_k = params
    dist = survjax.Weibull.from_log(log_shape=log_k, log_scale=jax.vmap(cnn)(images))
    return -dist.log_likelihood(data).mean()


class TrainingResult(NamedTuple):
    """Selected parameters, epoch summaries and within-epoch convergence trace."""

    params: tuple[CNN, jax.Array]
    training_nll: tuple[float, ...]
    validation_nll: tuple[float, ...]
    selected_epoch: int
    batch_nll: tuple[float, ...] = ()
    validation_progress: tuple[tuple[int, float], ...] = ()


_optimizer = optax.adam(1e-3)
_value_grad = eqx.filter_jit(eqx.filter_value_and_grad(loss))
_batch_loss = eqx.filter_jit(loss)


@eqx.filter_jit
def _train_step(params, state, images, data):
    value, grads = eqx.filter_value_and_grad(loss)(params, images, data)
    updates, state = _optimizer.update(grads, state, params)
    return eqx.apply_updates(params, updates), state, value


def _require_finite(value_and_grads, stage: str) -> None:
    arrays = jax.tree.leaves(eqx.filter(value_and_grads, eqx.is_inexact_array))
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in arrays]))
    if not bool(finite):
        raise FloatingPointError(f"nonfinite loss or gradients {stage}")


def _validation_nll(params, images, data) -> float:
    # 03 Training: cover the full validation cohort without a giant CNN batch.
    total = jnp.zeros((), dtype=jnp.float32)
    for start in range(0, len(data), 128):
        stop = min(start + 128, len(data))
        mean = _batch_loss(params, images[start:stop], data[start:stop])
        total = total + mean * jnp.float32(stop - start)
    return float(total / jnp.float32(len(data)))


def train(
    images: jax.Array,
    data: survjax.SurvivalData,
    validation_images: jax.Array,
    validation_data: survjax.SurvivalData,
    *,
    initial_intercept: ArrayLike,
    config: str = "smoke",
    log_every_batches: int | None = None,
) -> TrainingResult:
    """Train on float32 cohorts and select by validation NLL only.

    Args:
        images: Training pixels (N,1,28,28), N >= 128; no digit labels.
        data: Aligned training observations (N,), from float32 outcomes.
        validation_images: Validation pixels (M,1,28,28), M > 0.
        validation_data: Aligned float32 validation observations (M,).
        initial_intercept: Float64 training-only BFGS estimate (log shape,
            log scale), loaded from the separate fitting process. Cast to float32
            here to initialize shared log shape and the CNN's final bias.
        config: 'smoke' for one epoch, 'full' for ten. Caller supplies the
            prescribed 5k/1k or 50k/10k cohorts; test rows are never accepted
            separately or consulted. Batch size is always 128.
        log_every_batches: Positive validation logging interval, or None for
            epoch-only validation. With an interval, include initialization and
            every epoch end. Intermediate values never select a checkpoint.
    Returns:
        Best parameters, epoch-mean training/full-validation NLL histories,
        and 1-based selected epoch; ties retain the earlier checkpoint.
        batch_nll records each pre-update minibatch; validation_progress holds
        (completed_updates, full_validation_nll) diagnostic samples.
    Raises:
        ValueError: Config, logging interval, cohort shapes, precision or observations are invalid.
        FloatingPointError: Initial/post-step1 loss/gradients or epoch NLL fail.
    """
    if config not in ("smoke", "full"):
        raise ValueError("config must be 'smoke' or 'full'")
    if log_every_batches is not None and (
        type(log_every_batches) is not int or log_every_batches < 1
    ):
        raise ValueError("log_every_batches must be a positive integer or None")
    images, validation_images = jnp.asarray(images), jnp.asarray(validation_images)
    for pixels, observations, minimum in (
        (images, data, 128),
        (validation_images, validation_data, 1),
    ):
        if (
            pixels.ndim != 4
            or pixels.shape[1:] != (1, 28, 28)
            or observations.shape != (len(pixels),)
            or len(pixels) < minimum
        ):
            raise ValueError(
                "cohorts must be aligned NCHW images and observation vectors"
            )
        if pixels.dtype != jnp.float32 or observations.lower.dtype != jnp.float32:
            raise ValueError("training and validation cohorts must be float32")
        observations.issues.raise_if_invalid()
    intercept = jnp.asarray(initial_intercept, dtype=jnp.float32)
    if intercept.shape != (2,) or not bool(jnp.all(jnp.isfinite(intercept))):
        raise ValueError(
            "initial_intercept must contain finite log shape and log scale"
        )
    uc = jax.random.fold_in(jax.random.key(20261025), 3)
    params = (
        CNN(intercept[1], key=jax.random.fold_in(uc, 2)),
        intercept[0],
    )
    state = _optimizer.init(eqx.filter(params, eqx.is_inexact_array))
    best_params, best_nll, selected_epoch = params, float("inf"), 0
    training_history, validation_history = [], []
    batch_history, validation_progress = [], []
    batches_per_epoch = len(data) // 128
    if log_every_batches is not None:
        initial_nll = _validation_nll(params, validation_images, validation_data)
        if not np.isfinite(initial_nll):
            raise FloatingPointError("nonfinite initial validation NLL")
        validation_progress.append((0, initial_nll))
    first_step = True
    # 03 Splits and keys: shuffle epochs are zero-based; selection is one-based.
    for epoch in range(1 if config == "smoke" else 10):
        key = jax.random.fold_in(jax.random.fold_in(uc, 3), epoch)
        permutation = jax.random.permutation(key, len(data))
        losses = []
        for start in range(0, len(data) - 127, 128):
            indices = permutation[start : start + 128]
            batch_images, batch_data = images[indices], data[indices]
            if first_step:
                _require_finite(
                    _value_grad(params, batch_images, batch_data), "at initialization"
                )
            params, state, value = _train_step(params, state, batch_images, batch_data)
            if first_step:
                _require_finite(
                    _value_grad(params, batch_images, batch_data), "after step 1"
                )
                first_step = False
            losses.append(value)
            batch_history.append(float(value))
            batch = start // 128 + 1
            if log_every_batches is not None and (
                batch % log_every_batches == 0 or batch == batches_per_epoch
            ):
                diagnostic = _validation_nll(params, validation_images, validation_data)
                if not np.isfinite(diagnostic):
                    raise FloatingPointError("nonfinite progress validation NLL")
                validation_progress.append(
                    (epoch * batches_per_epoch + batch, diagnostic)
                )
        training_nll = float(jnp.mean(jnp.stack(losses)))
        validation_nll = (
            validation_progress[-1][1]
            if log_every_batches is not None
            else _validation_nll(params, validation_images, validation_data)
        )
        if not np.isfinite(training_nll) or not np.isfinite(validation_nll):
            raise FloatingPointError("nonfinite epoch training or validation NLL")
        training_history.append(training_nll)
        validation_history.append(validation_nll)
        if validation_nll < best_nll:
            best_params, best_nll, selected_epoch = params, validation_nll, epoch + 1
    return TrainingResult(
        best_params,
        tuple(training_history),
        tuple(validation_history),
        selected_epoch,
        tuple(batch_history),
        tuple(validation_progress),
    )


def _intercept_nll(theta: jax.Array, data: survjax.SurvivalData) -> jax.Array:
    dist = survjax.Weibull.from_log(log_shape=theta[0], log_scale=theta[1])
    return -dist.log_likelihood(data).mean()


def fit_intercept(time: ArrayLike, event: ArrayLike) -> np.ndarray:
    """Fit the intercept-only Weibull on supplied training rows in float64.

    Run in a separate JAX_ENABLE_X64=1 process; save returned log parameters
    and explicitly cast them to float32 in the neural/evaluation process.

    Args:
        time: Nonempty vector of observed training times, exact or right censored.
        event: Matching 0/1 or boolean vector; at least one event is required.
    Returns:
        Host float64 array (2,) containing log shape and log scale. Uses
        Stock Optimistix BFGS with rtol/atol=1e-5,
        at most 256 iterations.
    Raises:
        RuntimeError: x64 is disabled, solver fails or fit results are nonfinite.
        ValueError: Shapes, observation values or event count are invalid.
        TypeError: Observation types are invalid under SurvivalData's policy.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError(
            "intercept fitting requires JAX_ENABLE_X64=1 in a separate process"
        )
    times, events = jnp.asarray(time), jnp.asarray(event)
    if times.ndim != 1 or times.size == 0 or events.shape != times.shape:
        raise ValueError("fit requires nonempty matching time/event vectors")
    data = survjax.SurvivalData(times, events, dtype=jnp.float64)
    data.issues.raise_if_invalid()
    # spec 3.3: count in the calculation dtype before division.
    event_count = data.exact.sum(dtype=jnp.float64)
    if not bool(event_count > 0):
        raise ValueError("intercept fitting requires at least one event")
    initial_scale = data.lower.sum() / event_count
    initial = jnp.stack((jnp.zeros((), dtype=jnp.float64), jnp.log(initial_scale)))
    solution = optx.minimise(
        _intercept_nll,
        optx.BFGS(rtol=1e-5, atol=1e-5),
        initial,
        args=data,
        max_steps=256,
        throw=False,
    )
    if solution.result != optx.RESULTS.successful:
        raise RuntimeError(f"intercept BFGS failed: {solution.result}")
    value, gradient = jax.value_and_grad(_intercept_nll)(solution.value, data)
    if not bool(
        jnp.all(jnp.isfinite(solution.value))
        & jnp.isfinite(value)
        & jnp.all(jnp.isfinite(gradient))
    ):
        raise RuntimeError(
            "intercept BFGS returned nonfinite parameters, loss or gradient"
        )
    return np.asarray(solution.value, dtype=np.float64)


def evaluate(
    log_shape: ArrayLike, log_scale: ArrayLike, data: survjax.SurvivalData
) -> dict[str, float | int]:
    """Report float32 metrics for supplied CNN, intercept or oracle predictions.

    Args:
        log_shape: Scalar shared log shape, explicitly cast to float32.
        log_scale: Scalar intercept or aligned (N,) predictions; cast to float32.
        data: Nonempty float32 observations (N,); all rows are evaluated.
    Returns:
        n, mean_log_likelihood, shape, concordance, concordance_seconds.
        C uses higher risk = -log scale. Wall time includes any first-call
        compilation and waits for completion; subsequent calls may reuse it.
    Raises:
        ValueError: Cohort dtype/shape or parameter shapes are invalid.
        FloatingPointError: Any metric is nonfinite, including undefined C.
    """
    log_k = jnp.asarray(log_shape, dtype=jnp.float32)
    log_b = jnp.asarray(log_scale, dtype=jnp.float32)
    if data.lower.ndim != 1 or len(data) == 0 or data.lower.dtype != jnp.float32:
        raise ValueError("evaluation requires nonempty float32 observations (N,)")
    if log_k.shape != () or log_b.shape not in ((), data.shape):
        raise ValueError(
            "evaluation requires scalar log shape and scalar or (N,) log scale"
        )
    data.issues.raise_if_invalid()
    log_b = jnp.broadcast_to(log_b, data.shape)
    dist = survjax.Weibull.from_log(log_shape=log_k, log_scale=log_b)
    mean_ll = float(dist.log_likelihood(data).mean())
    shape = float(jnp.exp(log_k))
    start = perf_counter()
    concordance = float(survjax.concordance_index(-log_b, data))
    seconds = perf_counter() - start
    if not np.all(np.isfinite([mean_ll, shape, concordance])):
        raise FloatingPointError("evaluation produced nonfinite or undefined metrics")
    return {
        "n": len(data),
        "mean_log_likelihood": mean_ll,
        "shape": shape,
        "concordance": concordance,
        "concordance_seconds": seconds,
    }
