"""Survival MNIST outcome preparation (use-case 03 details)."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from examples.weibull_simulation.study import (
    GeneratedData,
    calibrate_censoring,
    generate_weibull,
)

TRUE_SHAPE = np.float64(1.38)
# 03 Precision: compute oracle parameters in host float64, cast for evaluation.
TRUE_LOG_SCALES = np.log(np.float64(10)) - np.float64(0.2) * np.arange(10)


def calibrate_mnist(training_labels: np.ndarray) -> float:
    """Calibrate 50% expected censoring from the 60k training-source digits.

    Args:
        training_labels: Integer digits (60000,), before train/validation split.
            Official test labels and smoke subsets must never be supplied.
    Returns:
        Uniform censoring limit c, computed by the shared host float64 solver.
    Raises:
        ValueError: Labels are not exactly 60k integer digits 0 through 9.
        RuntimeError: Shared calibration does not converge.
    """
    labels = np.asarray(training_labels)
    if (
        labels.shape != (60000,)
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any((labels < 0) | (labels > 9))
    ):
        raise ValueError("calibration requires the 60000 training-source digit labels")
    scales = np.exp(TRUE_LOG_SCALES[labels])
    return calibrate_censoring(TRUE_SHAPE, scales, 0.5)


def generate_outcomes(
    labels: ArrayLike,
    censor_limit: ArrayLike,
    *,
    global_indices: ArrayLike | None = None,
) -> GeneratedData:
    """Generate fixed float32 outcomes, before selecting full/smoke cohorts.

    Args:
        labels: Integer digits (N,), values 0 through 9 as verified by the loader.
        censor_limit: Positive finite scalar c from float64 calibration.
        global_indices: Optional integer (N,) indices in 0..69999. Defaults to
            0..N-1; omit only for rows in official global order. Explicit indices
            preserve outcomes when selecting or reordering rows. Values are
            preconditions, following the shared JIT-compatible generator.
    Returns:
        Shared GeneratedData with (N,) float32 time/event_time/censor_time and
        boolean event. Keys depend on global index, never on split or batch.
    Raises:
        ValueError: Labels/indices are not matching vectors or c is not scalar.
    """
    labels = jnp.asarray(labels)
    indices = (
        jnp.arange(labels.size, dtype=jnp.uint32)
        if global_indices is None
        else jnp.asarray(global_indices, dtype=jnp.uint32)
    )
    c = jnp.asarray(censor_limit, dtype=jnp.float32)
    if labels.ndim != 1 or indices.shape != labels.shape or c.ndim != 0:
        raise ValueError("labels/indices must be matching vectors and c must be scalar")
    uc = jax.random.fold_in(jax.random.key(20261025), 3)
    outcome_key = jax.random.fold_in(uc, 0)
    keys = jax.vmap(lambda index: jax.random.fold_in(outcome_key, index))(indices)
    log_scales = jnp.asarray(TRUE_LOG_SCALES, dtype=jnp.float32)[labels]
    shape = jnp.asarray(TRUE_SHAPE, dtype=jnp.float32)
    return jax.vmap(lambda key, scale: generate_weibull(key, shape, scale, c))(
        keys, jnp.exp(log_scales)
    )
