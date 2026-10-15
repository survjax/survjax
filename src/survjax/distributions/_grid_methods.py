"""Time-first adapters for registered direct methods (spec 5)."""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from survjax.distributions.base import SurvivalDistribution


def _reshape_grid(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    try:
        time = jnp.asarray(t)
    except (TypeError, ValueError) as exc:
        raise TypeError("grid time must contain real numeric values") from exc
    if time.ndim != 1:
        raise TypeError("grid time must be one-dimensional")
    # spec 5: use current batch rank under vmap; component axes are not batches.
    return time.reshape((time.shape[0],) + (1,) * len(self.batch_shape))


def survival_grid(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return survival on a shared time-first grid (spec 5).

    A one-dimensional real t of length K returns shape (K, *batch_shape),
    excluding component axes. Rows retain input order, including repeats.
    Conversion and values follow survival. Nonreal or non-1-D t raises
    TypeError, including during tracing.
    """
    return self.survival(_reshape_grid(self, t))


def log_survival_grid(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return log survival on a shared time-first grid (spec 5).

    A one-dimensional real t of length K returns shape (K, *batch_shape),
    excluding component axes. Rows retain input order, including repeats.
    Conversion and values follow log_survival. Nonreal or non-1-D t raises
    TypeError, including during tracing.
    """
    return self.log_survival(_reshape_grid(self, t))


def log_prob_grid(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return log density on a shared time-first grid (spec 5).

    A one-dimensional real t of length K returns shape (K, *batch_shape),
    excluding component axes. Rows retain input order, including repeats.
    Conversion and values follow log_prob. Nonreal or non-1-D t raises
    TypeError, including during tracing.
    """
    return self.log_prob(_reshape_grid(self, t))


def log_cdf_grid(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return log CDF on a shared time-first grid (spec 5).

    A one-dimensional real t of length K returns shape (K, *batch_shape),
    excluding component axes. Rows retain input order, including repeats.
    Conversion and values follow log_cdf. Nonreal or non-1-D t raises
    TypeError, including during tracing.
    """
    return self.log_cdf(_reshape_grid(self, t))


def cumulative_hazard_grid(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return cumulative hazard on a shared time-first grid (spec 5).

    A one-dimensional real t of length K returns shape (K, *batch_shape),
    excluding component axes. Rows retain input order, including repeats.
    Conversion and values follow cumulative_hazard. Nonreal or non-1-D t
    raises TypeError, including during tracing.
    """
    return self.cumulative_hazard(_reshape_grid(self, t))


def log_hazard_grid(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return log hazard on a shared time-first grid (spec 5).

    A one-dimensional real t of length K returns shape (K, *batch_shape),
    excluding component axes. Rows retain input order, including repeats.
    Conversion and values follow log_hazard. Nonreal or non-1-D t raises
    TypeError, including during tracing.
    """
    return self.log_hazard(_reshape_grid(self, t))


def hazard_grid(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return hazard on a shared time-first grid (spec 5).

    A one-dimensional real t of length K returns shape (K, *batch_shape),
    excluding component axes. Rows retain input order, including repeats.
    Conversion and values follow hazard. Nonreal or non-1-D t raises
    TypeError, including during tracing.
    """
    return self.hazard(_reshape_grid(self, t))
