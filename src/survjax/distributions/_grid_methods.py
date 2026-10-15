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
    """Return the survival probability for every time in t and every batch entry.

    The grid twin of ``survival``. Where ``survival(t)`` pairs times with batch
    entries element by element, this evaluates one shared vector of K times
    against the whole batch, time axis first: for batch shape (D, N) it
    equals ``survival(t[:, None, None])``, with shape (K, D, N). Values, including
    at t <= 0, t = +inf and NaN, are exactly those of ``survival``.

    Args:
        t: A one-dimensional array of K times. They need not be sorted or
            distinct; output row i belongs to ``t[i]``.

    Returns:
        An array of shape (K, *batch_shape), without component axes.

    Raises:
        TypeError: If ``t`` is not one-dimensional or not real, also inside
            ``jax.jit``.

    >>> import jax.numpy as jnp
    >>> import survjax
    >>> arms = survjax.Weibull(shape=1.5, scale=jnp.array([12.0, 18.0]))
    >>> arms.survival_grid(jnp.array([0.0, 6.0, 12.0, 24.0])).shape
    (4, 2)
    """
    return self.survival(_reshape_grid(self, t))


def log_survival_grid(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return the log survival probability for every time in t and every batch entry.

    The grid twin of ``log_survival``. Where ``log_survival(t)`` pairs times with batch
    entries element by element, this evaluates one shared vector of K times
    against the whole batch, time axis first: for batch shape (D, N) it
    equals ``log_survival(t[:, None, None])``, with shape (K, D, N). Values, including
    at t <= 0, t = +inf and NaN, are exactly those of ``log_survival``.

    Args:
        t: A one-dimensional array of K times. They need not be sorted or
            distinct; output row i belongs to ``t[i]``.

    Returns:
        An array of shape (K, *batch_shape), without component axes.

    Raises:
        TypeError: If ``t`` is not one-dimensional or not real, also inside
            ``jax.jit``.
    """
    return self.log_survival(_reshape_grid(self, t))


def log_prob_grid(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return the log density for every time in t and every batch entry.

    The grid twin of ``log_prob``. Where ``log_prob(t)`` pairs times with batch
    entries element by element, this evaluates one shared vector of K times
    against the whole batch, time axis first: for batch shape (D, N) it
    equals ``log_prob(t[:, None, None])``, with shape (K, D, N). Values, including
    at t <= 0, t = +inf and NaN, are exactly those of ``log_prob``.

    Args:
        t: A one-dimensional array of K times. They need not be sorted or
            distinct; output row i belongs to ``t[i]``.

    Returns:
        An array of shape (K, *batch_shape), without component axes.

    Raises:
        TypeError: If ``t`` is not one-dimensional or not real, also inside
            ``jax.jit``.
    """
    return self.log_prob(_reshape_grid(self, t))


def log_cdf_grid(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return the log CDF for every time in t and every batch entry.

    The grid twin of ``log_cdf``. Where ``log_cdf(t)`` pairs times with batch
    entries element by element, this evaluates one shared vector of K times
    against the whole batch, time axis first: for batch shape (D, N) it
    equals ``log_cdf(t[:, None, None])``, with shape (K, D, N). Values, including
    at t <= 0, t = +inf and NaN, are exactly those of ``log_cdf``.

    Args:
        t: A one-dimensional array of K times. They need not be sorted or
            distinct; output row i belongs to ``t[i]``.

    Returns:
        An array of shape (K, *batch_shape), without component axes.

    Raises:
        TypeError: If ``t`` is not one-dimensional or not real, also inside
            ``jax.jit``.
    """
    return self.log_cdf(_reshape_grid(self, t))


def cumulative_hazard_grid(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return the cumulative hazard for every time in t and every batch entry.

    The grid twin of ``cumulative_hazard``. Where ``cumulative_hazard(t)`` pairs times with batch
    entries element by element, this evaluates one shared vector of K times
    against the whole batch, time axis first: for batch shape (D, N) it
    equals ``cumulative_hazard(t[:, None, None])``, with shape (K, D, N). Values, including
    at t <= 0, t = +inf and NaN, are exactly those of ``cumulative_hazard``.

    Args:
        t: A one-dimensional array of K times. They need not be sorted or
            distinct; output row i belongs to ``t[i]``.

    Returns:
        An array of shape (K, *batch_shape), without component axes.

    Raises:
        TypeError: If ``t`` is not one-dimensional or not real, also inside
            ``jax.jit``.
    """
    return self.cumulative_hazard(_reshape_grid(self, t))


def log_hazard_grid(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return the log hazard for every time in t and every batch entry.

    The grid twin of ``log_hazard``. Where ``log_hazard(t)`` pairs times with batch
    entries element by element, this evaluates one shared vector of K times
    against the whole batch, time axis first: for batch shape (D, N) it
    equals ``log_hazard(t[:, None, None])``, with shape (K, D, N). Values, including
    at t <= 0, t = +inf and NaN, are exactly those of ``log_hazard``.

    Args:
        t: A one-dimensional array of K times. They need not be sorted or
            distinct; output row i belongs to ``t[i]``.

    Returns:
        An array of shape (K, *batch_shape), without component axes.

    Raises:
        TypeError: If ``t`` is not one-dimensional or not real, also inside
            ``jax.jit``.
    """
    return self.log_hazard(_reshape_grid(self, t))


def hazard_grid(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return the hazard for every time in t and every batch entry.

    The grid twin of ``hazard``. Where ``hazard(t)`` pairs times with batch
    entries element by element, this evaluates one shared vector of K times
    against the whole batch, time axis first: for batch shape (D, N) it
    equals ``hazard(t[:, None, None])``, with shape (K, D, N). Values, including
    at t <= 0, t = +inf and NaN, are exactly those of ``hazard``.

    Args:
        t: A one-dimensional array of K times. They need not be sorted or
            distinct; output row i belongs to ``t[i]``.

    Returns:
        An array of shape (K, *batch_shape), without component axes.

    Raises:
        TypeError: If ``t`` is not one-dimensional or not real, also inside
            ``jax.jit``.
    """
    return self.hazard(_reshape_grid(self, t))
