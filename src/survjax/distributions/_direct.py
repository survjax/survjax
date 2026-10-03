"""Pointwise defaults and registration-time policy (spec 4.2, 4.5)."""

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from survjax.distributions.base import SurvivalDistribution


def survival(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return P(T > t), with the broadcast shape of t and batch_shape.

    The default is exp(log_survival(t)). Equal-length vectors pair entries;
    use singleton axes for all pairs. Shared time-first grids are separate.
    Negative times return one; NaN times return NaN.
    """
    raw = getattr(type(self).log_survival, "_survjax_raw", type(self).log_survival)
    return jnp.exp(raw(self, t))


def cumulative_hazard(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return -log P(T > t), broadcasting t against batch_shape.

    Negative times return zero; NaN times return NaN. This is pointwise,
    rather than a shared time-first grid operation.
    """
    raw = getattr(type(self).log_survival, "_survjax_raw", type(self).log_survival)
    return -raw(self, t)


def log_cdf(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return log P(T <= t), broadcasting t against batch_shape.

    Uses log(-expm1(log_survival(t))) without subtracting nearly equal
    probabilities. Precision is retained until the family's log survival
    underflows to zero (roughly CDF 1e-38 in float32 or 1e-308 in float64).
    Override for CDF-natural families or smaller lower tails. Negative times
    and true zero CDF return -inf; NaN times return NaN. Evaluation is pointwise,
    not a shared time-first grid operation.
    """
    # spec 4.5: the outer wrapper already substituted safe times.
    raw = getattr(type(self).log_survival, "_survjax_raw", type(self).log_survival)
    log_s = raw(self, t)
    zero_mass = log_s == 0
    # spec 7.2: substitute before log; an unselected log(0) poisons gradients.
    safe_log_s = jnp.where(
        zero_mass, jax.lax.stop_gradient(-jnp.ones_like(log_s)), log_s
    )
    value = jnp.log(-jnp.expm1(safe_log_s))
    return jnp.where(zero_mass, -jnp.inf, value)


def apply_policy(
    method: Callable[["SurvivalDistribution", ArrayLike], jax.Array],
    negative_value: float,
) -> Callable[["SurvivalDistribution", ArrayLike], jax.Array]:
    """Wrap one direct method, preserving its raw implementation on inheritance."""
    raw = getattr(method, "_survjax_raw", method)

    @wraps(raw)
    def evaluate(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
        time = jnp.asarray(t)
        try:
            output_shape = jnp.broadcast_shapes(time.shape, self.batch_shape)
        except ValueError as exc:
            raise TypeError(
                "time and distribution batch shapes are incompatible"
            ) from exc
        time = jnp.broadcast_to(time, output_shape)
        negative = time < 0
        nan_time = jnp.isnan(time)
        inactive = negative | nan_time
        # spec 4.5/7.2: stop only substituted inputs, never real parameters.
        safe_time = jnp.where(
            inactive, jax.lax.stop_gradient(self.reference_time()), time
        )
        value = raw(self, safe_time)
        policy_value = jnp.where(negative, negative_value, value)
        # D-013: preserve numerical NaNs without inventing parameter diagnostics.
        policy_value = jnp.where(jnp.isnan(value), value, policy_value)
        return jnp.where(nan_time, jnp.nan, policy_value)

    # A decorated subclass reuses the raw override, not its parent's wrapper.
    evaluate._survjax_raw = raw  # type: ignore[attr-defined]
    return evaluate
