"""Pointwise survival methods: generic defaults (4.2) and value policy (4.5)."""

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from survjax.distributions.base import SurvivalDistribution


def _raw_log_survival(
    self: "SurvivalDistribution",
) -> Callable[["SurvivalDistribution", ArrayLike], jax.Array]:
    # spec 4.5: defaults call the raw primitive on already-substituted safe times.
    return getattr(type(self).log_survival, "_survjax_raw", type(self).log_survival)


def survival(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return P(T > t), with the broadcast shape of t and batch_shape.

    The default is exp(log_survival(t)). Equal-length vectors pair entries;
    use singleton axes for all pairs. Shared time-first grids are separate.
    Negative times return one; NaN times return NaN.
    """
    raw = _raw_log_survival(self)
    return jnp.exp(raw(self, t))


def cumulative_hazard(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return -log P(T > t), broadcasting t against batch_shape.

    Negative times return zero; NaN times return NaN. This is pointwise,
    rather than a shared time-first grid operation.
    """
    raw = _raw_log_survival(self)
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
    raw = _raw_log_survival(self)
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
        # spec 4.5: reject dtype misuse during tracing, without inspecting values.
        try:
            time = jnp.asarray(t)
        except (TypeError, ValueError) as exc:
            raise TypeError("time must contain real numeric values") from exc
        if not (
            jnp.issubdtype(time.dtype, jnp.integer)
            or jnp.issubdtype(time.dtype, jnp.floating)
        ):
            raise TypeError("time must contain real numeric values")
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
        # spec 4.5: a NaN family value takes precedence over policy constants.
        policy_value = jnp.where(jnp.isnan(value), value, policy_value)
        return jnp.where(nan_time, jnp.nan, policy_value)

    # A decorated subclass reuses the raw override, not its parent's wrapper.
    evaluate._survjax_raw = raw  # type: ignore[attr-defined]
    return evaluate
