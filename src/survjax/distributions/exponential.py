"""Exponential family in natural rate parameters (spec 4.1)."""

from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from survjax._dtypes import normalize_parameters
from survjax.distributions.base import SurvivalDistribution, param, register_family


@register_family
@dataclass(frozen=True)
class Exponential(SurvivalDistribution):
    """Exponential survival S(t)=exp(-rate*t), with positive finite rate.

    Args:
        rate: Required keyword, in inverse time units; rate=1/mean.
            Its array shape is batch_shape, and it is the only PyTree leaf.

    Log-rate a gives rate=exp(a); log-mean m gives rate=exp(-m).
    For example, Exponential(rate=1/5) has mean five time units.
    Methods broadcast t against batch_shape, pairing equal-length vectors.
    Construction checks dtypes, not parameter values; nonreal inputs and
    positional arguments raise TypeError. Direct methods raise TypeError for
    nonreal t or incompatible shapes, and apply the spec 4.5 time policy.
    """

    rate: jax.Array = param()

    @classmethod
    def from_log(cls, *, log_rate: ArrayLike) -> Self:
        """Return this family with natural rate=exp(log_rate) as its leaf.

        Args:
            log_rate: Keyword-only log rate, with arbitrary batch shape.
        Returns:
            Exponential with the same values/gradients as an exp-rate model.
        Raises:
            TypeError: Input is nonreal. Dtype promotion follows construction.

        Overflow stores an infinite rate; no log-space representation is kept.
        """
        values = normalize_parameters({"log_rate": log_rate}, warning_stacklevel=3)
        return cls(rate=jnp.exp(values["log_rate"]))

    def reference_time(self) -> jax.Array:
        """Return a positive finite reference of batch_shape in rate.dtype.

        Uses 1/rate when representable, otherwise one time unit. This changes
        only inactive evaluation inputs, never rates or active probabilities.
        Invalid-rate references have no value guarantee (spec 4.2).
        """
        limits = jnp.finfo(self.rate.dtype)
        usable = (self.rate >= limits.tiny) & (self.rate <= 1 / limits.tiny)
        safe_rate = jnp.where(usable, self.rate, jnp.ones_like(self.rate))
        return jnp.reciprocal(safe_rate)

    def _cumulative_hazard(self, t: ArrayLike) -> jax.Array:
        time = jnp.asarray(t)
        infinite = jnp.isposinf(time)
        # spec 7.2: don't multiply an inactive infinity by a live parameter.
        safe_time = jnp.where(infinite, jnp.zeros_like(time), time)
        limit = jnp.where(self.rate > 0, jnp.inf, jnp.nan)
        limit = jnp.where(self.rate < 0, -jnp.inf, limit)
        return jnp.where(infinite, limit, self.rate * safe_time)

    def log_survival(self, t: ArrayLike) -> jax.Array:
        """Return -rate*t, broadcasting t with batch_shape (spec 4.1/4.5).

        Negative t gives zero; +inf gives -inf for valid rates; NaNs propagate.
        """
        return -self._cumulative_hazard(t)

    def cumulative_hazard(self, t: ArrayLike) -> jax.Array:
        """Return rate*t with the broadcast shape; negative t gives zero.

        The infinity limit is +inf for valid rates; NaNs propagate (spec 4.5).
        """
        return self._cumulative_hazard(t)

    def log_prob(self, t: ArrayLike) -> jax.Array:
        """Return log(rate)-rate*t with the broadcast shape (spec 4.1/4.5).

        Negative/+inf t gives -inf, except NaN-rate precedence; NaN t gives NaN.
        """
        return jnp.log(self.rate) - self._cumulative_hazard(t)

    def log_hazard(self, t: ArrayLike) -> jax.Array:
        """Return log(rate), broadcast against t, including at zero and +inf.

        Negative t gives -inf; NaNs propagate. Uses no tail subtraction.
        """
        return jnp.broadcast_to(jnp.log(self.rate), jnp.shape(t))

    def hazard(self, t: ArrayLike) -> jax.Array:
        """Return rate, broadcast against t, including at zero and +inf.

        Negative t gives zero; NaNs propagate. The infinity rate derivative is 1.
        """
        return jnp.broadcast_to(self.rate, jnp.shape(t))
