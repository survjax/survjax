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
        # spec 4.2: unlike a stored time scale, 1/rate may not be representable.
        # This guard changes only the inactive reference, not model parameters.
        limits = jnp.finfo(self.rate.dtype)
        usable = (self.rate >= limits.tiny) & (self.rate <= 1 / limits.tiny)
        safe_rate = jnp.where(usable, self.rate, jnp.ones_like(self.rate))
        return jnp.reciprocal(safe_rate)

    def _cumulative_hazard(self, t: ArrayLike) -> jax.Array:
        time = jnp.asarray(t)
        infinite = jnp.isposinf(time)

        # spec 7.2: don't multiply an inactive infinity by a live parameter.
        safe_time = jnp.where(infinite, jnp.zeros_like(time), time)

        # spec 4.1: interior H(t)=rate*t, so log S(t)=-H(t).
        value = self.rate * safe_time

        # spec 4.5: restore the infinity limit separately from the formula.
        limit = jnp.where(self.rate > 0, jnp.inf, jnp.nan)
        limit = jnp.where(self.rate < 0, -jnp.inf, limit)
        return jnp.where(infinite, limit, value)

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

    def _log_cumulative_hazard(self, t: ArrayLike) -> jax.Array:
        time = jnp.asarray(t)
        zero = time == 0
        infinite = jnp.isposinf(time)
        boundary = zero | infinite

        # spec 7.2: substitute before log(t), including at both boundaries.
        safe_time = jnp.where(boundary, jnp.ones_like(time), time)

        # spec 4.1: log H = log(rate) + log(t), without forming rate*t.
        value = jnp.log(self.rate) + jnp.log(safe_time)
        zero_limit = jnp.where(jnp.isfinite(self.rate), -jnp.inf, jnp.nan)
        infinity_limit = jnp.where(self.rate > 0, jnp.inf, jnp.nan)
        value = jnp.where(zero, zero_limit, jnp.where(infinite, infinity_limit, value))
        return value

    def log_cdf(self, t: ArrayLike) -> jax.Array:
        """Return log(1-exp(-H(t))) while preserving tiny exponential tails.

        Args: t: Real times, broadcast with the parameter batch.
        Returns: Broadcast log CDF; zero at +inf for valid positive rates.
        Raises: TypeError: Nonreal times or incompatible shapes.
        """
        log_h = self._log_cumulative_hazard(t)
        dtype = log_h.dtype
        log_eps = jnp.log(jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))
        log_max = jnp.log(jnp.asarray(jnp.finfo(dtype).max, dtype=dtype))
        tiny = log_h < log_eps
        upper = log_h >= log_max

        # spec 7.2: sanitize both inactive branches before exp/expm1.
        safe_log_h = jnp.where(tiny | upper, jnp.zeros_like(log_h), log_h)
        standard = jnp.log(-jnp.expm1(-jnp.exp(safe_log_h)))
        value = jnp.where(tiny, log_h, standard)
        return jnp.where(upper, jnp.zeros_like(value), value)

    def log_prob(self, t: ArrayLike) -> jax.Array:
        """Return log(rate)-rate*t with the broadcast shape (spec 4.1/4.5).

        Negative/+inf t gives -inf, except NaN-rate precedence; NaN t gives NaN.
        """
        # spec 4.1: log f(t)=log(rate)-H(t), since h(t)=rate.
        return jnp.log(self.rate) - self._cumulative_hazard(t)

    def log_hazard(self, t: ArrayLike) -> jax.Array:
        """Return log(rate), broadcast against t, including at zero and +inf.

        Negative t gives -inf; NaNs propagate. Uses no tail subtraction.
        """
        # spec 4.1/5: log h(t)=log(rate) is constant in time but must broadcast.
        output_shape = jnp.broadcast_shapes(jnp.shape(t), self.batch_shape)
        return jnp.broadcast_to(jnp.log(self.rate), output_shape)

    def hazard(self, t: ArrayLike) -> jax.Array:
        """Return rate, broadcast against t, including at zero and +inf.

        Negative t gives zero; NaNs propagate. The infinity rate derivative is 1.
        """
        # spec 4.1/5: h(t)=rate is constant in time but must broadcast.
        output_shape = jnp.broadcast_shapes(jnp.shape(t), self.batch_shape)
        return jnp.broadcast_to(self.rate, output_shape)
