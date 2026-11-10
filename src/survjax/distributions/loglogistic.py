"""Log-logistic family in natural shape and scale parameters."""

from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from survjax import constraints
from survjax._dtypes import normalize_parameters
from survjax.distributions.base import SurvivalDistribution, param, register_family


@register_family
@dataclass(frozen=True)
class LogLogistic(SurvivalDistribution):
    """Log-logistic distribution, S(t) = 1 / (1 + (t / alpha)**beta).

    Pointwise times broadcast with ``batch_shape``; grid twins return
    time-first arrays of shape ``(K, *batch_shape)``.

    This matches flexsurv's ``llogis(shape=beta, scale=alpha)``. Shape beta
    is dimensionless; scale alpha has time units and is the median. The
    log time has logistic location log(alpha) and scale 1/beta. For example,
    log-time location log(10) and scale 0.5 give shape=2 and scale=10.

    Args:
        shape: Positive finite beta, keyword only.
        scale: Positive finite alpha, keyword only. Parameters broadcast to
            ``batch_shape`` and are stored without value validation.

    Raises:
        TypeError: Nonreal parameters, positional arguments or incompatible
            batch shapes. Direct methods reject nonreal/incompatible times.
    """

    shape: jax.Array = param(constraint=constraints.positive)
    scale: jax.Array = param(constraint=constraints.positive)

    @classmethod
    def from_log(cls, *, log_shape: ArrayLike, log_scale: ArrayLike) -> Self:
        """Construct with beta=exp(log_shape), alpha=exp(log_scale).

        Args:
            log_shape: Real log shape, keyword only.
            log_scale: Real log scale, broadcasting with log_shape.
        Returns:
            The ordinary family storing natural parameters, with identical
            values and parameter gradients. Overflow produces infinite,
            invalid parameters, reported by ``parameter_issues()``.
        Raises:
            TypeError: Nonreal inputs or incompatible batch shapes.
        """
        values = normalize_parameters(
            {"log_shape": log_shape, "log_scale": log_scale}, warning_stacklevel=3
        )
        return cls(
            shape=jnp.exp(values["log_shape"]), scale=jnp.exp(values["log_scale"])
        )

    def reference_time(self) -> jax.Array:
        """Return alpha, positive and finite for valid parameters.

        Returns:
            Scale with shape ``batch_shape`` in the parameter dtype.
        """
        return self.scale

    def log_survival(self, t: ArrayLike) -> jax.Array:
        """Return -softplus(beta * (log(t) - log(alpha))).

        Args:
            t: Real times broadcasting with ``batch_shape``.
        Returns:
            Broadcast log survival; 0 at t<=0, -inf at +inf, NaN at NaN.
        Raises:
            TypeError: Nonreal times or incompatible shapes.
        """
        z = self.shape * (jnp.log(t) - jnp.log(self.scale))
        return -jax.nn.softplus(z)

    def log_cdf(self, t: ArrayLike) -> jax.Array:
        """Return -softplus(-beta * (log(t) - log(alpha))), retaining tails.

        Args:
            t: Real times broadcasting with ``batch_shape``.
        Returns:
            Broadcast log CDF; -inf at t<=0, 0 at +inf, NaN at NaN.
        Raises:
            TypeError: Nonreal times or incompatible shapes.
        """
        z = self.shape * (jnp.log(t) - jnp.log(self.scale))
        return -jax.nn.softplus(-z)

    def log_hazard(self, t: ArrayLike) -> jax.Array:
        """Return the analytic log hazard without subtracting log f and log S.

        Args:
            t: Real times broadcasting with ``batch_shape``.
        Returns:
            Broadcast log hazard. At zero it is +inf, -log(alpha), or -inf for
            beta<1, beta=1, beta>1; at +inf or negative times: -inf.
            NaN inputs propagate. Finite boundary limits retain parameter
            derivatives; the shape derivative at beta=1, t=0 is undefined.
        Raises:
            TypeError: Nonreal times or incompatible shapes.
        """
        time = jnp.asarray(t)
        zero, infinite = time == 0, jnp.isposinf(time)
        # spec 7.2: substitute before log and multiplication at both limits.
        safe_time = jnp.where(zero | infinite, 1.0, time)
        log_scale = jnp.log(self.scale)
        safe_scale = jnp.where(zero | infinite, 1.0, self.scale)
        safe_shape = jnp.where(zero | infinite, 1.0, self.shape)
        ratio = jnp.log(safe_time) - jnp.log(safe_scale)
        z = safe_shape * ratio
        # spec 4.1: upper tail h~beta/t; avoid cancelling beta*log(t/alpha).
        power = jnp.where(z > 0, -ratio, (safe_shape - 1) * ratio)
        tail = jax.nn.softplus(jnp.where(z > 0, -z, z))
        value = jnp.log(safe_shape) - jnp.log(safe_scale) + power - tail
        limit = jnp.where(
            self.shape == 1, -log_scale, jnp.where(self.shape < 1, jnp.inf, -jnp.inf)
        )
        value = jnp.where(zero, limit, jnp.where(infinite, -jnp.inf, value))
        undefined = jnp.isnan(jnp.log(self.shape)) | jnp.isnan(log_scale)
        return jnp.where(undefined, jnp.nan, value)

    def log_prob(self, t: ArrayLike) -> jax.Array:
        """Return log density log h(t) + log S(t).

        Args:
            t: Real times broadcasting with ``batch_shape``.
        Returns:
            Broadcast log density, with the log-hazard limit at zero,
            -inf at negative times and +inf, and NaN for NaN times.
        Raises:
            TypeError: Nonreal times or incompatible shapes.
        """
        return self.log_hazard(t) + self.log_survival(t)
