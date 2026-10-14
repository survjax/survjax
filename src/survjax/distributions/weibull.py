"""Weibull family in natural shape and scale parameters (spec 4.1)."""

from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from survjax._dtypes import normalize_parameters
from survjax.distributions.base import SurvivalDistribution, param, register_family


@register_family
@dataclass(frozen=True)
class Weibull(SurvivalDistribution):
    """Weibull survival S(t)=exp(-(t/b)**k), with positive finite k=shape, b=scale.

    Args:
        shape: Required keyword k, dimensionless, stored as a natural leaf.
        scale: Required keyword b, in time units, stored as a natural leaf.

    Leaves broadcast to batch_shape in a common dtype. Methods broadcast t with
    that shape, pairing equal-length vectors; shared grids are time-first.
    Construction checks dtypes, not values. Nonreal inputs, positional arguments
    and incompatible shapes raise TypeError. Times follow the spec 4.5 policy.

    R/flexsurv use shape=k, scale=b; NumPyro uses concentration=k, scale=b.
    survival::survreg uses log-time location mu=log(b), reported scale sigma=1/k.
    flexsurv WeibullPH uses shape=k, scale=m=b**(-k), S(t)=exp(-m*t**k).
    Inverse scale r gives b=1/r; log inputs give k=exp(log_shape), b=exp(log_scale).
    For example, survreg mu=log(10), sigma=0.5 gives shape=2, scale=10.
    With common k, AFT log(b)=X@beta corresponds to PH gamma=-k*beta;
    varying shape removes this correspondence. flexsurv coef(fit) logs baseline
    shape/scale, unlike its natural estimates (spec 4.6).
    Discontinuous density/hazard shape derivatives at k=1 are excluded from the finite-boundary guarantee.
    """

    shape: jax.Array = param()
    scale: jax.Array = param()

    @classmethod
    def from_log(cls, *, log_shape: ArrayLike, log_scale: ArrayLike) -> Self:
        """Return this family with exponentiated natural shape/scale leaves.

        Args:
            log_shape: Keyword-only log shape, with arbitrary batch shape.
            log_scale: Keyword-only log scale, broadcast with log_shape.
        Returns:
            Weibull with common-dtype natural leaves; no log state is retained.
        Raises:
            TypeError: Inputs are nonreal or their batch shapes incompatible.

        Finite exp gives the ordinary constructor's values/gradients; overflow
        stores infinite natural parameters with spec 4.5 behavior.
        """
        values = normalize_parameters(
            {"log_shape": log_shape, "log_scale": log_scale}, warning_stacklevel=3
        )
        return cls(
            shape=jnp.exp(values["log_shape"]), scale=jnp.exp(values["log_scale"])
        )

    def reference_time(self) -> jax.Array:
        """Return scale, exactly batch_shape in the common parameter dtype.

        Positive and finite for valid parameters; only inactive inputs use it.
        """
        # spec 4.2: valid scale is already a positive finite time; Exponential's
        # 1/rate needs a reciprocal representability guard.
        return self.scale

    def _cumulative_hazard(self, t: ArrayLike) -> jax.Array:
        time = jnp.asarray(t)
        zero = time == 0
        infinite = jnp.isposinf(time)
        boundary = zero | infinite

        # spec 7.2: substitute before logs/products, including the scale operand.
        safe_time = jnp.where(boundary, jnp.ones_like(time), time)
        safe_scale = jnp.where(boundary, jnp.ones_like(self.scale), self.scale)

        # spec 4.1: interior H(t)=(t/scale)**shape. Logs avoid forming a ratio
        # that could underflow/overflow before the power becomes representable.
        log_ratio = jnp.log(safe_time) - jnp.log(safe_scale)
        value = jnp.exp(self.shape * log_ratio)

        # spec 4.5: restore boundary limits without hiding undefined parameters.
        zero_limit = jnp.where(self.shape > 0, 0.0, jnp.inf)
        infinity_limit = jnp.where(self.shape > 0, jnp.inf, 0.0)
        value = jnp.where(zero, zero_limit, jnp.where(infinite, infinity_limit, value))
        value = jnp.where(boundary & (self.shape == 0), 1.0, value)
        undefined = jnp.isnan(self.shape) | jnp.isnan(jnp.log(self.scale))
        return jnp.where(undefined, jnp.nan, value)

    def log_survival(self, t: ArrayLike) -> jax.Array:
        """Return -(t/b)**k, broadcasting t with batch_shape (spec 4.1/4.5).

        Args: t: Real times, broadcast with the parameter batch.
        Returns: Log survival; zero at t<=0 and -inf at +inf for valid k, b.
        Raises: TypeError: Nonreal times or incompatible shapes.
        """
        return -self._cumulative_hazard(t)

    def cumulative_hazard(self, t: ArrayLike) -> jax.Array:
        """Return (t/b)**k, with the broadcast shape (spec 4.1/4.5).

        Args: t: Real times, broadcast with the parameter batch.
        Returns: H; zero at t<=0 and +inf at +inf for valid parameters.
        Raises: TypeError: Nonreal times or incompatible shapes.
        """
        return self._cumulative_hazard(t)

    def _log_cumulative_hazard(self, t: ArrayLike) -> jax.Array:
        time = jnp.asarray(t)
        zero = time == 0
        infinite = jnp.isposinf(time)
        boundary = zero | infinite

        # spec 7.2: substitute before logs, including the scale operand.
        safe_time = jnp.where(boundary, jnp.ones_like(time), time)
        safe_scale = jnp.where(boundary, jnp.ones_like(self.scale), self.scale)

        # spec 4.1: log H = k * log(t / b), without forming H in the lower tail.
        log_ratio = jnp.log(safe_time) - jnp.log(safe_scale)
        value = self.shape * log_ratio

        # spec 4.5: restore the limits implied by H=(t/b)**k.
        zero_limit = jnp.where(
            self.shape > 0,
            -jnp.inf,
            jnp.where(self.shape == 0, 0.0, jnp.inf),
        )
        infinity_limit = jnp.where(
            self.shape > 0,
            jnp.inf,
            jnp.where(self.shape == 0, 0.0, -jnp.inf),
        )
        value = jnp.where(zero, zero_limit, jnp.where(infinite, infinity_limit, value))
        undefined = jnp.isnan(self.shape) | jnp.isnan(jnp.log(self.scale))
        return jnp.where(undefined, jnp.nan, value)

    def log_cdf(self, t: ArrayLike) -> jax.Array:
        """Return log(1-exp(-H(t))) while preserving tiny Weibull lower tails.

        Args: t: Real times, broadcast with the parameter batch.
        Returns: Broadcast log CDF; zero at +inf for valid shape and scale.
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

    def log_hazard(self, t: ArrayLike) -> jax.Array:
        """Return log(k)-log(b)+(k-1)*log(t/b), with analytic limits.

        Args: t: Real times, broadcast with the parameter batch.
        Returns: Broadcast log hazard. At zero: +inf, -log(b), -inf for
            k<1, k=1, k>1; at +inf those infinite limits reverse.
        Raises: TypeError: Nonreal times or incompatible shapes.

        Negative times give -inf; NaNs propagate. See the class derivative limits.
        """
        return self._log_hazard(t)

    def _log_hazard(self, t: ArrayLike) -> jax.Array:
        time = jnp.asarray(t)
        zero = time == 0
        infinite = jnp.isposinf(time)
        boundary = zero | infinite

        # spec 4.1/7.2: xlogy-style zero coefficients need safe AD operands too.
        safe_time = jnp.where(boundary, jnp.ones_like(time), time)
        safe_scale = jnp.where(boundary, jnp.ones_like(self.scale), self.scale)
        safe_shape = jnp.where(boundary, jnp.ones_like(self.shape), self.shape)

        # spec 4.1: interior h(t)=(shape/scale)*(t/scale)**(shape-1), in log
        # space; computing it directly avoids cancellation when H is large.
        log_ratio = jnp.log(safe_time) - jnp.log(safe_scale)
        value = jnp.log(safe_shape) - jnp.log(safe_scale) + (safe_shape - 1) * log_ratio

        # spec 4.5: shape=1 retains its finite -log(scale) boundary value.
        log_scale = jnp.log(self.scale)
        power_limit = jnp.where(self.shape < 1, jnp.inf, -jnp.inf)
        zero_limit = jnp.where(self.shape == 1, -log_scale, power_limit)
        infinity_limit = jnp.where(self.shape == 1, -log_scale, -power_limit)
        value = jnp.where(zero, zero_limit, jnp.where(infinite, infinity_limit, value))
        undefined = jnp.isnan(jnp.log(self.shape)) | jnp.isnan(log_scale)
        return jnp.where(undefined, jnp.nan, value)

    def log_prob(self, t: ArrayLike) -> jax.Array:
        """Return log(k)-log(b)+(k-1)*log(t/b)-(t/b)**k (spec 4.1).

        Args: t: Real times, broadcast with the parameter batch.
        Returns: Broadcast log density, with the zero-time hazard limits.
            Negative/+inf times give -inf unless NaN family values take precedence.
        Raises: TypeError: Nonreal times or incompatible shapes.
        """
        # spec 4.1: log f(t)=log h(t)-H(t); the base guards density at infinity.
        return self._log_hazard(t) - self._cumulative_hazard(t)
