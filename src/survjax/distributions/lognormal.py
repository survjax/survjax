"""Log-normal family in natural log-time location and standard deviation."""

from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, ndtr
from jax.typing import ArrayLike

from survjax import constraints
from survjax._dtypes import normalize_parameters
from survjax.distributions.base import SurvivalDistribution, param, register_family


def _log_normal_cdf(x: jax.Array) -> jax.Array:
    # spec 4.1/7.2: log_ndtr loses relative accuracy on the positive side;
    # keep both branches' operands safe even when vmap evaluates them all.
    positive = x > 0
    upper_x = jnp.where(positive, x, jnp.ones_like(x))
    lower_x = jnp.where(positive, -jnp.ones_like(x), x)
    return jnp.where(positive, jnp.log1p(-ndtr(-upper_x)), log_ndtr(lower_x))


@register_family
@dataclass(frozen=True)
class LogNormal(SurvivalDistribution):
    """Log-normal event time: ``log(T) ~ Normal(meanlog, sdlog)``.

    With mu=meanlog and sigma=sdlog, S(t)=Phi((mu-log(t))/sigma)
    and f(t)=exp(-((log(t)-mu)/sigma)**2/2)/(t*sigma*sqrt(2*pi)).
    Mu is a log-time location in the chosen time unit; sigma is dimensionless.
    R/flexsurv use meanlog/sdlog; NumPyro uses loc=meanlog, scale=sdlog.
    For example loc=2, scale=0.5 gives ``LogNormal(meanlog=2, sdlog=0.5)``.

    Args:
        meanlog: Finite real log-time location, keyword only.
        sdlog: Positive finite log-time standard deviation, keyword only.
            Parameters broadcast to ``batch_shape`` and are stored without
            value validation; use ``parameter_issues()`` for diagnostics.

    Raises:
        TypeError: Positional/nonreal inputs or incompatible batch shapes.
            Direct methods also reject nonreal or nonbroadcasting times.

    ``_grid`` methods return time-first arrays of shape ``(K,) + batch_shape``.
    """

    meanlog: jax.Array = param(constraint=constraints.real)
    sdlog: jax.Array = param(constraint=constraints.positive)

    @classmethod
    def from_log(cls, *, meanlog: ArrayLike, log_sdlog: ArrayLike) -> Self:
        """Build from a log standard deviation, storing natural parameters.
        Args:
            meanlog: Log-time location, unchanged (already unconstrained).
            log_sdlog: Log of the standard deviation. Both inputs broadcast.
        Returns:
            LogNormal(meanlog=meanlog, sdlog=exp(log_sdlog)), with identical
            values and parameter derivatives to that ordinary constructor.
            Overflow produces an infinite, diagnostically invalid sdlog.
        Raises:
            TypeError: Nonreal inputs or incompatible batch shapes.
        """
        values = normalize_parameters(
            {"meanlog": meanlog, "log_sdlog": log_sdlog}, warning_stacklevel=3
        )
        return cls(meanlog=values["meanlog"], sdlog=jnp.exp(values["log_sdlog"]))

    def reference_time(self) -> jax.Array:
        """Return ones of batch shape/parameter dtype, even if exp(meanlog) overflows."""
        return jnp.ones_like(self.meanlog)

    def log_survival(self, t: ArrayLike) -> jax.Array:
        """Return log S(t), retaining relative precision in both tails.
        Args:
            t: Times broadcasting against ``batch_shape``.
        Returns:
            Broadcast log survival; zero for t<=0, -inf at +inf, NaN for NaN t.
        Raises:
            TypeError: Nonreal times or times incompatible with batch_shape.
        """
        x = (self.meanlog - jnp.log(jnp.asarray(t))) / self.sdlog
        return _log_normal_cdf(x)

    def log_cdf(self, t: ArrayLike) -> jax.Array:
        """Return log F(t), using the symmetric normal tail formula.
        Args:
            t: Times broadcasting against ``batch_shape``.
        Returns:
            Broadcast log CDF; -inf for t<=0, zero at +inf, NaN for NaN t.
        Raises:
            TypeError: Nonreal times or times incompatible with batch_shape.
        """
        x = (jnp.log(jnp.asarray(t)) - self.meanlog) / self.sdlog
        return _log_normal_cdf(x)

    def log_prob(self, t: ArrayLike) -> jax.Array:
        """Return the analytic log density.
        Args:
            t: Times broadcasting against ``batch_shape``.
        Returns:
            Broadcast log density; -inf for t<=0 or +inf, NaN for NaN t.
        Raises:
            TypeError: Nonreal times or times incompatible with batch_shape.
        """
        time = jnp.asarray(t)
        log_time = jnp.log(jnp.where(time == 0, jnp.ones_like(time), time))
        z = (log_time - self.meanlog) / self.sdlog
        density = -0.5 * z**2 - log_time - jnp.log(self.sdlog)
        density = density - 0.5 * jnp.log(2 * jnp.pi)
        value = jnp.where(jnp.asarray(t) == 0, -jnp.inf, density)
        return jnp.where(jnp.isnan(density), jnp.nan, value)

    def log_hazard(self, t: ArrayLike) -> jax.Array:
        """Return log h(t)=log f(t)-log S(t), including the limiting hazard.
        Args:
            t: Times broadcasting against ``batch_shape``.
        Returns:
            Broadcast log hazard; -inf for t<=0 or +inf, NaN for NaN t.
        Raises:
            TypeError: Nonreal times or times incompatible with batch_shape.
        """
        boundary = (jnp.asarray(t) == 0) | jnp.isposinf(t)
        time = jnp.asarray(t)
        log_time = jnp.log(jnp.where(boundary, jnp.ones_like(time), time))
        z = (log_time - self.meanlog) / self.sdlog
        tail = z > 2
        # spec 7.2: the discarded density-survival difference must not form
        # inf-inf or differentiate log_ndtr at an overflowing squared z.
        lower_z = jnp.where(tail, jnp.zeros_like(z), z)
        density = -0.5 * lower_z**2 - log_time - jnp.log(self.sdlog)
        density = density - 0.5 * jnp.log(2 * jnp.pi)
        survival = _log_normal_cdf(-lower_z)
        safe_z = jnp.where(tail, z, jnp.full_like(z, 2))
        # Laplace's continued fraction for the inverse normal Mills ratio:
        # lambda(z)=z+1/(z+2/(z+3/...)). At z>=2, 128 terms converge
        # below float64 rounding; unlike log f-log S, its AD does not cancel.
        fraction = jax.lax.fori_loop(
            0, 128, lambda i, f: (128 - i) / (safe_z + f), jnp.zeros_like(z)
        )
        tail_hazard = jnp.log(safe_z + fraction) - log_time - jnp.log(self.sdlog)
        value = jnp.where(tail, tail_hazard, density - survival)
        value = jnp.where(boundary, -jnp.inf, value)
        return jnp.where(jnp.isnan(density) | jnp.isnan(survival), jnp.nan, value)
