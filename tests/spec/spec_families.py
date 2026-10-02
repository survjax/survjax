"""Custom families defined by the tests through the public extension interface (spec 4.2).

A4 tests the distribution base with families written here, never with the
built-ins (those arrive at A5). Every family follows the documented pattern
exactly: ``@survjax.register_family`` over a frozen dataclass subclass of
``survjax.SurvivalDistribution`` using ``survjax.param``, ``survjax.static``
and ``survjax.distribution`` fields, supplying only ``log_survival`` and
``reference_time`` unless the test is about an override.

Each family's ``log_survival`` itself meets the direct-method value policy of
spec 4.5 (``log S = 0`` for ``t < 0``, NaN for NaN ``t``, correct limits at
``0`` and ``+inf``), using safe inputs so that unselected branches carry no
NaN tangents. Since spec rev 12 the base applies that policy itself and calls
family code only on ``[0, +inf]`` (Q-019, resolved); the guarded families keep
the earlier tests independent of that, and ``naive_power`` and
``naive_log_prob_override`` test it directly.

Classes are built lazily and cached: a missing survjax name then fails only
the A4 tests that use it, not collection of earlier steps, and every call
returns the same class, so tree structures compare equal across calls.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp

import survjax


def _safe_time(t):
    """Nonnegative time for active branches; NaN and +inf pass through unchanged."""
    t = jnp.asarray(t)
    return t, jnp.where(t < 0, 0.0, t)


@functools.cache
def const_hazard():
    """Exponential-like: ``S(t) = exp(-rate * t)``; ``log f = log(rate) - rate * t``."""

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class ConstHazard(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()

        def log_survival(self, t):
            t, ts = _safe_time(t)
            return jnp.where(t < 0, 0.0, -self.rate * ts)

        def reference_time(self):
            return 1.0 / self.rate

    return ConstHazard


@functools.cache
def power_hazard():
    """Weibull-like, with the Weibull keyword ``shape`` (spec 4.1): ``S(t) = exp(-(t/scale)**shape)``.

    With ``k = shape``, ``b = scale``, ``z = t/b``:
    ``log f = log k - log b + (k-1) log z - z**k``.
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class PowerHazard(survjax.SurvivalDistribution):
        shape: jax.Array = survjax.param()
        scale: jax.Array = survjax.param()

        def log_survival(self, t):
            t, ts = _safe_time(t)
            return jnp.where(t < 0, 0.0, -((ts / self.scale) ** self.shape))

        def reference_time(self):
            return self.scale

    return PowerHazard


@functools.cache
def component_rate():
    """A batch parameter, a trailing component axis and a static field (the spec 4.2 example).

    ``S(t) = exp(-rate * c * t)`` with ``c = sum(w)`` (``method="sum"``) or
    ``c = max(w)`` (``method="max"``) over the component axis.
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class ComponentRate(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()
        w: jax.Array = survjax.param(component_ndim=1)
        method: str = survjax.static(default="sum")

        def _c(self):
            if self.method == "sum":
                return jnp.sum(self.w, axis=-1)
            return jnp.max(self.w, axis=-1)

        def log_survival(self, t):
            t, ts = _safe_time(t)
            return jnp.where(t < 0, 0.0, -self.rate * self._c() * ts)

        def reference_time(self):
            return 1.0 / (self.rate * self._c())

    return ComponentRate


@functools.cache
def scaled_hazard():
    """A nested distribution field (``survjax.distribution()``): ``log S = exp(log_ratio) * log S_base``."""

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class ScaledHazard(survjax.SurvivalDistribution):
        base: survjax.SurvivalDistribution = survjax.distribution()
        log_ratio: jax.Array = survjax.param()

        def log_survival(self, t):
            return jnp.exp(self.log_ratio) * self.base.log_survival(t)

        def reference_time(self):
            return self.base.reference_time()

    return ScaledHazard


OVERRIDE_SENTINEL = 7.0


@functools.cache
def log_prob_override():
    """ConstHazard with a ``log_prob`` override returning a sentinel, to observe dispatch only."""

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class LogProbOverride(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()

        def log_survival(self, t):
            t, ts = _safe_time(t)
            return jnp.where(t < 0, 0.0, -self.rate * ts)

        def reference_time(self):
            return 1.0 / self.rate

        def log_prob(self, t):
            shape = jnp.broadcast_shapes(jnp.shape(t), jnp.shape(self.rate))
            return jnp.full(shape, OVERRIDE_SENTINEL)

    return LogProbOverride


def unregistered():
    """A frozen dataclass subclass deliberately not passed to ``register_family``."""

    @dataclasses.dataclass(frozen=True)
    class Unregistered(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()

        def log_survival(self, t):
            t, ts = _safe_time(t)
            return jnp.where(t < 0, 0.0, -self.rate * ts)

        def reference_time(self):
            return 1.0 / self.rate

    return Unregistered


@functools.cache
def naive_power():
    """Weibull-like with no input guards: ``log S = -(t/scale)**shape`` as written for ``t >= 0``.

    Spec 4.2/4.5 (rev 12): family code need only be correct on ``[0, +inf]``;
    the base applies the value policy to other inputs. For ``t < 0`` this
    formula gives NaN (non-integer ``shape``) or a positive ``log S``.
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class NaivePower(survjax.SurvivalDistribution):
        shape: jax.Array = survjax.param()
        scale: jax.Array = survjax.param()

        def log_survival(self, t):
            return -((t / self.scale) ** self.shape)

        def reference_time(self):
            return self.scale

    return NaivePower


@functools.cache
def naive_log_prob_override():
    """NaivePower plus an unguarded ``log_prob`` override (the Weibull log density for ``t >= 0``)."""

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class NaiveLogProbOverride(survjax.SurvivalDistribution):
        shape: jax.Array = survjax.param()
        scale: jax.Array = survjax.param()

        def log_survival(self, t):
            return -((t / self.scale) ** self.shape)

        def reference_time(self):
            return self.scale

        def log_prob(self, t):
            k, b = self.shape, self.scale
            return jnp.log(k / b) + (k - 1.0) * jnp.log(t / b) - (t / b) ** k

    return NaiveLogProbOverride


@functools.cache
def log_hazard_override():
    """ConstHazard with an analytic ``log_hazard`` override: ``log h = log(rate)`` for every ``t >= 0``."""

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class LogHazardOverride(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()

        def log_survival(self, t):
            return -self.rate * t

        def reference_time(self):
            return 1.0 / self.rate

        def log_hazard(self, t):
            return jnp.broadcast_to(
                jnp.log(self.rate),
                jnp.broadcast_shapes(jnp.shape(t), jnp.shape(self.rate)),
            )

    return LogHazardOverride


@functools.cache
def bounded_support():
    """Uniform on ``(0, end]``: ``S(t) = 1 - t/end`` for ``t < end`` and ``0`` from ``end`` on.

    ``log S = log1p(-t/end)`` for ``t < end`` (``-inf`` otherwise), so
    ``h(t) = 1/(end - t)`` for ``t < end``. Only ``log_survival`` and
    ``reference_time`` are supplied (``support()`` is not needed by A4 tests).
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class BoundedSupport(survjax.SurvivalDistribution):
        end: jax.Array = survjax.param()

        def log_survival(self, t):
            inside = t < self.end
            ts = jnp.where(inside, t, 0.0)
            return jnp.where(inside, jnp.log1p(-ts / self.end), -jnp.inf)

        def reference_time(self):
            return 0.5 * self.end

    return BoundedSupport


@functools.cache
def kw_only_static_first():
    """Spec 4.2 (rev 19) recipe variant: ``kw_only=True`` lets a defaulted static field precede a parameter."""

    @survjax.register_family
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class KwOnlyStaticFirst(survjax.SurvivalDistribution):
        method: str = survjax.static(default="sum")
        rate: jax.Array = survjax.param()

        def log_survival(self, t):
            return -self.rate * t

        def reference_time(self):
            return 1.0 / self.rate

    return KwOnlyStaticFirst
