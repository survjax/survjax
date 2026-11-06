"""Parameter domains for diagnostics, never transforms.

Call a constraint on an array to get a boolean array: True means inside the
domain. Scalar constraints preserve the input shape; ordered/simplex reduce
the last axis (event_dim=1). Arrays use the parameter's floating dtype.
"""

import math
from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def real(x: ArrayLike) -> jax.Array:
    """Check finite values. Args: x, parameter array. Returns: same-shape mask."""
    return jnp.isfinite(x)


def nonnegative(x: ArrayLike) -> jax.Array:
    """Check finite x >= 0. Args: x, parameter array. Returns: same-shape mask."""
    x = jnp.asarray(x)
    return jnp.isfinite(x) & (x >= 0)


def unit_interval(x: ArrayLike) -> jax.Array:
    """Check 0 <= x <= 1. Args: x, parameter array. Returns: same-shape mask."""
    x = jnp.asarray(x)
    return nonnegative(x) & (x <= 1)


@dataclass(frozen=True)
class _GreaterThan:
    bound: float

    def __call__(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        bound = jnp.asarray(self.bound, dtype=x.dtype)
        return jnp.isfinite(x) & (x > bound)


@dataclass(frozen=True)
class _Interval:
    lower: float
    upper: float
    closed: bool

    def __call__(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        lower = jnp.asarray(self.lower, dtype=x.dtype)
        upper = jnp.asarray(self.upper, dtype=x.dtype)
        if self.closed:
            return (x >= lower) & (x <= upper)
        return (x > lower) & (x < upper)


@dataclass(frozen=True)
class _Ordered:
    event_dim: ClassVar[int] = 1

    def __call__(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.all(jnp.isfinite(x), axis=-1) & jnp.all(
            x[..., 1:] > x[..., :-1], axis=-1
        )


@dataclass(frozen=True)
class _Simplex:
    event_dim: ClassVar[int] = 1

    def __call__(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        # spec 4.7: allow summation error in the parameter dtype, including f32.
        tolerance = 4 * x.shape[-1] * jnp.finfo(x.dtype).eps
        return jnp.all(nonnegative(x), axis=-1) & (
            jnp.abs(jnp.sum(x, axis=-1) - 1) <= tolerance
        )


def _bound(value: float) -> float:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise TypeError("constraint bounds must be finite Python numbers")
    return value


def greater_than(a: float) -> _GreaterThan:
    """Return a constraint checking finite x > a.

    Args:
        a: Finite Python bound.
    Returns:
        Callable preserving the input array shape.
    Raises:
        TypeError: If the bound is not a finite Python number.
    """
    return _GreaterThan(_bound(a))


def interval(a: float, b: float) -> _Interval:
    """Return a constraint checking a < x < b.

    Args:
        a: Finite Python lower bound.
        b: Finite Python upper bound, greater than a.
    Returns:
        Callable preserving the input array shape.
    Raises:
        TypeError: If bounds are not finite Python numbers with a < b.
    """
    a, b = _bound(a), _bound(b)
    if a >= b:
        raise TypeError("constraint bounds require a < b")
    return _Interval(a, b, closed=False)


def closed_interval(a: float, b: float) -> _Interval:
    """Return a constraint checking a <= x <= b.

    Args:
        a: Finite Python lower bound.
        b: Finite Python upper bound, greater than a.
    Returns:
        Callable preserving the input array shape.
    Raises:
        TypeError: If bounds are not finite Python numbers with a < b.
    """
    bounds = interval(a, b)
    return _Interval(bounds.lower, bounds.upper, closed=True)


#: Check finite x > 0, returning a boolean mask with the input shape.
positive = greater_than(0)
#: Check finite strict order along the last axis; event_dim is 1.
ordered = _Ordered()
#: Check nonnegative unit sums within 4*n*eps along the last axis; event_dim is 1.
simplex = _Simplex()
