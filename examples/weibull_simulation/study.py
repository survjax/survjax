"""Shared demo generation; no fitting or public survjax sampling API."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


class GeneratedData(NamedTuple):
    """Observed time/event and latent event/censor times, all of equal shape."""

    time: jax.Array
    event: jax.Array
    event_time: jax.Array
    censor_time: jax.Array


def generate_weibull(
    key: jax.Array,
    shape: ArrayLike,
    scale: ArrayLike,
    censor_limit: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> GeneratedData:
    """Draw Weibull events and independent Uniform(0, c) censoring.

    Args:
        key: Explicit key, split in U-then-C order; callers own fold_in.
        shape: Positive finite k, broadcast against scale and censor_limit.
        scale: Positive finite b in time units; PH uses b0*exp(-gamma*arm/k).
        censor_limit: Positive finite c, in the same time units as scale.
        sample_shape: Static sample prefix before the broadcast parameter axes.

    Returns:
        Arrays of sample_shape + broadcast_shape; boolean event is T <= C.
        JAX precision and [0,1) U apply: U=0 gives latent T=inf, censored at C.
        Extreme parameters can underflow T to zero, invalid as an exact time.

    Raises:
        ValueError: Shapes cannot broadcast. Values are positive/finite
            preconditions, not host-validated inside this JIT-compatible function.
    """
    dtype = jnp.result_type(shape, scale, censor_limit, float)
    k, b, c = jnp.broadcast_arrays(
        *(jnp.asarray(value, dtype=dtype) for value in (shape, scale, censor_limit))
    )
    draw_shape = sample_shape + k.shape
    event_key, censor_key = jax.random.split(key)
    u = jax.random.uniform(event_key, draw_shape, dtype=dtype)
    event_time = b * (-jnp.log(u)) ** (1 / k)
    censor_time = c * jax.random.uniform(censor_key, draw_shape, dtype=dtype)
    return GeneratedData(
        jnp.minimum(event_time, censor_time),
        event_time <= censor_time,
        event_time,
        censor_time,
    )


def calibrate_censoring(shape: ArrayLike, scale: ArrayLike, target: float) -> float:
    """Solve mean_i integral_0^c S_i(t) dt / c = target on the host.

    Args:
        shape: Concrete positive finite k, broadcast against scale per row.
        scale: Concrete positive finite b; broadcast entries have equal weight.
        target: Expected censored fraction strictly between zero and one.

    Returns:
        Positive c via host-only NumPy float64 quadrature/bisection, in the same
        time units as scale, not differentiable. Substitution x**8 smooths for k < 1.

    Raises:
        ValueError: Empty, nonfinite, nonpositive or incompatible inputs/target.
        RuntimeError: Bracketing, root-finding or quadrature checks fail.
    """
    k, b = np.broadcast_arrays(
        np.asarray(shape, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    )
    if not k.size or not np.all(np.isfinite(k) & (k > 0)):
        raise ValueError("shape must be nonempty, positive and finite")
    if not np.all(np.isfinite(b) & (b > 0)) or not 0 < target < 1:
        raise ValueError("scale must be positive/finite and 0 < target < 1")

    def fraction(c: float, order: int) -> float:
        nodes, weights = rules[order]
        # 00-common: t=c*x**8, dt/c=8*x**7 dx; mean over broadcast rows.
        t = c * nodes[:, None] ** 8
        with np.errstate(over="ignore"):
            survival = np.exp(-((t / b.ravel()) ** k.ravel()))
        return float(np.dot(weights * 8 * nodes**7, survival.mean(axis=1)))

    rules = {}
    for order in (64, 128):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        rules[order] = ((nodes + 1) / 2, weights / 2)
    low, high = 0.0, float(np.median(b))
    for _ in range(128):
        if fraction(high, 128) < target:
            break
        high *= 2
        if not np.isfinite(high):
            raise RuntimeError("censoring calibration bracket overflowed")
    else:
        raise RuntimeError("censoring calibration could not bracket target")
    for _ in range(100):
        c = low + (high - low) / 2
        value = fraction(c, 128)
        if abs(value - target) <= 1e-12:
            if abs(fraction(c, 64) - value) > 1e-10:
                raise RuntimeError("censoring quadrature did not converge")
            return c
        if value > target:
            low = c
        else:
            high = c
    raise RuntimeError("censoring calibration did not converge")
