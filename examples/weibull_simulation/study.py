"""Shared demo generation and single-replicate PH fitting; no library API."""

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

import survjax


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
    log_hazard_ratio: ArrayLike = 0.0,
) -> GeneratedData:
    """Draw Weibull events and independent Uniform(0, c) censoring.

    Args:
        key: Explicit key, split in U-then-C order; callers own fold_in.
        shape: Positive finite k, broadcast against scale and censor_limit.
        scale: Positive finite baseline b0, in time units.
        log_hazard_ratio: Finite PH linear predictor eta (e.g. gamma*arm),
            broadcast per row; H(t)=exp(eta)*(t/b0)**k. Zero gives scale b0.
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
    parameters = (shape, scale, censor_limit, log_hazard_ratio)
    dtype = jnp.result_type(*parameters, float)
    k, b, c, eta = jnp.broadcast_arrays(
        *(jnp.asarray(value, dtype=dtype) for value in parameters)
    )
    # 01-details DGP: PH is Weibull with scale b0*exp(-eta/k).
    b = b * jnp.exp(-eta / k)
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


def calibrate_censoring(
    shape: ArrayLike,
    scale: ArrayLike,
    target: float,
    *,
    log_hazard_ratio: ArrayLike = 0.0,
) -> float:
    """Solve mean_i integral_0^c S_i(t) dt / c = target on the host.

    Run once per parameter cohort/target, before compiled replicate generation.
    Host float64 quadrature and Python root-finding are deliberately not jitted.

    Args:
        shape: Concrete positive finite k, broadcast against scale per row.
        scale: Concrete positive finite baseline b0.
        target: Expected censored fraction strictly between zero and one.
        log_hazard_ratio: Concrete finite PH eta; all broadcast rows have equal
            weight. Effective scales b0*exp(-eta/k) must be positive and finite.

    Returns:
        Positive c via host-only NumPy float64 quadrature/bisection, in the same
        time units as scale, not differentiable. Substitution x**8 smooths for k < 1.

    Raises:
        ValueError: Empty, nonfinite, nonpositive or incompatible inputs/target.
        RuntimeError: Bracketing, root-finding or quadrature checks fail.
    """
    k, b, eta = np.broadcast_arrays(
        *(
            np.asarray(value, dtype=np.float64)
            for value in (shape, scale, log_hazard_ratio)
        )
    )
    if not k.size or not np.all(np.isfinite(k) & (k > 0)):
        raise ValueError("shape must be nonempty, positive and finite")
    if not np.all(np.isfinite(b) & (b > 0) & np.isfinite(eta)) or not 0 < target < 1:
        raise ValueError("scale must be positive/finite, eta finite and 0 < target < 1")
    with np.errstate(over="ignore", under="ignore"):
        b = b * np.exp(-eta / k)
    if not np.all(np.isfinite(b) & (b > 0)):
        raise ValueError("effective PH scales must be positive and finite")

    rules = {}
    # Host-only calibration checks two quadrature orders once before JAX batching.
    for order in (64, 128):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        rules[order] = ((nodes + 1) / 2, weights / 2)

    def fraction(c: float, order: int) -> float:
        nodes, weights = rules[order]
        # 00-common: t=c*x**8, dt/c=8*x**7 dx; mean over broadcast rows.
        t = c * nodes[:, None] ** 8
        with np.errstate(over="ignore"):
            survival = np.exp(-((t / b.ravel()) ** k.ravel()))
        return float(np.dot(weights * 8 * nodes**7, survival.mean(axis=1)))

    low, high = 0.0, float(np.median(b))
    # Host bracketing converges before the compiled simulation is started.
    for _ in range(128):
        if fraction(high, 128) < target:
            break
        high *= 2
        if not np.isfinite(high):
            raise RuntimeError("censoring calibration bracket overflowed")
    else:
        raise RuntimeError("censoring calibration could not bracket target")
    # Host bisection is calibration work, never a loop over simulated replicates.
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


class FitResult(NamedTuple):
    """One fit: raw diagnostics, acceptance flags and accepted-fit Wald inference."""

    theta: jax.Array
    objective: jax.Array
    gradient: jax.Array
    hessian: jax.Array
    log_hr_se: jax.Array
    hr_interval: jax.Array
    solver_success: jax.Array
    hessian_ok: jax.Array
    estimable: jax.Array
    accepted: jax.Array


def _assess_fit(
    theta: jax.Array,
    value: jax.Array,
    gradient: jax.Array,
    hessian: jax.Array,
    solver_success: jax.Array,
) -> FitResult:
    """Assess an estimable optimizer result, retaining diagnostics on rejection."""
    identity = jnp.eye(theta.size, dtype=theta.dtype)
    finite_hessian = jnp.all(jnp.isfinite(hessian))
    chol = jnp.linalg.cholesky(jnp.where(finite_hessian, hessian, identity))
    hessian_ok = finite_hessian & jnp.all(jnp.isfinite(chol))
    finite = jnp.all(jnp.isfinite(theta)) & jnp.isfinite(value)
    accepted = (
        solver_success & finite & (jnp.max(jnp.abs(gradient)) < 1e-5) & hessian_ok
    )
    # 01-details: [H^-1]_gamma,gamma = ||L^-1 e_gamma||^2 for H=L L^T.
    safe_chol = jnp.where(hessian_ok, chol, identity)
    column = jax.scipy.linalg.solve_triangular(safe_chol, identity[:, -1], lower=True)
    se = jnp.sqrt(jnp.sum(column**2))
    safe_gamma = jnp.where(accepted, theta[-1], 0.0)
    safe_se = jnp.where(accepted, se, 1.0)
    interval = jnp.exp(safe_gamma + jnp.array([-1.96, 1.96]) * safe_se)
    return FitResult(
        theta,
        value,
        gradient,
        hessian,
        jnp.where(accepted, se, jnp.nan),
        jnp.where(accepted, interval, jnp.nan),
        solver_success,
        hessian_ok,
        jnp.asarray(True),
        accepted,
    )


def negative_log_likelihood(
    theta: jax.Array,
    data: survjax.SurvivalData,
    arm: jax.Array,
    *,
    model: Literal["weibull", "exponential"],
) -> jax.Array:
    """Return the summed PH negative log likelihood for a single (N,) cohort.

    Args:
        theta: (log k, log b0, gamma) for Weibull; (log rate0, gamma) for Exponential.
        data: Validated exact/right untruncated observations, shape (N,).
        arm: Matching binary arm indicators, shape (N,).
        model: Static model name, 'weibull' or 'exponential'.
    Returns:
        Scalar negative log likelihood, differentiable in theta.
    Raises:
        ValueError: Unknown model name.
    """
    if model == "weibull":
        base = survjax.Weibull.from_log(log_shape=theta[0], log_scale=theta[1])
    elif model == "exponential":
        base = survjax.Exponential.from_log(log_rate=theta[0])
    else:
        raise ValueError("model must be 'weibull' or 'exponential'")
    lp = theta[-1] * arm
    dist = survjax.ProportionalHazards(base, log_hazard_ratio=lp)
    return -dist.log_likelihood(data).sum()


def fit_replicate(
    data: survjax.SurvivalData,
    arm: jax.Array,
    *,
    model: Literal["weibull", "exponential"],
) -> FitResult:
    """Fit one float64 cohort with the use-case 01 BFGS/acceptance settings.

    Args:
        data: Validated nonempty (N,) exact/right data without truncation.
        arm: Matching binary indicators; no-event arms are unestimable.
        model: Static model name. Starts use log mean observed time and gamma=0.
    Returns:
        Raw theta/gradient (P,), Hessian (P,P), scalar flags and SE, and HR
        interval (2,). P=3 for Weibull, 2 for Exponential. Rejected fits retain
        raw diagnostics; their inference is NaN. Unestimable fits skip fitting
        and have NaN diagnostics. Supports jit/vmap; no derivative-through-fit
        guarantee. Caller performs data.issues.raise_if_invalid() before JIT.
    Raises:
        ValueError: Unknown model or data not float64.
    """
    import optimistix as optx

    if data.lower.dtype != jnp.float64:
        raise ValueError("simulation fitting requires float64 (JAX_ENABLE_X64=1)")
    if model not in ("weibull", "exponential"):
        raise ValueError("model must be 'weibull' or 'exponential'")
    size = 3 if model == "weibull" else 2
    estimable = jnp.any(data.exact & (arm == 0)) & jnp.any(data.exact & (arm == 1))

    def solve(_: None) -> FitResult:
        log_mean = jnp.log(jnp.mean(data.lower))
        start = (
            jnp.array([0.0, log_mean, 0.0])
            if model == "weibull"
            else jnp.array([-log_mean, 0.0])
        )

        def objective(theta: jax.Array, _: None = None) -> jax.Array:
            return negative_log_likelihood(theta, data, arm, model=model)

        solution = optx.minimise(
            objective,
            optx.BFGS(rtol=1e-8, atol=1e-8),
            start,
            max_steps=1000,
            throw=False,
        )
        theta = solution.value
        value, gradient = jax.value_and_grad(objective)(theta)
        hessian = jax.hessian(objective)(theta)
        return _assess_fit(
            theta,
            value,
            gradient,
            hessian,
            solution.result == optx.RESULTS.successful,
        )

    def unestimable(_: None) -> FitResult:
        missing = jnp.full((size,), jnp.nan, dtype=data.lower.dtype)
        false = jnp.asarray(False)
        return FitResult(
            missing,
            jnp.asarray(jnp.nan, dtype=data.lower.dtype),
            missing,
            jnp.full((size, size), jnp.nan, dtype=data.lower.dtype),
            jnp.asarray(jnp.nan, dtype=data.lower.dtype),
            jnp.full((2,), jnp.nan, dtype=data.lower.dtype),
            false,
            false,
            false,
            false,
        )

    return jax.lax.cond(estimable, solve, unestimable, None)
