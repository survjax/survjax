"""Tiny synthetic Slice A workflow, with exact events and right censoring.

Run from the checkout with JAX_ENABLE_X64=1 (float64) or =0 (float32):
    uv run python examples/slice_a.py
"""

import jax
import jax.numpy as jnp

import survjax


def run_example() -> dict[str, jax.Array]:
    """Validate synthetic data and return likelihood, derivative and grid arrays.

    Returns:
        Named arrays: observations and per-row contributions have shape (3,),
        Weibull parameters/gradient have shape (2,), its Hessian has shape
        (2, 2), and its summed objective is scalar. The time-first survival
        grid has shape (4, 2): four times by two scale parameters.

    Raises:
        survjax.InvalidDataError: At the explicit host checkpoint if the
            observations are edited to contain invalid values.
    """
    time = jnp.array([1.0, 2.0, 3.0])
    event = jnp.array([True, False, True])
    data = survjax.SurvivalData(time, event)
    data.issues.raise_if_invalid()

    rate = jnp.asarray(0.5)
    exponential = survjax.Exponential(rate=rate)
    parameters = jnp.array([2.0, 2.0])  # Weibull shape, scale.
    weibull = survjax.Weibull(shape=parameters[0], scale=parameters[1])

    def total(p: jax.Array) -> jax.Array:
        dist = survjax.Weibull(shape=p[0], scale=p[1])
        return dist.log_likelihood(data).sum()

    # Spec 7.3: derivatives are in parameters, on a finite smooth interior case.
    value, gradient = jax.jit(jax.value_and_grad(total))(parameters)
    hessian = jax.jit(jax.hessian(total))(parameters)

    grid_time = jnp.array([0.0, 1.0, 2.0, 3.0])
    grid_scale = jnp.array([1.0, 2.0])
    curves = survjax.Weibull(shape=parameters[0], scale=grid_scale)
    return {
        "time": time,
        "event": event,
        "exponential_rate": rate,
        "exponential_log_likelihood": exponential.log_likelihood(data),
        "weibull_parameters": parameters,
        "weibull_log_likelihood": weibull.log_likelihood(data),
        "weibull_total": value,
        "weibull_gradient": gradient,
        "weibull_hessian": hessian,
        "grid_time": grid_time,
        "grid_scale": grid_scale,
        "weibull_survival_grid": curves.survival_grid(grid_time),
    }


def main() -> None:
    """Print the example's inputs and computed arrays."""
    for name, value in run_example().items():
        print(f"{name}:\n{value}")


if __name__ == "__main__":
    main()
