"""Float32 reference fitting on synthetic observations."""

import sys
from pathlib import Path

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.survival_mnist import model  # noqa: E402


def test_intercept_fit_is_finite_float32():
    """03: BFGS fits the training-only reference in neural precision."""
    fitted = model.fit_intercept(
        jnp.array([1.0, 2.0, 4.0, 8.0]), jnp.array([1, 0, 1, 1])
    )
    assert fitted.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(fitted))
