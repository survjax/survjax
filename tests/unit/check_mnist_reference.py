"""Float32 reference fitting on synthetic observations."""

import sys
from pathlib import Path

import jax.numpy as jnp

import survjax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.survival_mnist import model  # noqa: E402


def test_intercept_fit_is_finite_float32():
    """03: BFGS fits the training-only reference in neural precision."""
    data = survjax.SurvivalData([1.0, 2.0, 4.0, 8.0], [1, 0, 1, 1], dtype=jnp.float32)
    log_k, log_b = model.fit_intercept(data)
    assert log_k.dtype == jnp.float32 and log_b.dtype == jnp.float32
    assert jnp.isfinite(log_k) and jnp.isfinite(log_b)


def test_full_cohort_float32_fit_matches_profile_optimum():
    """03: the full-cohort fit must terminate, not stall near the optimum."""
    import impl_tolerances as tol
    import numpy as np

    from examples.survival_mnist import data

    _, labels = data.load_mnist()
    outcomes = data.generate_outcomes(labels)
    idx = data.split_indices()["train"]
    fitted = model.fit_intercept(
        survjax.SurvivalData(outcomes.time, outcomes.event)[idx]
    )
    # Independent profile-score bisection on these fixed synthetic outcomes:
    # b**k = sum(t**k)/d; d/k + sum_event(log t)
    # - d*sum(t**k*log t)/sum(t**k) = 0. Host NumPy calculation, not survjax.
    expected = [0.17198064720109116, 1.4325021367797313]
    np.testing.assert_allclose(
        fitted, expected, atol=tol.ATOL[False], rtol=tol.RTOL[False]
    )
