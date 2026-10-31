"""Optional offline evaluation checks for use-case 03 References and metrics."""

import sys
from pathlib import Path

import equinox as eqx
import impl_tolerances as tol
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import survjax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.survival_mnist import model  # noqa: E402
from examples.survival_mnist.data import TRUE_LOG_SCALES, TRUE_SHAPE  # noqa: E402


def observations():
    return survjax.SurvivalData(jnp.array([1, 2, 4], dtype=jnp.float32), [1, 1, 1])


def test_exponential_predictions_have_correct_likelihood_and_risk_direction():
    """03: exact exponential NLL is t/b+log b; smaller b predicts earlier events."""
    row = model.evaluate(0, np.log([1, 2, 4]), observations())
    assert row["n"] == 3 and row["shape"] == 1 and row["concordance"] == 1
    np.testing.assert_allclose(
        row["mean_log_likelihood"],
        -(1 + np.log(2)),
        atol=tol.ATOL[False],
        rtol=tol.RTOL[False],
    )
    assert row["concordance_seconds"] >= 0
    reversed_row = model.evaluate(0, np.log([4, 2, 1]), observations())
    assert reversed_row["concordance"] == 0


def test_constant_intercept_has_half_concordance_and_mixed_likelihood():
    """03: scalar intercept broadcasts; exact/right exponential algebra."""
    data = survjax.SurvivalData(jnp.array([1, 2, 4], dtype=jnp.float32), [1, 0, 1])
    row = model.evaluate(0, np.log(2), data)
    # Sum log likelihood = -sum(t)/b - d*log(b), d=2.
    np.testing.assert_allclose(
        row["mean_log_likelihood"],
        -(3.5 + 2 * np.log(2)) / 3,
        atol=tol.ATOL[False],
        rtol=tol.RTOL[False],
    )
    assert row["concordance"] == 0.5


def test_oracle_uses_digit_scales_and_generating_shape():
    """03: oracle uses k=1.38, log b_d=log(10)-.2*d, cast for evaluation."""
    digits = np.array([9, 4, 0])
    log_b = np.log(10) - 0.2 * digits  # Independent generating rule.
    times = np.array([1, 2, 4])
    expected = (
        np.log(1.38)
        - 1.38 * log_b
        + 0.38 * np.log(times)
        - np.exp(1.38 * (np.log(times) - log_b))
    )
    row = model.evaluate(np.log(TRUE_SHAPE), TRUE_LOG_SCALES[digits], observations())
    np.testing.assert_allclose(
        [row["shape"], row["mean_log_likelihood"]],
        [1.38, expected.mean()],
        atol=tol.ATOL[False],
        rtol=tol.RTOL[False],
    )
    assert row["concordance"] == 1


def test_cnn_predictions_are_evaluated_without_labels():
    """03: actual CNN output feeds the same evaluation as the reference models."""
    cnn = model.CNN(np.log(2), key=jax.random.key(0))
    cnn = eqx.tree_at(lambda m: m.dense2.weight, cnn, jnp.zeros_like(cnn.dense2.weight))
    pixels = jnp.zeros((3, 1, 28, 28), dtype=jnp.float32)
    row = model.evaluate(0, jax.vmap(cnn)(pixels), observations())
    # All three exact; log likelihood=-sum(t)/2-3*log(2).
    np.testing.assert_allclose(
        row["mean_log_likelihood"],
        -7 / 6 - np.log(2),
        atol=tol.ATOL[False],
        rtol=tol.RTOL[False],
    )
    assert row["concordance"] == 0.5


def test_metrics_include_the_final_partial_batch():
    """03: evaluation uses every row, with observation rather than batch weighting."""
    times = jnp.concatenate(
        (jnp.ones(128, dtype=jnp.float32), jnp.array([1001], dtype=jnp.float32))
    )
    row = model.evaluate(0, 0, survjax.SurvivalData(times, jnp.ones(129, dtype=bool)))
    np.testing.assert_allclose(
        row["mean_log_likelihood"],
        -1129 / 129,
        atol=tol.ATOL[False],
        rtol=tol.RTOL[False],
    )
    assert row["n"] == 129 and row["concordance"] == 0.5


def test_float64_predictions_are_cast_before_ranking():
    """03 Precision: distinct float64 scales round to exact float32 score ties."""
    row = model.evaluate(
        np.float64(0),
        np.array([1, 1 + 1e-9, 1 + 2e-9], dtype=np.float64),
        observations(),
    )
    assert row["concordance"] == 0.5


@pytest.mark.parametrize("log_k,log_b", [([0], 0), (0, [0, 1]), (0, [[0, 1, 2]])])
def test_parameter_shapes_are_rejected(log_k, log_b):
    """03: shared shape is scalar; scales are scalar or aligned per image."""
    with pytest.raises(ValueError, match="scalar"):
        model.evaluate(log_k, log_b, observations())


@pytest.mark.parametrize("log_k,log_b", [(np.nan, 0), (0, [0, np.inf, 0])])
def test_nonfinite_predictions_fail_loudly(log_k, log_b):
    """03 acceptance: investigate failed metrics instead of writing plausible results."""
    with pytest.raises(FloatingPointError, match="nonfinite"):
        model.evaluate(log_k, log_b, observations())


def test_undefined_concordance_fails_instead_of_reporting_half():
    """Spec 9.1: an all-censored cohort has no comparable pairs, even for intercept."""
    data = survjax.SurvivalData(jnp.array([1, 2], dtype=jnp.float32), [0, 0])
    with pytest.raises(FloatingPointError, match="undefined"):
        model.evaluate(0, 0, data)


def test_invalid_and_empty_observations_are_rejected():
    """03: no dropping invalid rows or fabricating empty-cohort metrics."""
    data = survjax.SurvivalData(jnp.array([-1, 2], dtype=jnp.float32), [1, 0])
    with pytest.raises(ValueError):
        model.evaluate(0, 0, data)
    with pytest.raises(ValueError, match="nonempty"):
        model.evaluate(0, 0, observations()[:0])


def test_float64_observations_are_rejected(x64_enabled):
    """03 Precision: evaluation must not promote likelihood/C to float64."""
    if not x64_enabled:
        pytest.skip("float64 observations require x64 mode")
    data = survjax.SurvivalData(jnp.array([1, 2], dtype=jnp.float64), [1, 1])
    with pytest.raises(ValueError, match="float32"):
        model.evaluate(0, 0, data)
