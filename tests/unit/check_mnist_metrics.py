"""Optional offline evaluation checks for use-case 03 References and metrics."""

import sys
from pathlib import Path

import equinox as eqx
import impl_tolerances as tol
import jax
import jax.numpy as jnp
import numpy as np

import survjax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.survival_mnist import model, run  # noqa: E402
from examples.survival_mnist.data import TRUE_LOG_SCALES, TRUE_SHAPE  # noqa: E402


def observations():
    return survjax.SurvivalData(jnp.array([1, 2, 4], dtype=jnp.float32), [1, 1, 1])


def test_exponential_predictions_have_correct_likelihood_and_risk_direction():
    """03: exact exponential NLL is t/b+log b; smaller b predicts earlier events."""
    row = model.evaluate(0, np.log([1, 2, 4]), observations())
    assert row["shape"] == 1 and row["concordance"] == 1
    np.testing.assert_allclose(
        row["mean_log_likelihood"],
        -(1 + np.log(2)),
        atol=tol.ATOL[False],
        rtol=tol.RTOL[False],
    )
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


def test_constant_cnn_vector_output_reaches_evaluation():
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
    assert row["concordance"] == 0.5


def test_predict_keeps_the_final_partial_chunk():
    """03: chunked prediction includes all images and matches a constant head."""
    cnn = model.CNN(np.log(2), key=jax.random.key(0))
    cnn = eqx.tree_at(lambda m: m.dense2.weight, cnn, jnp.zeros_like(cnn.dense2.weight))
    predictions = model.predict(cnn, jnp.zeros((1001, 1, 28, 28), dtype=jnp.float32))
    assert predictions.shape == (1001,)
    np.testing.assert_allclose(
        predictions, np.log(2), atol=tol.ATOL[False], rtol=tol.RTOL[False]
    )


def test_concordance_timing_excludes_the_compile_call(monkeypatch):
    """03: time one synchronized call after warming the public C callable."""
    clock, completed = [0.0], []
    costs = iter([11.0, 2.0])

    class DeferredScalar:
        def __float__(self):
            cost = next(costs)
            clock[0] += cost
            completed.append(cost)
            return 0.5  # Constant scores give half credit to every comparable pair.

    monkeypatch.setattr(run, "_concordance", lambda *args: DeferredScalar())
    monkeypatch.setattr(run, "perf_counter", lambda: clock[0])
    assert run._concordance_seconds(1.5, observations()) == 2.0
    assert completed == [11.0, 2.0]
