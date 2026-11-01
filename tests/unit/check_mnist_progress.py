"""Within-epoch logging requested by the maintainer; selection stays per epoch."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import survjax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.survival_mnist import model  # noqa: E402


def train(interval):
    images = jnp.zeros((385, 1, 28, 28), dtype=jnp.float32)
    times = jnp.linspace(0.5, 3, 385, dtype=jnp.float32)
    data = survjax.SurvivalData(times, jnp.arange(385) % 2 == 0)
    return model.train(
        images,
        data,
        images[:2],
        data[:2],
        initial_intercept=np.array([0.2, 1.5]),
        log_every_batches=interval,
    )


def test_extra_validation_does_not_change_real_adam_updates_or_selection():
    """Logging is observational: identical keys, parameters and epoch summaries."""
    sparse, dense = train(None), train(2)
    assert sparse.selected_epoch == dense.selected_epoch == 1
    assert sparse.training_nll == dense.training_nll
    assert sparse.validation_nll == dense.validation_nll
    assert sparse.batch_nll == dense.batch_nll and len(dense.batch_nll) == 3
    assert [step for step, _ in dense.validation_progress] == [0, 2, 3]
    assert dense.validation_progress[-1][1] == dense.validation_nll[-1]
    for before, after in zip(
        jax.tree.leaves(sparse.params), jax.tree.leaves(dense.params), strict=True
    ):
        np.testing.assert_array_equal(before, after)


def test_intermediate_validation_never_selects_a_checkpoint(monkeypatch):
    """An intermediate best is a diagnostic, not a new selection opportunity."""
    snapshots = []
    values = iter([4.0, 0.1, 3.0])  # Initial, after update2, and epoch-end update3.

    def step(params, state, pixels, data):
        updated = (params[0], params[1] + jnp.float32(0.01))
        snapshots.append(updated)
        return updated, state, jnp.float32(len(snapshots))

    monkeypatch.setattr(model, "_train_step", step)
    monkeypatch.setattr(model, "_validation_nll", lambda *args: next(values))
    result = train(2)
    assert result.batch_nll == (1.0, 2.0, 3.0)
    assert result.training_nll == (2.0,)  # Hand mean of the three minibatches.
    assert result.validation_progress == ((0, 4.0), (2, 0.1), (3, 3.0))
    assert result.validation_nll == (3.0,)
    assert result.params is snapshots[-1]


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_invalid_logging_intervals_are_rejected(interval):
    """Logging interval must count batches, not booleans or fractions."""
    with pytest.raises(ValueError, match="log_every_batches"):
        train(interval)


@pytest.mark.parametrize("bad_at", [0, 1])
def test_nonfinite_initial_or_intermediate_validation_aborts(monkeypatch, bad_at):
    """Failed diagnostics cannot be plotted as finite convergence evidence."""
    values = iter([np.nan if bad_at == 0 else 4.0, np.nan])
    monkeypatch.setattr(model, "_validation_nll", lambda *args: next(values))
    with pytest.raises(FloatingPointError, match="validation NLL"):
        train(2)
