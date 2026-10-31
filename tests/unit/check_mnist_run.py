"""Offline optional CLI/process and cohort-wiring checks for use-case 03."""

import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.survival_mnist import data, model, run  # noqa: E402


@pytest.mark.parametrize("fail", [False, True])
def test_cli_isolates_precision_and_stops_on_preparation_failure(
    monkeypatch, tmp_path, fail
):
    """03 Precision: CPU children start x64 then float32; failed fit never trains."""
    monkeypatch.setattr(
        sys, "argv", ["run", "--config", "smoke", "--out", str(tmp_path)]
    )
    calls = []

    def child(command, *, env, check):
        calls.append((command, env))
        assert check and env["JAX_PLATFORMS"] == "cpu"
        if fail:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(run.subprocess, "run", child)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            run.main()
        assert len(calls) == 1
    else:
        run.main()
        assert [env["JAX_ENABLE_X64"] for _, env in calls] == ["1", "0"]
        assert "_prepare" in calls[0][0][2] and "_run" in calls[1][0][2]
        assert calls[0][0][3] == calls[1][0][3] == "smoke"
        assert calls[0][0][4] == calls[1][0][4]
        assert calls[1][0][-1] == str(tmp_path)


def fake_cohorts(monkeypatch):
    labels = np.arange(70000, dtype=np.int64) % 10
    pixels = np.broadcast_to(
        np.arange(70000, dtype=np.float32)[:, None, None, None], (70000, 1, 28, 28)
    )
    indices = {
        "train": np.arange(100, 228),
        "validation": np.arange(5, 7),
        "test": np.arange(60000, 60129),
    }
    outcomes = SimpleNamespace(
        time=jnp.arange(1, 70001, dtype=jnp.float32), event=jnp.ones(70000, dtype=bool)
    )
    monkeypatch.setattr(data, "load_mnist", lambda: (pixels, labels))
    monkeypatch.setattr(data, "split_indices", lambda config: indices)
    monkeypatch.setattr(data, "generate_outcomes", lambda actual, c: outcomes)
    return labels, indices


def test_preparation_calibrates_source_and_fits_training_only(monkeypatch, tmp_path):
    """03: all60k source digits calibrate; only selected training outcomes fit."""
    labels, indices = fake_cohorts(monkeypatch)

    def calibrate(actual):
        np.testing.assert_array_equal(actual, labels[:60000])
        return 7.0

    def fit(time, event):
        np.testing.assert_array_equal(time, indices["train"] + 1)
        assert event.all()
        return np.array([0.25, 1.5], dtype=np.float64)

    monkeypatch.setattr(data, "calibrate_mnist", calibrate)
    monkeypatch.setattr(model, "fit_intercept", fit)
    run._prepare("smoke", tmp_path / "prepared.json")
    prepared = json.loads((tmp_path / "prepared.json").read_text())
    assert prepared["censor_limit"] == 7 and prepared["intercept"] == [0.25, 1.5]


def test_runner_uses_selected_predictions_tail_and_records(
    monkeypatch, tmp_path, x64_enabled
):
    """03: fixed cohorts, selected checkpoint,129 test rows in order, three records."""
    if x64_enabled:
        with pytest.raises(RuntimeError, match="float32"):
            run._run("smoke", tmp_path / "missing", tmp_path)
        return
    labels, indices = fake_cohorts(monkeypatch)
    prepared = tmp_path / "prepared.json"
    prepared.write_text(
        json.dumps(
            {"censor_limit": 7, "intercept": [0.25, 1.5], "preparation_seconds": 1}
        )
    )

    def train(
        pixels,
        observations,
        validation_pixels,
        validation,
        *,
        initial_intercept,
        config,
    ):
        np.testing.assert_array_equal(pixels[:, 0, 0, 0], indices["train"])
        np.testing.assert_array_equal(observations.lower, indices["train"] + 1)
        np.testing.assert_array_equal(validation.lower, indices["validation"] + 1)
        np.testing.assert_array_equal(
            validation_pixels[:, 0, 0, 0], indices["validation"]
        )
        np.testing.assert_array_equal(initial_intercept, [0.25, 1.5])
        assert initial_intercept.dtype == np.float64 and config == "smoke"
        return model.TrainingResult(
            (lambda x: x[0, 0, 0], jnp.float32(0.3)), (2.0,), (3.0,), 1
        )

    predictions = []

    def evaluate(shape, scales, observations):
        np.testing.assert_array_equal(observations.lower, indices["test"] + 1)
        predictions.append(np.asarray(scales))
        return {
            "n": 129,
            "mean_log_likelihood": -2.0,
            "shape": 1.38,
            "concordance": 0.6,
            "concordance_seconds": 0.1,
        }

    monkeypatch.setattr(model, "train", train)
    monkeypatch.setattr(model, "evaluate", evaluate)
    run._run("smoke", prepared, tmp_path)
    np.testing.assert_array_equal(predictions[0], indices["test"])
    assert predictions[1] == 1.5
    np.testing.assert_array_equal(
        predictions[2], data.TRUE_LOG_SCALES[labels[indices["test"]]]
    )
    with (tmp_path / "metrics.csv").open() as stream:
        assert [row["model"] for row in csv.DictReader(stream)] == [
            "cnn",
            "intercept",
            "oracle",
        ]
    assert (tmp_path / "history.csv").read_text().splitlines() == [
        "epoch,training_nll,validation_nll",
        "1,2.0,3.0",
    ]
