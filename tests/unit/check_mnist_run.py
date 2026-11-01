"""Offline optional CLI/process and cohort-wiring checks for use-case 03."""

import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
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
        assert env["MPLBACKEND"] == "Agg"
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
        time=jnp.arange(1, 70001, dtype=jnp.float32), event=jnp.arange(70000) % 2 == 0
    )
    monkeypatch.setattr(data, "load_mnist", lambda: (pixels, labels))
    monkeypatch.setattr(data, "split_indices", lambda config: indices)

    def generate(actual, c):
        np.testing.assert_array_equal(actual, labels)
        assert c == 7.0
        return outcomes

    monkeypatch.setattr(data, "generate_outcomes", generate)
    return labels, indices


def test_preparation_calibrates_source_and_fits_training_only(monkeypatch, tmp_path):
    """03: all60k source digits calibrate; only selected training outcomes fit."""
    labels, indices = fake_cohorts(monkeypatch)
    clock = [100.0]
    load = data.load_mnist

    def load_with_cost():
        clock[0] += 50.0
        return load()

    monkeypatch.setattr(data, "load_mnist", load_with_cost)
    monkeypatch.setattr(run, "perf_counter", lambda: clock[0])

    def calibrate(actual):
        np.testing.assert_array_equal(actual, labels[:60000])
        clock[0] += 2.0
        return 7.0

    def fit(time, event):
        np.testing.assert_array_equal(time, indices["train"] + 1)
        np.testing.assert_array_equal(event, indices["train"] % 2 == 0)
        clock[0] += 3.0
        return np.array([0.25, 1.5], dtype=np.float64)

    monkeypatch.setattr(data, "calibrate_mnist", calibrate)
    monkeypatch.setattr(model, "fit_intercept", fit)
    run._prepare("smoke", tmp_path / "prepared.json")
    prepared = json.loads((tmp_path / "prepared.json").read_text())
    assert prepared["censor_limit"] == 7 and prepared["intercept"] == [0.25, 1.5]
    # Only calibration2s+fit3s count; the50s load must be excluded.
    assert prepared["preparation_seconds"] == 5.0


def test_runner_uses_selected_predictions_tail_and_records(
    monkeypatch, tmp_path, x64_enabled
):
    """03: fixed cohorts, selected checkpoint,129 test rows in order, three records."""
    if x64_enabled:
        with pytest.raises(RuntimeError, match="float32"):
            run._run("smoke", tmp_path / "missing", tmp_path)
        return
    labels, indices = fake_cohorts(monkeypatch)
    ticks = iter([10.0, 14.0, 20.0, 23.0])  # Independent wall-clock boundary values.
    monkeypatch.setattr(run, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(run, "version", lambda name: f"{name}-test")
    monkeypatch.setattr(run.platform, "python_version", lambda: "python-test")
    monkeypatch.setattr(run.platform, "platform", lambda: "platform-test")
    monkeypatch.setattr(run.platform, "processor", lambda: "cpu-test")
    monkeypatch.setattr(run.os, "cpu_count", lambda: 8)
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
        expected_shapes = (np.float32(0.3), 0.25, np.log(1.38))
        assert float(shape) == float(expected_shapes[len(predictions)])
        predictions.append(np.asarray(scales))
        return {
            "n": 129,
            "mean_log_likelihood": -2.0,
            "shape": 1.38,
            "concordance": 0.6,
            "concordance_seconds": 0.1,
        }

    figure_calls = []

    def figures(pixels, actual_labels, shape, scales, result, out):
        figure_calls.append(out)
        np.testing.assert_array_equal(pixels[:, 0, 0, 0], indices["test"])
        np.testing.assert_array_equal(actual_labels, labels[indices["test"]])
        assert float(shape) == float(np.float32(0.3)) and result.selected_epoch == 1
        np.testing.assert_array_equal(scales, indices["test"])
        assert result.training_nll == (2.0,) and result.validation_nll == (3.0,)

    monkeypatch.setattr(run, "_figures", figures)
    monkeypatch.setattr(model, "train", train)
    monkeypatch.setattr(model, "evaluate", evaluate)
    run._run("smoke", prepared, tmp_path)
    assert figure_calls == [tmp_path]
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

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["config"] == "smoke"
    assert metadata["censor_limit"] == 7
    assert metadata["intercept_log_parameters"] == [0.25, 1.5]
    assert metadata["oracle"]["shape"] == 1.38
    # Host float64 generating rule, retained before any evaluation cast.
    np.testing.assert_array_equal(
        metadata["oracle"]["log_scales_by_digit"], np.log(10.0) - 0.2 * np.arange(10)
    )
    assert metadata["sizes"] == {"train": 128, "validation": 2, "test": 129}
    # Alternating event flags, starting even; test has65 events and64 censored.
    assert metadata["realized_censored"] == {
        "train": 0.5,
        "validation": 0.5,
        "test": 64 / 129,
    }
    assert metadata["selected_epoch"] == 1
    assert metadata["selected_epoch_indexing"] == "one-based"
    assert metadata["training"] == {
        "epochs": 1,
        "batch_size": 128,
        "drop_partial_batch": True,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "architecture": "Conv(1,16,3)-ReLU-Pool2/2-Conv(16,32,3)-ReLU-Pool2/2-Linear(800,64)-ReLU-Linear(64,1)",
    }
    assert metadata["root_seed"] == 20261025
    assert metadata["keys"] == {
        "uc": "fold_in(key(root_seed),3)",
        "validation": "fold_in(uc,1)",
        "outcomes": "fold_in(fold_in(uc,0),global_index)",
        "initialization": "fold_in(uc,2)",
        "shuffle": "fold_in(fold_in(uc,3),epoch)",
        "epoch_indexing": "zero-based",
    }
    assert metadata["precision"] == {
        "calibration_oracle_fit": "float64",
        "outcomes_training_evaluation": "float32",
    }
    assert metadata["timings_seconds"] == {
        "preparation": 1,
        "training": 4,
        "prediction": 3,
        "concordance": {"cnn": 0.1, "intercept": 0.1, "oracle": 0.1},
    }
    assert "excludes data loading" in metadata["timing_notes"]["preparation"]
    assert "synchronized" in metadata["timing_notes"]["prediction"]
    assert "compilation" in metadata["timing_notes"]["concordance"]
    assert metadata["versions"] == {
        name: f"{name}-test"
        for name in (
            "survjax",
            "jax",
            "jaxlib",
            "numpy",
            "equinox",
            "optax",
            "optimistix",
            "matplotlib",
        )
    }
    assert metadata["environment"] == {
        "python": "python-test",
        "platform": "platform-test",
        "cpu": "cpu-test",
        "cpu_count": 8,
        "devices": [str(device) for device in jax.devices()],
    }
    assert metadata["source_sha256"] == {
        name: hashes[1] for name, hashes in data.RESOURCES.items()
    }
