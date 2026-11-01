"""CPU survival MNIST runner; launch with python -m examples.survival_mnist.run."""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from time import perf_counter


def _prepare(config: str, destination: Path) -> None:
    from examples.survival_mnist.data import (
        calibrate_mnist,
        generate_outcomes,
        load_mnist,
        split_indices,
    )
    from examples.survival_mnist.model import fit_intercept

    _, labels = load_mnist()
    start = perf_counter()
    c = calibrate_mnist(labels[:60000])
    outcomes = generate_outcomes(labels, c)
    indices = split_indices(config)["train"]
    fitted = fit_intercept(outcomes.time[indices], outcomes.event[indices])
    destination.write_text(
        json.dumps(
            {
                "censor_limit": c,
                "intercept": fitted.tolist(),
                "preparation_seconds": perf_counter() - start,
            }
        )
    )


def _figures(pixels, labels, log_k, log_scales, result, out: Path) -> None:
    """Write the three use-case03 SVGs from official-order selected test predictions."""
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import survjax
    from examples.survival_mnist.data import TRUE_LOG_SCALES, TRUE_SHAPE

    labels, log_scales = np.asarray(labels), np.asarray(log_scales)
    matches = [np.flatnonzero(labels == digit) for digit in range(10)]
    if any(len(indices) == 0 for indices in matches):
        raise ValueError("figures require a test image for every digit")
    first = np.array([indices[0] for indices in matches])
    times = jnp.linspace(0, 20, 201, dtype=jnp.float32)
    predicted = survjax.Weibull.from_log(
        log_shape=jnp.asarray(log_k, dtype=jnp.float32),
        log_scale=jnp.asarray(log_scales[first], dtype=jnp.float32),
    ).survival_grid(times)
    truth = survjax.Weibull.from_log(
        log_shape=jnp.log(jnp.asarray(TRUE_SHAPE, dtype=jnp.float32)),
        log_scale=jnp.asarray(TRUE_LOG_SCALES, dtype=jnp.float32),
    ).survival_grid(times)
    fig = plt.figure(figsize=(16, 6), layout="constrained")
    panels = fig.add_gridspec(2, 5)
    for digit in range(10):
        pair = panels[digit // 5, digit % 5].subgridspec(1, 2, width_ratios=(1, 2))
        image, curve = fig.add_subplot(pair[0]), fig.add_subplot(pair[1])
        image.imshow(pixels[first[digit], 0], cmap="gray", vmin=0, vmax=1)
        image.set_title(f"Digit {digit}")
        image.set_axis_off()
        curve.plot(times, predicted[:, digit], label="CNN")
        curve.plot(times, truth[:, digit], "--", label="True")
        curve.set(xlabel="Time", ylabel="Survival", xlim=(0, 20), ylim=(0, 1))
        if digit == 0:
            curve.legend(fontsize=8)
    fig.suptitle("First test image per digit: predicted and true survival")
    fig.savefig(out / "fig1.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.boxplot(
        [log_scales[labels == digit] for digit in range(10)], tick_labels=range(10)
    )
    ax.scatter(range(1, 11), TRUE_LOG_SCALES, marker="D", color="tab:red", label="True")
    ax.set(
        xlabel="True digit", ylabel="Log scale", title="Predicted log scale by digit"
    )
    ax.legend()
    fig.savefig(out / "fig2.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    epochs = np.arange(1, len(result.training_nll) + 1)
    selected = result.selected_epoch
    dense = bool(result.batch_nll and result.validation_progress)
    if dense:
        ax.plot(
            np.arange(len(result.batch_nll)),
            result.batch_nll,
            label="Training minibatch",
        )
        steps, values = zip(*result.validation_progress, strict=True)
        ax.plot(steps, values, "o-", label="Full validation")
        selected *= len(result.batch_nll) // len(epochs)
    else:
        ax.plot(epochs, result.training_nll, "o-", label="Training")
        ax.plot(epochs, result.validation_nll, "s-", label="Validation")
    ax.axvline(selected, linestyle="--", color="black", label="Selected epoch")
    ax.set(
        xlabel="Optimizer updates" if dense else "Epoch",
        ylabel="Negative log likelihood",
        title="Training convergence",
    )
    ax.legend()
    fig.savefig(out / "fig3.svg")
    plt.close(fig)


def _run(config: str, preparation: Path, out: Path) -> None:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    import survjax
    from examples.survival_mnist import data, model

    if jax.config.x64_enabled:
        raise RuntimeError("training/evaluation require a separate float32 process")
    prepared = json.loads(preparation.read_text())
    pixels, labels = data.load_mnist()
    cohorts = data.split_indices(config)
    # Reproduce preparation outcomes exactly: same labels/c/global keys and
    # explicit float32 generator arithmetic, independent of process x64 mode.
    outcomes = data.generate_outcomes(labels, prepared["censor_limit"])
    observations = survjax.SurvivalData(outcomes.time, outcomes.event)
    training, validation, test = (
        cohorts[name] for name in ("train", "validation", "test")
    )
    log_every_batches = max(1, (len(training) // 128 + 9) // 10)
    training_started = perf_counter()
    result = model.train(
        pixels[training],
        observations[training],
        pixels[validation],
        observations[validation],
        initial_intercept=np.array(prepared["intercept"], dtype=np.float64),
        config=config,
        log_every_batches=log_every_batches,
    )
    training_seconds = perf_counter() - training_started
    prediction_started = perf_counter()
    cnn, log_k = result.params
    predict = eqx.filter_jit(eqx.filter_vmap(cnn))
    log_scales = jnp.concatenate(
        [
            predict(pixels[test[start : start + 128]])
            for start in range(0, len(test), 128)
        ]
    )
    log_scales.block_until_ready()
    prediction_seconds = perf_counter() - prediction_started
    # 03: only the oracle accesses test digit labels, after checkpoint selection.
    oracle_scales = data.TRUE_LOG_SCALES[labels[test]]
    rows = []
    for name, shape, scales in (
        ("cnn", log_k, log_scales),
        ("intercept", prepared["intercept"][0], prepared["intercept"][1]),
        ("oracle", np.log(data.TRUE_SHAPE), oracle_scales),
    ):
        rows.append(
            {"model": name, **model.evaluate(shape, scales, observations[test])}
        )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "training_nll", "validation_nll"))
        writer.writerows(
            (epoch, train, valid)
            for epoch, (train, valid) in enumerate(
                zip(result.training_nll, result.validation_nll, strict=True), 1
            )
        )
    with (out / "progress.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "training_batch_nll", "validation_nll"))
        validation_progress = dict(result.validation_progress)
        writer.writerows(
            (
                step,
                result.batch_nll[step] if step < len(result.batch_nll) else "",
                validation_progress.get(step, ""),
            )
            for step in range(len(result.batch_nll) + 1)
        )
    metadata = {
        "config": config,
        "censor_limit": prepared["censor_limit"],
        "intercept_log_parameters": prepared["intercept"],
        "oracle": {
            "shape": float(data.TRUE_SHAPE),
            "log_scales_by_digit": data.TRUE_LOG_SCALES.tolist(),
        },
        "sizes": {name: len(indices) for name, indices in cohorts.items()},
        "realized_censored": {
            name: float(np.mean(~np.asarray(outcomes.event[indices])))
            for name, indices in cohorts.items()
        },
        "selected_epoch": result.selected_epoch,
        "selected_epoch_indexing": "one-based",
        "training": {
            "epochs": len(result.training_nll),
            "log_every_batches": log_every_batches,
            "batch_size": 128,
            "drop_partial_batch": True,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "architecture": "Conv(1,16,3)-ReLU-Pool2/2-Conv(16,32,3)-ReLU-Pool2/2-Linear(800,64)-ReLU-Linear(64,1)",
        },
        "root_seed": 20261025,
        "keys": {
            "uc": "fold_in(key(root_seed),3)",
            "validation": "fold_in(uc,1)",
            "outcomes": "fold_in(fold_in(uc,0),global_index)",
            "initialization": "fold_in(uc,2)",
            "shuffle": "fold_in(fold_in(uc,3),epoch)",
            "epoch_indexing": "zero-based",
        },
        "precision": {
            "calibration_oracle_fit": "float64",
            "outcomes_training_evaluation": "float32",
        },
        "timings_seconds": {
            "preparation": prepared["preparation_seconds"],
            "training": training_seconds,
            "prediction": prediction_seconds,
            "concordance": {row["model"]: row["concordance_seconds"] for row in rows},
        },
        "timing_notes": {
            "preparation": "calibration, outcomes, fit; excludes data loading",
            "training": "includes initialization, compilation and validation",
            "prediction": "selected CNN, all test rows, includes compilation; synchronized",
            "concordance": "per model, full test cohort; includes possible first-call compilation; synchronized",
        },
        "versions": {
            name: version(name)
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
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "devices": [str(device) for device in jax.devices()],
        },
        "source_sha256": {name: hashes[1] for name, hashes in data.RESOURCES.items()},
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    _figures(pixels[test], labels[test], log_k, log_scales, result, out)
    print(
        json.dumps(
            {
                **prepared,
                "out": str(out),
                "selected_epoch": result.selected_epoch,
                "metrics": rows,
                "training_seconds": training_seconds,
            }
        ),
        flush=True,
    )


def main() -> None:
    """Run the fixed smoke/full configuration through isolated precision processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1", MPLBACKEND="Agg")
    with tempfile.TemporaryDirectory(prefix="survjax-mnist-") as temporary:
        preparation = str(Path(temporary) / "preparation.json")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; from examples.survival_mnist.run import _prepare; _prepare(sys.argv[1], Path(sys.argv[2]))",
                args.config,
                preparation,
            ],
            env=env,
            check=True,
        )
        env = dict(env, JAX_ENABLE_X64="0")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; from examples.survival_mnist.run import _run; _run(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))",
                args.config,
                preparation,
                str(args.out),
            ],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
