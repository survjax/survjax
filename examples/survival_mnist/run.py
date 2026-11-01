"""Train a CNN with a survjax likelihood on synthetic MNIST survival times."""

import argparse
import csv
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import survjax
from examples.survival_mnist import data, model


def _figures(pixels, labels, log_k, log_scales, result, out: Path) -> None:
    """Plot predicted survival, digit scales and training progress."""
    labels, log_scales = np.asarray(labels), np.asarray(log_scales)
    matches = [np.flatnonzero(labels == digit) for digit in range(10)]
    first = np.array([indices[0] for indices in matches])
    times = jnp.linspace(0, 20, 201, dtype=jnp.float32)
    predicted = survjax.Weibull.from_log(
        log_shape=log_k,
        log_scale=log_scales[first],
    ).survival_grid(times)
    truth = survjax.Weibull.from_log(
        log_shape=jnp.log(data.TRUE_SHAPE),
        log_scale=data.TRUE_LOG_SCALES,
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
    ax.scatter(
        range(1, 11), data.TRUE_LOG_SCALES, marker="D", color="tab:red", label="True"
    )
    ax.set(
        xlabel="True digit", ylabel="Log scale", title="Predicted log scale by digit"
    )
    ax.legend()
    fig.savefig(out / "fig2.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    epochs = np.arange(1, len(result.training_nll) + 1)
    selected = result.selected_epoch
    ax.plot(epochs, result.training_nll, "o-", label="Training")
    ax.plot(epochs, result.validation_nll, "s-", label="Validation")
    ax.axvline(selected, linestyle="--", color="black", label="Selected epoch")
    ax.set(
        xlabel="Epoch",
        ylabel="Negative log likelihood",
        title="Training convergence",
    )
    ax.legend()
    fig.savefig(out / "fig3.svg")
    plt.close(fig)


def main() -> None:
    """Load images, train, evaluate and plot the fixed smoke or full experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pixels, labels = data.load_mnist()
    outcomes = data.generate_outcomes(labels)
    observations = survjax.SurvivalData(outcomes.time, outcomes.event)
    cohorts = data.split_indices(args.config)
    training, validation, test = (
        cohorts[name] for name in ("train", "validation", "test")
    )
    intercept = model.fit_intercept(outcomes.time[training], outcomes.event[training])
    started = perf_counter()
    result = model.train(
        pixels[training],
        observations[training],
        pixels[validation],
        observations[validation],
        initial_intercept=intercept,
        epochs=1 if args.config == "smoke" else 10,
    )
    training_seconds = perf_counter() - started
    cnn, log_k = result.params
    log_scales = model.predict(cnn, pixels[test])
    rows = [
        dict(model=name, **model.evaluate(shape, scales, observations[test]))
        for name, shape, scales in (
            ("cnn", log_k, log_scales),
            ("intercept", *intercept),
            ("oracle", jnp.log(data.TRUE_SHAPE), data.TRUE_LOG_SCALES[labels[test]]),
        )
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("epoch", "training_nll", "validation_nll"))
        writer.writerows(
            (i, train, valid)
            for i, (train, valid) in enumerate(
                zip(result.training_nll, result.validation_nll, strict=True), 1
            )
        )
    metadata = {
        "config": args.config,
        "sizes": {name: len(idx) for name, idx in cohorts.items()},
        "realized_censored": {
            name: float(np.mean(~np.asarray(outcomes.event[idx])))
            for name, idx in cohorts.items()
        },
        "selected_epoch": result.selected_epoch,
        "training_seconds": training_seconds,
        "concordance_seconds": {
            row["model"]: row["concordance_seconds"] for row in rows
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
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _figures(pixels[test], labels[test], log_k, log_scales, result, args.out)
    print(json.dumps(dict(metrics=rows, **metadata)), flush=True)


if __name__ == "__main__":
    main()
