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
    training_started = perf_counter()
    result = model.train(
        pixels[training],
        observations[training],
        pixels[validation],
        observations[validation],
        initial_intercept=np.array(prepared["intercept"], dtype=np.float64),
        config=config,
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
        "training": {
            "epochs": len(result.training_nll),
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
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
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
