"""Verified MNIST pixels and fixed global indices (use-case 03 details)."""

import gzip
import hashlib
import os
import struct
import tempfile
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from examples.weibull_simulation.study import (
    GeneratedData,
    calibrate_censoring,
    generate_weibull,
)

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
# torchvision/datasets/mnist.py publishes these MD5s. SHA-256s measured after
# verifying all four first downloads against those MD5s on 2026-10-30.
RESOURCES = {
    "train-images-idx3-ubyte.gz": (
        "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    ),
    "train-labels-idx1-ubyte.gz": (
        "d53e105ee54ea40749a09fcbcd1e9432",
        "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    ),
    "t10k-images-idx3-ubyte.gz": (
        "9fb629c4189551a2d022fa330f9573f3",
        "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    ),
    "t10k-labels-idx1-ubyte.gz": (
        "ec29112dd5afa0611ce80d1b7f02629c",
        "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
    ),
}


def _fetch(name: str, cache: Path) -> bytes:
    path = cache / name
    cached = path.exists()
    if cached:
        payload = path.read_bytes()
    else:
        with urllib.request.urlopen(MIRROR + name, timeout=60) as response:
            payload = response.read()
    md5, sha256 = RESOURCES[name]
    if hashlib.sha256(payload).hexdigest() != sha256 or (
        not cached and hashlib.md5(payload).hexdigest() != md5
    ):
        action = "remove it and retry" if cached else "download rejected; retry"
        raise ValueError(f"MNIST checksum mismatch: {path}; {action}")
    if not cached:
        cache.mkdir(parents=True, exist_ok=True)
        # A failed/interrupted transfer must never become a trusted cache entry.
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temp:
            temporary = Path(temp.name)
            try:
                temp.write(payload)
                temp.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    return payload


def _parse_idx(payload: bytes, *, images: bool, count: int) -> np.ndarray:
    raw = gzip.decompress(payload)
    dimensions = (count, 28, 28) if images else (count,)
    header = struct.pack(
        ">" + "I" * (len(dimensions) + 1), 2051 if images else 2049, *dimensions
    )
    if not raw.startswith(header) or len(raw) != len(header) + int(np.prod(dimensions)):
        raise ValueError("MNIST IDX header, dimensions or payload length is invalid")
    values = np.frombuffer(raw, dtype=np.uint8, offset=len(header)).reshape(dimensions)
    if not images and np.any(values > 9):
        raise ValueError("MNIST labels must be digits 0 through 9")
    return values


def load_mnist(cache: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load images and oracle labels in official global-index order.

    Args:
        cache: Cache directory; defaults to SURVJAX_EXAMPLES_DATA, otherwise
            ~/.cache/survjax-examples/mnist. Missing files are downloaded.
    Returns:
        Float32 [0, 1] images (70000, 1, 28, 28), uint8 labels (70000,).
        Training-source rows precede official test rows; labels are for
        outcome generation and oracle evaluation only.
    Raises:
        ValueError: A checksum, IDX format or digit label is invalid.
        OSError: Download, decompression or filesystem access fails.
    """
    if cache is None:
        cache = Path(
            os.environ.get("SURVJAX_EXAMPLES_DATA", "~/.cache/survjax-examples/mnist")
        )
    cache = cache.expanduser()
    images, labels = [], []
    for prefix, count in (("train", 60000), ("t10k", 10000)):
        images.append(
            _parse_idx(
                _fetch(f"{prefix}-images-idx3-ubyte.gz", cache),
                images=True,
                count=count,
            )
        )
        labels.append(
            _parse_idx(
                _fetch(f"{prefix}-labels-idx1-ubyte.gz", cache),
                images=False,
                count=count,
            )
        )
    pixels = np.concatenate(images).astype(np.float32)[:, None, :, :] / np.float32(255)
    return pixels, np.concatenate(labels)


def split_indices(config: str = "full") -> dict[str, np.ndarray]:
    """Return global indices for fixed train/validation/test cohorts.

    Args:
        config: 'full' (50k/10k/10k) or 'smoke' (5k/1k/2k prefixes).
    Returns:
        Mapping of cohort names to indices into load_mnist's arrays.
    Raises:
        ValueError: config is neither 'full' nor 'smoke'.
    """
    if config not in ("full", "smoke"):
        raise ValueError("config must be 'full' or 'smoke'")
    uc = jax.random.fold_in(jax.random.key(20261025), 3)
    permutation = np.asarray(
        jax.random.permutation(jax.random.fold_in(uc, 1), 60000), dtype=np.int64
    )
    cohorts = {
        "train": permutation[10000:],
        "validation": permutation[:10000],
        "test": np.arange(60000, 70000),
    }
    if config == "smoke":
        cohorts = {
            name: indices[:size]
            for (name, indices), size in zip(
                cohorts.items(), (5000, 1000, 2000), strict=True
            )
        }
    return cohorts


TRUE_SHAPE = np.float64(1.38)
# 03 Precision: compute oracle parameters in host float64, cast for evaluation.
TRUE_LOG_SCALES = np.log(np.float64(10)) - np.float64(0.2) * np.arange(10)


def calibrate_mnist(training_labels: np.ndarray) -> float:
    """Calibrate 50% expected censoring from the 60k training-source digits.

    Args:
        training_labels: Integer digits (60000,), before train/validation split.
            Official test labels and smoke subsets must never be supplied.
    Returns:
        Uniform censoring limit c, computed by the shared host float64 solver.
    Raises:
        ValueError: Labels are not exactly 60k integer digits 0 through 9.
        RuntimeError: Shared calibration does not converge.
    """
    labels = np.asarray(training_labels)
    if (
        labels.shape != (60000,)
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any((labels < 0) | (labels > 9))
    ):
        raise ValueError("calibration requires the 60000 training-source digit labels")
    scales = np.exp(TRUE_LOG_SCALES[labels])
    return calibrate_censoring(TRUE_SHAPE, scales, 0.5)


def generate_outcomes(
    labels: ArrayLike,
    censor_limit: ArrayLike,
    *,
    global_indices: ArrayLike | None = None,
) -> GeneratedData:
    """Generate fixed float32 outcomes, before selecting full/smoke cohorts.

    Args:
        labels: Integer digits (N,), values 0 through 9 as verified by the loader.
        censor_limit: Positive finite scalar c from float64 calibration.
        global_indices: Optional integer (N,) indices in 0..69999. Defaults to
            0..N-1; omit only for rows in official global order. Explicit indices
            preserve outcomes when selecting or reordering rows. Values are
            preconditions, following the shared JIT-compatible generator.
    Returns:
        Shared GeneratedData with (N,) float32 time/event_time/censor_time and
        boolean event. Keys depend on global index, never on split or batch.
    Raises:
        ValueError: Labels/indices are not matching vectors or c is not scalar.
    """
    labels = jnp.asarray(labels)
    indices = (
        jnp.arange(labels.size, dtype=jnp.uint32)
        if global_indices is None
        else jnp.asarray(global_indices, dtype=jnp.uint32)
    )
    c = jnp.asarray(censor_limit, dtype=jnp.float32)
    if labels.ndim != 1 or indices.shape != labels.shape or c.ndim != 0:
        raise ValueError("labels/indices must be matching vectors and c must be scalar")
    uc = jax.random.fold_in(jax.random.key(20261025), 3)
    outcome_key = jax.random.fold_in(uc, 0)
    keys = jax.vmap(lambda index: jax.random.fold_in(outcome_key, index))(indices)
    log_scales = jnp.asarray(TRUE_LOG_SCALES, dtype=jnp.float32)[labels]
    shape = jnp.asarray(TRUE_SHAPE, dtype=jnp.float32)
    return jax.vmap(lambda key, scale: generate_weibull(key, shape, scale, c))(
        keys, jnp.exp(log_scales)
    )
