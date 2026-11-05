"""Verified MNIST pixels and fixed global indices (use-case 03 details)."""

import gzip
import hashlib
import os
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from examples.weibull_simulation.study import GeneratedData, generate_weibull

# Generating rule: shared Weibull shape, digit-dependent log scale, Uniform censoring.
TRUE_SHAPE = jnp.float32(1.38)
TRUE_LOG_SCALES = jnp.log(jnp.float32(10)) - jnp.float32(0.2) * jnp.arange(10)
CENSOR_LIMIT = 7.0
# Use-case 03 key; splits, outcomes, initialization and shuffling fold in from it.
UC_KEY = jax.random.fold_in(jax.random.key(20261025), 3)

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
RESOURCES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}


def _fetch(name: str, cache: Path) -> bytes:
    path = cache / name
    if path.exists():
        payload = path.read_bytes()
    else:
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MIRROR + name, timeout=60) as response:
            payload = response.read()
        path.write_bytes(payload)
    if hashlib.sha256(payload).hexdigest() != RESOURCES[name]:
        raise ValueError(f"MNIST checksum mismatch: {path}; remove it and retry")
    return payload


def _parse_idx(payload: bytes, *, images: bool, count: int) -> np.ndarray:
    dimensions = (count, 28, 28) if images else (count,)
    return np.frombuffer(
        gzip.decompress(payload), np.uint8, offset=16 if images else 8
    ).reshape(dimensions)


def load_mnist(cache: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load images and oracle labels in official global-index order."""
    if cache is None:
        cache = Path(
            os.environ.get("SURVJAX_EXAMPLES_DATA", "~/.cache/survjax-examples/mnist")
        )
    cache = cache.expanduser()
    images, labels = [], []
    for prefix, count in (("train", 60000), ("t10k", 10000)):
        image_bytes = _fetch(f"{prefix}-images-idx3-ubyte.gz", cache)
        label_bytes = _fetch(f"{prefix}-labels-idx1-ubyte.gz", cache)
        images.append(_parse_idx(image_bytes, images=True, count=count))
        labels.append(_parse_idx(label_bytes, images=False, count=count))
    pixels = np.concatenate(images).astype(np.float32)[:, None, :, :] / np.float32(255)
    return pixels, np.concatenate(labels)


def split_indices(config: str = "full") -> dict[str, np.ndarray]:
    """Return global indices for fixed train/validation/test cohorts."""
    permutation = np.asarray(
        jax.random.permutation(jax.random.fold_in(UC_KEY, 1), 60000), dtype=np.int64
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


def generate_outcomes(labels: ArrayLike) -> GeneratedData:
    """Generate outcomes once for all 70k images in official order."""
    outcome_key = jax.random.fold_in(UC_KEY, 0)
    keys = jax.vmap(lambda index: jax.random.fold_in(outcome_key, index))(
        jnp.arange(70000)
    )
    scales = jnp.exp(TRUE_LOG_SCALES[jnp.asarray(labels)])
    return jax.vmap(
        lambda key, scale: generate_weibull(key, TRUE_SHAPE, scale, CENSOR_LIMIT)
    )(keys, scales)
