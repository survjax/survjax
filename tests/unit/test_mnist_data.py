"""Offline synthetic IDX checks for use-case 03 Data acquisition and Splits."""

import gzip
import hashlib
import io
import struct
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.survival_mnist import data  # noqa: E402


def idx(images=True, count=2):
    dimensions = (count, 28, 28) if images else (count,)
    values = np.zeros(dimensions, dtype=np.uint8)
    values.flat[-1] = 255 if images else 9
    return gzip.compress(
        struct.pack(
            ">" + "I" * (len(dimensions) + 1), 2051 if images else 2049, *dimensions
        )
        + values.tobytes()
    )


def test_idx_pixels_and_labels():
    """03: uint8 IDX payloads retain pixel and digit values exactly."""
    images = data._parse_idx(idx(), images=True, count=2)
    labels = data._parse_idx(idx(False), images=False, count=2)
    assert images.shape == (2, 28, 28) and images.dtype == np.uint8
    assert images[1, -1, -1] == 255
    np.testing.assert_array_equal(labels, [0, 9])


def test_download_then_checked_cache(tmp_path, monkeypatch):
    """03: download SHA checks and warm-cache use need no real network."""
    payload = idx()
    name = "synthetic.gz"
    monkeypatch.setitem(
        data.RESOURCES,
        name,
        hashlib.sha256(payload).hexdigest(),
    )
    calls = []
    monkeypatch.setattr(
        data.urllib.request,
        "urlopen",
        lambda url, timeout: calls.append(url) or io.BytesIO(payload),
    )
    assert data._fetch(name, tmp_path) == payload
    assert data._fetch(name, tmp_path) == payload
    assert calls == [data.MIRROR + name]
    (tmp_path / name).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        data._fetch(name, tmp_path)
    assert len(calls) == 1


@pytest.mark.parametrize("env_cache", [True, False])
def test_loader_preprocessing_and_global_order(tmp_path, monkeypatch, env_cache):
    """03: fixed sizes/order, float32 NCHW and division by 255, in either mode."""
    if env_cache:
        monkeypatch.setenv("SURVJAX_EXAMPLES_DATA", str(tmp_path))
    else:
        monkeypatch.delenv("SURVJAX_EXAMPLES_DATA", raising=False)
    expected_cache = (
        tmp_path if env_cache else Path.home() / ".cache/survjax-examples/mnist"
    )

    def fetch(name, cache):
        assert cache == expected_cache
        return idx("images" in name, 60000 if name.startswith("train") else 10000)

    monkeypatch.setattr(data, "_fetch", fetch)
    images, labels = data.load_mnist()
    assert images.shape == (70000, 1, 28, 28) and images.dtype == np.float32
    np.testing.assert_array_equal(images[[59999, 69999], 0, -1, -1], [1, 1])
    np.testing.assert_array_equal(labels[[0, 59999, 60000, 69999]], [0, 9, 0, 9])


def test_fixed_splits_and_smoke_prefixes():
    """03: official holdout, fixed validation key, no overlap, stable prefixes."""
    full, smoke = data.split_indices(), data.split_indices("smoke")
    key = jax.random.fold_in(jax.random.fold_in(jax.random.key(20261025), 3), 1)
    expected = np.asarray(jax.random.permutation(key, 60000))
    np.testing.assert_array_equal(full["validation"], expected[:10000])
    np.testing.assert_array_equal(full["train"], expected[10000:])
    np.testing.assert_array_equal(full["test"], np.arange(60000, 70000))
    # Reviewed JAX 0.11.2 cohort anchor (MNIST.1 review N-3), not new expected data.
    np.testing.assert_array_equal(
        full["validation"][:5], [16629, 39411, 48950, 18316, 47095]
    )
    assert all(indices.dtype == np.int64 for indices in full.values())
    assert len(np.unique(np.concatenate(list(full.values())))) == 70000
    for name, size in zip(full, (5000, 1000, 2000), strict=True):
        np.testing.assert_array_equal(smoke[name], full[name][:size])
