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


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        struct.pack(">II", 2049, 2) + b"\x00",
        struct.pack(">II", 2049, 1) + b"\x0a",
        struct.pack(">II", 2051, 1) + b"\x00",
        struct.pack(">II", 2049, 1) + b"\x00\x00",
    ],
)
def test_idx_rejects_malformed_payload(raw):
    """03: reject truncated, wrong magic/count, trailing bytes and bad digits."""
    with pytest.raises(ValueError, match="MNIST"):
        data._parse_idx(gzip.compress(raw), images=False, count=1)


def test_download_then_checked_cache(tmp_path, monkeypatch):
    """03: first-fetch MD5/SHA checks and warm-cache use need no real network."""
    payload = idx()
    name = "synthetic.gz"
    monkeypatch.setitem(
        data.RESOURCES,
        name,
        (hashlib.md5(payload).hexdigest(), hashlib.sha256(payload).hexdigest()),
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


@pytest.mark.parametrize(
    "hashes",
    [
        ("wrong", hashlib.sha256(b"bad").hexdigest()),
        (hashlib.md5(b"bad").hexdigest(), "wrong"),
    ],
)
def test_bad_download_is_never_cached(tmp_path, monkeypatch, hashes):
    """03: either failed digest prevents publishing a cache entry."""
    monkeypatch.setitem(data.RESOURCES, "bad.gz", hashes)
    monkeypatch.setattr(
        data.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"bad")
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        data._fetch("bad.gz", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_loader_preprocessing_and_global_order(tmp_path, monkeypatch):
    """03: fixed sizes/order, float32 NCHW and division by 255, in either mode."""
    monkeypatch.setenv("SURVJAX_EXAMPLES_DATA", str(tmp_path))

    def fetch(name, cache):
        assert cache == tmp_path
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
    assert len(np.unique(np.concatenate(list(full.values())))) == 70000
    for name, size in zip(full, (5000, 1000, 2000), strict=True):
        np.testing.assert_array_equal(smoke[name], full[name][:size])
    with pytest.raises(ValueError, match="config"):
        data.split_indices("typo")
