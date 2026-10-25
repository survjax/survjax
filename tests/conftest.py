"""Shared test fixtures; dtype modes are selected by scripts/run_tests.sh."""

import csv
import hashlib
import io
import json
import os
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def x64_enabled() -> bool:
    """Read the process mode without changing JAX configuration."""
    return os.environ["JAX_ENABLE_X64"] == "1"


_R_FIXTURES = Path(__file__).parent / "fixtures"


def _checked_payloads(directory: Path, names: set[str]) -> dict[str, bytes]:
    hashes = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        entry = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if entry is None:
            raise ValueError("malformed fixture manifest")
        checksum, filename = entry.groups()
        if filename in hashes:
            raise ValueError(f"duplicate manifest entry: {filename}")
        hashes[filename] = checksum
    if set(hashes) != names:
        raise ValueError("manifest must list exactly the known fixture payloads")
    payloads = {}
    for filename, checksum in hashes.items():
        content = (directory / filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != checksum:
            raise ValueError(f"SHA256 mismatch: {filename}")
        payloads[filename] = content
    return payloads


def _reject_json_constant(word: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {word}")


def _load_r_fixture(name: str, root: Path = _R_FIXTURES) -> dict:
    if name not in ("exponential", "bc_like"):
        raise ValueError(f"unknown R fixture: {name}")
    is_bc = name == "bc_like"
    names = {"data.csv", "expected.json"}
    if is_bc:
        names.add("bc_synthetic_weibull.json")
    payloads = _checked_payloads(root / name, names)
    expected = json.loads(
        payloads["expected.json"], parse_constant=_reject_json_constant
    )
    header = expected["header"]
    count, seed = (686, 8201) if is_bc else (16, 8101)
    if header["schema_version"] != 1 or header["units"] != "years":
        raise ValueError("invalid fixture schema version or units")
    if (
        header["seed"] != seed
        or header["r_version"] != "4.4.2"
        or header["packages"] != {"flexsurv": "2.3.2", "survival": "3.7.0"}
    ):
        raise ValueError("invalid fixture provenance")
    if header["generating_script"] != f"scripts/fixtures/generate_{name}.R":
        raise ValueError("invalid generating script provenance")
    if (
        header["checksums"]["algorithm"] != "md5"
        or header["checksums"]["data.csv"]
        != hashlib.md5(payloads["data.csv"]).hexdigest()
    ):
        raise ValueError("CSV provenance checksum mismatch")
    if header["event_coding"] != {"exact": 1, "right": 0}:
        raise ValueError("invalid event coding")

    reader = csv.DictReader(io.StringIO(payloads["data.csv"].decode()))
    columns = ["row_id", "time", "event"] + (["group"] if is_bc else [])
    if reader.fieldnames != columns:
        raise ValueError("invalid CSV columns")
    rows = list(reader)
    dtypes = {
        "row_id": np.int64,
        "time": np.float64,
        "event": np.int64,
        "group": np.str_,
    }
    data = {
        key: np.asarray([row[key] for row in rows], dtype=dtypes[key])
        for key in columns
    }
    if (
        len(rows) != count
        or not np.array_equal(data["row_id"], np.arange(1, count + 1))
        or expected["row_id"] != data["row_id"].tolist()
    ):
        raise ValueError("invalid row count or row alignment")
    if (
        not np.isfinite(data["time"]).all()
        or not (data["time"] > 0).all()
        or set(data["event"]) != {0, 1}
    ):
        raise ValueError("invalid observation times or events")
    events = int(data["event"].sum())
    if header["counts"] != {"exact": events, "right": count - events}:
        raise ValueError("event counts disagree with CSV")
    labels = (
        ("fitted", "unit_shape", "alternative")
        if is_bc
        else ("slow", "generating", "fast")
    )
    if set(expected["cases"]) != set(labels):
        raise ValueError("invalid parameter case labels")
    for case in expected["cases"].values():
        keys = (
            ("shape", "scale_Good", "beta_Medium", "beta_Poor") if is_bc else ("rate",)
        )
        parameters = np.asarray([case[key] for key in keys], dtype=np.float64)
        if (
            parameters.shape != (len(keys),)
            or not np.isfinite(parameters).all()
            or not (parameters[: 2 if is_bc else 1] > 0).all()
        ):
            raise ValueError("invalid reference parameters")
        for source in ("flexsurv", "survreg"):
            quantities = case[source]
            total = np.asarray(quantities["total_log_likelihood"], dtype=np.float64)
            if total.shape != () or not np.isfinite(total):
                raise ValueError("invalid reference total")
            for quantity in ("log_likelihood", "log_survival", "survival"):
                values = np.asarray(quantities[quantity], dtype=np.float64)
                if values.shape != (count,) or not np.isfinite(values).all():
                    raise ValueError(f"invalid reference array: {quantity}")
                quantities[quantity] = values
    expected["data"] = data
    return expected


@pytest.fixture
def r_fixture() -> Callable[..., dict]:
    """Load a manifest-checked R set, with an optional temporary fixture root.

    Returns header/cases/grid plus data arrays; all floating arrays retain
    NumPy float64 precision independently of the process's JAX dtype mode.
    The additional bc group/grid/export consistency checks follow in A8.4a.2.
    """
    return _load_r_fixture
