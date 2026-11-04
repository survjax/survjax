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


def pytest_collection_modifyitems(items):
    """Mark all project tests except isolated wheel-install tests for GPU CI."""
    gpu_marker = pytest.mark.gpu
    for item in items:
        if "install" not in item.path.parts:
            item.add_marker(gpu_marker)


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
    if name == "concordance":
        return _load_concordance_fixture(root / name)
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
    if is_bc:
        _check_bc_fixture(expected, data, payloads["bc_synthetic_weibull.json"])
    expected["data"] = data
    return expected


def _check_bc_fixture(expected: dict, data: dict, payload: bytes) -> None:
    groups, sizes = ["Good", "Medium", "Poor"], [229, 229, 228]
    header = expected["header"]
    if not np.array_equal(data["group"], np.repeat(groups, sizes)) or header[
        "group_sizes"
    ] != dict(zip(groups, sizes, strict=True)):
        raise ValueError("invalid bc group order or sizes")
    events = {
        group: int(data["event"][data["group"] == group].sum()) for group in groups
    }
    if header["events_by_group"] != events:
        raise ValueError("bc event counts disagree with CSV")

    grid = expected["grid"]
    times = np.asarray(grid["time"], dtype=np.float64)
    if grid["groups"] != groups or grid["shape"] != [141, 3]:
        raise ValueError("invalid bc grid metadata")
    if (
        times.shape != (141,)
        or not np.isfinite(times).all()
        or not (np.diff(times) > 0).all()
        or times[0] != 0
        or times[-1] != 7
    ):
        raise ValueError("invalid bc grid times")
    grid["time"] = times
    for case in expected["cases"].values():
        for source in ("flexsurv", "survreg"):
            curves = np.asarray(case[source]["survival_grid"], dtype=np.float64)
            if curves.shape != (141, 3) or not np.isfinite(curves).all():
                raise ValueError("invalid bc reference grid")
            if not (curves[0] == 1).all():
                raise ValueError("invalid bc survival at zero")
            case[source]["survival_grid"] = curves

    exported = json.loads(payload, parse_constant=_reject_json_constant)
    if set(exported) != {"meta", "data", "parameters", "results", "grid"}:
        raise ValueError("invalid acceptance export schema")
    meta = {
        "seed": header["seed"],
        "generator": header["generating_script"],
        "R": header["r_version"],
        "flexsurv": "2.3.2",
        "survival": "3.7-0",
    }
    if exported["meta"] != meta:
        raise ValueError("acceptance export provenance mismatch")
    if exported["data"] != {
        key: data[key].tolist() for key in ("time", "event", "group")
    }:
        raise ValueError("acceptance export rows mismatch")
    if (
        set(exported["grid"]) != {"time", "survival"}
        or exported["grid"]["time"] != times.tolist()
    ):
        raise ValueError("acceptance export grid times mismatch")
    mapping = {"fixed_a": "unit_shape", "fixed_b": "alternative", "r_optimum": "fitted"}
    parameter_names = {
        "shape": "shape",
        "good_scale": "scale_Good",
        "beta_medium": "beta_Medium",
        "beta_poor": "beta_Poor",
    }
    labels = set(mapping)
    if (
        set(exported["parameters"]) != labels
        or set(exported["results"]) != labels
        or set(exported["grid"]["survival"]) != labels
    ):
        raise ValueError("acceptance export case labels mismatch")
    for label, original in mapping.items():
        case = expected["cases"][original]
        if exported["parameters"][label] != {
            key: case[value] for key, value in parameter_names.items()
        }:
            raise ValueError("acceptance export parameters mismatch")
        sources = exported["results"][label]
        if set(sources) != {"flexsurv", "survreg"}:
            raise ValueError("acceptance export source labels mismatch")
        for source in ("flexsurv", "survreg"):
            values = case[source]
            if sources[source] != {
                "loglik": values["log_likelihood"].tolist(),
                "survival": values["survival"].tolist(),
            }:
                raise ValueError("acceptance export source values mismatch")
        if (
            exported["grid"]["survival"][label]
            != case["flexsurv"]["survival_grid"].tolist()
        ):
            raise ValueError("acceptance export curves mismatch")


def _load_concordance_fixture(directory: Path) -> dict:
    payloads = _checked_payloads(directory, {"data.csv", "concordance_synthetic.json"})
    fixture = json.loads(
        payloads["concordance_synthetic.json"], parse_constant=_reject_json_constant
    )
    meta = fixture["meta"]
    if (meta["R"], meta["survival"], meta["flexsurv"], meta["seed"]) != (
        "4.4.2",
        "3.7-0",
        "2.3.2",
        20261023,
    ) or meta["generator"] != "scripts/fixtures/generate_concordance.R":
        raise ValueError("invalid concordance provenance")
    if meta["csv_md5"] != hashlib.md5(payloads["data.csv"]).hexdigest():
        raise ValueError("CSV provenance checksum mismatch")
    rows = list(csv.DictReader(io.StringIO(payloads["data.csv"].decode())))
    expected_rows = []
    for case in fixture["cases"]:
        for i, (time, event) in enumerate(
            zip(case["time"], case["event"], strict=True)
        ):
            expected_rows.append(
                dict(
                    case=case["name"],
                    row_id=str(i + 1),
                    time=format(time, ".17g"),
                    event=str(event),
                    **{
                        f"score{d + 1}": format(scores[i], ".17g")
                        for d, scores in enumerate(case["scores"])
                    },
                )
            )
    if rows != expected_rows:
        raise ValueError("concordance CSV/JSON alignment mismatch")
    return fixture


@pytest.fixture
def r_fixture() -> Callable[..., dict]:
    """Load a manifest-checked R set, with an optional temporary fixture root.

    Returns header/cases/grid plus data arrays; all floating arrays retain
    NumPy float64 precision independently of the process's JAX dtype mode.
    The bc acceptance export must match the primary rows, vectors and grids.
    The concordance export must match every CSV observation and score row.
    """
    return _load_r_fixture
