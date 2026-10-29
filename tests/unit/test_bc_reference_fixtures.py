"""BC fixture structure and exact export mapping (spec 10.4/11.1/12.1)."""

import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def bc_root(tmp_path):
    shutil.copytree(
        Path(__file__).parents[1] / "fixtures/bc_like", tmp_path / "bc_like"
    )
    return tmp_path


def _write_rehashed(root, filename, payload):
    """Construct damaged temporary inputs with otherwise valid integrity links."""
    directory = root / "bc_like"
    (directory / filename).write_text(json.dumps(payload).replace('"1e999"', "1e999"))
    manifest = directory / "SHA256SUMS"
    entries = []
    for line in manifest.read_text().splitlines():
        _, name = line.split()
        entries.append(
            f"{hashlib.sha256((directory / name).read_bytes()).hexdigest()}  {name}\n"
        )
    manifest.write_text("".join(entries))


def test_bc_grids_keep_time_first_shape_and_original_r_doubles(r_fixture):
    """Spec 3.3/5/11.1: all sources/cases remain float64 with (141, 3) grids."""
    fixture = r_fixture("bc_like")
    np.testing.assert_array_equal(
        fixture["data"]["group"], np.repeat(["Good", "Medium", "Poor"], [229, 229, 228])
    )
    assert fixture["grid"]["time"].dtype == np.float64
    # Original R %a values from the A8.3b numeric audit, at time index 51.
    assert fixture["grid"]["time"][51] == float.fromhex("0x1.4666666666667p+1")
    for case in fixture["cases"].values():
        for source in ("flexsurv", "survreg"):
            curves = case[source]["survival_grid"]
            assert curves.shape == (141, 3)
            assert curves.dtype == np.float64
            np.testing.assert_array_equal(curves[0], np.ones(3))
    for source in ("flexsurv", "survreg"):
        assert fixture["cases"]["fitted"][source]["survival_grid"][
            51, 1
        ] == float.fromhex("0x1.7d5e6d720ec5cp-1")


def test_rehashed_csv_group_reordering_fails(r_fixture, bc_root):
    """Spec 11.1: swapping Good/Medium row labels violates the declared blocks."""
    path = bc_root / "bc_like/data.csv"
    rows = [line.split(",") for line in path.read_text().splitlines()]
    rows[1][-1], rows[230][-1] = rows[230][-1], rows[1][-1]
    path.write_text("\n".join(",".join(row) for row in rows) + "\n")
    payload = json.loads(path.with_name("expected.json").read_text())
    payload["header"]["checksums"]["data.csv"] = hashlib.md5(
        path.read_bytes()
    ).hexdigest()
    _write_rehashed(bc_root, "expected.json", payload)
    with pytest.raises(ValueError, match="^invalid bc group order or sizes$"):
        r_fixture("bc_like", root=bc_root)


@pytest.mark.parametrize(
    "damage,message",
    [
        ("groups", "invalid bc group order or sizes"),
        ("counts", "bc event counts disagree with CSV"),
        ("labels", "invalid bc grid metadata"),
        ("metadata_shape", "invalid bc grid metadata"),
        ("time_shape", "invalid bc grid times"),
        ("time_nonfinite", "invalid bc grid times"),
        ("time_order", "invalid bc grid times"),
        ("time_start", "invalid bc grid times"),
        ("time_end", "invalid bc grid times"),
        ("grid_shape", "invalid bc reference grid"),
        ("grid_nonfinite", "invalid bc reference grid"),
        ("grid_zero", "invalid bc survival at zero"),
    ],
)
def test_rehashed_bc_structure_is_checked(r_fixture, bc_root, damage, message):
    """Spec 11.1: rehashed malformed metadata or grids still fail precisely."""
    payload = json.loads((bc_root / "bc_like/expected.json").read_text())
    grid = payload["grid"]
    curves = payload["cases"]["fitted"]["survreg"]["survival_grid"]
    if damage == "groups":
        payload["header"]["group_sizes"]["Good"] = 1
    elif damage == "counts":
        payload["header"]["events_by_group"]["Good"] = 0
    elif damage == "labels":
        grid["groups"].reverse()
    elif damage == "metadata_shape":
        grid["shape"] = [3, 141]
    elif damage == "time_shape":
        grid["time"].pop()
    elif damage == "time_nonfinite":
        grid["time"][1] = "1e999"
    elif damage == "time_order":
        grid["time"][1] = 0
    elif damage == "time_start":
        grid["time"][0] = -0.05
    elif damage == "time_end":
        grid["time"][-1] = 6.99
    elif damage == "grid_shape":
        curves.pop()
    elif damage == "grid_nonfinite":
        curves[1][0] = "1e999"
    else:
        curves[0][0] = 0.5
    _write_rehashed(bc_root, "expected.json", payload)
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        r_fixture("bc_like", root=bc_root)


@pytest.mark.parametrize(
    "damage,message",
    [
        ("schema", "invalid acceptance export schema"),
        ("meta", "acceptance export provenance mismatch"),
        ("time", "acceptance export rows mismatch"),
        ("event", "acceptance export rows mismatch"),
        ("group", "acceptance export rows mismatch"),
        ("grid_time", "acceptance export grid times mismatch"),
        ("case", "acceptance export case labels mismatch"),
        ("case_results", "acceptance export case labels mismatch"),
        ("case_curves", "acceptance export case labels mismatch"),
        ("parameter", "acceptance export parameters mismatch"),
        ("source", "acceptance export source labels mismatch"),
        ("flexsurv.loglik", "acceptance export source values mismatch"),
        ("survreg.loglik", "acceptance export source values mismatch"),
        ("flexsurv.survival", "acceptance export source values mismatch"),
        ("survreg.survival", "acceptance export source values mismatch"),
        ("curves", "acceptance export curves mismatch"),
    ],
)
def test_rehashed_acceptance_export_must_match_primary_files(
    r_fixture, bc_root, damage, message
):
    """Spec 10.4/11.1: the additional schema preserves every source field."""
    filename = "bc_synthetic_weibull.json"
    payload = json.loads((bc_root / "bc_like" / filename).read_text())
    if damage == "schema":
        del payload["data"]
    elif damage == "meta":
        payload["meta"]["R"] = "wrong"
    elif damage in ("time", "event", "group"):
        payload["data"][damage].reverse()
    elif damage == "grid_time":
        payload["grid"]["time"][1] = 0
    elif damage == "case":
        payload["parameters"]["wrong"] = payload["parameters"].pop("fixed_a")
    elif damage == "case_results":
        payload["results"]["wrong"] = payload["results"].pop("fixed_a")
    elif damage == "case_curves":
        curves = payload["grid"]["survival"]
        curves["wrong"] = curves.pop("fixed_a")
    elif damage == "parameter":
        payload["parameters"]["fixed_a"]["shape"] = 2
    elif damage == "source":
        payload["results"]["fixed_a"]["wrong"] = payload["results"]["fixed_a"].pop(
            "survreg"
        )
    elif "." in damage:
        source, quantity = damage.split(".")
        payload["results"]["fixed_a"][source][quantity][0] += 1
    else:
        payload["grid"]["survival"]["r_optimum"] = [
            row[::-1] for row in payload["grid"]["survival"]["r_optimum"]
        ]
    _write_rehashed(bc_root, filename, payload)
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        r_fixture("bc_like", root=bc_root)
