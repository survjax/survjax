"""Committed R fixture integrity and common schema (spec 10.4/12.1)."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fixture_root(tmp_path):
    """Corrupt only temporary copies; committed reference bytes are read-only."""
    root = tmp_path / "fixtures"
    shutil.copytree(Path(__file__).parents[1] / "fixtures", root)
    return root


def _rehash(path):
    """Simulate an intentionally updated manifest to exercise schema checks."""
    manifest = path.parent / "SHA256SUMS"
    entries = []
    for line in manifest.read_text().splitlines():
        _, filename = line.split()
        checksum = hashlib.sha256((path.parent / filename).read_bytes()).hexdigest()
        entries.append(f"{checksum}  {filename}\n")
    manifest.write_text("".join(entries))


@pytest.mark.parametrize("name,count", [("exponential", 16), ("bc_like", 686)])
def test_committed_sets_load_without_losing_stored_precision(r_fixture, name, count):
    """Spec 3.3/10.4: host reference doubles survive either JAX dtype mode."""
    fixture = r_fixture(name)
    assert fixture["data"]["time"].shape == (count,)
    assert fixture["data"]["time"].dtype == np.float64
    # Both generators serialize original R doubles (A8.3b numeric audit).
    for case in fixture["cases"].values():
        for source in ("flexsurv", "survreg"):
            assert case[source]["log_likelihood"].dtype == np.float64
            assert case[source]["survival"].shape == (count,)


@pytest.mark.parametrize(
    "name,label,time_hex,loglik_hex",
    [
        ("exponential", "slow", "0x1.4907fd8249249p+0", "-0x1.372eb5350c2ccp+1"),
        ("bc_like", "fitted", "0x1.f61c1366p+1", "-0x1.d4b8992127333p-3"),
    ],
)
def test_values_are_not_rounded_through_float32(
    r_fixture, name, label, time_hex, loglik_hex
):
    """Spec 3.3/10.4: preserve the original R doubles, not just their dtype."""
    # Original R %a values audited in A8.3/A8.3b, for row 1 of each fixture.
    fixture = r_fixture(name)
    assert fixture["data"]["time"][0] == float.fromhex(time_hex)
    assert fixture["cases"][label]["flexsurv"]["log_likelihood"][0] == float.fromhex(
        loglik_hex
    )


def test_missing_manifest_fails(r_fixture, fixture_root):
    """Spec 10.4: missing reference files fail, never skip."""
    (fixture_root / "exponential/SHA256SUMS").unlink()
    with pytest.raises(FileNotFoundError, match="SHA256SUMS"):
        r_fixture("exponential", root=fixture_root)


@pytest.mark.parametrize(
    "damage", ["malformed", "missing_entry", "duplicate", "unknown_entry"]
)
def test_manifest_must_describe_exactly_the_known_payloads(
    r_fixture, fixture_root, damage
):
    """Spec 10.4: malformed, incomplete or ambiguous manifests are rejected."""
    manifest = fixture_root / "exponential/SHA256SUMS"
    lines = manifest.read_text().splitlines(keepends=True)
    changes = {
        "malformed": "not a hash\n",
        "missing_entry": lines[0],
        "duplicate": "".join(lines + lines[:1]),
        "unknown_entry": "".join(lines) + "0" * 64 + "  other.json\n",
    }
    manifest.write_text(changes[damage])
    with pytest.raises(ValueError, match="manifest"):
        r_fixture("exponential", root=fixture_root)


def test_missing_payload_fails(r_fixture, fixture_root):
    """Spec 10.4: a valid manifest does not excuse a missing payload."""
    (fixture_root / "bc_like/bc_synthetic_weibull.json").unlink()
    with pytest.raises(FileNotFoundError, match="bc_synthetic_weibull"):
        r_fixture("bc_like", root=fixture_root)


@pytest.mark.parametrize(
    "name,filename",
    [
        ("exponential", "data.csv"),
        ("exponential", "expected.json"),
        ("bc_like", "data.csv"),
        ("bc_like", "expected.json"),
        ("bc_like", "bc_synthetic_weibull.json"),
    ],
)
def test_each_payload_is_verified_before_any_json_parsing(
    r_fixture, fixture_root, monkeypatch, name, filename
):
    """Spec 10.4: independent CSV/JSON tampering fails before parsing anything."""
    path = fixture_root / name / filename
    path.write_bytes(path.read_bytes() + b"broken")

    def forbid_parsing(*args, **kwargs):
        pytest.fail("JSON parsed before every payload hash was verified")

    monkeypatch.setattr(json, "loads", forbid_parsing)
    with pytest.raises(ValueError, match=f"SHA256 mismatch: {filename}"):
        r_fixture(name, root=fixture_root)


@pytest.mark.parametrize(
    "damage", ["schema", "rows", "provenance", "shape", "nonfinite"]
)
def test_rehashed_json_still_has_to_satisfy_common_schema(
    r_fixture, fixture_root, damage
):
    """Spec 10.4/12.1: a fresh hash cannot hide malformed or misaligned values."""
    path = fixture_root / "exponential/expected.json"
    # Parsing here creates malformed test inputs, never reference expectations.
    payload = json.loads(path.read_text())
    if damage == "schema":
        payload["header"]["schema_version"] = 2
    elif damage == "rows":
        payload["row_id"][:2] = [2, 1]
    elif damage == "provenance":
        payload["header"]["packages"]["flexsurv"] = "wrong"
    elif damage == "shape":
        payload["cases"]["slow"]["flexsurv"]["survival"].pop()
    else:
        payload["cases"]["slow"]["survreg"]["log_survival"][0] = float("nan")
    path.write_text(json.dumps(payload))
    _rehash(path)
    with pytest.raises(ValueError, match="schema|alignment|provenance|array|nonfinite"):
        r_fixture("exponential", root=fixture_root)
