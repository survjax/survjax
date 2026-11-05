"""JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q <this file>."""

import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.weibull_simulation import _results as output  # noqa: E402
from examples.weibull_simulation.run import ScenarioRun  # noqa: E402

if not jax.config.x64_enabled:
    raise RuntimeError("Run metadata checks with JAX_ENABLE_X64=1")


def test_cpu_probe_reads_proc_model_or_reports_unavailable(tmp_path, monkeypatch):
    """Host probing: mocked Linux text/fallback is not runtime matrix evidence."""
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor : 0\nmodel name : Fixture Linux CPU\n")
    monkeypatch.setattr(output.platform, "system", lambda: "Linux")
    monkeypatch.setattr(output.platform, "processor", lambda: "")
    monkeypatch.setattr(output, "_CPUINFO", cpuinfo)
    assert output._cpu_model() == "Fixture Linux CPU"
    cpuinfo.unlink()
    assert output._cpu_model() is None


def test_metadata_roundtrip_keeps_each_batch_and_rejects_nonfinite_values(tmp_path):
    """01 metadata: shared group timings, exact n/R/indices/c/realized censoring."""
    batches = []
    # Explicit host fixtures have different sizes, replicate counts and model timings.
    for group, n in enumerate((6, 10)):
        models = {
            m: SimpleNamespace(
                fits=SimpleNamespace(theta=np.zeros((2, group + 3, p))),
                first_call_seconds=float(n * p),
                warm_seconds=float(p),
                fits_per_second=2 * (group + 3) / p,
            )
            for m, p in (("weibull", 3), ("exponential", 2))
        }
        scenario = ScenarioRun(
            np.array([40.0, 20.0]),
            np.array([0.2 + group * 0.1, 0.5 + group * 0.1]),
            models,
        )
        batches.append((np.array([2 * group, 2 * group + 1]), n, scenario))
    snapshot = {"configuration": "fixture", "source": {"commit": "before"}}
    output.write_metadata(tmp_path, snapshot, {0.25: 40.0, 0.55: 20.0}, batches)
    data = json.loads((tmp_path / "metadata.json").read_text())
    assert data["source"] == snapshot["source"] and "batches" not in snapshot
    assert data["calibration"] == [
        {"target_censoring": 0.25, "censor_limit": 40.0},
        {"target_censoring": 0.55, "censor_limit": 20.0},
    ]
    for group, (indices, n, batch) in zip(data["batches"], batches, strict=True):
        assert (
            group["n"] == n
            and group["replicates_per_scenario"]
            == batch.models["weibull"].fits.theta.shape[1]
        )
        assert group["timed_scenarios"] == 2
        for level, entry in enumerate(group["scenarios"]):
            assert entry == {
                "index": int(indices[level]),
                "target_censoring": [0.25, 0.55][level],
                "censor_limit": float(batch.censor_limit[level]),
                "realized_censoring": float(batch.realized_censoring[level]),
            }
        for model, values in group["models"].items():
            assert values == {
                key: getattr(batch.models[model], key)
                for key in ("first_call_seconds", "warm_seconds", "fits_per_second")
            }
    batches[0][2].realized_censoring[0] = np.nan
    with pytest.raises(ValueError, match="JSON compliant"):
        output.write_metadata(tmp_path, snapshot, {0.25: 40.0, 0.55: 20.0}, batches)


def assert_smoke_metadata(out: Path, rows: list[dict[str, str]]) -> None:
    """Inspect the real CLI metadata without running a second smoke subprocess."""
    metadata = json.loads((out / "metadata.json").read_text())
    assert (
        metadata["configuration"] == "smoke" and metadata["seeds"]["root"] == 20261025
    )
    assert (
        metadata["runtime"]["x64_enabled"] and metadata["runtime"]["backend"] == "cpu"
    )
    assert metadata["invocation"]["python_argv"][1:] == [
        str(ROOT / "examples/weibull_simulation/run.py"),
        "--config",
        "smoke",
        "--out",
        str(out),
    ]
    group = metadata["batches"][0]
    assert (
        len(metadata["batches"]) == 1
        and group["n"] == 100
        and group["replicates_per_scenario"] == 20
    )
    assert group["timed_scenarios"] == 2
    for row in rows:
        scenario = group["scenarios"][int(row["scenario"])]
        for key in ("target_censoring", "censor_limit", "realized_censoring"):
            assert scenario[key] == float(row[key])
        for key in ("first_call_seconds", "warm_seconds", "fits_per_second"):
            assert group["models"][row["model"]][key] == float(row[key])

    runtime = metadata["runtime"]
    assert runtime["thread_environment"]["OMP_NUM_THREADS"] == "2"
    assert runtime["thread_environment"]["MKL_NUM_THREADS"] is None
    assert (
        metadata["source"]["commit"]
        == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )
    assert (
        metadata["source"]["lock_sha256"]
        == hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    )
    assert metadata["source"]["spec_seen"] == (ROOT / ".spec-seen").read_text().strip()
    assert (
        metadata["dgp"]["shape"] == 1.38 and metadata["dgp"]["baseline_scale"] == 11.42
    )
    assert metadata["fitting"]["rtol"] == metadata["fitting"]["atol"] == 1e-8

    assert runtime["packages"] == {
        name: version(name)
        for name in ("survjax", "jax", "jaxlib", "numpy", "optimistix", "matplotlib")
    }
    if sys.platform == "darwin":
        assert (
            runtime["cpu_model"]
            == subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        )
