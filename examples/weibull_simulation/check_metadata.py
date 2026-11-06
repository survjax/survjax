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


@pytest.mark.parametrize("system", ["Darwin", "Linux"])
def test_denied_cpu_queries_report_unavailable_consistently(monkeypatch, system):
    """Host-query failures report null CPU identity, never an invented model."""

    def denied(*args, **kwargs):
        if system == "Darwin":
            raise subprocess.CalledProcessError(1, ["sysctl"])
        raise PermissionError("cpuinfo denied")

    monkeypatch.setattr(output.platform, "system", lambda: system)
    monkeypatch.setattr(output.subprocess, "check_output", denied)
    monkeypatch.setattr(
        output, "_CPUINFO", SimpleNamespace(exists=lambda: True, read_text=denied)
    )
    assert output._cpu_model() is None


def test_metadata_solver_settings_match_the_actual_fitter_call(monkeypatch):
    """01 solver/provenance: catch a call-site override or shared-setting drift."""
    import optimistix as optx

    import survjax
    from examples.weibull_simulation import study

    settings = output.capture_metadata("smoke", 20261025, float(np.log(0.7)))["fitting"]
    original, seen = optx.minimise, []

    def spy(objective, solver, start, **kwargs):
        assert type(solver).__name__ == settings["solver"].split(".")[-1]
        seen.append((solver.rtol, solver.atol, kwargs["max_steps"], kwargs["throw"]))
        return original(objective, solver, start, **kwargs)

    monkeypatch.setattr(optx, "minimise", spy)
    data = survjax.SurvivalData(
        np.array([1.0, 2.0, 3.0, 2.0, 4.0, 6.0]),
        event=np.array([1, 0, 1, 1, 1, 0], dtype=bool),
    )
    data.issues.raise_if_invalid()
    fit = jax.jit(
        lambda d: study.fit_replicate(
            d, np.array([0, 0, 0, 1, 1, 1]), model="exponential"
        )
    )(data)
    assert fit.accepted
    assert seen == [
        (settings["rtol"], settings["atol"], settings["max_steps"], settings["throw"])
    ]
    assert seen == [(1e-8, 1e-8, 1000, False)]
    assert settings["gradient_tolerance"] == 1e-5
    score = jax.numpy.array([settings["gradient_tolerance"] / 2, 0.0])
    assessed = study._assess_fit(
        jax.numpy.zeros(2),
        jax.numpy.array(1.0),
        score,
        jax.numpy.eye(2),
        jax.numpy.array(True),
    )
    assert assessed.accepted


@pytest.mark.parametrize("dirty", [False, True])
def test_reducer_provenance_tracks_its_own_source_file(tmp_path, monkeypatch, dirty):
    """SIM.8c.1 F-2: source path and dirty state cannot refer to another file."""
    from examples.weibull_simulation import _evidence
    from examples.weibull_simulation.check_results import _compact_fixture

    _compact_fixture(tmp_path)
    expected_path = "examples/weibull_simulation/_evidence.py"
    seen = []

    def git_output(command, **kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            return "a" * 40 + "\n"
        # The reducer is dirty in one case; PLAN.md remains clean in both.
        assert command[:4] == ["git", "status", "--porcelain", "--"]
        seen.append(command[-1])
        return (
            " M " + expected_path + "\n"
            if dirty and command[-1] == expected_path
            else ""
        )

    monkeypatch.setattr(_evidence.subprocess, "check_output", git_output)
    _evidence.write_evidence(tmp_path, tmp_path)
    manifest = json.loads((tmp_path / "evidence.json").read_text())
    assert manifest["reducer_path"] == expected_path
    assert manifest["reducer_dirty"] is dirty
    assert seen == [expected_path]


def test_committed_full_bundle_hashes_and_source_provenance():
    """D-044: bind every compact payload to original clean full-run provenance."""
    root = ROOT / "examples/weibull_simulation/results"
    manifest = json.loads((root / "evidence.json").read_text())
    assert set(manifest["output_sha256"]) == {
        "metadata.json",
        "performance.csv",
        "failures.csv",
        "illustration.csv",
        "sampling_histograms.json",
    }
    # These loops read/hash committed files only, never run the simulation.
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    for name in ("metadata.json", "performance.csv"):
        assert manifest["source_sha256"][name] == manifest["output_sha256"][name]
    assert len(manifest["source_sha256"]["replicates.csv"]) == 64
    assert manifest["reducer_dirty"] is False
    source = subprocess.check_output(
        ["git", "show", f"{manifest['reducer_commit']}:{manifest['reducer_path']}"],
        cwd=ROOT,
    )
    assert hashlib.sha256(source).hexdigest() == manifest["reducer_sha256"]
    assert (
        len(manifest["reducer_commit"]) == 40 and len(manifest["reducer_sha256"]) == 64
    )
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["configuration"] == "full" and metadata["runtime"]["x64_enabled"]
    assert metadata["source"]["worktree_dirty_at_start"] is False
    assert metadata["seeds"]["root"] == 20261025
