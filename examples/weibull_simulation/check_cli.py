"""JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q -s <this file>."""

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import impl_tolerances as tol  # noqa: E402

from examples.weibull_simulation import run as runner  # noqa: E402
from examples.weibull_simulation.check_metadata import (  # noqa: E402
    assert_smoke_metadata,
)

if not jax.config.x64_enabled:
    raise RuntimeError("Run CLI checks with JAX_ENABLE_X64=1")

ATOL, RTOL = tol.ATOL[True], tol.RTOL[True]


@pytest.mark.parametrize("config", ["smoke", "full"])
def test_configuration_grid_fixed_limits_and_csv_fields(config, tmp_path, monkeypatch):
    """01 DGP/seeds/files: prespecified limits, n-then-limit order, lossless table."""
    calls = []
    metrics = {
        "bias": 0.1,
        "bias_mcse": 0.02,
        "empirical_se": 0.3,
        "empirical_se_mcse": 0.04,
        "model_se": 0.5,
        "coverage": 0.6,
        "coverage_mcse": 0.07,
        "power": 0.8,
        "power_mcse": 0.09,
    }

    def scenario(indices, n, r, c):
        calls.append((indices.tolist(), n, r, c.tolist()))
        counts = {
            "attempted": r,
            "accepted": r - 4,
            "failed": 4,
            "solver": 1,
            "nonfinite": 2,
            "gradient": 3,
            "hessian": 2,
            "unestimable": 1,
        }
        flags = np.zeros((2, r), dtype=bool)
        flags[:, :4] = True
        # Fake numerical outputs exercise serialization without running the full grid.
        from examples.weibull_simulation.study import FitResult

        p = 3
        fits = FitResult(
            np.zeros((2, r, p)),
            np.zeros((2, r)),
            np.zeros((2, r, p)),
            np.zeros((2, r, p, p)),
            np.zeros((2, r)),
            np.zeros((2, r, 2)),
            ~flags,
            ~flags,
            ~flags,
            ~flags,
        )
        record = SimpleNamespace(
            fits=fits,
            failure_masks={
                name: np.broadcast_to(
                    np.isin(
                        np.arange(r),
                        [3] if name == "unestimable" else np.arange(counts[name]),
                    ),
                    (2, r),
                )
                for name in counts
                if name not in ("attempted", "accepted")
            },
            failures={name: np.full(2, value) for name, value in counts.items()},
            performance={name: np.full(2, value) for name, value in metrics.items()},
            first_call_seconds=10.0,
            warm_seconds=2.0,
            fits_per_second=2 * r / 2,
        )
        return runner.ScenarioRun(
            c, 0.3 + 0.01 * indices, {"weibull": record, "exponential": record}
        )

    monkeypatch.setattr(runner, "run_scenario", scenario)
    out = tmp_path / "nested" / config
    capture = runner.capture_metadata

    def before_work(*args):
        assert not calls and out.is_dir()
        return capture(*args)

    monkeypatch.setattr(runner, "capture_metadata", before_work)
    runner.run_config(config, out)
    sizes, r = ([100], 20) if config == "smoke" else ([100, 300, 1000], 1000)
    expected = [
        (i, n, r, c)
        for i, (n, c) in enumerate((n, c) for n in sizes for c in [50.0, 20.0])
    ]
    assert calls == [
        ([2 * i, 2 * i + 1], n, r, [50.0, 20.0]) for i, n in enumerate(sizes)
    ]
    with (out / "performance.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * len(expected)
    for row, (index, n, _, c) in zip(
        rows, [x for x in expected for _ in range(2)], strict=True
    ):
        assert int(row["scenario"]) == index and int(row["n"]) == n
        assert float(row["censor_limit"]) == c
        assert float(row["realized_censoring"]) == 0.3 + 0.01 * index
        counts = {
            "attempted": r,
            "accepted": r - 4,
            "failed": 4,
            "solver": 1,
            "nonfinite": 2,
            "gradient": 3,
            "hessian": 2,
            "unestimable": 1,
        }
        for name, value in {
            **counts,
            **metrics,
            "first_call_seconds": 10.0,
            "warm_seconds": 2.0,
            "fits_per_second": 2 * r / 2,
            "timed_scenarios": 2,
        }.items():
            assert float(row[name]) == value
        assert set(row) == {
            "scenario",
            "n",
            "censor_limit",
            "realized_censoring",
            "model",
            *counts,
            *metrics,
            "timed_scenarios",
            "first_call_seconds",
            "warm_seconds",
            "fits_per_second",
        }
    assert [row["model"] for row in rows] == ["weibull", "exponential"] * len(expected)


def test_unknown_configuration_does_not_start_work(tmp_path, monkeypatch):
    """01 configurations: invalid requests must fail before generation/output."""

    def unexpected(*args, **kwargs):
        pytest.fail("invalid configuration started work")

    monkeypatch.setattr(runner, "run_scenario", unexpected)
    with pytest.raises(ValueError, match="config must be"):
        runner.run_config("unknown", tmp_path / "absent")
    assert not (tmp_path / "absent").exists()
    with pytest.raises(SystemExit):
        runner.main(["--config", "unknown", "--out", str(tmp_path)])
    blocked = tmp_path / "file"
    blocked.write_text("keep")
    with pytest.raises(FileExistsError):
        runner.run_config("smoke", blocked)
    assert blocked.read_text() == "keep"


def test_import_keeps_precision_and_imported_runner_refuses_float32(tmp_path):
    """Common float64 convention: only CLI main selects precision, not imports."""
    code = (
        "import jax; from pathlib import Path; "
        "from examples.weibull_simulation import run; "
        "assert not jax.config.x64_enabled; "
        f"run.run_config('smoke', Path({str(tmp_path / 'absent')!r}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "JAX_ENABLE_X64": "0"},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "simulation configurations require JAX_ENABLE_X64=1" in result.stderr
    assert not (tmp_path / "absent").exists()


def test_direct_script_smoke_writes_four_complete_summary_rows(tmp_path):
    """01 smoke: documented script entry, two scenarios, 20 fits/model, float64."""
    start = perf_counter()
    env = {**os.environ, "JAX_ENABLE_X64": "0", "OMP_NUM_THREADS": "2"}
    env.pop("MKL_NUM_THREADS", None)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/weibull_simulation/run.py"),
            "--config",
            "smoke",
            "--out",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=120,
    )
    print(f"direct-script smoke: {perf_counter() - start:.2f}s")
    with (tmp_path / "performance.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [
        (int(x["scenario"]), int(x["n"]), float(x["censor_limit"]), x["model"])
        for x in rows
    ] == [
        (i, 100, p, m)
        for i, p in enumerate([50.0, 20.0])
        for m in ["weibull", "exponential"]
    ]
    assert float(rows[0]["censor_limit"]) > float(rows[2]["censor_limit"]) > 0
    arm = np.repeat([0, 1], 50)
    for i, row in enumerate(rows[::2]):
        c = float(row["censor_limit"])
        keys = jax.random.split(jax.random.fold_in(jax.random.key(20261025), i), 20)
        censored = 0
        for key in keys:
            uk, ck = jax.random.split(key)
            u = np.asarray(jax.random.uniform(uk, (100,), dtype="float64"))
            t = 11.42 * (-np.log(u) * np.exp(-np.log(0.7) * arm)) ** (1 / 1.38)
            censor = c * np.asarray(jax.random.uniform(ck, (100,), dtype="float64"))
            censored += np.count_nonzero(t > censor)
        np.testing.assert_allclose(
            float(row["realized_censoring"]), censored / 2000, atol=ATOL, rtol=RTOL
        )
    # Host-side assertions inspect already serialized scenario rows.
    for row in rows:
        assert int(row["timed_scenarios"]) == 2
        attempted, accepted, failed = (
            int(row[x]) for x in ("attempted", "accepted", "failed")
        )
        assert attempted == 20 and accepted + failed == attempted
        for reason in ("solver", "nonfinite", "gradient", "hessian", "unestimable"):
            assert 0 <= int(row[reason]) <= failed
        for timing in ("first_call_seconds", "warm_seconds", "fits_per_second"):
            assert np.isfinite(float(row[timing])) and float(row[timing]) > 0
        np.testing.assert_allclose(
            float(row["fits_per_second"]),
            40 / float(row["warm_seconds"]),
            atol=ATOL,
            rtol=RTOL,
        )

    with (tmp_path / "replicates.csv").open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    assert len(raw) == 80
    assert len({(r["scenario"], r["model"], r["replicate"]) for r in raw}) == 80
    assert all(
        int(r["parameter_count"]) == (3 if r["model"] == "weibull" else 2) for r in raw
    )
    # Compare serialized flags to each summary; this is host-side verification only.
    for row in rows:
        cohort = [
            r
            for r in raw
            if (r["scenario"], r["model"]) == (row["scenario"], row["model"])
        ]
        for (
            name
        ) in "accepted failed solver nonfinite gradient hessian unestimable".split():
            assert sum(int(r[name]) for r in cohort) == int(row[name])

    # Inspect emitted files only; no extra smoke fit/subprocess is needed.
    for name in (
        "evidence.json",
        "failures.csv",
        "sampling_histograms.json",
        "illustration.csv",
        "fig1_survival.svg",
        "fig2_sampling.svg",
        "fig3_coverage.svg",
    ):
        assert (tmp_path / name).is_file()
    manifest = json.loads((tmp_path / "evidence.json").read_text())
    assert (
        manifest["source_sha256"]["replicates.csv"]
        == hashlib.sha256((tmp_path / "replicates.csv").read_bytes()).hexdigest()
    )
    with (tmp_path / "illustration.csv").open(newline="") as handle:
        illustration = list(csv.DictReader(handle))
    assert not illustration or all(
        r["scenario"] == "1"
        and any(
            (r["scenario"], r["model"], r["replicate"])
            == (fit["scenario"], fit["model"], fit["replicate"])
            and fit["accepted"] == "1"
            for fit in raw
        )
        for r in illustration
    )
    assert_smoke_metadata(tmp_path, rows)
