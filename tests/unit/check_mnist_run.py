"""Smoke execution and committed full-run acceptance."""

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_smoke_cli(tmp_path):
    """03: the single-process runner writes all three figures and cohort metrics."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.survival_mnist.run",
            "--config",
            "smoke",
            "--out",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
    )
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["sizes"] == {"train": 5000, "validation": 1000, "test": 2000}
    assert metadata["selected_epoch"] == 1
    assert all((tmp_path / f"fig{i}.svg").is_file() for i in (1, 2, 3))
    # The CSV layout is what the docs page and acceptance check read.
    results = ROOT / "examples/survival_mnist/results"
    for name in ("metrics.csv", "history.csv"):
        written = (tmp_path / name).read_bytes()
        assert b"\r" not in written
        header = written.split(b"\n", 1)[0]
        assert header == (results / name).read_bytes().split(b"\n", 1)[0]


def test_full_results_meet_acceptance():
    """03: fixed full experiment beats intercept and learns k within 5%."""
    results = ROOT / "examples/survival_mnist/results"
    with (results / "metrics.csv").open() as stream:
        cnn, intercept, oracle = list(csv.DictReader(stream))
    for metric in ("mean_log_likelihood", "concordance"):
        assert float(cnn[metric]) > float(intercept[metric])
    assert abs(float(cnn["shape"]) / 1.38 - 1) <= 0.05
    assert all(int(row["n"]) == 10000 for row in (cnn, intercept, oracle))
    metadata = json.loads((results / "metadata.json").read_text())
    assert metadata["sizes"] == {"train": 50000, "validation": 10000, "test": 10000}
