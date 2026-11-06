"""Optional: the Bayesian runner passes its acceptance checks and reproduces the committed draws."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / "examples/bayesian_rmst/results/posterior.npz"


def test_run_reproduces_committed_posterior(tmp_path):
    """02: a fresh float64 run passes the acceptance asserts and matches the draws bit for bit."""
    output = tmp_path / "run"
    # A separate process: the runner sets the device count and float64 before JAX starts.
    subprocess.run(
        [sys.executable, "-m", "examples.bayesian_rmst.run", str(output)],
        cwd=ROOT,
        env=dict(os.environ, JAX_ENABLE_X64="1"),
        check=True,
        capture_output=True,
        timeout=600,
    )
    with np.load(output / "posterior.npz") as new, np.load(COMMITTED) as old:
        assert sorted(new.files) == sorted(old.files)
        for name in old.files:
            np.testing.assert_array_equal(new[name], old[name], err_msg=name)
