"""Exercise the docs runner's whole-build certification without Sphinx/JAX."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/build_docs.sh"


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "checkout with spaces"
    repo.mkdir()
    for args in (
        ["init", "-q"],
        ["config", "user.email", "test@example.invalid"],
        ["config", "user.name", "Test"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True)
    (repo / ".gitignore").write_text("docs/_build/\nbin/\n")
    (repo / "scripts").mkdir()
    shutil.copy(SCRIPT, repo / "scripts/build_docs.sh")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    (repo / "bin").mkdir()
    uv = repo / "bin/uv"
    uv.write_text("""#!/usr/bin/env bash
set -eu
for name in JUPYTER_RUNTIME_DIR IPYTHONDIR MPLCONFIGDIR; do
  test -d "${!name}"
  case "${!name}" in "$PWD"/docs/_build/*) ;; *) exit 7 ;; esac
done
test "$SURVJAX_EXAMPLES_DATA" = /explicit/mnist
if [ "$JAX_ENABLE_X64" = 1 ]; then
  case "${ACTION-}" in
    fail) exit 9 ;;
    interrupt) kill -TERM "$PPID" ;;
    edit) echo changed >> .gitignore ;;
  esac
fi
""")
    uv.chmod(0o755)
    return repo


def run_build(repo, action=""):
    return subprocess.run(
        ["bash", "scripts/build_docs.sh"],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{repo / 'bin'}:{os.environ['PATH']}",
            "SURVJAX_EXAMPLES_DATA": "/explicit/mnist",
            "ACTION": action,
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_success_records_checkout_and_dirty_tree(checkout):
    (checkout / "local-note").write_text("untracked")
    result = run_build(checkout)
    assert result.returncode == 0, result.stderr
    record = (checkout / "docs/_build/build-identity.txt").read_text()
    assert record.endswith("state: success\n")
    for field in ("commit:", "started_utc:", "completed_utc:", "dtypes: x32 x64"):
        assert field in record
    assert f"checkout: {checkout}" in record
    assert "tree: dirty\n?? local-note" in record
    assert str(checkout / "docs/_build/html-x64/index.html") in result.stdout


@pytest.mark.parametrize("action", ["fail", "interrupt", "edit"])
def test_unsuccessful_rebuild_invalidates_old_success(checkout, action):
    assert run_build(checkout).returncode == 0
    result = run_build(checkout, action)
    assert result.returncode != 0
    record = (checkout / "docs/_build/build-identity.txt").read_text()
    assert "state: success" not in record
    assert "state: failed" in record
