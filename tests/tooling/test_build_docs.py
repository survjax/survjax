"""Exercise the docs runner's whole-build certification without Sphinx/JAX."""

import os
import shutil
import subprocess
import sys
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
    uv.write_text(
        """#!/usr/bin/env bash
set -eu
if [[ " $* " == *" python -c "* ]]; then
  while [ "$1" != python ]; do shift; done
  shift
  exec "PYTHON_EXECUTABLE" "$@"
fi
root="$PWD"
if [[ "$OSTYPE" == msys* ]]; then root="$(cygpath -m "$root")"; fi
for name in JUPYTER_RUNTIME_DIR IPYTHONDIR MPLCONFIGDIR; do
  test -d "${!name}"
  case "${!name}" in "$root"/docs/_build/.cache/*) ;; *) exit 7 ;; esac
done
test "$SURVJAX_EXAMPLES_DATA" = /explicit/mnist
echo "$JAX_ENABLE_X64" >> bin/dtype-calls
if [ "$JAX_ENABLE_X64" = 1 ]; then
  case "${ACTION-}" in
    fail) exit 9 ;;
    interrupt) kill -TERM "$PPID" ;;
    edit) echo changed >> .gitignore ;;
  esac
fi
""".replace("PYTHON_EXECUTABLE", Path(sys.executable).as_posix()),
        newline="\n",
    )
    uv.chmod(0o755)
    return repo


def run_build(repo, bash_executable, action=""):
    return subprocess.run(
        [bash_executable, "scripts/build_docs.sh"],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{repo / 'bin'}{os.pathsep}{os.environ['PATH']}",
            "SURVJAX_EXAMPLES_DATA": "/explicit/mnist",
            "ACTION": action,
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_success_records_checkout_and_dirty_tree(checkout, bash_executable):
    (checkout / "local-note").write_text("untracked")
    result = run_build(checkout, bash_executable)
    assert result.returncode == 0, result.stderr
    record = (checkout / "docs/_build/build-identity.txt").read_text()
    assert record.endswith("state: success\n")
    assert (checkout / "bin/dtype-calls").read_text().splitlines() == ["0", "1"]
    for field in ("commit:", "started_utc:", "completed_utc:", "dtypes: x32 x64"):
        assert field in record
    assert f"checkout: {checkout.as_posix()}" in record
    assert "tree: dirty\n?? local-note" in record
    assert (checkout / "docs/_build/html-x64/index.html").as_posix() in result.stdout


@pytest.mark.parametrize("action", ["fail", "interrupt", "edit"])
def test_unsuccessful_rebuild_invalidates_old_success(
    checkout, bash_executable, action
):
    assert run_build(checkout, bash_executable).returncode == 0
    result = run_build(checkout, bash_executable, action)
    assert result.returncode != 0
    record = (checkout / "docs/_build/build-identity.txt").read_text()
    assert "state: success" not in record
    assert "state: failed" in record


def test_further_edit_to_already_dirty_file_invalidates_success(
    checkout, bash_executable
):
    tracked = checkout / ".gitignore"
    tracked.write_text(tracked.read_text() + "# dirty before build\n")
    assert run_build(checkout, bash_executable).returncode == 0
    status_before = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=checkout, text=True
    )
    result = run_build(checkout, bash_executable, "edit")
    status_after = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=checkout, text=True
    )
    assert status_before == status_after
    assert result.returncode != 0
    assert "Checkout changed during build" in result.stderr
    record = (checkout / "docs/_build/build-identity.txt").read_text()
    assert "state: success" not in record
    assert "state: failed" in record
