"""Build-entry-point isolation and failure behavior without executing notebooks."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def docs_builder(tmp_path, monkeypatch):
    repo = tmp_path / "selected worktree"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[2] / "scripts/build_docs.sh"
    script = scripts / source.name
    shutil.copy2(source, script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys\n"
        "if '-c' in sys.argv:\n"
        "    args = sys.argv[sys.argv.index('python') + 1:]\n"
        "    sys.exit(subprocess.run([sys.executable, *args]).returncode)\n"
        "record = dict(argv=sys.argv[1:], cwd=os.getcwd(), env=dict(os.environ))\n"
        "with open(os.environ['DOCS_BUILD_TEST_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(record) + '\\n')\n"
        "sys.exit(int(os.environ.get('DOCS_BUILD_TEST_EXIT', '0')))\n"
    )
    uv.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DOCS_BUILD_TEST_LOG", str(log))
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(tmp_path / "wrong-env"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "wrong-source"))
    return script, repo, log


@pytest.mark.parametrize("explicit", [False, True])
def test_docs_build_uses_script_worktree_and_selected_output(
    docs_builder, tmp_path, explicit
):
    """A portal's cwd/environment must not select another worktree's code."""
    script, repo, log = docs_builder
    args = [str(script)]
    output = repo / "docs/_build"
    if explicit:
        args += ["--output-dir", "portal output"]
        output = tmp_path / "portal output"
    result = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2
    for mode, suffix, call in zip(("0", "1"), ("x32", "x64"), calls, strict=True):
        assert call["cwd"] == str(repo)
        assert call["argv"][:3] == ["run", "--project", str(repo)]
        assert call["argv"][-1] == str(output / f"html-{suffix}")
        env = call["env"]
        assert env["JAX_ENABLE_X64"] == mode
        assert env["UV_PROJECT_ENVIRONMENT"] == str(repo / ".venv")
        assert env["PYTHONPATH"] == f"{repo / 'src'}:{repo}"
        assert env["JUPYTER_PATH"] == str(repo / ".venv/share/jupyter")
        assert env["JAX_COMPILATION_CACHE_DIR"] == str(output / ".cache/jax")
        assert env["MPLCONFIGDIR"] == str(output / ".cache/matplotlib")
        assert env["IPYTHONDIR"] == str(output / ".cache/ipython")
    assert (output / ".gitignore").read_text() == "*\n"


def test_docs_build_stops_after_failed_dtype(docs_builder, tmp_path, monkeypatch):
    """A failed float32 build must not be reported as a successful pair."""
    script, _, log = docs_builder
    monkeypatch.setenv("DOCS_BUILD_TEST_EXIT", "7")
    result = subprocess.run([str(script)], cwd=tmp_path, capture_output=True)
    assert result.returncode == 7
    assert len(log.read_text().splitlines()) == 1


@pytest.mark.parametrize(
    "args",
    [["--unknown"], ["--output-dir"], ["--output-dir", ""], ["--output-dir", "--help"]],
)
def test_docs_build_rejects_invalid_arguments_before_building(
    docs_builder, tmp_path, args
):
    script, _, log = docs_builder
    result = subprocess.run([str(script), *args], cwd=tmp_path, capture_output=True)
    assert result.returncode == 2
    assert not log.exists()
