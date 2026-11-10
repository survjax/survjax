"""Build-entry-point isolation and failure behavior without executing notebooks."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def docs_builder(tmp_path, monkeypatch, bash_executable):
    repo = tmp_path / "selected worktree"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[2] / "scripts/build_docs.sh"
    script = scripts / source.name
    shutil.copy2(source, script)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("docs/_build/\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    stub = bin_dir / "fake_uv.py"
    stub.write_text(
        "import json, os, subprocess, sys\n"
        "if '-c' in sys.argv:\n"
        "    if os.environ.get('DOCS_BUILD_TEST_SETUP_EXIT'):\n"
        "        sys.exit(int(os.environ['DOCS_BUILD_TEST_SETUP_EXIT']))\n"
        "    args = sys.argv[sys.argv.index('python') + 1:]\n"
        "    sys.exit(subprocess.run([sys.executable, *args]).returncode)\n"
        "record = dict(argv=sys.argv[1:], cwd=os.getcwd(), env=dict(os.environ))\n"
        "with open(os.environ['DOCS_BUILD_TEST_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(record) + '\\n')\n"
        "sys.exit(int(os.environ.get('DOCS_BUILD_TEST_EXIT', '0')))\n"
    )
    uv.write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(Path(sys.executable).as_posix())} "
        f'{shlex.quote(stub.as_posix())} "$@"\n',
        newline="\n",
    )
    uv.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DOCS_BUILD_TEST_LOG", str(log))
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(tmp_path / "wrong-env"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "wrong-source"))
    return [bash_executable, script.as_posix()], repo, log


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("shell_os", [None, "msys", "cygwin"])
def test_docs_build_uses_script_worktree_and_selected_output(
    docs_builder, tmp_path, monkeypatch, explicit, shell_os
):
    """A portal's cwd/environment must not select another worktree's code."""
    script, repo, log = docs_builder
    path_separator = os.pathsep
    if shell_os is not None:
        startup = tmp_path / "bash-startup"
        startup.write_text(f"OSTYPE={shell_os}\n", newline="\n")
        monkeypatch.setenv("BASH_ENV", startup.as_posix())
        path_separator = ";"
        if sys.platform != "win32":
            # POSIX hosts exercise platform selection without drive-letter paths.
            cygpath = tmp_path / "bin/cygpath"
            cygpath.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$2\"\n", newline="\n"
            )
            cygpath.chmod(0o755)
    args = list(script)
    output = repo / "docs/_build"
    if explicit:
        args += ["--output-dir", "portal output"]
        output = tmp_path / "portal output"
    result = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2
    for mode, suffix, call in zip(("0", "1"), ("x32", "x64"), calls, strict=True):
        assert Path(call["cwd"]) == repo
        assert call["argv"][:2] == ["run", "--project"]
        assert Path(call["argv"][2]) == repo
        assert Path(call["argv"][-1]) == output / f"html-{suffix}"
        env = call["env"]
        assert env["JAX_ENABLE_X64"] == mode
        assert Path(env["UV_PROJECT_ENVIRONMENT"]) == repo / ".venv"
        assert [Path(p) for p in env["PYTHONPATH"].split(path_separator)] == [
            repo / "src",
            repo,
        ]
        assert Path(env["JUPYTER_PATH"]) == repo / ".venv/share/jupyter"
        assert Path(env["JAX_COMPILATION_CACHE_DIR"]) == output / ".cache/jax"
        assert Path(env["MPLCONFIGDIR"]) == output / ".cache/matplotlib"
        assert Path(env["IPYTHONDIR"]) == output / ".cache/ipython"
        assert Path(env["JUPYTER_RUNTIME_DIR"]) == output / ".cache/jupyter-runtime"
    assert (output / ".gitignore").read_text() == "*\n"
    identity = (output / "build-identity.txt").read_text()
    assert identity.endswith("state: success\n")
    assert f"checkout: {repo.as_posix()}" in identity
    if explicit:
        assert not (repo / "docs/_build/build-identity.txt").exists()


def test_docs_build_stops_after_failed_dtype(docs_builder, tmp_path, monkeypatch):
    """A failed float32 build must not be reported as a successful pair."""
    script, _, log = docs_builder
    monkeypatch.setenv("DOCS_BUILD_TEST_EXIT", "7")
    result = subprocess.run(script, cwd=tmp_path, capture_output=True)
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
    result = subprocess.run([*script, *args], cwd=tmp_path, capture_output=True)
    assert result.returncode == 2
    assert not log.exists()


def test_docs_build_setup_failure_invalidates_existing_identity(
    docs_builder, tmp_path, monkeypatch
):
    """Dependency setup must not leave an earlier custom-root success certified."""
    script, _, _ = docs_builder
    output = tmp_path / "portal output"
    args = [*script, "--output-dir", output.as_posix()]
    result = subprocess.run(args, cwd=tmp_path, capture_output=True)
    assert result.returncode == 0
    monkeypatch.setenv("DOCS_BUILD_TEST_SETUP_EXIT", "8")
    result = subprocess.run(args, cwd=tmp_path, capture_output=True)
    assert result.returncode == 8
    assert (output / "build-identity.txt").read_text() == "state: building\n"
