"""Runner regressions: default gates cannot inherit stale step/feature filters."""

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path, bash_executable, script, args=()):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    target = scripts / script
    target.write_text((ROOT / "scripts" / script).read_text(), newline="\n")
    (scripts / "spec-features.txt").write_text(
        "# completed only\n observation.censoring # comment\n\nfamily.lognormal\n"
    )
    runner = scripts / "run_tests.sh"
    runner.write_text("#!/bin/sh\nexit 0\n", newline="\n")
    runner.chmod(0o755)
    (tmp_path / "tests").mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text(
        '#!/bin/sh\ncase "$*" in\n'
        '"rev-parse --show-toplevel") pwd ;;\n'
        '"rev-parse --abbrev-ref HEAD") echo meta-feature-gate ;;\n'
        "*) exit 1 ;;\nesac\n"
    )
    git.chmod(0o755)
    uv = bin_dir / "uv"
    stub = bin_dir / "fake_uv.py"
    stub.write_text(
        "import json, os, sys\n"
        "with open(os.environ['CALL_LOG'], 'a') as f:\n"
        "    f.write(json.dumps([sys.argv[1:], os.environ.get('SPEC_THROUGH'), "
        "os.environ.get('SPEC_FEATURES'), os.environ.get('JAX_ENABLE_X64')]) + '\\n')\n"
    )
    uv.write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(Path(os.sys.executable).as_posix())} "
        f'{shlex.quote(stub.as_posix())} "$@"\n',
        newline="\n",
    )
    uv.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    result = subprocess.run(
        [bash_executable, target.as_posix(), *args],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "CALL_LOG": str(log),
            "SURVJAX_WHEEL": "/unused/prebuilt.whl",
            "SURVJAX_JAX_CACHE": "off",
            "SURVJAX_X64_MODES": "1 0",
            "SPEC_THROUGH": "A1",
            "SPEC_FEATURES": "summary.rmst",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    calls = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    return result, calls


@pytest.mark.parametrize("script", ["check.sh", "update_slow_tests.sh"])
@pytest.mark.parametrize("args,limit", [((), ""), (("--through", "A3"), "A3")])
def test_gate_selection_uses_committed_features(
    tmp_path, bash_executable, script, args, limit
):
    result, calls = _run(tmp_path, bash_executable, script, args)
    assert result.returncode == 0, result.stderr
    spec = [call for call in calls if "tests/spec" in call[0]]
    assert [(call[1], call[2], call[3]) for call in spec] == [
        (limit, "observation.censoring,family.lognormal", mode) for mode in ("1", "0")
    ]


@pytest.mark.parametrize(
    "args", [("--through",), ("--through", "summary.rmst"), ("--unknown",)]
)
def test_check_rejects_bad_arguments_before_running_tools(
    tmp_path, bash_executable, args
):
    result, calls = _run(tmp_path, bash_executable, "check.sh", args)
    assert result.returncode == 2
    assert not calls
