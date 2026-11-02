"""Run the current commit's GPU-marked tests on a short-lived Modal T4."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

import modal

HOURS = 12
BUDGET_USD = 10.0
GPU_PER_SECOND = 0.000164
CPU_PER_CORE_SECOND = 0.0000131
MEMORY_PER_GIB_SECOND = 0.00000222
CPU_CORES = 2
MEMORY_GIB = 8
MODAL_VERSION = "1.6.3"
UV_VERSION = "0.11.19"

app = modal.App("survjax-gpu-tests")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(f"uv=={UV_VERSION}")
)


def _run(
    command: list[str], *, cwd: Path, env: dict[str, str], log: Path, deadline: float
) -> int:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        log.write_text("Configured runtime limit reached before this command.\n")
        return 124
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=remaining,
            check=False,
        )
        output = result.stdout + result.stderr
        log.write_text(output)
        print(output, flush=True)
        return result.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        log.write_text(str(output) + "\nConfigured runtime limit reached.\n")
        print(log.read_text(), flush=True)
        return 124


@app.function(
    image=image,
    gpu="T4",
    cpu=CPU_CORES,
    memory=MEMORY_GIB * 1024,
    timeout=HOURS * 3600,
    startup_timeout=600,
)
def run_gpu_tests(commit: str, source_archive: bytes, runtime_hours: float) -> bytes:
    import os
    import tempfile

    with tempfile.TemporaryDirectory(prefix="survjax-gpu-") as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        with tarfile.open(fileobj=io.BytesIO(source_archive), mode="r:gz") as archive:
            archive.extractall(repo, filter="data")

        results = root / "results"
        results.mkdir()
        (results / "commit.txt").write_text(f"{commit}\n")
        env = dict(os.environ)
        env.update(UV_LOCKED="true", UV_PYTHON_DOWNLOADS="never")
        deadline = time.monotonic() + runtime_hours * 3600
        status = 0

        commands = [
            ("environment", ["uv", "sync", "--locked", "--group", "dev"]),
            ("cuda-install", []),
        ]
        for name, command in commands:
            if name == "cuda-install":
                import tomllib

                lock = tomllib.loads((repo / "uv.lock").read_text())
                jax_version = next(
                    package["version"]
                    for package in lock["package"]
                    if package["name"] == "jax"
                )
                jaxlib_version = next(
                    package["version"]
                    for package in lock["package"]
                    if package["name"] == "jaxlib"
                )
                command.extend(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        ".venv/bin/python",
                        f"jax[cuda12]=={jax_version}",
                        f"jaxlib=={jaxlib_version}",
                    ]
                )
            code = _run(
                command,
                cwd=repo,
                env=env,
                log=results / f"{name}.log",
                deadline=deadline,
            )
            if code:
                status = code
                break

        if status == 0:
            import tomllib

            lock = tomllib.loads((repo / "uv.lock").read_text())
            versions = {
                package["name"]: package["version"]
                for package in lock["package"]
                if package["name"] in {"jax", "jaxlib"}
            }
            (results / "versions.json").write_text(json.dumps(versions, indent=2))
            build_dir = results / "wheel"
            build_dir.mkdir()
            code = _run(
                ["uv", "build", "--wheel", "--out-dir", str(build_dir)],
                cwd=repo,
                env=env,
                log=results / "wheel-build.log",
                deadline=deadline,
            )
            if code:
                status = code

        wheel_paths = list((results / "wheel").glob("*.whl"))
        if status == 0 and len(wheel_paths) != 1:
            status = 1
        if wheel_paths:
            env["SURVJAX_WHEEL"] = str(wheel_paths[0])

        if status == 0:
            for mode in ("1", "0"):
                mode_env = dict(
                    env,
                    JAX_ENABLE_X64=mode,
                    JAX_PLATFORMS="cuda",
                    XLA_PYTHON_CLIENT_PREALLOCATE="false",
                )
                label = "x64" if mode == "1" else "x32"
                probe = [
                    ".venv/bin/python",
                    "-c",
                    "import jax,jax.numpy as j; x=jax.jit(lambda x:x*x+1)"
                    "(j.arange(4.)); x.block_until_ready(); "
                    "assert all(d.platform=='gpu' for d in x.devices()),x.devices(); "
                    "print(jax.__version__,jax.devices(),x.dtype)",
                ]
                runs = [
                    ("device", probe),
                    (
                        "tests",
                        [
                            ".venv/bin/python",
                            "-m",
                            "pytest",
                            "tests",
                            "-m",
                            "gpu",
                            "--ignore=tests/install",
                            "-q",
                            "-n",
                            "0",
                        ],
                    ),
                    (
                        "doctests",
                        [
                            ".venv/bin/python",
                            "-m",
                            "pytest",
                            "src",
                            "--doctest-modules",
                            "-q",
                        ],
                    ),
                    (
                        "install",
                        [
                            ".venv/bin/python",
                            "-m",
                            "pytest",
                            "tests/install",
                            "-q",
                            "-n",
                            "0",
                        ],
                    ),
                ]
                for name, command in runs:
                    test_env = dict(mode_env)
                    if name == "install":
                        test_env["JAX_PLATFORMS"] = "cpu"
                    code = _run(
                        command,
                        cwd=repo,
                        env=test_env,
                        log=results / f"{name}-{label}.log",
                        deadline=deadline,
                    )
                    if code:
                        status = code
                if status == 124:
                    break

        (results / "exit-code.txt").write_text(f"{status}\n")
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in results.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(results))
        return output.getvalue()


@app.local_entrypoint()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-runtime-hours", type=float, default=HOURS)
    parser.add_argument("--max-cost-usd", type=float, default=BUDGET_USD)
    arguments = parser.parse_args()
    if not 0 < arguments.max_runtime_hours <= HOURS:
        raise SystemExit(f"Runtime must be greater than 0 and at most {HOURS} hours.")
    estimated_cost = (arguments.max_runtime_hours * 3600 + 600) * (
        GPU_PER_SECOND
        + CPU_CORES * CPU_PER_CORE_SECOND
        + MEMORY_GIB * MEMORY_PER_GIB_SECOND
    )
    if arguments.max_cost_usd <= 0 or estimated_cost > arguments.max_cost_usd:
        raise SystemExit(
            f"Configured {arguments.max_runtime_hours:g}-hour estimate is "
            f"${estimated_cost:.2f}; "
            f"requested cap is ${arguments.max_cost_usd:.2f}. No job started."
        )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if dirty:
        raise SystemExit("Commit or stash changes before a GPU run; only HEAD is sent.")
    source = subprocess.check_output(["git", "archive", "--format=tar.gz", "HEAD"])
    print(
        f"Commit {commit}; T4; hard timeout {arguments.max_runtime_hours:g} h; "
        f"estimated maximum ${estimated_cost:.2f}."
    )
    result = run_gpu_tests.remote(commit, source, arguments.max_runtime_hours)
    artifact = Path(tempfile.gettempdir()) / f"survjax-gpu-{commit[:12]}.zip"
    artifact.write_bytes(result)
    print(f"Results: {artifact}")
    with zipfile.ZipFile(artifact) as archive:
        status = int(archive.read("exit-code.txt").decode().strip())
    if status:
        raise SystemExit(status)


if __name__ == "__main__":
    print(f"Modal SDK {MODAL_VERSION}; maximum runtime {HOURS} hours.")
