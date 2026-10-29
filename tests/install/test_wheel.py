"""Install and type-check outside the source tree, offline (spec 10.2–10.3)."""

import json
import os
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

ROOT = Path(__file__).resolve().parents[2]


def _run(args, *, cwd, env):
    result = subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="session")
def wheel(tmp_path_factory):
    supplied = os.environ.get("SURVJAX_WHEEL")
    if supplied is not None:
        path = Path(supplied)
        assert path.is_absolute() and path.is_file() and path.suffix == ".whl", supplied
        return path.resolve()
    directory = tmp_path_factory.mktemp("wheel")
    _run(
        [
            "uv",
            "build",
            "--offline",
            "--no-python-downloads",
            "--wheel",
            "--out-dir",
            directory,
        ],
        cwd=ROOT,
        env=os.environ.copy(),
    )
    wheels = list(directory.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0].resolve()


@pytest.fixture(scope="session")
def installed(wheel, tmp_path_factory):
    directory = tmp_path_factory.mktemp("installed-wheel").resolve()
    assert ROOT not in directory.parents
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "MYPYPATH"):
        env.pop(name, None)
    env.update(UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never")
    # Spec 10.3: keep locked artifact URLs/hashes; avoid registry rediscovery.
    requirements = directory / "pylock.runtime.toml"
    _run(
        [
            "uv",
            "export",
            "--locked",
            "--no-default-groups",
            "--no-emit-project",
            "--format",
            "pylock.toml",
            "--output-file",
            requirements,
        ],
        cwd=ROOT,
        env=env,
    )
    _run(
        ["uv", "venv", "--python", sys.executable, directory / "venv"],
        cwd=directory,
        env=env,
    )
    python = directory / "venv" / "bin" / "python"
    _run(
        [
            "uv",
            "pip",
            "sync",
            "--python",
            python,
            "--only-binary",
            ":all:",
            requirements,
        ],
        cwd=directory,
        env=env,
    )
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--no-deps",
            "--only-binary",
            ":all:",
            wheel,
        ],
        cwd=directory,
        env=env,
    )
    _run(["uv", "pip", "check", "--python", python], cwd=directory, env=env)
    return python, directory, env


def test_wheel_contains_pure_python_package_and_type_marker(wheel):
    """Spec 10.2: ship a pure Python wheel with the public package and py.typed."""
    assert wheel.name.endswith("-py3-none-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert {"survjax/__init__.py", "survjax/py.typed"} <= set(archive.namelist())


def test_installed_core_uses_wheel_without_demo_dependencies(installed, x64_enabled):
    """Spec 10.1–10.3/12.2: installed runtime, metadata and exports are usable."""
    python, directory, env = installed
    script = textwrap.dedent("""
        import importlib.metadata as metadata
        import importlib.resources as resources
        import importlib.util
        import json
        import re
        import sys
        from pathlib import Path

        import jax
        import jax.numpy as jnp
        import numpy as np
        import survjax

        assert Path(sys.prefix) in Path(survjax.__file__).resolve().parents
        assert set(survjax.__all__) == {
            "constraints", "Exponential", "Weibull", "SurvivalData",
            "InvalidDataError", "InvalidParameterError", "SurvivalDistribution",
            "register_family", "param", "distribution", "static",
            "concordance_index", "concordance_index_with_info",
        }
        assert all(hasattr(survjax, name) for name in survjax.__all__)
        assert not hasattr(survjax, "log_likelihood")
        assert resources.files("survjax").joinpath("py.typed").is_file()
        meta = metadata.metadata("survjax")
        assert meta["Name"] == "survjax"
        assert meta["Version"] == "0.0.1.dev0"
        assert meta["Requires-Python"] == ">=3.12"
        assert meta["License-Expression"] == "Apache-2.0"
        assert {re.match("[A-Za-z0-9_.-]+", req).group(0)
                for req in meta.get_all("Requires-Dist")} == {"jax", "numpy"}
        for name in ("numpyro", "equinox", "optax", "optimistix", "matplotlib",
                     "pytest", "mypy", "sphinx"):
            assert importlib.util.find_spec(name) is None, name
            assert not any(m.split(".")[0] == name for m in sys.modules), name
        data = survjax.SurvivalData(jnp.array([1.0, 2.0]), jnp.array([True, False]))
        assert data.issues.ok
        rows = survjax.Exponential(rate=jnp.asarray(0.5)).log_likelihood(data)
        print(json.dumps({"rows": rows.tolist(), "dtype": str(rows.dtype),
                          "python": sys.version.split()[0], "jax": jax.__version__,
                          "numpy": np.__version__, "package": survjax.__file__}))
    """)
    report = json.loads(_run([python, "-I", "-c", script], cwd=directory, env=env))
    print(report)
    # Exact log(r)-r*t at t=1 and right log S=-r*t at t=2, r=0.5.
    np.testing.assert_allclose(
        report["rows"],
        [np.log(0.5) - 0.5, -1.0],
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    assert report["dtype"] == ("float64" if x64_enabled else "float32")


def test_installed_custom_family_is_typed_and_differentiable(installed, x64_enabled):
    """Spec 4.2/7.3/10.3: a typed external family uses installed generic defaults."""
    python, directory, env = installed
    example = directory / "custom_family.py"
    shutil.copyfile(Path(__file__).with_name("custom_family.py"), example)
    report = json.loads(_run([python, "-I", example], cwd=directory, env=env))
    # log S=-r*t**2, log f=log(2*r*t)-r*t**2. At r=.5, t=[1,2],
    # e=[1,0]: rows=[-.5,-2], sum'=1/r-5=-3, sum''=-1/r**2=-4.
    for key, expected in (
        ("rows", [-0.5, -2.0]),
        ("gradient", -3.0),
        ("hessian", -4.0),
    ):
        np.testing.assert_allclose(
            report[key], expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )
    print(
        _run(
            [
                sys.executable,
                "-I",
                "-m",
                "mypy",
                "--strict",
                "--no-incremental",
                "--python-executable",
                python,
                example,
            ],
            cwd=directory,
            env=env,
        )
    )
