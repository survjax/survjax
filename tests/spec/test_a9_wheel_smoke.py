"""A9: wheel smoke test outside the source tree (spec 10.2, 10.3, 12.1, 12.2; rev 34).

10.3: "Smoke-test wheel installation and imports outside the source tree in
Slice A"; 12.2 lists a "wheel smoke test" in the Slice A gate. The spec tests
do not build the wheel: the runner builds it (``uv build``) and passes its
path in ``SURVJAX_WHEEL`` (Q-040). Without that variable these tests fail, so
the gate cannot pass without a wheel.

A pure-Python wheel is installed by unpacking it into a directory that then
goes first on ``sys.path``; this is what an installer does for such a wheel and
needs no network ("Regular tests and CI never need network", 10.3). Runtime
dependencies (``jax``, ``numpy``) come from the test environment.
"""

import os
import re
import subprocess
import sys
import textwrap
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

WHEEL_VARIABLE = "SURVJAX_WHEEL"


@pytest.fixture(scope="module")
def wheel():
    value = os.environ.get(WHEEL_VARIABLE, "").strip()
    if not value:
        pytest.fail(f"{WHEEL_VARIABLE} must name a built survjax wheel (Q-040)")
    path = Path(value).resolve()
    if not path.is_file() or path.suffix != ".whl":
        pytest.fail(f"{WHEEL_VARIABLE}={value!r} is not a wheel file")
    return path


@pytest.fixture(scope="module")
def installed(wheel, tmp_path_factory):
    target = tmp_path_factory.mktemp("site")
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(target)
    return target


def _metadata(wheel):
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        assert len(names) == 1, names
        return Parser().parsestr(archive.read(names[0]).decode("utf-8"))


@pytest.mark.step("A9")
def test_wheel_is_pure_python(wheel):
    """spec 10.2, rev 4: "no custom build hooks; no compiled extensions" (a `py3-none-any` wheel)."""
    assert wheel.name.endswith("-py3-none-any.whl")


@pytest.mark.step("A9")
def test_wheel_contains_package_and_type_marker(wheel):
    """spec 10.2, rev 4: the `src/survjax` layout ships `survjax/__init__.py` and `py.typed`."""
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "survjax/__init__.py" in names
    assert "survjax/py.typed" in names


@pytest.mark.step("A9")
def test_wheel_metadata_names_survjax_and_python_312(wheel):
    """spec 10.2 and 10.3, rev 4: metadata `Name: survjax`, `Requires-Python: >=3.12`."""
    meta = _metadata(wheel)
    assert meta["Name"] == "survjax"
    assert meta["Requires-Python"].replace(" ", "") == ">=3.12"


@pytest.mark.step("A9")
def test_wheel_requires_no_demo_dependencies(wheel):
    """spec 10.3, rev 4: "Demo dependencies live in development groups" — not in the wheel's requirements."""
    requires = _metadata(wheel).get_all("Requires-Dist") or []
    names = {re.match(r"[A-Za-z0-9_.-]+", r.strip()).group(0).lower() for r in requires}
    assert "jax" in names
    for demo in ("numpyro", "equinox", "optax", "optimistix", "matplotlib"):
        assert demo not in names


@pytest.mark.step("A9")
def test_installed_wheel_imports_and_evaluates_outside_the_source_tree(installed, tmp_path):
    """spec 10.3 and 12.2, rev 34: the installed wheel imports from outside the source tree and evaluates a likelihood.

    Runs in a fresh interpreter whose working directory is an empty
    directory; ``survjax`` must come from the unpacked wheel.
    """
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        site = Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(site))

        import numpy as np
        import survjax

        assert site in Path(survjax.__file__).resolve().parents, survjax.__file__
        data = survjax.SurvivalData(np.array([1.0, 2.0]), np.array([True, False]))
        value = np.asarray(survjax.Exponential(rate=np.array(0.5)).log_likelihood(data))
        expected = np.array([np.log(0.5) - 0.5, -1.0])
        assert np.allclose(value, expected, rtol=1e-4, atol=1e-5), value
        assert bool(data.issues.ok)
        """
    )
    workdir = tmp_path / "elsewhere"
    workdir.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", script, str(installed)],
        capture_output=True,
        text=True,
        cwd=workdir,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
