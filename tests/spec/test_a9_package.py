"""A9: public exports, distribution metadata, type marker and core imports (spec 10.1–10.3, 12.1; rev 34).

12.1 (Extension and install): "installed wheel and source archive; public
exports and type marker; core imports without demo dependencies". The wheel
itself is checked in ``test_a9_wheel_smoke.py``; the source archive is
checked "before release" (10.3), not in Slice A.

Only names whose slice is Slice A are required here: 10.2 says "Create
modules when their slice needs them". Kaplan–Meier and the metrics other than
concordance (Slice E), wrappers and other families (Slice D) are not asserted.
The concordance exports are in ``test_a10_concordance_api.py``.
"""

import importlib.metadata
import importlib.resources
import subprocess
import sys
import textwrap

import pytest

import survjax

DEMO_DEPENDENCIES = ("numpyro", "equinox", "optax", "optimistix", "matplotlib")


@pytest.mark.step("A9")
@pytest.mark.parametrize(
    "name",
    [
        "SurvivalData",
        "Exponential",
        "Weibull",
        "InvalidDataError",
        "InvalidParameterError",
    ],
)
def test_slice_a_names_are_exported_at_top_level(name):
    """spec 10.2, rev 35: "Re-export the data object, distributions, ... `InvalidParameterError` beside `InvalidDataError`" at top level (no likelihood function since rev 35)."""
    assert hasattr(survjax, name)


@pytest.mark.step("A9")
def test_package_has_type_marker():
    """spec 10.2 and 10.3, rev 4: "Include `py.typed`"."""
    assert importlib.resources.files("survjax").joinpath("py.typed").is_file()


@pytest.mark.step("A9")
def test_distribution_name_is_survjax():
    """spec 10.2, rev 4: distribution metadata `name = "survjax"`."""
    assert importlib.metadata.metadata("survjax")["Name"] == "survjax"


@pytest.mark.step("A9")
def test_distribution_requires_python_312():
    """spec 10.3, rev 4: "Python 3.12 (`requires-python >=3.12`)"."""
    requires = importlib.metadata.metadata("survjax")["Requires-Python"]
    assert requires.replace(" ", "") == ">=3.12"


@pytest.mark.step("A9")
def test_distribution_version_is_not_before_the_initial_version():
    """spec 10.2, rev 4: "Initial version `0.0.1.dev0`": the installed version is a valid PEP 440 version no earlier than `0.0.1.dev0` (release bumps such as `0.0.1a2` are allowed; Q-041)."""
    from packaging.version import InvalidVersion, Version

    raw = importlib.metadata.version("survjax")
    try:
        version = Version(raw)
    except InvalidVersion:
        pytest.fail(f"survjax version {raw!r} is not a PEP 440 version")
    assert version >= Version("0.0.1.dev0"), raw


@pytest.mark.step("A9")
def test_distribution_license_metadata_is_apache():
    """spec 10.1, rev 4: original code is Apache-2.0, with "matching metadata" (Q-041)."""
    meta = importlib.metadata.metadata("survjax")
    fields = [meta.get("License-Expression") or "", meta.get("License") or ""]
    fields += meta.get_all("Classifier") or []
    assert any("Apache" in value for value in fields)


@pytest.mark.step("A9")
def test_core_imports_and_runs_without_demo_dependencies():
    """spec 10.3, rev 4: "core code imports and runs without NumPyro, Equinox, Optax, Optimistix, R or plotting libraries".

    A fresh interpreter in which importing any demo dependency fails imports
    survjax and its Slice A public modules and evaluates a likelihood.
    """
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        BLOCKED = {DEMO_DEPENDENCIES!r}

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError("demo dependency blocked: " + name)
                return None

        sys.meta_path.insert(0, Block())

        import numpy as np
        import survjax
        import survjax.constraints
        import survjax.distributions

        data = survjax.SurvivalData(np.array([1.0, 2.0]), np.array([True, False]))
        value = survjax.Exponential(rate=np.array(0.5)).log_likelihood(data)
        assert np.all(np.isfinite(np.asarray(value)))
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
        assert not loaded, loaded
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
