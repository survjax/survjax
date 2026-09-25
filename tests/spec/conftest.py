"""Self-contained configuration for the Chronox specification tests.

- Dtype mode: taken from the ``JAX_ENABLE_X64`` environment variable, which the
  runner sets. Tests never set it; the suite runs once per mode.
- ``step`` marker: every test carries exactly one ``@pytest.mark.step("A2")``.
  ``SPEC_THROUGH=A3`` deselects every test marked with a later step; unset
  means run everything.
- ``spec_fixture``: loader for committed reference fixtures in ``fixtures/``.
"""

import json
import os
import re
from pathlib import Path

import jax
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures"

# JAX's own parsing of boolean environment variables.
_TRUTHY = {"1", "true", "t", "yes", "y", "on"}
X64 = os.environ.get("JAX_ENABLE_X64", "").strip().lower() in _TRUTHY

_STEP_RE = re.compile(r"^([A-Z])([0-9]+)$")


def _step_key(step):
    match = _STEP_RE.match(str(step))
    if match is None:
        raise pytest.UsageError(f"step {step!r} is not of the form 'A2'")
    return match.group(1), int(match.group(2))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "step(name): delivery step a spec test belongs to, e.g. step('A2'); "
        "SPEC_THROUGH=A3 deselects later steps",
    )


def pytest_collection_modifyitems(config, items):
    through = os.environ.get("SPEC_THROUGH", "").strip()
    limit = _step_key(through) if through else None

    unmarked, keep, drop = [], [], []
    for item in items:
        if HERE not in Path(str(item.path)).resolve().parents:
            keep.append(item)
            continue
        marks = list(item.iter_markers(name="step"))
        if len(marks) != 1 or len(marks[0].args) != 1:
            unmarked.append(item.nodeid)
            continue
        key = _step_key(marks[0].args[0])
        (keep if limit is None or key <= limit else drop).append(item)

    if unmarked:
        raise pytest.UsageError(
            "spec tests need exactly one step marker: " + ", ".join(unmarked)
        )
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


@pytest.fixture(scope="session", autouse=True)
def _dtype_mode_matches_environment():
    """The runner's JAX_ENABLE_X64 and JAX's active mode must agree."""
    if bool(jax.config.jax_enable_x64) != X64:
        pytest.fail(
            "JAX_ENABLE_X64 environment variable and jax.config.jax_enable_x64 "
            "disagree; set the variable before starting pytest"
        )


@pytest.fixture(scope="session")
def x64():
    """True when the suite runs with JAX 64-bit mode enabled."""
    return X64


@pytest.fixture(scope="session")
def float_dtype():
    """The default floating dtype of the active mode."""
    return np.dtype(np.float64 if X64 else np.float32)


def load_fixture(name):
    """Load a committed reference fixture by file name.

    Supported: ``.json`` (parsed), ``.npz`` (dict of arrays), ``.csv`` (numpy
    structured array with a header row). A missing fixture fails the test: a
    test that needs reference values must never pass without them.
    """
    path = FIXTURE_DIR / name
    if not path.is_file():
        pytest.fail(f"committed fixture {name!r} not found in {FIXTURE_DIR}")
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix == ".npz":
        with np.load(path) as archive:
            return {key: archive[key] for key in archive.files}
    if path.suffix == ".csv":
        return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    pytest.fail(f"unsupported fixture format: {path.suffix}")


@pytest.fixture(scope="session")
def spec_fixture():
    """Return the committed-fixture loader, ``spec_fixture("name.npz")``."""
    return load_fixture
