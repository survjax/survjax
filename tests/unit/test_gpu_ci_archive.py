"""Regression tests for the Modal runner's source archive."""

import importlib.util
import io
import sys
import tarfile
from pathlib import Path
from types import ModuleType


def _load_runner(monkeypatch):
    """Load the runner without requiring the separately installed Modal SDK."""
    modal = ModuleType("modal")

    class FakeImage:
        def apt_install(self, *args, **kwargs):
            return self

        def pip_install(self, *args, **kwargs):
            return self

    class FakeApp:
        def function(self, **kwargs):
            return lambda function: function

        def local_entrypoint(self):
            return lambda function: function

    modal.App = lambda name: FakeApp()
    modal.Image = type("Image", (), {"debian_slim": lambda **kwargs: FakeImage()})
    monkeypatch.setitem(sys.modules, "modal", modal)

    runner_path = Path(__file__).parents[2] / "scripts" / "gpu_ci.py"
    spec = importlib.util.spec_from_file_location("survjax_gpu_ci", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def test_source_archive_can_be_safely_extracted(tmp_path, monkeypatch):
    runner = _load_runner(monkeypatch)
    source = runner._source_archive("HEAD")

    destination = tmp_path / "repo"
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(source), mode="r:gz") as archive:
        assert "spec" not in archive.getnames()
        archive.extractall(destination, filter="data")

    assert (destination / "pyproject.toml").is_file()
