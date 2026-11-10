"""Git Bash discovery for the shell verification tooling (spec 10.3)."""

from types import SimpleNamespace

import conftest
import pytest


@pytest.mark.parametrize(
    "git_path", ["cmd/git.exe", "bin/git.exe", "ucrt64/bin/git.exe"]
)
def test_finds_bash_from_git_installation(tmp_path, monkeypatch, git_path):
    """The Bash launcher lives at the installation root, regardless of PATH."""
    root = tmp_path / "Program Files" / "Git"
    git = root / git_path
    git.parent.mkdir(parents=True)
    git.touch()
    bash = root / "bin/bash.exe"
    bash.parent.mkdir(exist_ok=True)
    bash.touch()
    monkeypatch.setattr(conftest, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(conftest.shutil, "which", lambda name: str(git))

    assert conftest.bash_executable.__wrapped__() == str(bash)


def test_missing_git_bash_fails_explicitly(tmp_path, monkeypatch):
    """Never fall back to the system WSL Bash launcher (spec 10.3 tooling)."""
    git = tmp_path / "Git/cmd/git.exe"
    git.parent.mkdir(parents=True)
    git.touch()
    monkeypatch.setattr(conftest, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(conftest.shutil, "which", lambda name: str(git))

    with pytest.raises(AssertionError, match="Git Bash not found"):
        conftest.bash_executable.__wrapped__()
