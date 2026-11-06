"""Release artifacts must match each other and the explicit release tag."""

import io
import runpy
import tarfile
import zipfile
from pathlib import Path

import pytest

validate = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/validate_release.py")
)["validate"]


def artifacts(root: Path, version="0.0.1a1", source_version=None):
    # Synthetic archive metadata, independent of uv and the package source.
    readme = b"# survjax\n"
    headers = (
        "Metadata-Version: 2.4\nName: survjax\nVersion: {}\n"
        "Description-Content-Type: text/markdown\n\n"
    )
    metadata = headers.format(version).encode() + readme
    wheel = root / f"survjax-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"survjax-{version}.dist-info/METADATA", metadata)
    with tarfile.open(root / f"survjax-{version}.tar.gz", "w:gz") as archive:
        for name, data in {
            "PKG-INFO": headers.format(source_version or version).encode() + readme,
            "README.md": readme,
        }.items():
            member = tarfile.TarInfo(f"survjax-{version}/{name}")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return wheel


def test_alpha_release_returns_the_exact_testable_wheel(tmp_path):
    """Spec 10.3/13: a matching alpha wheel and sdist pass the release gate."""
    wheel = artifacts(tmp_path)
    assert validate(tmp_path, "v0.0.1a1") == wheel.resolve()


def test_release_tag_must_match_package_version(tmp_path):
    """Spec 13: the release tag must identify the built package version."""
    artifacts(tmp_path)
    with pytest.raises(ValueError, match="tag"):
        validate(tmp_path, "v0.0.2")


@pytest.mark.parametrize("version", ["0.0.1.dev0", "0.0.1+local"])
def test_development_and_local_versions_cannot_publish(tmp_path, version):
    """Spec 12.2: development builds cannot accidentally become releases."""
    artifacts(tmp_path, version)
    with pytest.raises(ValueError, match="release version"):
        validate(tmp_path, f"v{version}")


def test_wheel_and_source_metadata_must_agree(tmp_path):
    """Spec 13: testing a wheel does not excuse a mismatched source archive."""
    artifacts(tmp_path, source_version="0.0.2")
    with pytest.raises(ValueError, match="metadata differ"):
        validate(tmp_path, "v0.0.1a1")


def test_unexpected_artifacts_are_rejected(tmp_path):
    """Spec 13: only the intended pair may enter the publishing handoff."""
    artifacts(tmp_path)
    (tmp_path / "extra.whl").touch()
    with pytest.raises(ValueError, match="exactly one"):
        validate(tmp_path, "v0.0.1a1")
