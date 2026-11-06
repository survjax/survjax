"""Validate the exact distributions handed from release tests to publishing."""

import argparse
import re
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def validate(directory: Path, tag: str) -> Path:
    """Check release metadata and return the sole wheel's absolute path."""
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    files = [p for p in directory.iterdir() if not p.name.startswith(".")]
    if len(wheels) != 1 or len(sources) != 1 or len(files) != 2:
        raise ValueError("Expected exactly one wheel and one source archive")
    wheel, source = wheels[0], sources[0]
    with zipfile.ZipFile(wheel) as archive:
        (metadata_path,) = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        wheel_metadata = BytesParser().parsebytes(archive.read(metadata_path))
    version = wheel_metadata["Version"]
    if not version or not re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", version):
        raise ValueError("Expected a release version without dev/local suffixes")
    if wheel_metadata["Name"] != "survjax" or tag != f"v{version}":
        raise ValueError("Release tag must equal v<survjax version>")
    if wheel.name != f"survjax-{version}-py3-none-any.whl":
        raise ValueError("Unexpected wheel filename")
    if source.name != f"survjax-{version}.tar.gz":
        raise ValueError("Unexpected source archive filename")
    with tarfile.open(source) as archive:
        prefix = f"survjax-{version}"
        source_metadata = BytesParser().parsebytes(
            archive.extractfile(f"{prefix}/PKG-INFO").read()
        )
        readme = archive.extractfile(f"{prefix}/README.md").read()
    # Require identical metadata/description, independent of header ordering.
    if sorted(wheel_metadata.items()) != sorted(source_metadata.items()):
        raise ValueError("Wheel and source metadata differ")
    if wheel_metadata["Description-Content-Type"] != "text/markdown":
        raise ValueError("Expected Markdown project description")
    if not readme or wheel_metadata.get_payload(decode=True) != readme:
        raise ValueError("Source README differs from wheel description")
    if source_metadata.get_payload(decode=True) != readme:
        raise ValueError("Source README differs from source description")
    return wheel.resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("tag")
    args = parser.parse_args()
    print(validate(args.directory, args.tag))
