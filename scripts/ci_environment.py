"""Record the built wheel and the CPU environment of a CI job.

Usage: ``uv run python scripts/ci_environment.py`` after ``uv build --wheel``.

Prints the platform, Python, JAX and NumPy versions and the JAX devices, and
fails unless every device is a CPU. Exports ``SURVJAX_WHEEL`` (the single wheel
in ``dist/``) to later steps through ``$GITHUB_ENV`` when that is set, so
``scripts/check.sh`` tests that wheel instead of building another one.
"""

import os
import platform
import sys
from pathlib import Path

import jax
import numpy


def main() -> None:
    (wheel,) = Path("dist").glob("*.whl")
    wheel_path = wheel.resolve()
    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as env:
            env.write(f"SURVJAX_WHEEL={wheel_path}\n")
    print(f"wheel: {wheel_path}")
    print(platform.platform(), platform.machine(), sys.version)
    print(f"JAX {jax.__version__}, NumPy {numpy.__version__}")
    print(jax.devices())
    assert all(device.platform == "cpu" for device in jax.devices())


if __name__ == "__main__":
    main()
