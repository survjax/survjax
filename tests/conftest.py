"""Shared test fixtures; dtype modes are selected by scripts/run_tests.sh."""

import os

import pytest


@pytest.fixture
def x64_enabled() -> bool:
    """Read the process mode without changing JAX configuration."""
    return os.environ["JAX_ENABLE_X64"] == "1"
