#!/bin/sh
# The implementer's own tests, one process per dtype mode.
# PYTEST_WORKERS sets the pytest-xdist worker count (default 6; 0 runs serially).
set -eu

workers="${PYTEST_WORKERS:-6}"
JAX_ENABLE_X64=1 uv run --group test pytest -q -n "$workers" --ignore=tests/spec
JAX_ENABLE_X64=0 uv run --group test pytest -q -n "$workers" --ignore=tests/spec
