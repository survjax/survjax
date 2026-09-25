#!/bin/sh
set -eu

JAX_ENABLE_X64=1 uv run --group test pytest -q
JAX_ENABLE_X64=0 uv run --group test pytest -q
