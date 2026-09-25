# Slice A implementation plan

Status: **A0 implemented; awaiting maintainer review before A1.**

The maintainer also requested moving the read-only source documents from the
repository root to `spec/`. This changes their paths without editing their
contents. The initial Git commit already exists, so A0 does not reinitialize Git.

## Step A0 — package and quality-tool scaffold

One concept: make an empty `chronox-jax` package installable and establish the
required local checks. End the step with one commit and `REVIEW.md`, then stop.

Files to create:

- `spec/GOALS.md` and `spec/SPECIFICATION.md`: move the existing read-only
  documents unchanged to the documented layout.
- `pyproject.toml`: version `0.0.1.dev0`, Python 3.12, `uv_build`, Apache-2.0
  metadata, runtime dependencies (`jax`, `numpy`), test and development groups,
  Ruff rules, and mypy settings. Select compatible dependency bounds during the
  step, record the actual versions, and justify lower bounds with checks.
- `uv.lock` and `.python-version`: locked CPU environment and Python 3.12.
- `.pre-commit-config.yaml`: the section 10.1 hooks, including a local mypy
  hook through uv.
- `.gitignore` and `LICENSE`: generated-file exclusions and Apache-2.0 text.
- `src/chronox_jax/__init__.py` and `src/chronox_jax/py.typed`: empty public
  package and typing marker. No feature modules yet.
- `scripts/run_tests.sh`: run the same test suite in separate processes with
  `JAX_ENABLE_X64=1` and `JAX_ENABLE_X64=0`.
- `tests/unit/test_import.py`: one installed-package import smoke test.

Public API touched: the `chronox_jax` import namespace only; no names exported
from `__init__.py` yet.

Checks for the step: `uv run ruff format --check .`, `uv run ruff check .`,
`uv run mypy src`, `scripts/run_tests.sh`, and a package build/import check.
Report exact results and selected versions in `REVIEW.md`. The generated lockfile
may exceed the usual 300-line step target; the handwritten code and configuration
should stay small.

Spec coverage: 10.1 (build, license, quality tools), 10.2 (name, layout, version,
explicit exports), 10.3 (Python, dependencies, typing marker, two dtype modes),
and the installation foundation for 12.2 Slice A. This step establishes the
scaffold only; later steps supply the Slice A behavior and its evidence.

## Next steps after A0 review

Follow the A1–A9 sequence in `CLAUDE.md`. Write a fresh, concrete plan for each
step and wait for approval before its code changes. Each step ends with its own
commit and an overwritten `REVIEW.md`.
