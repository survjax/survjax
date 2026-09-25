# Review: A0 package scaffold

Read in this order:

1. `PLAN.md` for the approved scope and requested document move.
2. `pyproject.toml`, `.pre-commit-config.yaml`, and `scripts/run_tests.sh` for
   package metadata, tools, and dtype-mode execution.
3. `src/chronox_jax/__init__.py`, `src/chronox_jax/py.typed`, and
   `tests/unit/test_import.py` for the empty public package and smoke test.
4. `LICENSE`, `.python-version`, `.gitignore`, and generated `uv.lock`.
5. The Git rename diff for `spec/GOALS.md` and `spec/SPECIFICATION.md`.

## Invariants

- Distribution name is `chronox-jax`; import name is `chronox_jax`; there is no
  top-level `chronox` package. `__all__` is empty until a later step adds API.
- The package contains `py.typed` and includes the Apache-2.0 license in the
  wheel. The notice names Wouter A.C. van Amsterdam, as confirmed by the
  maintainer.
- `scripts/run_tests.sh` starts separate Python processes with
  `JAX_ENABLE_X64=1` and `JAX_ENABLE_X64=0`.
- The two specification documents were moved without content changes. Their
  Git blob hashes match the initial commit.
- `spec/` is excluded from Ruff formatting because its read-only Python code
  examples do not match the formatter's current style.

## Evidence

| Check | Result | Requirement |
| --- | --- | --- |
| `uv lock` and `uv sync --dev` | 33 packages resolved; environment installed | spec 10.3 |
| `uv run ruff format --check .` | `5 files already formatted` | spec 10.1 |
| `uv run ruff check .` | `All checks passed!` | spec 10.1 |
| `uv run mypy src` | `Success: no issues found in 1 source file` | spec 10.3 |
| `scripts/run_tests.sh` | `1 passed` in each of two dtype modes | spec 10.2–10.3; `test_public_package_imports` checks the package import and current exports |
| `pre-commit run --all-files` | All 11 hooks passed | spec 10.1 |
| `uv build` | Wheel and source archive built | spec 10.1–10.2 |
| Fresh environment outside the source tree, wheel installed with `--no-deps` | Import, version, exports, `py.typed`, and absence of a top-level `chronox` import passed | spec 10.2–10.3 |
| Git blob hashes for both moved documents | Both match the initial commit | Read-only source preservation |

The test runs used Python 3.12.9 on macOS ARM. The locked direct dependencies
are JAX 0.11.2, NumPy 2.5.3, pytest 9.1.1, mpmath 1.4.1, Hypothesis 6.168.1,
Ruff 0.16.9, mypy 2.3.1, and pre-commit 4.6.2. Those tested versions are the
declared lower bounds. The build check used uv 0.11.19, matching the declared
`uv_build` lower bound. No other Python/JAX combination was checked.

## Uncertainty and remaining work

- This is the empty A0 package. It does not establish any numerical or data
  behavior from the rest of Slice A.
- The source archive and wheel build, but the full wheel smoke test in A9 is
  still required. The temporary A0 environment installed the wheel without
  dependencies to check only the package contents and import.
- No R fixture generation or R checks were run; they belong to A8.
- A Linux CI job is still required by spec 10.3 because this repository has a
  remote. It needs its own approved step before Slice A is complete.
