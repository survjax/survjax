Independent review: round 1/3

# GPU.1 — Modal runner and GPU test marker

Read PLAN.md GPU.1, then scripts/gpu_ci.py, tests/conftest.py,
pyproject.toml, and this review. Base commit: a3519520588cb638aff475cb4bcdec14b3d26741.

Invariants:
- The local entrypoint sends only a clean committed HEAD source archive. It
  never includes uncommitted files or repository credentials.
- Runtime is bounded by the CLI limit (at most 12 hours) and by Modal's
  12-hour function timeout. The modeled resource cost includes a 10-minute
  startup allowance and the script refuses estimates above the requested cap.
- JAX and jaxlib versions come from uv.lock and are both explicitly pinned
  when installing the CUDA 12 extra. Tests assert a working GPU backend.
- All collected tests outside tests/install carry the gpu marker; isolated
  wheel-install tests stay CPU-only. Both dtype modes and the install smoke
  tests run in the remote job.
- No public package API, dependencies, lockfile, spec tests or numerical code
  changed. The GitHub workflow is a separate remaining step.

Validation:
- `uv run ruff format --check scripts/gpu_ci.py tests/conftest.py`:
  2 files already formatted.
- `uv run ruff check scripts/gpu_ci.py tests/conftest.py`: All checks passed.
- `uv run mypy src`: Success, 22 source files.
- `uv lock --check`: Resolved 171 packages; lock unchanged.
- `python3 -m py_compile scripts/gpu_ci.py`: passed.
- `uv run --no-sync pytest --collect-only -q -m gpu tests/unit`: 1169 tests
  collected in 1.41s. Tests were collected, not executed on a GPU.
- `git diff --check`: clean.
- No Modal credentials or remote GPU execution were available, so provider
  invocation and GPU runtime remain unverified.

Pricing estimate uses Modal's current standard Function rates: T4 $0.000164/s,
CPU $0.0000131/core/s, memory $0.00000222/GiB/s. Two cores, 8 GiB and 12 hours
plus 10 minutes estimate to $9.11. This is an estimate guard, not a provider
enforced billing ceiling; Starter includes $30/month in compute credits.
GitHub workflow trigger and fork guard remain for the next step.
