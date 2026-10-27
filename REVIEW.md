# A9.2 — Installed wheel and out-of-tree extension

Read `scripts/check_wheel.sh`, then `tests/install/test_wheel.py`, then
`tests/install/custom_family.py`, then the active A9.2 entry in PLAN.md.
Approval: "review passed, plan approved, on to A9.2". Branch `A9`, starting
commit `8e7a7f4`. New script/test/example content totals 281 lines. Library
code, exports, dependencies, lockfile, shared tolerances, spec, independent
tests and protected scripts are unchanged.

Invariants and spec-to-test mapping:

- `scripts/check_wheel.sh` (10.3, 12.1–12.2) builds one wheel into a fresh
  temporary directory, retains it for inspection and supplies its absolute
  path through SURVJAX_WHEEL to the unchanged gate. Default is through A9;
  explicit arguments permit the available A7 cumulative gate while A8 is
  absent. Offline execution, disabled Python downloads and a locked project
  are inherited by child processes. Gate errors propagate unchanged.
- `test_wheel_contains_pure_python_package_and_type_marker` (10.2) checks
  the py3-none-any wheel, import package and py.typed marker. Own tests build
  a wheel once per test session when SURVJAX_WHEEL is absent, so the standard
  tests runner needs no manual artifact setup. An explicitly supplied invalid
  path fails instead of silently building another artifact.
- The installed fixture (10.3) exports the existing lock's runtime closure as
  pylock.toml, preserving pinned versions, artifact URLs and hashes. It
  creates an unseeded virtual environment outside the checkout, syncs cached
  binary runtime artifacts offline, installs the wheel without dependency
  resolution and runs uv pip check. Test/docs/demo groups are excluded.
- `test_installed_core_uses_wheel_without_demo_dependencies` (10.1–10.3,
  12.2) starts isolated Python (-I) from the external directory. It verifies
  the package comes from that virtual environment, exact public exports,
  absence of the removed likelihood function, type marker, distribution
  name/version/Python/license metadata and exactly JAX/NumPy runtime
  requirements. It checks demo, test and docs packages are unavailable and
  unimported, then evaluates the current likelihood with hand-derived values
  and the runner-selected dtype. No source path or editable install is used.
- `test_installed_custom_family_is_typed_and_differentiable` (4.2, 7.3,
  10.3) copies the annotated custom family outside the checkout, runs it in
  the fresh environment, and checks generic-density values and jitted
  parameter gradient/Hessian against explicit algebra. Existing mypy runs
  in strict mode outside the checkout with the installed Python as its
  package-discovery target. The external example uses only public API.

Expected values are independent formulas stated beside assertions. All
comparison tolerances come from tests/impl_tolerances.py. Tests never change
JAX dtype mode; each process inherits JAX_ENABLE_X64. No network, R, optional
solver, plotting or demo dependency is used. A prepopulated locked runtime
cache remains an environment-setup prerequisite; a missing cache fails.
The build uses uv's native compatible uv_build implementation; the build
requirement in pyproject.toml is unchanged.

Verification results:

- `JAX_ENABLE_X64=<1|0> UV_CACHE_DIR=/private/tmp/survjax-uv-cache
  uv run --locked --offline pytest tests/install -q -s`: 3 passed in 17.16s
  (float64), 3 passed in 21.98s (float32), including external strict mypy:
  `Success: no issues found in 1 source file`. These focused runs exercised
  the default build-once fixture. The final extra runtime-requirement metadata
  assertion is exercised in the cumulative gate.
- `JAX_ENABLE_X64=<1|0> SPEC_THROUGH=A9 SURVJAX_WHEEL=<artifact below>
  UV_CACHE_DIR=/private/tmp/survjax-uv-cache uv run --locked --offline pytest
  tests/spec/test_a9_package.py tests/spec/test_a9_wheel_smoke.py -q`:
  16 passed in 2.74s per mode, no skips or failures. Independent tests remain
  unchanged and use the same runner-built wheel as the cumulative gate.
- `bash -n scripts/check_wheel.sh`: exit 0. Ruff: 76 files already formatted /
  All checks passed! Mypy core: no issues found in 17 source files.
  `git diff --check`: exit 0.

Artifact retained by the runner:
`/private/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.b3M6R5/survjax-0.0.1.dev0-py3-none-any.whl`.
SHA-256: `700418efd9bf12297ed2e0851d5c133767dc7abf575d146ed85b3d4da6f1df3f`.
Its WHEEL metadata says `Generator: uv 0.11.19`, `Root-Is-Purelib: true`,
`Tag: py3-none-any`. Tested Python 3.12.9, JAX/JAXlib 0.11.2, NumPy 2.5.3;
other pinned runtime packages: ml-dtypes 0.6.0, opt-einsum 3.4.0, SciPy 1.18.1.
No different Python/JAX combination or accelerator is claimed.

Focused logs: `/private/tmp/survjax-a92-install-x1.log`,
`/private/tmp/survjax-a92-install-x0.log`. Independent logs:
`/private/tmp/survjax-a92-spec-x1.log`, `/private/tmp/survjax-a92-spec-x0.log`.
Full gate: `UV_CACHE_DIR=/private/tmp/survjax-uv-cache
scripts/check_wheel.sh --through A7` exited 0:
`check: all gates passed through A7`. Guard: ok (A9).

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 792 passed, 7 skipped, 211.16s | 798 passed, 1 skipped, 192.31s |
| Source doctests | 4 passed, 0.89s | 4 passed, 0.82s |
| Independent through A7 | 2114 passed, 28 skipped, 218 deselected, 99.80s | 2118 passed, 24 skipped, 218 deselected, 95.99s |

Full log: [/private/tmp/survjax-a92-gate.log](/private/tmp/survjax-a92-gate.log).
All three new own cases pass in both modes; there are no new skips or
exclusions. The 218 deselections belong to later gate steps. A9 package/wheel
acceptance was run separately, without filters. Only PLAN.md/REVIEW.md result
records changed after this gate; final Ruff and whitespace checks follow.
Ruff 0.16.9 and mypy 2.3.1 were used; `jax.devices()` reports CpuDevice(id=0).

The first requirements.txt install attempt failed at setup (1 passed, 2 errors per
mode): cached wheels were present, but the registry metadata needed for a
new requirements resolution was absent. Exporting pylock.toml retained the
lock's artifact identities and fixed offline installation without downloading
anything, changing versions or modifying the lockfile.

An auxiliary verbose build diagnostic without UV_CACHE_DIR failed before
building because the sandbox denied access to the default user cache
(`Operation not permitted` at `/Users/wamster3/.cache/uv`). It is not build
evidence; the successful runner build uses the explicit permitted temporary
cache, and backend identity is read from the resulting wheel.

A8 fixtures remain absent: the cumulative through-A9 gate is pending A8.
The default runner does not skip or suppress that requirement. A9.3 example,
A9.4 benchmarks and A9.d1 docs have not started. Sdist and release checks are
outside this Slice A wheel step. No contract ambiguity/new deviation found.
Final result records are complete. No R, sdist or docs build ran. Commit this
step on `A9` and stop for review.
