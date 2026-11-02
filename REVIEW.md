Independent review: round 2/3

# GPU.1 — Modal runner and GPU test marker

Read PLAN.md GPU.1 follow-up, then scripts/gpu_ci.py,
tests/unit/test_gpu_ci_archive.py, tests/conftest.py, pyproject.toml, and this
review. Base commit: a3519520588cb638aff475cb4bcdec14b3d26741.

Invariants for the follow-up:
- The archive sent to Modal contains only paths from the dev-ci commit and no
  top-level link into the sibling design repository.
- Remote extraction continues using Python's `data` filter for every member.
- GPU tests and wheel building do not read the external top-level `spec` link;
  committed fixtures remain in the archive.

The archive regression test checks both that `spec` is absent and that the
result extracts under the same `data` filter used remotely.

Observed real Modal run before this fix: commit
6df96957329d012e37cf2023a89ef5a978d4829a failed at remote extraction with
`LinkOutsideDestinationError` for `spec -> ../design/spec`, before dependency
installation or tests. Modal run: https://modal.com/apps/vanamsterdam/main/ap-ORcnLOQqNmikgGNeO7dhtc.

Round 1 findings addressed:
- B-1: Modal CLI options are now typed parameters on `@app.local_entrypoint`;
  the exact one-command invocation is in the script docstring.
- F-1: Modal CPU and memory requests now have matching hard limits (2 cores,
  8 GiB). The $10 guard checks current modeled compute cost including a
  10-minute startup allowance. This limits compute charges to the model, not
  provider-independent fees or future rate changes. Modal can restart a
  preempted function, so repeated attempts could exceed the estimate.
- F-2: `jax` and `jaxlib` are pinned to uv.lock. The CUDA 12 extra resolves its
  plugin/library wheels at run time; `uv pip freeze` records installed package
  versions, and `nvidia-smi` records the driver and GPU in the result archive.
- N-2: The module docstring states that all tests outside tests/install are
  currently marked gpu; the marker is ready for future CPU/GPU test separation.
- N-3: Removed manual argparse, the placeholder command list, duplicate lock
  parsing, redundant imports and unused SDK-version print. The direct Python
  invocation retains the CUDA install in the environment; wheel smoke tests
  explicitly use CPU.
- N-1: No dedicated test was added for these guards; the Modal CLI guard probe
  could not run because the uvx download timed out. This remains unverified.

Validation after fixes:
- Ruff format/check on scripts/gpu_ci.py and tests/conftest.py: passed.
- `uv run mypy src`: Success, no issues in 22 source files.
- `python3 -m py_compile scripts/gpu_ci.py`: passed.
- `uv run --no-sync pytest --collect-only -q -m gpu tests/unit`: 1169 tests
  collected in 2.32s.
- `git diff --check`: clean.
- Modal CLI guard probe (`uvx --from modal==1.6.3 modal run scripts/gpu_ci.py
  --max-runtime-hours 12 --max-cost-usd 1`) was blocked downloading the Modal
  CLI dependency (`watchfiles` metadata request timed out); no Modal app ran.
- Direct guard harness with a stub Modal decorator: rejected a 12-hour run
  under a $1 cap before checking git state or contacting Modal.
- Round 1 independent gate: ruff/mypy passed; own suites passed (1282 x64,
  1287 x32); 5 spec wheel-test setup errors because `SURVJAX_WHEEL` was not
  provided. The reviewer could not run scripts/check_wheel.sh in its checkout.
- Follow-up regression test: `uv run --group test pytest
  tests/unit/test_gpu_ci_archive.py -q` — 1 passed.
- Ruff format/check on `scripts/gpu_ci.py` and the regression test — passed;
  `python3 -m py_compile scripts/gpu_ci.py tests/unit/test_gpu_ci_archive.py`
  passed; `git diff --check` clean.

Cost estimate at current Modal Function rates: T4 $0.000164/s, CPU
$0.0000131/core/s, memory $0.00000222/GiB/s. With fixed 2-core/8-GiB limits,
12 hours plus 10 minutes estimates to $9.11. Plugin and CUDA library versions
are recorded, not individually pinned; no project dependency or lock changed.
The 12-hour timeout is per Modal execution attempt. Modal documents that a
preempted function can be restarted, so the $10 figure is not an absolute
provider billing ceiling across repeated preemptions.

Remaining after this step: add the CPU-independent GitHub GPU workflow with
same-repository PR, nightly and manual triggers, and the fork-PR guard.
