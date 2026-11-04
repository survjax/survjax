# GPU.1 follow-up — Keep the Modal source archive extractable

The first real Modal run showed that `git archive` preserves the top-level
`spec -> ../design/spec` link, which Python's safe `tarfile` data filter
rejects. Keep the external design-repository link out of the transferred
archive; tests and wheel building use committed test fixtures and package
files, not the external spec checkout.

Files: `scripts/gpu_ci.py` (exclude the external `spec` symlink from the
source archive while retaining safe extraction); `tests/unit/test_gpu_ci_archive.py`
(regression check that builds an isolated git fixture, then verifies its
archive contains no `spec` path and extracts with the `data` filter);
`PLAN.md` and `REVIEW.md`.
Public API: none. Spec sections: none; this is CI runner transport only.

Keep the fix within 300 changed lines. Run the focused regression test, Ruff
format/check, Python compile, and whitespace check. Commit on `ci`, then
continue the independent-review process already in progress for GPU.1.

# GPU.1 — Modal runner and GPU test marker (2026-11-02)

Branch ci, base a3519520588cb638aff475cb4bcdec14b3d26741. User explicitly
requested a one-command paid Modal runner, 12-hour/$10 caps, GPU test marking,
PR-to-main/nightly/manual triggers, and fork exclusion. User approved Modal's
isolated SDK in conversation; do not change project dependencies or uv.lock.

Files: scripts/gpu_ci.py (single local Modal invocation; transfers the current
commit's source archive; JAX and jaxlib pinned to uv.lock, CUDA 12 installation
with resolved packages and driver recorded; both dtype suites; wheel-install
smoke on CPU; actual-device assertions; zipped logs/results returned locally);
tests/conftest.py (mark all collected tests gpu except tests/install),
pyproject.toml (register marker); REVIEW.md (overwrite for this step). No
workflow yet; the guarded GitHub trigger belongs in the next step. No
tests/spec, numerical code, public API, or dependencies/lock changes.

Budget: T4 + hard limits of 2 CPU cores and 8 GiB at current Modal rates is
about $0.749/hour. Twelve hours plus a 10-minute startup allowance is about
$9.11, below $10. No explicit retry policy, volumes, or other GPU workers.
The per-attempt code deadline and Modal timeout are both at most 12 hours;
reject a requested cap if current rates make the estimate exceed $10. Modal
may restart a preempted function, so the estimate is not an absolute spend
ceiling across restarts. Modal's CUDA plugin and CUDA library wheels are
resolved by the CUDA 12 extra at run time; `uv pip freeze` and `nvidia-smi`
record the actual package set and driver for each result. The follow-on hosted
workflow will use five hours because GitHub hosted jobs have a six-hour
maximum, with the remaining hour for result return.

GPU.1 stays within 300 changed implementation lines. Run static/targeted
checks, document exact outputs, commit and independently review before
stopping. The next step adds the GitHub workflow, including fork protection.
Hosted execution requires one-time Modal auth. No credentials are needed for a
local Modal run after login.
