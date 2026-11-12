# B4.1.COMMENTS — Explain likelihood pruning flags

Independent review: round 1/3

Base / previous step: f6a9bf33122451720465814224e3d0a7fbc7c7ee (B4.1).
Original worktree base: cc0e30f. Cumulative gate: A12 with existing features.

Read [PLAN.md](PLAN.md), the comment diff in src/survjax/likelihood.py, and
the flag order in src/survjax/_data_metadata.py. Spec 3.2 defines the metadata;
7.2 permits pruning only when the relevant work is known to be absent.

The comments identify indices 0 through 5, distinguish dataset-wide metadata
from per-row exactness, explain conservative tracing/slicing, and explain why
flags 1–3 require conditional set mass while flag 0 requires density/log Z.
They also explain the zero placeholders and final per-row selection.
No executable code, API, dependency, tolerance or protected path changed.

Validation: Python AST comparison against the base printed
"AST unchanged: comments only". scripts/check.sh --fast --through A12 exited 0:
all gates passed. Guard, formatting (266 files), lint and mypy (25 source files)
clean. Implementation: float64 1192 passed/8 skipped; float32 1193 passed/
7 skipped. Doctests: 4 passed per mode. Acceptance: float64 3385 passed/
32 skipped; float32 3373 passed/40 skipped, with the two existing PH arithmetic
warnings per mode. git diff --check: clean.
No new tests, benchmark, timing refresh, docs build or R regeneration is needed
for this comment-only change. Existing pruning tests retain behavioral coverage.

No uncertainties or new deviations. B4.2, family activation and likelihood docs
remain; reviewer-workflow investigation remains deferred.
