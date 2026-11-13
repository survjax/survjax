# NP.ADAPTER.1: experimental NumPyro observation adapter

Base and worktree starting commit: 6e779b6a6c05a53f46fec749b36d9b07d9d4562c.
Previous completed commit: 6e779b6 (fixture integration on main).
Dependencies: released likelihood/data interfaces and the existing optional
NumPyro 0.22.0 example dependency. Preceding baseline: [A12](../../REVIEW.md).
Cumulative gate: A12. No new post-release core feature is enabled.

The maintainer explicitly requests runnable discussion code for issue #1 after
agreeing on a single adapter: structured observations use log_likelihood,
arrays use log_prob, and future sampling generates underlying event times.
This authorizes an experimental example outside the v0 package API; it does
not authorize publishing, changing the spec, or adding core sampling.

Create examples/numpyro_adapter/adapter.py, a small runnable inference probe,
a short example README, and focused optional tests. About 300 Python lines;
the README adds run instructions and the compatibility boundary.
Expose no new survjax exports and add no dependency declarations. Implement
structured masking and preserve observation metadata for supported exact/right
data. Reuse NumPyro mask/expansion wrappers and survjax validation/scoring.
Keep sampling explicitly unavailable. Document untested generic wrappers and
obs_mask imputation as outside this prototype's demonstrated surface.

Spec coverage: 3.2 (shape/metadata), 4.5/4.7 (direct scoring/diagnostics),
7.1–7.3 (likelihood and safe substitution), 10.3 (optional integration), 12.2.
The adapter is an explicitly requested exploration of the deferred design.

Compatibility: the maintainer answered yes to the explicit request for a
prototype-only ndim shim. Scope it to a context manager used by the standalone
demo and optional tests; restore the class on exit. Spec 3.2 and the independent
acceptance test remain unchanged. A production property needs their coordinated
update later. No import-time monkey patch or change to core is authorized here.

Validation: analytical exponential likelihood, factor-model agreement,
pointwise extraction, parameter gradients/Hessians, JIT/vmap, both mask/plate
orders, nested masks, invalid excluded records and preserved issue metadata,
array scoring, and explicit sampling failure. Run both dtypes, a short seeded
NUTS demonstration, and the existing fast A12 gate before independent review.

Docs to-do: example instructions explain the observation/sample distinction,
run commands, compatibility shim if needed, and the observed integration limits.
Covered in examples/numpyro_adapter/README.md. Production docs and an issue
reply/possible draft PR follow maintainer review. The maintainer agreed to a
runnable prototype first and asked how to share it; recommend a draft PR linked
from issue #1. Publishing remains a separate action.

Integration: retain the experimental example on adapter; do not publish or
claim production compatibility. Stop after this step and independent review.

## Minimal review follow-up (2026-11-13)

The maintainer approved the proposed minimal response to round 1. Add one
eager-validation regression test: excluded invalid records must not emit a
support warning. Explain in the example README that core likelihood safeguards
already protect the tested gradients; replacement additionally makes excluded
records valid before adapter validation. Explain the stop_gradient convention
without claiming its necessity for these masked derivatives. Keep the scoped
ndim shim and the documented incremental timing-list update as prototype
boundaries. No API or numerical implementation changes. Verify both dtypes,
confirm the new test detects omitted replacement, run the fast A12 gate,
and request review round 2 after a fix-up commit.
