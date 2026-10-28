# Review: A10.2b2 — Public concordance entry points

## Reading order

1. `src/survjax/metrics.py`: `_evaluate_concordance`, then the two public
   wrappers. The existing counter/result layer is reused without changes.
2. `tests/unit/test_concordance_entry_points.py`: hand values, all five shape
   rows (including every info leaf), whole-cohort failures and precision cases.
3. `src/survjax/__init__.py`, its exact import-smoke export list, and the
   two entries in `docs/api.md`, then the
   current PLAN.md evidence and D-027 implementation note.

Authorized by "review approved ... get on with next subslice", on A10-cindex
from `715a7e3` in dev-a10. The maintainer explicitly approved only the two
API-reference entries as a code-step docs exception; the seven-line addition
is their heading/fence/directives. No tutorial or other docs edit.

## Invariants and requirements

Spec 6.1–6.2: both metric entry points use `_resolve_data`, so raw time/event
and prepared data share pure checked normalization. Missing/mixed forms and
static score-kind/shape misuse raise TypeError. Runtime nonfinite scores and
observation issues give NaN plus diagnostics. Scores are compared in their
JAX representation, preserving integer and available floating rank precision;
they are not cast to the data dtype. The ratio retains the data float dtype.

Spec 9.1–9.2: non-scalar observation axes must have identical lengths before
leading-axis broadcasting. Counts and masks are reduced per cohort, never
pooled over leading axes. Construction issue bits are retained and current
normalized fields are rechecked after pytree reconstruction. Support is read
from actual fields: exact/right observations only, left bound 0 and right
bound +inf. Metadata flags are not a substitute for value checks.

Invalid and unsupported masks are explicitly broadcast to the result shape
before counting (review N-7). Whole failed cohorts receive finite safe score/
time arrays and false exact masks; no partial cohort or representative time
for an unsupported observation is evaluated as a result (review N-3). The
existing finalizer guarantees zero failed count words and distinguishes
failed from evaluated zero-pair cohorts under accepted D-027.

Leading axes are flattened for vmap and restored afterward. Explicit batch
size/N dimensions handle empty cohorts without ambiguous reshape(-1,0).
The counter retains O(N squared) work/O(N) pair workspace per result; no
N-by-N pair matrix. No floors, clipping, epsilons, tracer-to-NumPy conversion,
value-dependent Python branching or global precision change. No useful
score-gradient guarantee. Frozen result field headings now use Attributes,
matching existing diagnostics (review N-6).

## Tests and expected values

Own new tests cover raw/prepared and plain/info forms on explicit C,T,D,D
pairs, giving (1+1/2)/4=3/8 with counts (1,2,1,4). Reversing scores gives
5/8, including half credit. Those exact binary fractions anchor all five
spec shape-table rows; every count-word/mask leaf must match result shape.
NaN-score, raw-event-code, left-truncation and overlapping-mask cohorts leave
a valid neighbor at 3/8. Edited-field validation, evaluated empty input, six
static axis errors and three nonnumeric/nonreal score kinds have own cases.
Float32 data retain distinct float64 ranks where available and adjacent large
int32 ranks; results still have float32 dtype. jit and jit-of-vmap compose.
No expected value comes from running the implementation as an oracle.

The unchanged independent API/value/diagnostic/shape/property modules cover
additional conventions, boundaries, call forms, compile reuse and permutations.
Their fixture module is outside this code step, not modified/skipped/xfail'ed.

## Checks actually run before the regression gate

Commands use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`, in separate
JAX_ENABLE_X64=1/0 processes. Working directory: dev-a10.

Own focused command:
`uv run --group test pytest -q tests/unit/test_concordance_entry_points.py tests/unit/test_concordance_results.py tests/unit/test_concordance_counts.py tests/unit/test_api_reference.py tests/unit/test_import.py`.
Float64 **64 passed in 6.87s**; float32 **64 passed in 6.27s**, no skips.
This includes all 23 new own entry-point cases and API-reference coverage.
Earlier runs passed 60 per mode before score-kind cases, then 63 per mode
before including the import smoke test.

Independent command, with SPEC_THROUGH=A10:
`uv run --group test pytest -q tests/spec/test_a10_concordance_api.py tests/spec/test_a10_concordance_values.py tests/spec/test_a10_concordance_diagnostics.py tests/spec/test_a10_concordance_shapes.py tests/spec/test_a10_concordance_properties.py`.
Float64 **113 passed in 45.13s**; float32 **112 passed, 1 skipped in 47.97s**.
The existing skip requires float64 scores alongside float32 data. No failures
and no new skips/exclusions. These are five selected modules, not the full A10
gate. All independent files are unchanged.

Ruff initially reformatted 2 files; lint passed. Mypy reported no issues in
20 source files. No focused numerical failure or tolerance change occurred.

The initial through-A7 attempt exited 1: its float64 own run had **1 failed,
848 passed, 7 skipped in 153.65s**. The failed test was
`test_public_package_imports`, whose exact expected export set still described
the previous API. Added the two intended metric names while retaining every
previous name; the exact-set assertion remains strict. This is an own test,
not a protected acceptance test. The 64-case focused rerun above passes with
that smoke test included. Optional initial log:
`/private/tmp/survjax-a102b2-initial-gate.log`. The final gate below is a new
run after that correction and this review edit, using a separate log.

Source/test changes total 291 lines, plus the seven authorized reference lines;
the public metrics module is 220 lines. No scope split was needed.

## Final gate and remaining work

Run the unchanged regression gate after source/tests and this review are final:
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`.
PLAN.md records actual status/counts/timings after completion; this review is
left unchanged afterward. Optional complete output:
`/private/tmp/survjax-a102b2-final-gate.log`. Durable summary is in PLAN.md.
Finish with final static/whitespace/staged/protected-path checks.

No merge, rebase, spec/.spec-seen/sync edit, likelihood API change, new
dependency or other-branch change. The shared spec is revision 37; the
separately owned likelihood alignment remains absent from this branch and is
not claimed here. The C-index statistical contract remains unchanged.

R fixtures, narrative documentation/playground/held-out example, installed
wheel evidence and the full Slice A gate remain pending. No full through-A10
pass, R comparison, docs build, benchmark or accelerator check is claimed.
Optional older workspace-walker compatibility nit N-5 is unchanged. Stop after
this code step's commit for review, without merging or starting the next step.
