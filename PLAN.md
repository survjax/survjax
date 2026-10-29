# PLAN.md: current sub-slice only

This file holds the plan of the sub-slice in progress, newest section first.
Earlier A9 work is in `plans/A9.md`; A10 records are preserved verbatim in
`plans/A10.md` while this branch returns to A9 at the maintainer's request.
A10 is unfinished; archiving its records does not claim completion.

---

# A9.4 — Reproducible CPU baseline (2026-10-29)

Authorized by "check the roadmap ... plan the needed work and go on to
implement it" on the maintainer-created `a9b` branch, initially clean at
`1c0fd06`. The separate archive commit preserves A10 records verbatim;
A10 is unfinished and outside this step. No approval exception is needed:
no new public API, dependency, protected path or spec interpretation.

## Roadmap audit and remaining sequence

The workspace `../ROADMAP.md` still describes A10.2a and missing A8/A9
integration. Current merged plans/evidence show A9.1 derivatives and A9.2
wheel/extension checks accepted, A8 fixtures integrated, and through-A9
combined-tree gates already passed during A10 integration/A10.3. A9.3's
standalone example was explicitly withdrawn; do not restore it. The roadmap
is maintained from the workspace root; report its needed status correction
in this branch's handoff instead of editing outside this repository.

Remaining A9 steps, each separately committed/reviewed with a stop:

1. **A9.4, now:** reproducible CPU likelihood/gradient measurements below.
2. **A9.d1, later docs step:** installed-subset verification page and
   navigation, benchmark reproduction, audit executable pages/playground,
   fresh two-dtype docs builds and rendered-page review (spec 10.5).
3. **A9 closure, later:** map every section 12.2 Slice A gate to concrete
   evidence and run the final combined-tree fresh-wheel gate. Keep the tiny
   synthetic example pending until the agreed replacement README/use-case
   material provides evidence. Use-case 1 needs the separately authorized
   PH work; it is not part of A9.4. Do not claim Slice A complete.

## Implementation plan

One concept: measure existing behavior without optimizing library code.
Spec: 7.2–7.3, 12.1–12.3. Public API touched: none.

Files: create `scripts/benchmark_slice_a.py` (at most 300 changed code lines)
and `benchmarks/slice_a_cpu.json` (machine-readable evidence); update this
plan and overwrite REVIEW.md. No source, docs, spec, protected files,
fixtures, lockfile or dependency changes.

- Fixed seed, CPU selected explicitly, separate x64/x32 processes, both
  Exponential/Weibull, N=1000/10000/100000, shared and row parameters.
- Measure a jitted summed log likelihood and parameter gradient together,
  returning data and parameter diagnostics as auxiliary outputs. Compare
  prepared SurvivalData with ordinary construction from traced time/event
  arrays. Record prepared versus conservative traced static metadata.
- Time host generation, initial placement and prepared construction
  separately. Synchronize all output leaves. Record compilation/first call,
  warmups and repeated warmed samples (minimum/median/maximum), never
  introduce timing thresholds in normal tests.
- Before recording a row, compare its objective/gradient to independently
  differentiated exact/right formulas evaluated with NumPy. Import existing
  ATOL/RTOL from tests/impl_tolerances.py; do not add/loosen tolerances.
  Assert diagnostics valid. Check changed dynamic values and equal-shaped
  minibatches from a parent reuse the compiled trace, with numerical checks;
  retain A9.1 tests as additional regression evidence.
- Record exact commands, seed, parameter/input shapes, dtype, locked versions,
  hardware/OS and actual CPU device, repetition count and logical array
  input/output bytes. Explicitly state peak memory unmeasured; do not infer
  peak memory from bytes. Summarize scaling/overhead and limitations in review.

Verification: execute the full matrix in separate dtype processes; inspect
all 48 rows/correctness/reuse checks and JSON metadata. Run the existing
likelihood composition tests (both modes), Ruff format/check, mypy src and
`scripts/check_wheel.sh --through A9` using a fresh wheel, then whitespace
and scope checks. No new regular tests: the benchmark's analytical checks
validate its full numerical workload and existing composition tests cover
recompilation behavior; no timing assertions. Record exact outputs/failures.
Commit on a9b with REVIEW.md saying "Independent review: round 1/3", invoke
the independent-review skill in ../review/code-a9b for step A9.4 and the
literal full commit hash. Address blocker/should-fix findings in separate
fix-ups within three rounds, then stop and report.

Docs to-do: A9.d1 described above. Example replacement and final gate map
remain pending; no A10 or later-slice work starts in this step.


## Implementation evidence

The 230-line harness and committed JSON cover all 48 cases on Apple M4 Max,
CPU only, Python 3.12.9, JAX/jaxlib 0.11.2 and NumPy 2.5.3. Each case passes
analytical value/full-gradient checks for its initial input and two changed
parent minibatches, and one-trace reuse. Ten samples follow three warmups.
Independent report audits verify case completeness, medians, shapes, flags,
finite samples and summary fields. Peak memory remains explicitly unmeasured.
Focused existing derivative/composition suites: 15 passed in 28.03s in each
dtype, no skips. Fresh-wheel `scripts/check_wheel.sh --through A9` exited 0, ending
`check: all gates passed through A9`: own suites 1044 passed/7 skipped
(float64), 1050 passed/1 skipped (float32); source doctests 4 passed per mode;
independent suites 2211 passed/28 skipped and 2202 passed/37 skipped.
Ruff reports 104 files already formatted / All checks passed; mypy has no
issues in 20 source files. Exact commands, timings, artifact identity, initial
harness corrections and limits are in REVIEW.md. Final whitespace/scope
checks, commit and independent review follow. No library,
docs, dependency, fixture or protected-path change. Remaining A9 steps above
are still pending; this evidence does not close Slice A.
