# D.LOGNORMAL.1 — built-in log-normal family

Base/worktree starting commit: 3abefc5683a3aee8169eb2deffe57602897a3af3.
Preceding completed step: A12 acceptance, ff07cf9 (checkpoint), merged at base.
Dependencies: existing registered families, direct-method policy and diagnostics.
Preceding records: [A12 acceptance](../../REVIEW.md),
[record isolation](../meta-parallel-records/PLAN.md).
Cumulative gate target: A12; D.LOGNORMAL.1 is not a gate selector.
The maintainer authorizes this Slice D family independently of Slice A.

## Implementation

- Add `src/survjax/distributions/lognormal.py`: registered natural parameters
  `meanlog` (real), `sdlog` (positive); `from_log(meanlog=..., log_sdlog=...)`
  exponentiates only the positive parameter. Supply analytic log survival,
  log CDF, density and hazard with safe branch operands and boundary limits.
  Use a 128-term inverse-Mills-ratio continued fraction above standardized
  time 2 for stable hazard Hessians; keep the specified survival/CDF formulas.
  Use constant reference time 1, positive/finite for every valid parameter,
  including locations whose exponential is unrepresentable.
- Export `LogNormal` from both package initializers; these overlap with the
  parallel LogLogistic step. Update `tests/unit/test_import.py` and
  `tests/install/test_wheel.py` export assertions.
- Add `tests/unit/test_lognormal.py`: mpmath tail values and parameter
  derivatives/Hessians, finite-difference second derivatives, boundaries,
  eager/jit/vmap, broadcasting/indexing/grid, diagnostics and likelihood.
- Keep code and tests together under approximately 300 implementation lines.
  No new dependencies, shared numerical changes, protected paths or docs edits.

Public API: `LogNormal`, its two constructors and registered direct methods.
Spec revision 47: sections 3.3, 4.1, 4.2, 4.5–4.7, 5, 7.1–7.3, 12.1.

## Validation

Tier 0: touched tests in both dtype processes with six workers, no marker filter.
Tier 1: `scripts/check.sh --fast --through A12`; report any failure honestly.
Measure slow tests and regenerate the slow list if required. Commit the step
and handoff, then run the independent-review skill, at most three rounds.

## Docs to-do

Add LogNormal to `docs/api.md` and document its parameterization and tail
behavior in a separate docs step. The current all-exports API-doc test will
detect the missing entry until that authorized follow-up lands.

## Remaining scope

This step completes the assigned family direct-method scope. Other Slice D
features (sampling, solvers, wrappers, summaries) are outside this assignment.
