# SIM.fixed-limits — Explicit generation and prespecified censoring

Read the latest PLAN section, the Weibull docs page, run.py, then the
_results/_evidence/_plots changes and their focused checks. The maintainer
explicitly authorized updating the use-case design and regenerating results.
Independent review and full test suites remain waived.

Design commit: 213cfb0 (spec revision 47); .spec-seen records it. Starting and
previous step final: e33d4b5. Original worktree base: 8757c0f.

The example now shows independent JAX uniforms, inverse Weibull survival,
observed times and event indicators, with a 20-year censoring limit. The full
runner uses limits (50, 20), without calibration. Exports drop target fractions;
metadata schema 2 records prespecified censoring_limits. Figure labels and the
illustration selector now use actual limits. No likelihood, solver, acceptance
threshold or random-key scheme changed. The calibration utility remains unused
by the runner; deleting that separately tested utility is not needed here.

Validation so far: Ruff check and format check passed (13 example files).
The three changed CLI/metadata/result check files passed: 25 tests in 12.79 s.
Earlier attempts exposed stale CSV column indices (fixed) and a sandbox-denied
sysctl query; the passing run used escalation. The figure-label check and full
regeneration follow. The earlier docs build passed with explicit generation;
final docs will be rebuilt after regenerating results.

Source changes are committed before the full run so provenance can identify
clean source. Remaining authorized work: regenerate the 12,000-fit bundle,
check its coverage and bookkeeping, update result prose, rebuild docs and
commit the compact evidence. Raw per-fit diagnostics remain local.
