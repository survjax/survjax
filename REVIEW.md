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

## Regenerated evidence

Full run: `uv run --locked --group examples python
examples/weibull_simulation/run.py --config full --out
/tmp/survjax-weibull-fixed-limits` exited 0. Source was clean commit ac0fa01,
using design 213cfb0. Compact exports and all three figures are regenerated;
raw diagnostics remain at that temporary path. No values were hand edited.

The 12 rows account for 12,000 attempted fits. There are 137 rejected fits,
all gradient-only. At n=1000, Weibull coverage is 0.9425887 and 0.9359504;
both are within three observed Monte Carlo SEs of 0.95. The corresponding
Exponential coverage is 0.8850575 and 0.9371197. Realized censoring is about
24% and 53%, rather than the previous calibrated target fractions.

The committed-bundle bookkeeping/coverage check and figure mapping check
passed: 2 tests in 11.73 s. The figure check had also passed before generation
(1 test, 8 deselected). All evidence hashes match. Ruff check and format check
passed again. Ordinary diff --check flags CSV CRLF terminators; the same check
with core.whitespace=cr-at-eol passes, retaining byte-exact generated hashes.

No full suites or independent review ran. Browser visual inspection is not
claimed. Final two-mode documentation build status follows.

Final validation: scripts/build_docs.sh exited 0 with both x32 and x64
builds successful. The example runs in float64 in both builds. The rendered
page reports HR 0.693, CI [0.482, 0.996], 20/20 accepted batch fits and 80
smoke fits. The local HTML URL responds with HTTP 200. All requested comments
are addressed; no remaining substeps. Final evidence commit follows ac0fa01;
the task started at e33d4b5, with original worktree base 8757c0f.
