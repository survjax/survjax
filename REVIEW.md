Independent review: round 1/3; verdict maintainer (2026-11-06)

# SIM.8c.1 — Full-run provenance and reducer placement

## Resumed review (2026-11-06)

The maintainer reported restored reviewer availability and authorized resuming
review of 4e6c475. A first written round now exists at
../review/slices/SIM.8c.1.md, verdict maintainer. It independently confirmed
both artifacts are byte-identical and the reducer move preserves outputs.

F-1 requires a maintainer module-layout decision. Recommended proposal:
_evidence.py owns write_evidence/its CLI; _plots.py owns figure rendering;
check_*.py remains independent checks. Both runtime helpers stay below about
300 lines. The maintainer was explicitly asked to approve these names.
Do not reinterpret the earlier audit-module proposal as producer approval.

F-2 is an implementation test gap: reducer_path/reducer_dirty need controlled
clean/dirty cases scoped to the actual reducer file, so forced-False and
wrong-path mutations fail. Prepare that fix after the F-1 module decision.
N-1 is nonblocking awareness of historical absolute invocation paths; these
are intentionally original metadata bytes, and later result notes explain it.
No code fix or dependent artifact/figure implementation proceeds while the
maintainer decision is pending. Reviewer process completed: focused checks
21 passed in 8.00 s; raw gate x64/x32 implementer suites 1282/1287 passed,
doctests 4 each; x64 acceptance 2328 passed with five SURVJAX_WHEEL setup
errors, so x32 acceptance was not reached. This is separate from the earlier
passing wheel-supplying cumulative gate. Earlier quota history follows verbatim.

## Review availability (2026-11-05)

Committed implementation/artifacts: 494abca2c1cd3473e6dc074c1effe7a657d17543.
The exact independent-review command for SIM.8c.1 was invoked twice, the
second being the skill's single permitted retry after no new round. Both
exited 1: "You've hit your 5-hour limit · resets 2:40am (Europe/Amsterdam)".
No SIM.8c.1 review file/round was written; there is no verdict to claim.
SIM.8b remains merge at 61c5cc6 after two completed written review rounds.
The maintainer has been asked whether to defer reviews and finish the
remaining compact artifacts/figures. No dependent implementation proceeds
before that waiver or restored review availability. This paragraph is status
bookkeeping, not an implementation change or a review finding.

Prepared outside Git: all compact files (23,634 bytes including manifest)
from clean reducer 494abca, reducer_dirty=false. Figure source prototype
/private/tmp/sim_figure_fragment.py and inspected SVG/PNG previews at
/private/tmp/survjax-sim-figures-preview. No bulk raw data is staged/tracked.
Data inputs remain /private/tmp/survjax-sim-full-a3792a7. The full compact
bundle draft and future committed-evidence check are prepared externally;
only metadata/performance are committed so far.


Read PLAN.md SIM.8c.1 and approved D-035, then results/metadata.json and
performance.csv, then the reducer rename/import changes and boundary test.
Previous final: 61c5cc62a7f890175b40aa6a1333566800d2e809 (SIM.8b round 2
merge; F-1/F-2/N-1/N-2/N-3 fixed). Integration base
 d2469e0a7e719dbcb276277e6fd80a447626ad4e; original worktree base
8757c0fda78e34e6e818249adedd2a144a33c33d. Remain release-simulation.
The maintainer explicitly authorized compact evidence AND the next figure
step; these are budgeted sub-steps with independent reviews, not a merge.

## Scope and invariants

Only 213 generated lines: original metadata and all 12 summary rows, copied
byte for byte from the clean a3792a7703ed539db777472d876bd00ac2d1568e full
run. Historical invocation/output_directory in metadata are deliberately
preserved; they describe the raw run, not its later copy to results/.
No raw replicate CSV is staged/copied. Other compact files follow in the
next split; no manifest yet claims this partial artifact set is complete.

Move the reviewed reducer from _plots.py to the previously proposed
check_full.py and update existing imports. This makes room for the authorized
figure step while keeping modules under ~300 lines. No public library or
shared generator import contract changes. Python loops remain host audits/I/O.
Two nonblocking round-2 nits addressed in this provenance sub-step:
N-4: parameterized low/high edge test, including nextafter(.45,1) with a
smaller minimum so the high-edge adjustment is necessary. No binning change.
N-5: reducer_path and reducer_dirty (file-specific Git status) now accompany
commit/hash. A dirty reducer is explicitly distinguishable from recorded HEAD.
No dependency/core/protected-path/docs/solver/seed/acceptance/timing change.

## Checks

Ruff format/check and mypy src pass. Focused command:

```bash
JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q examples/weibull_simulation/check_results.py examples/weibull_simulation/check_cli.py examples/weibull_simulation/check_metadata.py
```

21 passed in 10.40 s. Edge checks alone: 11 passed in 4.34 s.
Manual stdlib audit: all 12 summary rows, six scenarios, 12,000 attempts,
11,873 accepted/127 failed; each attempted=accepted+failed=1000. Metadata
is clean a3792a7, lock hash matches, both Weibull n=1000 coverage checks are
within three MCSE of .95. Byte hashes equal originals:
performance 33c9f420b5ecc80d3d75f8f097671569fbade13945cc4629f0205359760e175e;
metadata 475f578d9413bd3eed88aaa2b93abcd34fee3736394e87258f392f31dcba899f.
No raw CSV exists under results/. git diff --check passes.

No repeated cumulative gate for this host-only move/artifact/test step.
SIM.8b's scripts/check_wheel.sh --through A10 passed implementer tests
1282/1287, doctests 4 each and acceptance 2333/2324 (x64/x32), with their
expected skips. The reviewer's raw gate lacks SURVJAX_WHEEL and reports five
wheel-setup errors; this is separate from the passing wheel-supplying gate.

Remaining authorized work: derived failure/histogram/illustration/manifest
files and their committed checks, then three SVG figures. Executable docs
remain after these reviewed steps. No merge/push/publication.
