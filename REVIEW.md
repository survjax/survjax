Independent review: round 1/3

# SIM.8b — Compact, reproducible full-study evidence writer

Read in order:
1. PLAN.md SIM.8b, D-035 maintainer decision, use-case main/details outputs.
2. examples/weibull_simulation/_plots.py write_evidence (new approved helper).
3. run.py's post-export call, _results.py manifest invalidation, .gitignore.
4. check_results.py hand CSV/negative checks and check_cli.py fake-count repair.

Previous step final / source base: a3792a7703ed539db777472d876bd00ac2d1568e.
Maintainer integration: d2469e0a7e719dbcb276277e6fd80a447626ad4e.
Original worktree: 8757c0fda78e34e6e818249adedd2a144a33c33d.
The maintainer approved the compact replacement and explicitly requested the
next figure step as well. Both code steps receive independent review; source
and generated evidence are split to preserve normal review budgets. This
commit has no generated data, fitting/core API/dependency/protected-file edits.
Named _plots.py was proposed before approval; no additional unnamed helper.

## Contract and invariants

D-035 overrides the use case's committed raw-replicate requirement. Full raw
CSV remains local, and is ignored at the conventional results/replicates.csv
path. Existing CLI now also derives compact outputs after raw CSV/metadata.
Raw numerical generation/fitting/keys/acceptance remain unchanged and JAX
batched; Python loops here only audit and serialize existing host records.

Every group must have unique complete 0..attempted-1 IDs. Flags are binary,
accepted complements failed, failed is the OR of separate reasons, and each
accepted/failure count matches the performance table. Counts may overlap by
reason; failed is never computed by summing reasons. Every rejected ID with
all reason flags, max absolute gradient (including NaN if unavailable), and
log HR is retained. Accepted IDs are their complement in the recorded design.

Every accepted HR is positive/finite and included in common .025 bins.
Outward histogram edges modify display bins only, not fitted estimates.
Empty accepted sets get zero counts. One mutually accepted replicate is
selected at n=300/.55 (n=100/.55 for smoke); if none, illustration.csv has a
header only. Later plotting must handle that explicitly. Current full run
selects scenario 3, replicate 0 for both models. Metadata/performance bytes
are preserved; evidence.json binds all original raw/output hashes and the
reducer source hash. It is written last and invalidated with raw CSV replacement;
no stale manifest should certify a partial write. This is not a filesystem
transaction against concurrent writers, which are outside this single-run CLI.

## Checks and evidence

Ruff format/check and mypy src passed (148 files; 22 source files).
Optional focused command (CPU probing requires escalation in this sandbox):

```bash
JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q examples/weibull_simulation/check_results.py examples/weibull_simulation/check_cli.py examples/weibull_simulation/check_metadata.py
```

16 passed in 11.89 s. Initial sandboxed invocation: 15 passed/1 failed because
its existing CPU-identity expectation calls sysctl; the escalated rerun passed.
Final check_results.py rerun after the stale-manifest/empty-accepted checks
and fixture cleanup: 7 passed in 2.64 s. No tolerance was changed.

```bash
scripts/check_wheel.sh --through A10
uv run --locked --group examples python examples/weibull_simulation/_plots.py --results /private/tmp/survjax-sim-full-a3792a7 --out /private/tmp/survjax-sim-compact-implemented
```

Cumulative wheel gate exited 0: implementer tests x64 1282 passed/8 skipped
(185.37 s), float32 1287 passed/3 skipped (132.77 s); source doctests 4 passed
each mode; acceptance x64 2333 passed/27 skipped (78.84 s), float32 2324
passed/36 skipped (66.45 s). Final: "check: all gates passed through A10".
Wheel retained at /var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/
survjax-wheel.3hjJds/survjax-0.0.1.dev0-py3-none-any.whl. The reduction
command exited 0, generated 23,481 bytes including manifest. It keeps all 127
failures and histogram counts total 11,873; the 4.6 MiB raw CSV is not copied.
All five approved compact payloads have exactly the prototype bytes; source
metadata remains clean a3792a7, not falsely attributed to this postprocessor.

Requirement mapping: hand fixture checks overlapping failures and accepted-ID
complement, both lower/final-upper histogram inclusion, common replicate
selection, source/output checksums and no copied raw CSV. Four corruption
checks reject missing/duplicate IDs, reason-count and acceptance inconsistency
before creating output. The no-accepted-fit case retains all six failures,
zero histogram counts and an empty illustration. Existing roundtrip proves raw diagnostics still retain
rejections/unestimable rows and invalidates metadata/evidence on replacement.
Existing CLI checks both grids, calibrated constants/keys/export schema and
executes the smoke command. Fake reason masks now match their stated counts;
no assertions were weakened. These are optional explicit checks, not a silent
skip or new dependency in ordinary core tests.

## Full-run science already audited before reduction

The full CLI ran from clean a3792a7 into /private/tmp/survjax-sim-full-a3792a7.
Its metadata records CPU Apple M4 Max, Python 3.12.9, JAX/jaxlib 0.11.2,
NumPy 2.5.3, Optimistix 0.1.0, Matplotlib 3.11.2, x64 and unset curated thread
settings; effective worker count and peak memory not measured. Calibrated c
47.793149053990135/.25 and 19.17253543619806/.55. Full has 12,000 fits:
11,873 accepted, 127 gradient-only rejections, all solver-success/finite/SPD/
estimable. Strict-gradient residuals range 1.0008445445388515e-5 to
5.097075334365111e-5. Maximum rejected H^-1*g log-HR correction is
1.0307596577744073e-7; Exponential closed-form d_g/Y_g log-HR discrepancy
across all fits is <=1.0307596665093399e-7. Small local discrepancies despite
failed scores are observed; relative solver stopping on the summed objective
is a plausible explanation, not an instrumented causal proof. No gate changed.
Weibull n=1000 coverage .9434156378600823 +/- .007410825977435696 MCSE and
.9307851239669421 +/- .00815805747879105 MCSE: -.88848/-2.35532 MCSE from
.95, both within 3 MCSE.

Independent exploratory audit /private/tmp/check_full_expanded.py reconstructs
all PRNG draws with JAX primitives and NumPy inverse CDF, checks all objectives,
analytical scores/Hessians/inference/counts/summary MCSEs, and both calibration
integrals at 40-digit precision. Its two hand-derivative checks passed. This
exploratory audit is not claimed as a committed check; the new committed code
is a compact exporter/consistency audit, not another survival implementation.
Full raw SHA-256: 58bc3ae48a263f0f4ea9ab522bbdff0cf02627301bcbd0477e370e7785fbc7a4.

Remaining: commit compact artifacts in budgeted pieces, implement/review the
three figures (explicitly authorized next), then executable docs. The frozen
shared generator import contract is unchanged; no merge/push/publication.
