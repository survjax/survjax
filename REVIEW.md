Independent review: round 1/3

# SIM.8c.2 — Compact full-run evidence

Read newest PLAN section and approved D-035, results/README.md, evidence.json,
then the three small data payloads and new committed-bundle checks in
check_results.py/check_metadata.py. Previous final:
c08a14281e4aa42a72e3882e6f19445ada0eb401 (SIM.8c.1 round 2 merge,
F-1/F-2 fixed). Original base 8757c0fda78e34e6e818249adedd2a144a33c33d;
integration base d2469e0a7e719dbcb276277e6fd80a447626ad4e. User explicitly
authorized compact evidence and then figures; this is the budgeted data split.

## Provenance and invariants

All payloads derive from /private/tmp/survjax-sim-full-a3792a7, generated from
clean a3792a7703ed539db777472d876bd00ac2d1568e. No refit. Reduction used clean
reviewed c08a142, producer _evidence.py, reducer_dirty=false. Original metadata
and performance bytes are unchanged. No raw CSV is copied/staged; the ignored
raw file remains local. Core API/generator/dependencies/spec/protected paths
are unchanged. All new test loops inspect committed host records only.

failures.csv retains all 127 rejected IDs and each reason flag; accepted IDs
are the complement in each 0..999 group. Every rejection is gradient-only;
solver success never overrides the strict max|gradient|<1e-5 policy.
sampling_histograms.json retains exact counts of all 11,873 accepted HRs at
63 common stored bin edges (about .025 wide), including the last right edge.
illustration.csv contains the complete parameters for both models at the
first mutually accepted n=300/.55 replicate (scenario 3, ID 0).
evidence.json binds original raw checksums and all five compact payloads.
The complete six-file bundle is about 23 KiB; no external hosting/compression.
README explains original absolute paths, accepted denominators and regeneration.

## Validation

```bash
JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q examples/weibull_simulation/check_results.py examples/weibull_simulation/check_cli.py examples/weibull_simulation/check_metadata.py
uv run ruff format --check .
uv run ruff check .
uv run mypy src
```

25 passed in 10.40 s; Ruff/mypy pass (149 files / 22 source files).
New checks demonstrate all 12 groups, unique in-range failed IDs, individual
reason totals, attempted=accepted+failed, integer histogram totals including
every accepted fit, a common accepted illustration ID, both n=1000 Weibull
coverage gates, all output hashes and clean source/provenance/root seed.
These optional checks need only committed files, not the raw CSV or network.

Independent full-run reconstruction/analytical derivatives and the 40-digit
calibration audit were recorded in SIM.8b REVIEW (ecb0bdc/61c5cc6). Reviewed
reducer output was separately recounted by Claude in SIM.8b and SIM.8c.1;
this step preserves those bytes. Manual audit verified the historical reducer
Git object matches its source hash, and all output checksums match. No bitwise
reproducibility of timings or across machines is claimed.

No repeated full simulation or core gate for this artifact/test-only step.
The prior wheel-supplying through-A10 gate passed both modes: implementer
1282/1287, doctests 4 each, acceptance 2333/2324, with expected skips.
Reviewer raw gates lack SURVJAX_WHEEL (five wheel setup errors); distinguish
that environment issue from the passing wheel-supplying gate.

Remaining: three SVG figures (already authorized), then executable docs.
No merge/push/publication. Shared generator import contract remains frozen.
