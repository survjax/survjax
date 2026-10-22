# A8.3 review nits — 2026-10-22

Requested N-1/N-2 follow-up to `1051217`, on `a8-1-r-generator`.
One fix-up commit, then stop for review; no merge or A8.4 implementation.

## Read in this order

1. Both fixture READMEs: safe regeneration commands, committed-file comparisons
   and the explicit warning about no-argument overwrites.
2. `tests/fixtures/{exponential,bc_like}/SHA256SUMS`: two SHA-256 entries each,
   for the exact committed CSV/JSON bytes. These replace hashes embedded only
   in README text with a machine-readable source for integrity checks.
3. `PLAN.md`: A8.4 loader hash checks and corruption-test requirements.

## Findings addressed

- N-1: README commands compare run 1 with run 2, then run 1 with each committed
  payload. Hash verification covers both files too. No-argument execution is
  removed from the verification block and explicitly documented as overwriting
  committed files.
- N-2 (A8.4 handoff): manifests and a working `shasum -a 256 -c SHA256SUMS`
  command are available now. The planned shared loader must verify both files
  with hashlib before parsing, fail on missing/malformed manifests or mismatched
  content, and test independent CSV/JSON edits. No automatic pytest loader check
  is claimed yet: the loader belongs to A8.4 after A6/A7 integration.

## Verification

- In each fixture directory, `shasum -a 256 -c SHA256SUMS`: exit 0;
  `data.csv: OK` and `expected.json: OK` for both fixtures.
- `cmp` each committed payload against both original A8.3 generation runs:
  all eight comparisons exit 0. No R regeneration or payload changes.
- Temporary copies of each fixture: flip one CSV event, then independently
  replace one JSON total with zero while retaining valid JSON. In all four
  fixture/file combinations, shasum exits nonzero and reports the modified
  file FAILED while the other file remains OK. All temporary copies removed.
- `sh -n` on both README shell blocks: exit 0.
- `git diff --check`: exit 0, no output.

No full through-A5 gate, Ruff, mypy, R checks or docs build rerun for this
README/manifest/plan-only fix. No generator, library, test, dependency,
fixture-payload, spec or spec-marker changes. No new DEVIATIONS entry.
A8.4's actual loader checks remain future work; stop after committing this fix.
