# B3.2 — Commit and exercise the Slice B R fixtures

Feature: left/interval censoring and truncation reference evidence.
Base / preceding step: 6353c04655edab82a02bfdf42cd66b8ee5e6f422 (B3.1).
Original worktree base: cc0e30f. Dependencies: B1/B2 likelihood work and
[B3.1 generator](../B3.1/PLAN.md); merged LogNormal/LogLogistic families.
Cumulative gate: A12, existing observation feature flags unchanged.

Commit the byte-identical, pinned-R outputs from B3.1 in
tests/fixtures/slice_b/{data.csv,slice_b_likelihood.json}, with SHA256SUMS.
Install the same JSON at tests/spec/fixtures/slice_b_likelihood.json.
The maintainer's go-ahead for the announced fixture-integration step authorizes
this payload installation; no spec-test code, assertions or markers change.
This completes the intermediate directory state raised as B3.1 S-1.

Extend the existing tests/conftest.py loader for manifest validation, provenance
and CSV/JSON alignment. Add tests/reference/test_slice_b_r.py: six fixed vectors,
per-row likelihood agreement under eager/jit/vmap in float64, valid diagnostics,
and byte identity of the implementation/acceptance payloads. Keep host loading
and identity checks active in both dtype modes. Update the fixture README with
regeneration and verification instructions. No public API, library, dependency
or docs/ changes; implementation plus payloads targeted below 300 changed lines.

Spec coverage: sections 2, 6.1–6.2, 10.4, 12.1–12.2. Validation: existing
generator output hashes, touched reference tests in both modes, independent
fixture tests with all three families enabled, fast A12 gate, independent review.
If a new test exceeds one second, update the slow-test list with the repository
script before committing. Full nonfast gate remains the maintainer/CI tier.

Resume: the maintainer cleaned the reviewer checkout and requested restarting
review. Keep the generated 479-line timing refresh in a separate preparatory
commit, followed by the fixture integration and these records; both belong to
B3.2. The implementation/payload change remains below 300 lines.

## Docs to-do

Later likelihood docs can cite committed flexsurv agreement for these ten cells;
list the excluded conventions (B3.1 README). Remaining: audit mathematical matrix
coverage, B4 static-metadata pruning and docs. This step does not claim Slice B
completion or activate unrelated family feature gates.
