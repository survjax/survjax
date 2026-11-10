# B3.1 — Slice B fixed-parameter R generator

Implementation base: f9c9b0c; merged base / previous checkpoint: 6fafbaa.
Original worktree base: cc0e30f.
Dependencies: B1 interval hooks, B2 likelihood integration; spec revision 60
sections 2, 10.4, 12.1–12.2. Preceding records: ../B2.1/PLAN.md, ../B2.2/PLAN.md.
Cumulative gate: A12 with the two existing observation features.

Create scripts/fixtures/generate_slice_b.R and tests/fixtures/slice_b/README.md.
One shared seeded synthetic dataset: five rows in each of ten comparable cells;
Weibull (including shape 1), log-normal and log-logistic, two fixed vectors each.
Use pinned flexsurvreg fixedpars=TRUE outputs, check parameter preservation,
row order, sums, analytic likelihoods, hazard/mass/width constraints and boundary
coverage before export. Base R serialization; no dependency additions.
Schema comes from main c1a6a8d tests/spec/fixtures/README.md; do not modify or
copy protected tests. Generate temporary outputs only in this step. The next
step integrates committed payloads/shared loading and independent acceptance.
The pinned R environment was verified: R 4.4.2, flexsurv 2.3.2, survival 3.7.0.
Validation: R parse, generator self-checks, two byte-identical generation runs,
independent high-precision audit of the staged values, fast A12 gate, review.
No public API, library code, docs/ or protected path changes. About 300 lines
maximum for generator plus README; split if needed.

## Docs to-do

Later docs explain the ten comparable cells and excluded flexsurv conventions.
Remaining: B3 fixture payload/loader/acceptance integration, mathematical matrix
gaps if any, B4 metadata pruning and docs. Main's fixture tests were subsequently merged at the maintainer's request
(6fafbaa, main 491e611).

## Post-merge fixture check

Maintainer requested stopping generation, merging main, then checking fit to
requirements. Generation had already completed. Merge 6fafbaa preserves both
branches' plan-index links; no review-record conflict occurred.
The incoming independent fixture test file passed all 73 tests in float64 with
all four observation/family features enabled, using a temporary FIXTURE_DIR
pointing at /tmp/survjax-slice-b-run1 through the existing spec loader. No
independent assertion, marker, or protected file was edited. This is a staged
artifact check, not a passing committed-fixture gate. Two R runs were byte-
identical; all 300 values passed an 80-digit independent mpmath audit (maximum
absolute error 4.972287541523497e-15). CSV and JSON rows agree exactly.
The maintainer now requests independent review of the generator step. Commit
this generator and its records for review; payload installation into the
protected acceptance directory remains a separate integration step.
