# Review: specification revision 8 — pure SurvivalData construction

Read in this order:

1. `spec/SPECIFICATION.md` sections 3.1–3.2 and 6.2: pure construction,
   diagnostic property, unchanged static misuse and explicit host checkpoint.
2. Sections 7.1 and 10.2: stored raw-input issues and deliberate compatibility
   change. Ordinary eager construction no longer automatically raises for
   invalid observation values; migrate strict callers to explicit host raising.
3. Section 11.4: strict versus simulation workflows, pilot/batch guidance and
   illustrative compiled Newton solve with second parameter derivatives.
4. Sections 12.1–12.2: acceptance requirements and Slice A construction gate.
5. Section 13 and `DEVIATIONS.md` D-006: open compatibility questions.

## Invariants

- Both ordinary constructors and their convenience forms use the same
  normalization and issue semantics for valid and invalid observation values,
  eagerly and under transformations. Construction is not a validity certificate.
- Data retains exactly five observation fields and one issue-bitfield leaf;
  `data.issues` decodes it without duplicate leaves. Stored invalid input evidence
  reaches likelihood NaNs and `invalid_data` even after normalization.
- Static misuse raises `TypeError`; explicit unavailable dtype requests raise
  `ValueError`. Existing dtype warnings, endpoint rules and issue attribution
  remain. Concrete metadata uses normalized rows; any traced observation input
  requires conservative flags, including in ordinary construction.
- Structural invalidity, nonexistent estimates and optimizer failures are
  separate. All-censored exponential observations with positive follow-up are
  valid despite having no finite log-rate MLE.
- Newton on the likelihood gradient uses its Hessian; BFGS approximates
  curvature. Optimistix remains external. Example reason codes are local to
  the study, not new Chronox API. Pilots retain dataset size and failure records;
  continuation can use host batches or fully compiled stopping.

## Verification and acceptance handoff

No executable or independent acceptance tests changed. Section 12.1 now maps
construction checks to eager/JIT/vmap agreement, traced ordinary inputs,
static misuse and dtype rules (3.1, 3.3, 6.2); diagnostic/metadata checks to the
property, PyTree storage, host raising and convenience equivalence (3.2, 6.2);
and stored-input checks to likelihood propagation (7.1). These are requirements
for separate implementation and independent tester work, not passing claims.
The full optional simulation recipe is future example verification, not a new
core dependency or a claim that Slice A is complete.

Checks actually run:

- `git diff --check`: passed, no output.
- Constructor/exception reference audit with `rg` and review of the specification
  diff: active API text uses `data.issues`; the old spelling/rejection appears
  only in compatibility notes, open questions and explicitly historical text.
- Python `ast.parse` on the two section 11.4 Python blocks: both parsed.
  They were not executed: current source does not implement the specified
  diagnostic or likelihood APIs. No Optimistix dependency was installed.
- Checked Newton, root-finding failure results, BFGS and conditional batching
  against the primary Optimistix/JAX documentation linked in section 11.4.
- Initial `scripts/check.sh --through A1`: guard passed, then exit 2 because
  the sandbox denied uv cache access. Retried with approved cache access.
- Successful A1 gate: guard passed; Ruff format: `13 files already formatted`;
  Ruff lint: `All checks passed!`; mypy: `Success: no issues found in 2 source files`.
  Own tests: float64 **22 passed, 3 skipped**; float32 **25 passed**.
  Independent spec tests: float64 **58 passed, 13 skipped, 313 deselected**;
  float32 **66 passed, 5 skipped, 313 deselected**.
  Final output: `check: all gates passed through A1` (exit 0).
  This verifies the existing A1 code, not revision 8 behavior. R, wheel builds
  and later-slice checks were not run for this documentation-only revision.

## Open items and scope

D-006 leaves the legacy `TracedInputError` lifecycle open, as explicitly chosen
by the maintainer. The exception type for a traced host checkpoint and whether
to offer an alias for the old callable diagnostic spelling are also unspecified.
No such compatibility behavior was invented. These questions do not change the
approved pure construction contract.

Implementation, source docstrings and independent tests are separate work.
`spec/GOALS.md`, `src/`, `tests/`, dependency files and role instructions are
unchanged. The specification request authorizes this edit on `spec-constructor`.
This is one documentation concept; its contract replacement and workflow exceed
approximately 300 changed lines. Stop after this commit for maintainer review.
