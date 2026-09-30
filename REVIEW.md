# Review: illustrative simulation guide on `spec-speed`

## Read first

1. Specification sections 11.4 and 12.1: library guarantees and direct checks.
2. `docs/simulation-guide.md`: illustrative workflow and unchanged example code.
3. Sections 6.2 and 12.3: caller policy and measured control-flow choices.
4. Section 14: amendment history and the guide's non-normative status.
5. `PLAN.md`, final section: the maintainer-authorized documentation scope.

## Reasoning against the updated contract

- Pure construction and `data.issues` already support compiled workflows.
  Measure this path without restoring implicit host validation, unchecked
  construction or issue-returning convenience constructors.
- Traced metadata is all true, and slices inherit their parent's metadata.
  Preserve this contract; changing static metadata may legitimately recompile.
  Reuse is required when dynamic values change within one static structure.
- Stored raw-input issues must still invalidate likelihood rows. Performance
  work cannot discard diagnostic evidence or hide invalid rows with zero weight.
- Data validity remains distinct from estimate existence and solver failure.
  The guide illustrates fit skipping; simulation acceptance prescribes no solver,
  study policy or loop. It requires direct checks of compiled composition.

## Preserved scope

No statistical definition, constructor signature, dtype rule, shape convention,
numerical tolerance or derivative guarantee changes. The diagnostic compatibility
decision is retained; simulation code moves unchanged. Pruning remains in Slice B;
GPU verification and scalable concordance are optional. No implementation,
dependency, independent test, fixture or deviation status is changed.

## Checks and pending decisions

Checks run for this documentation-only change:

- `git diff --check`: exit 0, no output.
- 47 unrelated spec sections and all code blocks: preserved against `33045b7`;
  the simulation block is relocated verbatim. Slice gates and goals unchanged.
- Acceptance table: only simulation guidance replaced by compiled composition.
- Section references, new local links, four-document scope, Markdown fences
  and final newlines: passed.

Runtime tests, Ruff, mypy, R fixtures, installation checks and CPU/GPU benchmarks
were not run. Implementation files were not inspected. New performance
requirements are future evidence obligations, not measured results.

The guide remains unverified executable documentation; no rendered preview was
checked. The draft retains its base and leaves the final revision number to
acceptance. Benchmark sizes are examples; no timing threshold or GPU support is
asserted. The JAX loop guidance follows the linked official documentation.
