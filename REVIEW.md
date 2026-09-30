# Review: performance amendment on `spec-speed`

## Read first

1. `spec/GOALS.md`, quality goals, and specification section 1: performance
   priority and the limits of the v0 commitment.
2. Specification sections 7.2 and 12.3: execution requirements and measurement
   of prepared data versus pure ordinary construction from traced inputs.
3. Sections 9.2 and 10.3: exact concordance, explicit caller subsampling and
   optional accelerator verification with evidence before support claims.
4. Sections 12.1, 12.2 and 14: the new CPU baseline and draft history.
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
- Section 11.4's simulation workflow distinguishes invalid data, non-existent
  estimates and solver failures. Its scalar fit-skipping decisions must remain
  meaningful when profiling mapping choices; `vmap` is not mandated for speed.

## Preserved scope

No statistical definition, constructor signature, dtype rule, shape convention,
numerical tolerance or derivative guarantee changes. The diagnostic compatibility
decision and simulation examples are retained. Pruning remains in Slice B;
GPU verification and scalable concordance are optional. No implementation,
dependency, independent test, fixture or deviation status is changed.

## Checks and pending decisions

Checks run for this documentation-only change:

- `git diff --check`: exit 0, no output.
- Unchanged statistical/API sections: 35 verified against `9a403a3`.
- All specification code examples/signature blocks: unchanged.
- Existing acceptance criteria and slice gates: preserved; only CPU baseline added.
- Safe-input rules, concordance definition and existing environment requirements: preserved.
- Section numbering and textual references: passed.
- Draft base, four-document scope, Markdown fences and final newlines: passed.

Runtime tests, Ruff, mypy, R fixtures, installation checks and CPU/GPU benchmarks
were not run. Implementation files were not inspected. New performance
requirements are future evidence obligations, not measured results.

The base is `9a403a3`: its header says revision 9 while its contract/history
include revision 10. The draft names that base and leaves the final revision
number to acceptance. Benchmark sizes are examples; no timing threshold or GPU
support is asserted. The existing platform references were checked during the
2026-09-29 discussion. The earlier `codex/spec-performance` draft is retained
as history; this branch adapts the proposal without merging its older files.
