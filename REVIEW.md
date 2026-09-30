# Merge-resolution handoff — revision 9 on a2-constructors

Status: conflict resolution only; no revision 9 implementation performed and
no claim that A2 now passes the new contract. The maintainer completes the merge.

## Read first

1. `PLAN.md`: replacement A2 migration proposal; approval is required before code.
2. `spec/SPECIFICATION.md` sections 3.1–3.3, 6.2, 10.2 and 12.1: merged contract.
3. `src/chronox_jax/data.py` and `diagnostics.py`: current implementation still
   uses eager constructor checkpoints and the removed convenience method.
4. `DEVIATIONS.md`: duplicate D-006 identifiers and older questions to reconcile.

## Resolution and invariants

The old PLAN.md and REVIEW.md conflicted because they describe different work:
A2 implementation versus the specification revision. Both have been replaced
with a current plan and this handoff, as requested. Previous text remains in Git.
No runtime source, test or automatically merged file was edited during this
resolution. The staged specification is revision 9. Only the two conflicted
files are staged by this task.

The next implementation must preserve normalization, dtype rules and issue
bits while making ordinary construction pure. Data diagnostics use the
`data.issues` property; invalid values are distinct from static misuse. Metadata
completion and likelihood implementation retain their A3/A6 step boundaries.

## Checks for this resolution

- Initial `git status` / `git ls-files -u`: exactly PLAN.md and REVIEW.md unmerged.
- Read the merged specification, working agreement, deviations, current source
  and relevant own-test references to prepare the replacement plan.
- `git diff --check` and `git diff --cached --check`: passed with no output.
  After staging, `git ls-files -u` returned no output; `git status` reported
  "All conflicts fixed but you are still merging."
- No runtime checks, A2 acceptance gate, R checks or wheel builds were run for
  these planning documents. No historical test counts are presented as current.

## Remaining decisions

The automatically merged deviations contain two distinct D-006 entries. Their
identifiers/references need maintainer reconciliation; statuses remain untouched.
Spec 13's exception-lifecycle and traced-checkpoint error questions remain open.

CLAUDE.md's Git rule says agents must never merge, so no merge commit is made
here. Once the resolution is staged, the maintainer can finish the existing
merge with `git commit --no-edit`. CLAUDE.md's plan-first rule requires approval
of the replacement PLAN.md before implementation begins.
