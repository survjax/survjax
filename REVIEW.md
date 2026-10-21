# Revision 32 alignment and A6.4 replacement plan

Status: spec read and marked on maintainer-selected branch `A6`, based on
`21c2632`. This commit changes only `.spec-seen`, PLAN.md and REVIEW.md.
No source or test implementation has been redone yet. The new plan awaits
approval under working agreement section 1 before code changes.

## Read first

1. Spec 7.1–7.3 and 12.1 at `cf4fef9` (revision 32), and accepted decision
   0045: parameter-gradient isolation before family code.
2. `../review/slices/A6.4a.md`, B1 and its addendum, plus the designer request:
   the prior zero-only guards do not cover all declared violations.
3. PLAN.md's opening alignment and A6.4a rework / A6.4b1 / A6.4b2 / A6.4c:
   withdraw guards, prepare parameters centrally, integrate the evaluator,
   then expose the public function and method. Each part ends in a stop.
4. `.spec-seen`: `cf4fef9`; independent tests remain snapshot `f527f75`.

## Contract and planned evidence

- Spec 7.1: every own/nested parameter leaf is first broadcast to the
  current outer batch shape, preserving trailing component axes. At entries
  in the declared union mask, gradients are stopped before family code.
  Numerical parameter values remain unchanged; invalid contributions are NaN.
- Spec 7.2: this is the explicit exception to the older rule against stopping
  actual parameters. Valid entries remain differentiable. The protected
  CLAUDE.md has not been edited; revision 32 settles the exception.
- Spec 7.3/12.1: planned helper and evaluator tests compare valid rows' values,
  Jacobians and Hessians with those rows evaluated alone, in forward/reverse
  modes, eager/jit/vmap, both dtypes. Coverage includes zero, negative, NaN,
  both infinities and named-condition violations in built-in, custom and
  nested families; component axes, smaller rebuilt leaves and draws batches.
- Invalid-row gradients remain unspecified. Undeclared invalidity has no
  isolation guarantee. Direct-method policy and unsupported-mode policy
  remain unchanged. There is no promise of direct-method gradient isolation
  at invalid parameters.
- A6.4a rework will remove the superseded guards and their dedicated test
  module in a follow-up commit, retaining the original A5 boundary guards.
  Its existing direct-method suites and through-A5 gate must still pass.
  Complete likelihood evidence follows in A6.4b2/c, not in this alignment.

## Checks actually run

- Read the full normative diff `git -C ../design diff be7b253 cf4fef9 --
  spec/SPECIFICATION.md`, current changed sections and decision 0045.
- `../scripts/spec-since.sh dev --mark`: exit 0,
  `dev/.spec-seen -> cf4fef9; commit it in dev`.
- `../scripts/spec-since.sh dev`: exit 0,
  `dev: spec unchanged since cf4fef9`.
- `git diff --check`: exit 0, no output.

No pytest, Ruff, mypy, runtime gate or docs build was run for this metadata
and planning change. The earlier A6.4a results remain in commit `21c2632`;
they do not establish compliance with revision 32. No independent-test sync,
spec edit, dependency, docs edit or source/test change in this pass.

## Limits and uncertainties

No open design question: revision 32 and decision 0045 settle review B1.
The replacement plan is deliberately split to keep each code/test diff near
300 lines; replan before exceeding that budget. Independent revision-32
acceptance tests must come from their author through the maintainer's sync
workflow; their absence does not reduce our own planned test coverage.
No deviation was added or marked resolved.
