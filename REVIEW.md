# MNIST.17 — Apply walkthrough annotations

Independent review: round 1/3.

## Read first

1. PLAN.md MNIST.17: complete annotation checklist from maintainer's mnist2.md.
2. docs/use-cases/survival-mnist.md diff from 7840cd7.
3. Built docs/_build/html-x32/use-cases/survival-mnist.html and x64 equivalent.

The maintainer requested all deletions and comments in mnist2.md, including
removing the censor-fraction sentence, pixel-scaling/outcome-repeat prose and
oracle-gap sentence. Actual shapes replace the relative shape difference.
The loss is shown without annotations (docs only); Adam is called an optimizer;
timing starts “Calculating”; takeaways describe shared shape/per-image scale
as a choice, with both parameters per image as a possible alternative.
The annotation file is user-owned and remains untouched/untracked; the plan
records its full checklist for the reviewer checkout. No new spec decision:
metadata still reports censor fractions and the table retains oracle metrics.
No runtime/API/dependency/result changes. The note proposes no implemented
per-image shape head and preserves the shared-shape risk-score explanation.

## Verification and invariants

- Direct scripts/build_docs.sh exited 0; both modes report build succeeded.
  Live MNIST execution: 16.51s x32, 21.58s x64. IPKernelApp emits the existing
  TCP-without-encryption warning; no notebook execution errors.
- Parsed both rendered pages: untyped loss, shape 1.368 versus 1.380,
  flexible-parameter note and timing wording present; deleted passages absent.
- AST comparison: displayed loss body equals model.py loss body (excluding
  its docstring); no computation changes. All annotation comments removed.
- git diff --check: clean. No new tests for prose, no broad gate rerun by
  implementer. Docs build executes smoke; the previous full results are reused.

Starting/previous step 7840cd7; original worktree base ee72c83. One docs step,
no remaining sub-steps beyond independent review. No known implementation issue.
