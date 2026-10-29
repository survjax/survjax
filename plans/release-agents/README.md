# First-release worktree handoffs

Prepared 2026-10-29; assumes A10.4 and acceptance fixtures are merged.
The maintainer already assigned all remaining A9 work to another agent.
No agents, branches or worktrees are created by these plans.

## Assignments and launch order

| Agent / proposed branch | Handoff | Start condition |
| --- | --- | --- |
| Existing A9 agent | Existing A9 plan | Already assigned; retain that ownership |
| Existing A10 agent / later `a10-followup` | [A10 follow-up](concordance-docs.md) | Finish in-progress A10.d1 first |
| PH / `release-ph` | [PH](ph.md) | Integrated A10.4; PH/RMST ownership agreed |
| RMST / `release-rmst` | [RMST](rmst.md) | Integrated A10.4; PH/RMST ownership agreed |
| Demo foundation + simulation / `release-simulation` | [Simulation](simulation.md) | Dependency approval; fitting waits for PH |
| Survival MNIST / `release-mnist` | [MNIST](mnist.md) | Shared generator and dependency setup merged |
| Bayesian RMST / `release-bayes` | [Bayesian](bayesian.md) | Generator, dependencies, PH and RMST merged |

Continue the existing A10 guide; start PH and RMST independently. Foundation work may also start
before PH; once its reviewed generator and environment land, simulation and
MNIST can proceed independently. Bayesian delivery waits for PH/RMST closure.
These are separate user-run sessions; no resident subagents are required.

## Instructions included with every assignment

Read AGENTS.md, the current spec and the linked handoff in your own worktree.
The maintainer creates your branch and matching reviewer checkout. Start from
a clean agreed main commit and record it; do not create/merge branches, rebase,
push, or cherry-pick another agent's unfinished work. Request a maintainer
integration when prerequisites land. Never share a virtual environment between
worktrees. Confirm branch and status before writing.

This packet plans first-release items explicitly brought forward in spec 12.2;
it does not authorize the rest of slices B–F. Assignment authorizes planning;
implementation follows a concrete PLAN.md and the existing approval rules.
Dependencies, protected files, unnamed modules/API, ambiguity and oversized
steps still require the maintainer's decision. Read use-case main AND details.

Use one concept per step, about 300 changed lines including tests. Treat the
numbered steps below as proposed splits; refine before code and split again
when necessary, including generated results. Archive the completed previous
sub-slice verbatim in its own commit under the established PLAN.md workflow;
do not archive an unfinished sub-slice. PLAN.md and REVIEW.md are local to each
branch; preserve records by step during maintainer integration.
After each implementation/docs step: commit, run independent-review, report
and stop. These planning-only handoffs need no independent implementation review.

## Shared-file ownership and integration

- A9 agent: CPU baseline, general Slice A verification, replacement mini README
  example and general playground work, plus the final core evidence audit.
  Coordinate its teaching example with simulation; do not duplicate it.
- PH agent: wrappers.py, both distribution export files, exact public-export
  assertions/install expectations, and any required API-reference export entry.
- RMST agent initially: base.py, exponential.py, weibull.py and new justified
  RMST tolerances. PH does not edit those until RMST's core work merges.
- Simulation agent: shared demo generator in its study.py and pyproject.toml /
  uv.lock for the approved examples group. Other agents request dependency
  changes from this owner; they never regenerate competing lockfiles.
- Each docs/demo agent owns only its named page/example/tests/results.
  The existing A10.d1 agent retains its current guide/index/API edits until
  that step merges; no second agent starts those files from this packet.
  docs/index.md, docs/conf.py, build-tool changes and shared playground edits
  are serialized through the maintainer/A9 agent, after the relevant pages exist.
  Demo dependencies must be available to fresh docs builds before integration;
  propose the precise command/config change, do not edit protected scripts.
- Never edit tests/spec/ or spec/. Independent tests/fixture sync stay with the
  maintainer. Do not weaken acceptance selection or assertions.

## Common evidence and final gate

Record exact commands, outputs, versions and limits. Core steps run focused
tests in separate x64/float32 processes, Ruff, mypy and the unchanged available
cumulative gate. New core APIs need transformation/derivative/boundary checks
per spec 12.1; docs steps execute scripts/build_docs.sh and inspect rendered HTML.
Keep ordinary tests offline and independent of optional demo packages/data.
Give demo checks an explicit optional-environment invocation without changing
the standard runner or silently skipping failures. Full demos are separate runs.

After all branches integrate, the maintainer coordinates a final release audit:
through-A10 regression, new PH/RMST acceptance evidence once supplied, installed
wheel AND source-archive checks, fresh docs/playground, all three full demos,
and an evidence map for each spec 12.2 release item. A9 owns core closure;
it does not certify future demo/PH/RMST work. Publication remains separate.
