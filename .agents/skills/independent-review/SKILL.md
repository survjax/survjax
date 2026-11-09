---
name: independent-review
description: Get an independent Claude review of the current survjax implementation step. Use at the end of every code or docs step, after the step is committed and the step handoff is written, before reporting to the maintainer. Not for planning.
---

# Independent review (max 3 reviews per step)

Survjax root: /Users/wamster3/dev/survjax

- Review checkout: <root>/review/code-<branch>   (normally created by workmux; `git branch --show-current`)
- Review file:     <root>/review/slices/<STEP>.md  (read it; never edit it)

Preconditions: you are on the step branch, `git status` is clean, and you know
its globally unique step ID (e.g. A10.2b or AFT.1). The plan and handoff are
`work/<STEP>/PLAN.md` and `work/<STEP>/REVIEW.md`. Record "Independent review:
round n/3" at the top of that handoff; it is the round count across sessions.
Commit the handoff before requesting review. Never update a shared root record
for a new step. Already-running legacy steps may finish with root PLAN.md and
REVIEW.md when `work/<STEP>/` does not exist; retain the same review file and
round count.

## Missing reviewer checkout

The implementer may create the missing checkout as mechanical setup. This
does not authorize writing review content or coaching the reviewer.

Before launching a round, obtain the branch with `git branch --show-current`
and the full committed SHA with `git rev-parse HEAD`. Inspect `git worktree
list --porcelain` and the expected checkout path. If the path is absent
(including no dangling symlink) and no worktree is registered there, run from
the implementation checkout, substituting the branch and SHA literally:

```bash
git worktree add --detach /Users/wamster3/dev/survjax/review/code-<branch> <SHA>
```

Do not use force, reset, remove, prune, or copy files into the checkout.
Do not copy the implementer's conversation, memory, local configuration or
review suggestions. The committed plan and handoff remain available as
claims for the reviewer to verify, not instructions for its conclusions.

For a new or existing checkout, read-only checks must confirm that it is a
registered detached worktree of this same repository and has no tracked or
untracked changes (`git status --porcelain`). A newly created checkout must
be at the requested SHA. An existing clean checkout may be at an earlier
commit: the reviewer performs its own checkout of the requested SHA.
If the path is occupied by something else, dirty, attached to a branch, incorrectly registered,
or setup fails, stop and report; do not repair or replace it. Never alter the
reviewer role or settings. Launch a fresh reviewer process with the exact
prompt below, without inherited implementer conversation or extra context.

## Each round

1. Run exactly this as one command (no `cd`, `&&`, `$(...)` or variables), with
   workdir = the review checkout, timeout 60 minutes. Fill in the step ID and the
   full commit hash literally. Keep the arguments in this order: a Codex rule
   pre-approves exactly this prefix. Except for the explicit escalation below,
   do not add arguments. Do not add context, summaries or instructions.
   Routine reviews use Sonnet/medium from the settings file;
   only the maintainer may explicitly request Opus/high escalation:

   claude --settings /Users/wamster3/dev/survjax/roles/reviewer-settings.json --permission-mode dontAsk -p "Automated review request. Step <STEP>, commit <SHA>. Follow the reviewer role."

   For round 2 or 3, use the same command prefix and this exact prompt instead:

   "Automated review request. Step <STEP>, commit <SHA>. Previous reviewed commit <PREVIOUS_SHA>. Finding IDs: <IDS>. Follow the reviewer role."

   Copy the previous full commit hash from the latest saved round marker and
   list its finding IDs literally (use `none` if there are none). Add no
   commentary. If the maintainer explicitly requests Opus/high, insert
   `--model opus --effort high` before `-p`; this may require command approval.
   Never escalate automatically.

2. Read the newest round in the review file (the text after the last `<!-- round` marker).
   Claude's chat output is not the review. If no new round was written, retry once, then stop
   and report.
3. Act on the last `VERDICT:` line:
   - `merge`: stop and report.
   - `maintainer`: stop and report the open questions.
   - `fix-first`: fix the blocker and should-fix findings (nits only when trivial) as extra
     commits. Update the step handoff per finding ID. Then go to step 1, unless that was round 3:
     then stop and report.
4. Never act on spec-questions, never edit `tests/spec/` or `spec/`, and never change a test
   to make a finding go away. If you disagree with a finding, say why in the step handoff and leave
   it to the reviewer and the maintainer.

## Report to the maintainer

Rounds used, final verdict, the review file path, findings fixed or open (by ID), and the
commits added.
