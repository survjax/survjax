---
name: independent-review
description: Get an independent Claude review of the current survjax implementation step. Use at the end of every code or docs step, after the step is committed and REVIEW.md is written, before reporting to the maintainer. Not for planning.
---

# Independent review (max 3 reviews per step)

Survjax root: /Users/wamster3/dev/survjax

- Review checkout: <root>/review/code-<branch>   (created by workmux; `git branch --show-current`)
- Review file:     <root>/review/slices/<STEP>.md  (read it; never edit it)

Preconditions: you are on the step branch, `git status` is clean, REVIEW.md is written, and
you know the step ID (e.g. A10.2b) from PLAN.md. Record "Independent review: round n/3"
at the top of REVIEW.md; it is the round count across sessions.

## Each round

1. Run exactly this as one command (no `cd`, `&&`, `$(...)` or variables), with
   workdir = the review checkout, timeout 60 minutes. Fill in the step ID and the
   full commit hash literally. Keep the arguments in this order: a Codex rule
   pre-approves exactly this prefix. Do not add arguments, context, summaries or
   instructions:

   claude --settings /Users/wamster3/dev/survjax/roles/reviewer-settings.json --permission-mode dontAsk -p "Automated review request. Step <STEP>, commit <SHA>. Follow the reviewer role."

2. Read the newest round in the review file (the text after the last `<!-- round` marker).
   Claude's chat output is not the review. If no new round was written, retry once, then stop
   and report.
3. Act on the last `VERDICT:` line:
   - `merge`: stop and report.
   - `maintainer`: stop and report the open questions.
   - `fix-first`: fix the blocker and should-fix findings (nits only when trivial) as extra
     commits. Update REVIEW.md per finding ID. Then go to step 1, unless that was round 3:
     then stop and report.
4. Never act on spec-questions, never edit `tests/spec/` or `spec/`, and never change a test
   to make a finding go away. If you disagree with a finding, say why in REVIEW.md and leave
   it to the reviewer and the maintainer.

## Report to the maintainer

Rounds used, final verdict, the review file path, findings fixed or open (by ID), and the
commits added.
