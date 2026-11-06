# WHEEL.1 — Prepare alpha 2 and fix release version checks

Independent review: round 1/3

Read first: PLAN.md, pyproject.toml and uv.lock version-only diff,
tests/install/test_wheel.py, tests/spec/test_a9_package.py, D-048.

Maintainer explicitly authorized the protected spec version check change in
conversation. This is the only protected file changed. No new API or deps.
D-048 remains formally open; maintainer authorized metadata equality rather
than the pre-release initial-version literal. Spec itself is unchanged.

Invariants: project and installed versions agree exactly; isolated smoke
imports the installed wheel outside the source checkout. Host reads the
project version using stdlib tomllib and passes argv to the isolated Python.
The spec check independently reads project metadata. Runtime imports remain
free of test/demo deps. uv.lock changes only the local package version.

Evidence:
- Original a1 wheel tests: 4 passed per dtype after smoke fix.
- Alpha 2 touched tests: x64 install + release validator: 10 passed in 21.65s;
  x64 protected package suite: 11 passed in 2.97s; x32 all three files:
  21 passed in 26.32s. No marker filtering in these runs.
- uv lock --check --offline: resolved 166 packages, success.
- uv build --offline --out-dir /tmp/survjax-alpha2-wheel-fix: wheel rebuilt
  from sdist, both artifacts 0.0.1a2. Release validator accepts v0.0.1a2.
- scripts/check.sh --fast --through A10 before implementation commit:
  all gates passed. Ruff: 192 files formatted, no lint issues. mypy: no
  issues in 22 source files. Implementation x64: 1072 passed, 8 skipped;
  x32: 1077 passed, 3 skipped. Doctests: 4 passed per dtype.
  Spec x64: 2148 passed, 28 skipped; x32: 2154 passed, 22 skipped.
- git diff --check passes. Full tier-2 gate not run.

The branch guard checks committed main...HEAD only; after this commit it will
reject the explicitly authorized protected test on patch-wheel-test. Do not
change the guard or branch to conceal that. Independent reviewer should
report this branch policy conflict separately from the passing test evidence.

Spec requirements: existing installed-core test covers 10.2–10.3 metadata,
exports, marker, dependency isolation, numerical runtime. A9 package test
covers installed project-version equality for the release bump.

GitHub read-only inspection: v0.0.1a1 is a published prerelease; its build job
failed checks and publish job was skipped (run 42288066496). PyPI JSON request
returned 404. No upload, tag, push, remote or publishing mutation performed.

Starting/previous step/original worktree base: 8794005.
Plan archive: 69a66df. Remaining separate docs step: release guide candidate
and commands, changelog/citation status. Tag v0.0.1a2 must target the reviewed
commit after integration, under maintainer release procedure. Existing tags
remain untouched. Historical benchmark/example provenance must stay unchanged.

## Round 1 outcome — maintainer

Review file: ../review/slices/WHEEL.1.md. Verdict: maintainer. Stop per skill.
B-1: reviewer requires spec-author/sync routing for the protected test and
cannot inspect conversation approval. Maintainer's exact answer here was:
"Yes, update that version check" in response to the explicit question naming
tests/spec/test_a9_package.py and comparison with pyproject.toml. I implemented
that authorization. Do not undo the user's requested edit merely to change a
review verdict. The checked-in guard nevertheless has no conversation-approval
override: postcommit scripts/check.sh --fast --through A10 exits 1 at the
protected path. All test/static results above precede that commit and passed.
Maintainer must decide integration routing; no guard edits or branch changes.
S-1: D-048 remains open for maintainer resolution. Project metadata versus
installed artifact metadata checks distinct sources, so a wrong-version wheel
is detected; it intentionally does not impose the initial development version.
N-1: wording cleanup is deferred to maintainer resolution as review requests.
Reviewer could not run tests because its uv cache access and escalation were
blocked. No independent runtime verification claim is made.

Last implementation commit: 2e4971db20493123d1a3e4adb0c652a9d8930556.
Previous step/original worktree base: 8794005; archive: 69a66df.
Remaining: maintainer integration decision for B-1/S-1, separate docs step,
then final integrated CI and maintainer tagging/publishing. No new tag created.
