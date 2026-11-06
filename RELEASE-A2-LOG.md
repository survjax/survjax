# Alpha 2 release execution log

## Authorization and starting state

Maintainer explicitly authorized guard/test repair, continuing past the review
stop, main integration, new tag/push and GitHub prerelease (PyPI trigger).
Base 8794005; version/test fix 2e4971d; previous handoff 61b7dcc.
Existing v0.0.1a1 remains unchanged. GitHub release exists; run 42288066496
failed its build checks and skipped publishing. PyPI JSON returned 404.

## Preparation

1. Read version references, release workflow, Q-041 and installed tests.
2. Replaced two stale initial-version assertions with project-metadata checks.
   Protected A9 edit explicitly approved. Bumped pyproject and uv.lock to a2.
3. Built wheel from sdist offline; validate_release.py accepts v0.0.1a2.
4. Touched tests pass in both modes; fast A10 gate passed before commit.
5. Independent WHEEL.1 round 1 returned maintainer on protected-branch policy
   and D-048. Maintainer then explicitly approved continuing and guard repair.
6. Added exact patch-wheel-test exception for the approved test and guard;
   other protected files remain blocked. Updated release commands, candidate
   citation version and changelog. Historical benchmark/example versions kept.

Further commands/results are recorded below as executed. No new release has
been published by these preparation changes alone.

7. `scripts/check.sh --through A10`: all gates passed. Ruff: 193 formatted,
   no lint findings; mypy: 22 source files clean. Implementation x64:
   1312 passed/8 skipped, x32: 1317 passed/3 skipped. Doctests: 4 each.
   Spec x64: 2333 passed/27 skipped; x32: 2324 passed/36 skipped.
8. First direct docs build compiled both sites successfully, but rejected
   certification because I finished the guard edit after starting it. This
   is the intended checkout-change detector; rerun against stable files.

9. Stable sandbox docs run failed on notebook CPU sysctl inspection (and
   psutil shutdown restrictions). Re-ran scripts/build_docs.sh directly with
   escalation: both x32/x64 strict builds passed and identity is success.
   Checked rendered releasing.html in both modes for a2 tag and no old tag.
10. bash -n scripts/check.sh and git diff --check pass.
