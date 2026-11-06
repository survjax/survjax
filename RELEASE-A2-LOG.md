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

11. Preparation committed 38afc97. Independent RELEASE.A2 round 1 verdict:
    maintainer, repeating protected-policy B-1/S-1. Reviewer cannot see the
    maintainer's explicit "Authorize all these release actions" reply and
    could not run tests (sandbox). Proceed under that recorded authorization;
    no new implementation defect reported. Candidate wording retained until
    publication; root log retained per request. D-048 left formally open.
12. Rebuilt final wheel from sdist at 38afc97 in /tmp/survjax-a2-final;
    validate_release.py accepts v0.0.1a2. Full artifact gate running with
    SURVJAX_WHEEL set to that exact wheel; results recorded after completion.

13. Exact final wheel gate passed both dtypes, including all slow tests.
    x64 implementation: 1312 passed/8 skipped; x32: 1317 passed/3 skipped;
    source doctests: 4 per mode; final x32 spec: 2324 passed/36 skipped.
    Local wheel SHA256 d6e6b8616b018b95d162fff18bee906a55147b5f0305f84d5cb35734ab68d870;
    local sdist SHA256 3583c7a535721e50ce66174a720833abd3c9aae98fe339b4b49fd2c4965cdd00.
14. Verified main worktree clean at 8794005 and remote main matched.
    git -C ../dev merge --ff-only patch-wheel-test advanced to 56311e3.
    git push origin patch-wheel-test:main succeeded. Hosted full CPU CI:
    https://github.com/survjax/survjax/actions/runs/42293972832 (pending).
    No new tag/release yet; waiting for CI before publication.

15. Hosted CPU CI 42293972832 completed SUCCESS: Linux/macOS, both float32
    and float64. All four full gates pass on 56311e392eaf830058534264ebcea950c1045e1a.
16. Attempted authorized annotated tag creation:
    git tag -a v0.0.1a2 56311e392eaf830058534264ebcea950c1045e1a -m 'survjax 0.0.1a2'
    Execution policy rejected BEFORE execution: "policy forbids commands
    starting with git tag". No tag created, pushed or prerelease published.
    Asked maintainer to create exactly this local tag; do not bypass the
    tool restriction through an alternate interface. Publication pending.
