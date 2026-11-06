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

17. Maintainer created local annotated v0.0.1a2. Verified tag object
    75fe2b87cfeb299fb72135973e100eeb2d9e39b8 peels to the tested commit
    56311e392eaf830058534264ebcea950c1045e1a; remote main matched.
18. git push origin refs/tags/v0.0.1a2 succeeded. Published with:
    gh release create v0.0.1a2 --repo survjax/survjax --verify-tag --prerelease
    --title 'survjax 0.0.1a2' --notes-file /tmp/survjax-a2-release-notes.md
    GitHub prerelease published at 2026-11-06T21:27:21Z:
    https://github.com/survjax/survjax/releases/tag/v0.0.1a2
19. PyPI workflow 42296861084 started on the exact tested commit and reached
    full checks on its rebuilt artifacts. Await upload result before claiming
    PyPI publication. Existing v0.0.1a1 remains unchanged.

20. Workflow build job succeeded; full artifact checks completed at
    2026-11-06T21:58:24Z. Uploaded python-distributions artifact downloaded
    with gh run download into /tmp/survjax-a2-workflow-dist. Validator accepts
    v0.0.1a2. Both SHA256 values match the local tested artifacts exactly
    (see item 13). Thus no rebuild or replacement is needed.
21. Publish job failed before any steps started. GitHub check annotations:
    "Tag v0.0.1a2 is not allowed to deploy to pypi due to environment protection
    rules." Environment has custom branch policies with only main (branch).
    No upload attempted. gh --log-failed has no job log because no steps ran;
    read check-run 128178552520 annotations to diagnose instead.
22. Requested explicit authorization to add a tag deployment allow-rule and
    retry only the failed publish job. Keep other environment rules intact.
    GitHub prerelease remains live; PyPI publication is not yet complete.

23. Maintainer explicitly selected "Allow release tags v* for future releases
    and retry". POST environment deployment-branch-policies name=v* type=tag
    created rule 65796432. Readback confirms main branch and v* tag allowed;
    no other protections changed. gh run rerun 42296861084 --failed succeeded.
24. Retry succeeded: both build and publish jobs green. PyPI upload times:
    wheel 2026-11-06T22:02:49Z, sdist 22:02:51Z. Public PyPI JSON SHA256 values
    match both workflow and local tested artifacts exactly (item 13).
25. Fresh public-index install outside checkout:
    uv venv --python 3.12 /tmp/survjax-a2-public-verify
    uv pip install --python /tmp/survjax-a2-public-verify/bin/python
      --index-url https://pypi.org/simple --only-binary :all: survjax==0.0.1a2
    Resolved Python 3.12.9, JAX 0.12.0.1, NumPy 2.5.4. Temporary smoke script
    first rejected /tmp versus /private/tmp prefix alias; resolved both paths
    and reran. Both float64/float32 pass version, installed path, type marker,
    absence of demo/test deps, eager/JIT exponential survival (exp(-0.5)).
    This smoke is not a full compatibility-matrix test on these newer deps.
26. Follow-up source docs record actual publication and date. Published tag
    remains fixed at 56311e3; execution-record follow-up does not alter it.

27. Final publication-status docs build passed both strict dtype sites with
    success identity; rendered release status checked. Whitespace clean.

28. Follow-up review round 2: maintainer verdict, no new findings. N-1/N-2
    fixed. Prior protected-policy B-1/S-1 remain formally recorded; maintainer
    explicitly authorized those release actions. N-3 root log retained per
    maintainer request. Reviewer did not reproduce runtime/network checks.
29. Final record integration is docs/log/citation-date only. Release tag stays
    at 56311e3; package is 0.0.1a2. No further release substeps remain for this
    upload. RTD hosting and unfinished acceptance slices were not certified.

## Next release checklist

1. Change pyproject.toml version and run uv lock; version checks read metadata.
2. Update candidate release notes/citation and validate wheel rebuilt from sdist.
3. Run full artifact gate and docs build; review, integrate, confirm hosted CI.
4. Tag the tested commit with the matching v-prefixed version, push tag, publish
   a GitHub prerelease for an alpha. pypi environment now permits v* tags.
5. Inspect build/upload; diagnose failures before retry. Never move/reuse a
   published version. Verify public-index install and both artifact hashes.
6. Record actual publication/date/results. A failed-before-upload environment
   job can retry its publish job with the tested artifacts after rule repair.
