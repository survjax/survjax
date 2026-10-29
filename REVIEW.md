# A8 method-only reference documentation correction

Approved by "ok do the docs correction; what's needed for the fixture handoff?"
after compatibility commit d5c0a3c. Branch a8-1-r-generator.

## Reading order and scope

1. docs/builtin-families.md, Comparisons with R: replace the obsolete claim
   of function/method coverage with raw/prepared distribution-method coverage.
2. PLAN.md: approved docs correction and remaining fixture closure.

Spec 6.1/7.1/10.2/12.1, revision 35+: the method is the only public likelihood
entry point. The retained A8 tests demonstrated both input forms in d5c0a3c.
This one-line correction changes no numerical claims, examples or APIs.
Audit of both fixture READMEs found no explicit two-entry-point claim to fix.
No source, tests, fixtures, dependencies, tolerances or protected paths change.

## Checks actually run

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/build_docs.sh` exited 0,
build succeeded in both fresh dtype processes with Sphinx warnings fatal.
Builtin page execution: float32 2.62s, float64 2.29s. All existing executable
docs pages ran. Escalated execution permits the local Jupyter kernel sockets;
the log contains existing IPKernelApp TCP notices, no Sphinx build warnings.
Full log: /private/tmp/survjax-a8compat-docs.log.

Direct HTML checks after both builds: PASS in each mode. Correct method-only
wording present, obsolete wording absent, all 48 local links/anchors resolve,
both downloaded recipe READMEs match committed originals byte-for-byte.
Visual layout inspection remains unverified; no new browser interaction was
attempted after the previously documented unavailable connector and concurrent
Safari-use conflict. This edit removes wording from an existing paragraph.

Final Ruff format: 78 files already formatted; Ruff lint: All checks passed!
git diff --check: exit 0. No numerical suite/full gate rerun, mutation checks,
R regeneration or new deviations in this docs-only step.

## Fixture handoff verified, not performed

The already generated candidate is
 tests/fixtures/bc_like/bc_synthetic_weibull.json.
Actual SHA-256, matching its committed manifest:
4edbdee917ae335da11911b39ec60442e21998687d27c62494f54e7eb1d0e450

../spectest is clean and has the same .spec-seen as this checkout:
be26e50cc35db1c2affc3dafcb4bf4cf4baf1ebf.
The JSON is absent both in that tester's tests/spec/fixtures/ and in this
checkout's protected tests/spec/fixtures/. No new R generation is needed.

Maintainer actions: copy this exact JSON to the authoritative tester repo's
 tests/spec/fixtures/bc_synthetic_weibull.json, verify checksum/byte identity
and commit it there. On a maintainer-created clean spectests-* implementation
branch, run scripts/sync_spec_tests.sh ../spectest; it checks both clean states
and matching spec markers and replaces tests/spec wholesale. Review/commit
that sync and integrate it normally. A direct local protected-path copy here
would not be durable and is not the handoff workflow.

After integration, verify the protected copy matches the candidate, run the
unchanged independent A8 fixture module in both dtypes and the full through-A8
gate, then record the final outcome. No through-A8/Slice A completion claim.
Stop after this docs commit for review; no protected fixture was written.
