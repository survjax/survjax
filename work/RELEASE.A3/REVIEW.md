# RELEASE.A3 — Alpha 3 preparation handoff

Independent review: round 3/3

## Reading order and invariants

Read PLAN.md, the alpha-3 CHANGELOG.md entry, pyproject.toml/uv.lock and
CITATION.cff/CITING.md, then docs/releasing.md. Compare public feature claims
with src/survjax/__init__.py and the linked B7/B8/AFT evidence.

The package and lockfile agree on 0.0.1a3. This prepares an incremental alpha,
not completed v0. Citation metadata has no invented publication date; the
operator must assign it and date the changelog before final-commit checks.
Published alpha-2 history is preserved. No API, runtime code, dependencies,
protected paths, acceptance assertions or completed-feature selection changed.
The publisher owner matches the release workflow's vanamsterdam/survjax guard.
No tags, remote writes, merges, hosting configuration or publication performed.

## Validation

All checks below ran in /Users/wamster3/dev/survjax/dev-release-a3 on the
candidate changes based on cd383be; records are added after those checks.

- uv lock: resolved 166 packages; only survjax changed from 0.0.1a2 to 0.0.1a3.
  uv sync --locked --group dev --group docs --group examples: exit 0.
  Python 3.12.9, JAX/jaxlib 0.11.2, NumPy 2.5.3, CPU macOS arm64.
- uv build --build-constraints /private/tmp/survjax-release-a3/build-constraints.txt
  --out-dir /private/tmp/survjax-release-a3/dist: exit 0; pinned uv_build==0.11.19;
  built sdist then wheel from sdist.
- uv run python scripts/validate_release.py /private/tmp/survjax-release-a3/dist
  v0.0.1a3: exit 0, selected the sole alpha-3 wheel. This compares tag/version,
  wheel/sdist metadata and README descriptions (spec 10.3).
- Manual archive inspection: 34 entries in each distribution; wheel includes
  new-family/wrapper modules and py.typed; exact runtime requirements remain
  jax>=0.11.2 and numpy>=2.5.3; source pyproject version is 0.0.1a3.
- SURVJAX_WHEEL=/private/tmp/survjax-release-a3/dist/survjax-0.0.1a3-py3-none-any.whl
  scripts/check.sh: exit 0, all gates passed; no --fast or historical limit.
  Guard passed; Ruff format 292 files; lint clean; mypy clean on 25 files.
  Implementation float64: 1527 passed/8 skipped (111.04s);
  float32: 1526 passed/9 skipped (90.97s). Source doctests: 4 passed each mode.
  Acceptance float64: 4864 passed/46 skipped (54.20s);
  float32: 4776 passed/124 skipped (43.04s). Two existing NumPy warnings per
  mode at tests/spec/test_a11_ph_direct_methods.py:78. All baseline tests and
  all five committed features selected. Existing install tests verify the
  exact artifact, isolated imports/exports, metadata and optional-dependency
  absence; existing validator tests cover its rejection behavior.
- scripts/build_docs.sh: exit 0, strict executable builds in both dtypes,
  including all three use-case notebooks. Build identity completed
  2026-11-13T12:11:20Z, checkout dev-release-a3, dirty candidate at cd383be.
  HTMLParser inspection of releasing.html, changelog.html and citing.html in
  both sites verified candidate version/commands, canonical owner, baseline
  gate wording and omitted-date explanation. No browser layout inspection.
- git diff --check: exit 0. No numerical tests or tolerances changed; no new
  tests needed for metadata/prose. No R regeneration or performance rerun.

## Local rehearsal artifacts

Directory: /private/tmp/survjax-release-a3/dist (temporary, not committed).
SHA256:

- survjax-0.0.1a3-py3-none-any.whl:
  48d32c90a22c69e99d8d7672732c5ea2c9f7f9c89305a88928982df636fadc7b
- survjax-0.0.1a3.tar.gz:
  8f3dacc073bee0112464973b068dc31eb556b7d9d6be2d04760af09bd8f73ca0

These are local rehearsal files; publishing must use its own workflow-tested
artifact pair and record those hashes. No public-index installation is claimed.

## Remaining operator work

Independent review pending at commit time. No further implementation substep
planned if review clears. Maintainer integration, actual release-date assignment,
fresh full CPU CI/build checks on the final release commit, tagged hosted docs
and signed-out access checks, tags/publication and post-upload public install
verification remain. No GPU run or broader support claim is made here.
The user authorized preparation only. Follow docs/releasing.md for the separate
operator actions; do not treat this handoff as evidence that they happened.

Starting / previous-step final / original worktree base:
cd383be231739ee8851d2dac540b00289128f622.
Roadmap handoff: alpha-3 preparation ready for independent review; publication
is a separate maintainer action. No shared roadmap was edited.

## Round 1 corrections

Reviewer file: ../../../review/slices/RELEASE.A3.md; reviewed 66c3781;
verdict fix-first. Independent scripts/check.sh --fast passed. F-1 corrected:
the prerequisite paragraph now says all baseline acceptance tests plus the
completed features, matching the commands and rehearsal paragraph. N-1 fixed
by making the alpha-2 hosting caveat a separate sentence. N-2 needs no change;
published history remains above the candidate instructions as context.
The stale through-A12 CI step label is outside this preparation's files;
its command already selects the correct gate. Flag it for maintainer cleanup.

After these prose-only corrections, scripts/build_docs.sh again exited 0 for
both strict executable sites. HTMLParser checks in both rendered release pages
confirmed the corrected prerequisite/caveat and absence of through-A12 prose.
No package/runtime changes; no repeat full numerical gate was needed.
git diff --check passed. Round 2 requested after this fix-up commit.

## Authorized final metadata

Round 2 at f40fb07591ef2d8142fd428e7cac9b87a611cfef ended VERDICT: merge.
F-1/N-1 fixed, N-2 withdrawn; the optional N-3 line wrapping is fixed here.
The maintainer then approved proceeding as far as access allows, including
operator actions; see the scope extension in PLAN.md. Assign 2026-11-13 as the
scheduled release date in citation/changelog, updating adjacent prose while
keeping publication explicitly pending. These date statements supersede the
initial intentionally undated candidate invariant above. No API, code, tests,
dependencies or completed-feature list changes. Round 3 reviews this final
metadata before integration. Fresh final-snapshot checks are recorded below.

Final dated-snapshot validation: pinned sdist→wheel build and validator passed;
both SHA256 hashes exactly match the earlier rehearsal pair. Full exact-wheel
scripts/check.sh exited 0: Ruff format 293 files/lint/mypy clean;
implementation float64 1527 passed/8 skipped (196.87s), float32 1526 passed/9
skipped (167.84s); doctests 4 per mode. Both full acceptance runs passed;
float32 4776 passed/124 skipped (62.08s); the float64 summary was truncated
in captured output, so no separate count/timing is claimed for this rerun.
The same two existing PH reference warnings remain per mode. Both strict docs
builds passed, completed 2026-11-13T13:05:18Z. Parsed dated release/changelog/
citation pages passed in both outputs. git diff --check passed. Local main and
remote main still equal cd383be before requesting the final review round.
