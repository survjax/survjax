# RELEASE.A3 — Alpha 3 preparation handoff

Independent review: round 1/3

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
