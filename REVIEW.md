Independent review: round 1/3

# RELEASE.3 — Prepare PyPI Trusted Publishing

Reviewer substitution: maintainer explicitly requested a separate gpt-6-astra
agent at low reasoning because Claude is unavailable. Keep commit-first,
independent review, findings/fix-up commits, max three rounds and stop/report;
do not invoke Claude or edit the protected reviewer skill. Reviewer reads the
committed step independently; no implementation handoff/analysis is supplied.

## Read first

1. PLAN.md RELEASE.3 and current spec 10.1–10.3, 12.2, 13.
2. .github/workflows/release.yml: trigger, build/test/artifact job, isolated
   dependent publisher, environment and permissions.
3. scripts/validate_release.py: matching version/tag, filenames and archive
   metadata/description; tests/unit/test_validate_release.py: synthetic cases.
4. scripts/check.sh and tests/install/test_wheel.py (unchanged): gates and
   supplied-wheel support used by the new workflow. Do not edit protected paths.

## Invariants

- Local preparation only. No push, tag, release, remote workflow run, package
  upload, account/credential/visibility setup or publication occurs here.
- Only published GitHub Releases in vanAmsterdam/survjax build; no push, PR,
  schedule or dispatch trigger. Checkout the event SHA, without credentials.
- Existing Python 3.12 / uv 0.11.19 CPU baseline; locked dev/test environment;
  existing uv_build constrained to tested 0.11.19. No new dependencies.
- uv build creates a source archive and builds its wheel from that archive.
  One pure Python survjax wheel + matching sdist only. Current development
  artifacts cannot pass: tag is v<version> and version must be dotted
  major.minor.patch optionally followed by a/b/rc + number, without dev/local.
  This deliberately narrow release policy is for this initial release, not
  a general PEP 440 implementation or public library API.
- Validate matching wheel/sdist metadata and exact Markdown README bytes.
  Pass the validated absolute wheel through SURVJAX_WHEEL to the full A10 gate
  (both dtypes, including installed-wheel tests); gate does not rebuild it.
- Artifact upload follows passing tests; publish job only downloads the named
  artifact and invokes the PyPI action, no checkout/build/source execution.
  Only publish receives id-token: write, uses pypi environment; no token secret,
  alternate registry or skip-existing bypass. Action SHAs pinned.
- Existing shared CPU CI is unchanged. Through A10 does not verify A11/A12;
  acceptance integration stays with its owner and is a pre-release obligation.
  No docs/source/spec/protected-path/version/lockfile changes.

## Official interfaces checked (2026-11-06)

- https://docs.pypi.org/trusted-publishers/using-a-publisher/ and
  https://docs.pypi.org/trusted-publishers/adding-a-publisher/:
  OIDC permission, action and GitHub environment publisher identity.
- https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/:
  separate build/publish jobs, artifact transfer, pending publisher and
  protected pypi environment. Account setup remains an operator action.
- https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#release:
  release published trigger; event SHA identifies the tagged release commit.
- https://docs.astral.sh/uv/guides/package/: default wheel-from-sdist build.
- https://raw.githubusercontent.com/pypa/gh-action-pypi-publish/release/v1/action.yml:
  packages-dir and skip-existing inputs, metadata verification and attestations
  default on. Publish is Linux-only; selected Ubuntu 24.04 accordingly.
- git ls-remote verified action commit refs: checkout v5 fbc6f399..., upload
  v4 ea165f8d..., download v5 634f93cb..., publisher v1.14.3 7245906f...
  (annotated tag peeled). setup-uv SHA/baseline reused from existing CI.

## Checks and outcomes

- New tests first failed collection in both modes (scripts is not on pytest's
  import path). Fixed the test loader to runpy.run_path on the script, without
  adding a package, dependency or modifying global test import setup.
- JAX_ENABLE_X64=1 uv run --group test pytest -n 6 -q
  tests/unit/test_validate_release.py: 6 passed in 0.55s; float32: 6 / 0.49s.
  Covers valid alpha, mismatched tag, dev/local rejection, archive metadata
  disagreement and extra-artifact rejection. Tests all below one second.
- YAML parsed with existing environment PyYAML BaseLoader (preserves YAML 1.2
  on key); checked release-only trigger, job-scoped OIDC, environment pypi,
  same artifact name and no checkout/run steps in publisher. Passed.
- bash -n on each embedded run block: passed. No actionlint executable
  available; not installed as a new project dependency.
- uv build --out-dir /tmp/survjax-release3-dev: exit 0, sdist + wheel-from-sdist
  built. Validator rejects these actual 0.0.1.dev0 artifacts as intended.
- Separate temporary source copy with version 0.0.1a1 (only under /tmp):
  uv build /tmp/survjax-release3-candidate --build-constraints
  /tmp/survjax-release3-build-constraints.txt --out-dir /tmp/survjax-release3-alpha:
  exit 0. Constraints contain uv_build==0.11.19 as in the workflow.
  Validator accepts both real alpha archives for v0.0.1a1 and returns the
  exact wheel path. Repository version/lock/assertions remain unchanged.
- Both full tests/install/test_wheel.py runs used SURVJAX_WHEEL pointing to
  /tmp/survjax-release3-dev/survjax-0.0.1.dev0-py3-none-any.whl:
  float64: 4 passed in 40.42s; float32: 4 passed in 39.72s. Includes fresh
  out-of-tree installation, runtime-only imports, derivatives, external typing.
- Same SURVJAX_WHEEL supplied to scripts/check.sh --fast --through A10:
  exit 0, all gates passed. Ruff 175 files formatted / checks passed;
  mypy 22 files clean. Implementer float64: 1066 passed / 8 skipped;
  float32: 1071 / 3. Source doctests: 4 each. Spec float64: 2149 / 27;
  float32: 2156 / 20. Fast gate deselects slow cases; full touched install
  checks above include them. No A11/A12 claim.
- git diff --check: passed. Full numerical tier-2 gate not run locally;
  configured for the future release workflow, existing CI/maintainer gate.

## Limits / remaining work

GitHub-hosted Linux workflow, artifact service, OIDC identity, environment
protection, PyPI metadata rendering/attestations and upload remain untested.
No TestPyPI/PyPI upload or successful-hosted claim. No new Twine dependency;
publishing action performs its own metadata verification when run later.
The local alpha build validates packaging/tag policy, not a released/tested
version: installed-wheel assertions still require the development version.

RELEASE.4 must identify pending publisher: PyPI project survjax; GitHub owner
vanAmsterdam, repository survjax, filename release.yml, environment pypi.
Operator must create/protect that environment with required reviewers and
allowed release tags, configure the pending publisher under their authorized
account, update version/lock/install assertion together, and complete all
release/demos/acceptance/hosting checks before publishing a GitHub Release.
RELEASE.5a–5c (changelog, citation, contribution), license/provenance/version
checks from spec 13 and public clean-install/docs checks still gate launch.
No account/project ownership inferred from URLs; no account work done here.

## Commit trace

Previous step / clean start: 2ae2d37cdc22dbcc8681f64325df9249b73837b5.
Original worktree base: 3cb4cd7fbba3313f00dfabed0028ee6f404384df.
RELEASE.3 commit follows these checks, then independent Astra review and stop.
