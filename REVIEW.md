Independent review: round 1/3

# RELEASE.5b — Citation guidance and planned organization identity

Independent reviewer: Astra, low reasoning, authorized substitute for Claude.

## Read first

1. PLAN.md RELEASE.5b: citation topic and maintainer's new organization note.
2. CITATION.cff / CITING.md / docs/citing.md: metadata and single prose source.
3. docs/releasing.md diff: intended organization transfer and future publisher
   fields, distinguished from the currently prepared old-owner bindings.
4. CHANGELOG.md status update; pyproject.toml author/title/license/URLs and
   .github/workflows/release.yml repository guard (both unchanged).
5. Current spec revision 47: 10.1/10.5 and 13 citation/release requirements.

## Invariants / decisions

- Maintainer says GitHub organization survjax exists and repository will move
  there. Intended target survjax/survjax keeps its current repository name.
  No transfer, organization/account/remote/visibility action is performed.
- Existing package URLs, release guard and installed metadata test still use
  vanAmsterdam/survjax. New CFF repository-code matches that actual prepared
  metadata. Do not present an uncompleted transfer as current canonical state.
- Handoff's future publisher owner is survjax, repository survjax, workflow
  release.yml, environment pypi. Text explicitly labels these post-transfer
  targets and blocks launch until coordinated identity update RELEASE.5d.
  The old workflow guard would skip under the new owner until updated.
- Planned 5d covers package repository/issues URLs, README badge links, CFF
  URL, release guard, metadata test expectations and final handoff consistency.
  It does not grant remote transfer/publication authority. Operator must
  recheck org permissions, Actions/environment protection, RTD App/webhook,
  publisher owner and authorized clones' remotes for the transferred repo.
- CFF author comes directly from pyproject.toml: Wouter A.C. van Amsterdam,
  with given/family names split accordingly. Title combines existing project
  name/description. Existing Apache-2.0 expression reused, no ownership claims
  inferred from organization creation. Confirm final agreed credit at release.
- No DOI, ORCID, release date, released version, article or archive invented.
  Cite actual installed version and/or full source SHA, including local edits
  in the reproducibility record. CFF omits optional unassigned fields.
- Guide uses a labeled text template and a read-only installed-version command;
  none of its placeholders is stored as a real identifier in CFF. No new tool
  dependency, citation service/account connection or global installation.
- Orphan include adds citing.html/search without modifying navigation. Root
  guide is the single prose source. No public API, runtime, dependency,
  configuration, lockfile, version, protected path or specification change.

## Official sources checked (2026-11-06)

- https://citation-file-format.github.io/ and
  https://raw.githubusercontent.com/citation-file-format/citation-file-format/1.2.0/schema.json:
  CFF software metadata and official 1.2.0 JSON Schema.
- https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files:
  root CITATION.cff/default branch integration and software citation formats.
- https://docs.github.com/en/repositories/creating-and-managing-repositories/transferring-a-repository:
  transfer permissions, target name, redirects and organization policy.
- https://docs.pypi.org/trusted-publishers/adding-a-publisher/:
  publisher identifies GitHub owner/repo/workflow/environment; a repository URL
  redirect is not a replacement for configuring the actual publisher identity.

## Verification

- Official CFF 1.2.0 schema downloaded read-only into /tmp with curl -L --fail.
  Existing docs environment PyYAML/jsonschema Draft7Validator + FormatChecker:
  schema checked, CITATION.cff validated; exit 0. Author/title/repository/license
  match pyproject metadata; DOI/version/date/identifier/preferred-citation fields
  and author ORCID are absent. No cffconvert/new dependency installed.
- scripts/check.sh --fast --through A10: exit 0, all gates passed. Ruff:
  180 files already formatted / all checks passed; mypy: 22 source files clean.
  Both dtype modes ran. Captured output includes implementer float64
  1066 passed / 8 skipped and spec float32 2156 passed / 20 skipped;
  intermediate count lines were truncated in the capture, so are not inferred.
  Fast gate excludes slow cases; no new tests or full install/numerical rerun
  for metadata/prose. No A11/A12 acceptance claim.
- scripts/build_docs.sh directly with approved escalation: exit 0; float32
  and float64 both report build succeeded, forced notebook execution retained.
  Existing Bayesian and MNIST pages execute; no static-output substitution.
- uv run --locked --group docs python /tmp/survjax-release-audit.py:
  each site has 26 HTML pages, 1,663 local links/fragments/assets, 14 downloads,
  zero failures. Existing downloadable notebooks byte-match source; use-case
  images/navigation present. Shared sequential execution tree: 18 notebooks /
  153 code cells; logs separately show both dtype executions.
- Existing changelog include audit rerun: complete included prose, headings,
  bullets, wrapper source and search entry match root CHANGELOG.md, both modes.
- Citation include audit: complete normalized prose matches root CITING.md;
  wrapper source bytes and search entry pass, both modes. Handoff source copy
  matches; rendered table owner is survjax and prose explicitly includes the
  planned new/current old identities, pending RELEASE.5d and guard-skip effect.
  Citation shell block passes bash -n; no future operator actions executed.
  Programmatic HTML/content inspection only, no browser visual-layout claim.
- git diff --check: passed.

## Limits and remaining steps

No actual repository transfer or canonical URL/organization permission check,
GitHub citation UI rendering, account setup, OIDC, upload, hosted Linux/RTD,
public docs or public clean install verified. No tags, remote edits or pushes.
Current CFF URL and active bindings remain old-owner values intentionally;
this step alone is not ready for publication under the new organization.

Remaining local docs: RELEASE.5c contribution guidance. RELEASE.5d coordinates
identity-dependent code/metadata once the canonical destination is confirmed.
Operator transfer and other owners' completion gates remain prerequisites.
Changelog now records citation guidance prepared and transfer/identity update
pending, without closing simulation/docs/acceptance ownership.

## Commit trace

Previous completed step / clean start:
1494cd18d2d192c7070513c376c08c17aa00262c (RELEASE.5a).
Original worktree base: 3cb4cd7fbba3313f00dfabed0028ee6f404384df.
Commit the bounded step, independent Astra review, then stop.
