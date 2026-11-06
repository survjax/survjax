# RELEASE.5b — Add citation guidance and record the planned organization move

Start / previous step: 1494cd18d2d192c7070513c376c08c17aa00262c.
Original worktree base: 3cb4cd7fbba3313f00dfabed0028ee6f404384df.
Files: CITATION.cff, CITING.md, docs/citing.md (orphan include),
docs/releasing.md (planned identity transfer), CHANGELOG.md (remaining gates),
PLAN.md, REVIEW.md. One citation topic plus the maintainer's explicit transfer
note; below 300 implementation lines. No API, dependency, runtime/configuration,
version, lockfile, protected path or spec edits. Spec revision 47: 10.1/10.5/13.

Reuse the explicit package-author metadata Wouter A.C. van Amsterdam; split
CFF given/family names without inventing authors, ORCID, DOI, publication date
or a released version. Minimal CFF 1.2.0 software citation plus prose explaining
exact installed version/source SHA and dependency/data citations. Keep the
current repository-code URL until the planned repository transfer is completed
and a coordinated identity update is reviewed. Add the guide as a standalone
include/search page without changing another agent's navigation.

Maintainer created the GitHub organization survjax and plans to transfer this
repository there (intended target survjax/survjax, keeping the repository name).
Record this as a launch prerequisite in handoff/changelog, not a completed
transfer. Handoff must distinguish current old-owner bindings from future
publisher fields (owner survjax; repo survjax; release.yml; pypi), and require
GitHub App/webhook/environment/permission checks after transfer. No remote,
account, organization, credentials, visibility, git remote or transfer action.

Add RELEASE.5d as the coordinated local identity-update step: pyproject URLs,
README badge links, citation repository URL, release workflow repository guard,
metadata test expectations and final handoff synchronization after confirming
the canonical identity. No active binding changes in this docs/citation step;
the old guard would skip publishing after transfer until 5d is complete.

Verify official CFF/GitHub/PyPI interfaces; validate CFF against the official
1.2.0 schema using existing docs-environment YAML/jsonschema (no dependency
added), and compare author/title/repository/license with pyproject metadata.
Run scripts/build_docs.sh directly in both modes, inspect rendered citation /
changelog/handoff text and local links/assets/downloads; required fast A10 gate;
git diff --check. No new unit tests for metadata/prose. Independent Astra/low
review after commit, max three rounds with fix-ups; stop/report.

Remaining: RELEASE.5c contribution guidance; RELEASE.5d identity update; actual
operator transfer/account setup and all separately owned launch gates.

---

# RELEASE.5a — Draft the initial changelog

Start / previous step: b0090b6b49c3d59dca6636a4f8d3893699352f96.
Original worktree base: 3cb4cd7fbba3313f00dfabed0028ee6f404384df.
Files: CHANGELOG.md, docs/changelog.md (standalone orphan include), PLAN.md,
REVIEW.md. One docs topic under 300 implementation lines, no navigation or
other-agent edits. No API, dependency, runtime, configuration, version,
lockfile, protected path or spec changes. Spec revision 46: 10.5, 12.2, 13.

Write an Unreleased initial-alpha entry, with no release date or published tag.
Candidate remains 0.0.1a1; preparation baseline remains 0.0.1.dev0. Describe
only merged/current capabilities: SurvivalData and diagnostics, exact/right
likelihood subset, direct/grid methods, Exponential/Weibull/PH, concordance,
staged RMST, extension support, Bayesian/MNIST examples, executable docs and
local release infrastructure. State important likelihood and derivative limits,
unavailable later-slice APIs and minimal runtime/tested baseline. Clearly mark
publication, hosting, simulation/docs/acceptance completion and citation /
contribution guidance as remaining gates. Do not import unfinished work or
turn local/A10 checks into a broader release claim. Link to existing guides
and operator handoff without touching their pages.

Keep CHANGELOG.md as the single prose source; docs/changelog.md includes it
for rendering/search at changelog.html, with no toctree change. Run
scripts/build_docs.sh directly in both dtypes, inspect included text/source,
links and rendered headings, existing-site asset/download audit; required fast
A10 gate; git diff --check. No new unit tests for prose. Independent Astra/low
review after commit, max three rounds with fix-ups as needed; stop/report.

Remaining docs: RELEASE.5b citation and RELEASE.5c contribution guidance.
External release and other agents' unfinished completion gates remain open.

---

# RELEASE.4 — Write the release operator handoff

Start / previous step: 035f4e001f46d999a36e035b9db91f0bd6f2bc95.
Original worktree base: 3cb4cd7fbba3313f00dfabed0028ee6f404384df.
Files: docs/releasing.md, PLAN.md, REVIEW.md. One docs topic under 300 lines;
no API, dependency, configuration, protected path, spec or version changes.
Current spec revision 46: 10.1/10.3/10.5, 12.2, 13 release requirements.
Historical revision numbers in previous review records remain historical.

Write a standalone Sphinx/MyST orphan page (no notebook cells), accessible as
releasing.html and through search, leaving the docs agent's pages/navigation
untouched. Cover: current unverified hosting/publication status; outstanding
release gates/owners; RTD slug/account confirmation and GitHub connection;
private-repository hosting constraints, active/default/public version settings;
locked executable builds, cold MNIST network and budget/download checks;
PyPI pending publisher exact fields and pypi environment protections; candidate
version/tag/source/docs sequence; build/test the same distributions; explicit
GitHub Release trigger and final environment approval; fresh public-install and
unauthenticated docs checks. Do not perform the operator's remote/account work.

Use candidate 0.0.1a1 / v0.0.1a1 consistently, keeping the repository version at
0.0.1.dev0. Initial-alpha README guide links deliberately retain /en/latest/;
state they move and give the release version URL separately. Verify actual RTD
version slug/default and public access; do not rely on stable for an alpha.
List RELEASE.5a–5c and other agents' unfinished demo/acceptance work as blockers,
not new implementation scope. Never claim A10 proves A11/A12 or broader release.

Verify current official RTD/PyPI/GitHub docs. Run scripts/build_docs.sh directly
(both dtype processes), inspect rendered handoff/links/code blocks and audit
local site links/assets/downloads. Run the required fast gate through A10 and
git diff --check. No new unit tests for prose; no remote account actions,
publication or public-install command execution. Independent Astra/low reviewer
as authorized, after commit, at most three rounds; fix if needed, then stop.

Remaining: RELEASE.5a initial changelog, 5b citation, 5c contribution guidance;
external hosting evidence, release version commit, remote account setup and
other agents' completion gates are required before any public launch.

---

# RELEASE.3 — Prepare Trusted Publishing workflow

Start / previous step: 2ae2d37cdc22dbcc8681f64325df9249b73837b5.
Original worktree base: 3cb4cd7fbba3313f00dfabed0028ee6f404384df.
Files: .github/workflows/release.yml, scripts/validate_release.py,
tests/unit/test_validate_release.py, PLAN.md, REVIEW.md. No public API,
dependency, protected path, shared CI, docs source or spec changes.
Spec: 10.1–10.3 packaging/runtime isolation, 12.2 publication separate,
13 wheel/sdist pre-release checks. Aim below 300 implementation/test lines.

Use only release: published (includes prereleases), intended repository guard,
checkout of event SHA, pinned existing Python/uv baseline and action SHAs.
Build once with pinned existing uv_build baseline: uv build produces sdist
and a wheel built from it. Validate exactly one of each, matching metadata
and v<version> tag; reject development/local versions. Run the unchanged full
through-A10 gate with SURVJAX_WHEEL pointing to that exact wheel. This does not
claim A11/A12 acceptance; their integration retains independent ownership.
Upload only the two tested distributions to a named workflow artifact. A
separate dependent Ubuntu job downloads that artifact without checkout or
build, uses environment pypi and job-only id-token: write, then invokes pinned
pypa publishing action (no credentials or skip-existing). No ordinary CI event
publishes. No TestPyPI, remote workflow dispatch, upload, tag or account action.

The small stdlib validator checks the tag/version/name, filenames and wheel /
sdist metadata + README agreement, returning the tested wheel path. Unit tests
exercise positive alpha archives and tag/dev/archive-mismatch failures. Run
touched unit tests fully in both dtypes, local uv build + real artifact audit,
prove current dev artifacts are rejected; validate a temporary alpha build
without changing this checkout's version. Parse YAML and inspect trigger,
permissions and artifact flow using existing tools; run required fast A10 gate.
No new actionlint/Twine dependency. Do not claim remote execution/upload.

Independent review: user explicitly replaces unavailable Claude with a separate
gpt-6-astra reviewer at low reasoning. Keep commit-first, independent review,
maximum three rounds, fix-up commits, and stop/report. No Claude invocation.

Docs to-do for RELEASE.4: workflow release.yml, environment pypi; pending PyPI
publisher fields (project survjax, owner vanAmsterdam, repository survjax,
workflow release.yml, environment pypi); required environment reviewers/tag
restrictions and remaining release checks. RELEASE.5a–5c, public hosting,
version bump and outstanding demo/acceptance gates remain before publication.

---

# RELEASE.2 — Prepare package release metadata

Starting SHA / previous completed step: ac50796fc82ff087df3c8973724992a4ec2e2eb5.
Original worktree base: 3cb4cd7fbba3313f00dfabed0028ee6f404384df.
RELEASE.1 independent review: round 2, merge; hosted/Linux checks remain open.

Files: pyproject.toml, README.md (two link targets only),
tests/install/test_wheel.py, PLAN.md, REVIEW.md. Public API: none.
Spec: 10.1–10.3 packaging/runtime isolation, 12.2 separate publication,
13 release metadata. No dependency, protected path, docs page/navigation,
license, numerical or spec edits. Remain within 300 changed implementation lines.

Add README.md as Markdown package description, Repository/Documentation/Issues
URLs and the Development Status :: 3 - Alpha classifier. Add only the tested
Python 3.12 classifier, not an untested version/platform/GPU matrix. Preserve
Apache-2.0 expression and JAX/NumPy-only runtime requirements. Replace the two
README guide links with absolute intended hosted URLs so the PyPI description
has no checkout-relative links; this is a link correction, not a prose rewrite.

Proposed first release: 0.0.1a1, tag v0.0.1a1, RTD tag version v0.0.1a1.
Keep the development version 0.0.1.dev0 in this preparation step. The release
operator must change pyproject.toml, refresh uv.lock and update the installed
version assertion together before building/tagging that release; no tag or
version-specific docs URL points to a nonexistent release now. Final version
choice and the exact operator sequence belong to RELEASE.4.

Add a wheel-artifact test for Markdown description equal to README, public URLs,
alpha/tested-Python classifiers and absence of relative Markdown links. Existing
installed-wheel test continues verifying version/license and runtime deps.
Validate touched install tests in both dtype processes, build wheel + sdist,
inspect their metadata and README inclusion locally, uv lock --check,
scripts/check.sh --fast --through A10 and git diff --check. No docs build:
Sphinx sources/config/executable cells unchanged. No new tools/dependencies.
Use independent-review after commit (RELEASE.2), then stop.

Docs to-do: RELEASE.4 operator topic must carry the version/tag/docs-version
sequence and pending RTD/PyPI account actions. RELEASE.3 workflow/artifact
validation and RELEASE.4 remain; historical RELEASE.1 plan follows unchanged.

## Required pre-release documentation follow-ups (round 1 F-1)

Spec 10.1/13 applies before any public alpha upload, including proposed
0.0.1a1. The infrastructure sequence alone is not a complete release gate.
Assign these separate bounded, reviewed docs steps before publication:

- RELEASE.5a: initial changelog / release notes, describing the implemented
  subset and limitations for the chosen version (CHANGELOG.md).
- RELEASE.5b: citation guidance with maintainer-confirmed attribution and
  bibliographic details; do not invent a DOI or publication.
- RELEASE.5c: contribution guidance covering the actual development/check
  workflow and issue/PR entry points without changing protected agreements.

These are planned follow-ups, not implemented in RELEASE.2. Each requires
its own PLAN/REVIEW, appropriate checks, commit and stop. RELEASE.3 may prepare
an inert publishing workflow, but the RELEASE.4 operator handoff must list
5a–5c as blocking public release until complete, alongside the other spec 13
pre-release checks and remaining demo/acceptance work owned by others.

RELEASE.4 must also state that README links currently target moving /en/latest/
docs, not frozen tag docs. Before building release artifacts, choose whether
to retain that alpha policy or change both guide links to the activated release
version, consistently with project version, tag and RTD settings.

---

# RELEASE.1 — Prepare Read the Docs builds

Clean starting SHA / original worktree base / previous completed step:
3cb4cd7fbba3313f00dfabed0028ee6f404384df (DOCS.6).
Plan-only archive: 2c991d0; integrated docs records moved verbatim to
plans/pre-release-prepare.md without closing other workstreams.

## Workstream sequence (one reviewed step at a time)

1. RELEASE.1: RTD configuration; execute and audit existing docs pipeline.
2. RELEASE.2: package description, URLs and alpha metadata; propose 0.0.1a1,
   subject to maintainer choice, with consistent version/tag instructions.
3. RELEASE.3: explicit-release-only Trusted Publishing workflow; validate
   wheel and sdist locally and publish the same tested artifacts later.
4. RELEASE.4: one documentation topic for the operator handoff, including RTD
   connection/public-version settings, PyPI pending publisher and release checks.

## This step

Files: .readthedocs.yaml, PLAN.md, REVIEW.md. No public API, dependency,
lockfile, docs page/navigation, protected path or specification changes.
Spec coverage: 10.3 (locked Python 3.12 CPU environment), 10.4 (MNIST hash
cache), 10.5 (executable documentation), 12.2 (publication separate).

Use supported Ubuntu 24.04/Python 3.12, pin uv to the local 0.11.19 baseline,
sync docs/examples with --locked in this checkout's own .venv, then invoke
scripts/build_docs.sh directly. Both dtypes execute; stage the contents of
html-x64 into READTHEDOCS_OUTPUT/html only after both succeed. Keep downloads,
figures and relative navigation intact. Do not require the spec symlink for
hosting. Use current official RTD custom-job/output documentation.

Checks: YAML structure and command execution/staging with existing tools;
existing build-script unit tests; direct both-dtype executable docs build with
a cold MNIST cache; rendered local link/fragment/download/image audit;
scripts/check.sh --fast --through A10; git diff --check. No new implementation
or tests needed for this reversible configuration-only step. Record timings
and compare with RTD's documented budget; do not equate macOS execution with
Linux/hosted verification. Check available Linux tooling and record any limit.
Independent review required (RELEASE.1), then stop. No publication/account work.

## Docs to-do

RELEASE.4 owns user-facing instructions; leave existing docs-agent pages alone.
Before public launch, run a fresh Linux/RTD build (including cold MNIST fetch),
verify project-specific time/memory limits and public access/downloads, and
record evidence. RTD project/account ownership is unverified.

Round 1 follow-up: make install explicit and reshim the pip-installed uv;
validate command lifecycle and build an archived checkout with dangling spec.
No new scope, dependencies or protected paths.
