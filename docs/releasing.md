---
orphan: true
---

# First release operator handoff

Version **`0.0.1a2`**, Git tag **`v0.0.1a2`**, was published on GitHub and
PyPI on 6 November 2026. The earlier `v0.0.1a1` GitHub prerelease remains;
its PyPI build failed before upload. The alpha-2 artifact checks passed and
its upload succeeded after allowing release tags in the `pypi` environment.
The public-index install smoke check passed in both dtype modes.

This page retains the reusable release procedure. Examples use `0.0.1a2`;
select a new version for the next release and never move an existing tag.
See `RELEASE-A2-LOG.md` in the repository for the execution record. Read the
Docs hosting and broader acceptance gates are not certified by this release.

## Complete the release prerequisites

The draft changelog, citation metadata and contribution guidance are prepared.
Before publishing, reconcile them with the final release commit and agreed
credits. Confirm license/notices, fixture provenance, package-name availability and the
actually tested Python/JAX configurations. The current CPU CI and publishing
workflow run acceptance through A12 plus the completed features listed in
`scripts/spec-features.txt`. Consult the merged step evidence, including
`work/B7.AUDIT/REVIEW.md` for Slice B, and obtain fresh checks on the release
commit. These gates do not certify all planned slices or the whole release.

Run full CPU CI on the final release commit. If advertising the recorded
NVIDIA T4/CUDA 12 configuration, also obtain the manual GPU check for that
commit; do not infer accelerator support from CPU checks. Record outstanding
limitations in the release notes. Proceed only after these prerequisites and
the hosting checks below have evidence.

## Check repository and publishing access

The canonical repository is **`https://github.com/survjax/survjax`**.
Package repository/issues URLs, README badge/links, `CITATION.cff`, release
guard and artifact metadata tests use that identity. The guard permits only
that namespace. Confirm organization access, `pypi` environment protections,
Actions permissions and GitHub App access. Configure the PyPI publisher with
that owner; URL redirects do not replace publisher identity. Check other
clones' remotes separately. Verify repository pages, issues and badges while
signed out before launch.

## Connect Read the Docs

The intended Community address is `https://survjax.readthedocs.io`. Confirm
the project slug in the dashboard and complete a hosted build before
advertising that address as live documentation.

1. Ensure the reviewed `.readthedocs.yaml`, lockfile and docs/build changes are
   pushed to the GitHub branch RTD will build, normally `main`. Local commits
   in a worktree are not automatically available to RTD. Confirm the root
   config exists on the remote branch before importing the project.
2. Use Community hosting for the public repository. Confirm signed-out
   repository access before connecting the service.
3. Sign in to the chosen RTD service with GitHub, authorize/install its GitHub
   App for the `survjax` organization and this repository, then select
   **Add project**, choose **survjax/survjax**, and **Continue**. Confirm the
   repository/default branch and project slug, then **Next**. When asked about
   configuration, select **This file exists** for root `.readthedocs.yaml`.
4. Creation triggers a build. Inspect the build log for the correct source
   commit, pinned setup and both dtype executions; keep the existing pipeline.
   Verify `latest` is active and points to the intended default branch.
5. Use the project's actual dashboard URL/domain. Community defaults to
   `<project>.readthedocs.io`; Business uses
   `<organization>-<project>.readthedocs-hosted.com`. A different slug/domain
   requires updating package Documentation metadata and advertised URLs before
   release. Do not infer the RTD account/organization name from the GitHub name.

[Adding a project](https://docs.readthedocs.com/platform/stable/intro/add-project.html),
[Git integration](https://docs.readthedocs.com/platform/stable/reference/git-integration.html)
and [default domains](https://docs.readthedocs.com/platform/stable/custom-domains.html)
describe this flow. GitHub App integration subscribes to the required events;
manual imports need their own webhook setup. Project creation/builds can publish
documentation, so these are operator actions, not work performed by this guide.

The config selects Ubuntu 24.04, Python 3.12 and uv 0.11.19, synchronizes the
locked docs/examples groups, and runs `scripts/build_docs.sh`. Both dtype
processes execute notebooks; only the complete `html-x64` site is staged for
hosting after both succeed. First confirm this path on a fresh Linux worker,
including the `asdf reshim python` bootstrap and `uv` command lookup.

Allow initial package downloads and HTTPS access to
`https://ossci-datasets.s3.amazonaws.com/mnist/`. MNIST fetches four compressed
files, verifies SHA256 hashes and caches them at
`~/.cache/survjax-examples/mnist`; `SURVJAX_EXAMPLES_DATA` can select a writable
cache directory. A persistent cache is optional: prove a cold build succeeds.
Keep live execution enabled, including the MNIST smoke run. Cells retain the
120-second timeout. Check actual project time/memory limits; the documented
Community [build budget](https://docs.readthedocs.com/platform/stable/builds.html#build-resources)
is 15 minutes/7 GB. Local macOS timings do not prove Linux
hosting fits that budget. MathJax rendering also uses the existing external CDN.

Activate and build `latest` from `main`. After the release tag exists, sync RTD
versions, activate its tag version, verify the actual URL slug, and build that
exact commit. For the intended Community slug, the release URL would be
`https://survjax.readthedocs.io/en/v0.0.1a2/`; use the actual dashboard domain. Choose the checked tag version as
the site's default for the launch, keeping `latest` available. For Business
hosting, set the version privacy to **Public**; project privacy separately
controls public dashboard/build visibility. An active or hidden version is not
a substitute for the appropriate privacy setting. Do not rely on the `stable`
alias choosing an alpha.

Until hosting is live, README guide links point to the repository sources.
Root citation/contribution/changelog links also work in the source checkout
and are resolved within the built site. Once hosting is verified, choose
versioned hosted URLs consistently; use the verified tag URL when citing the
release's documentation. Check the root redirect/default, version
selector if present, navigation, search, rendered formulas/figures, and both
use-case notebook/model/metadata downloads while signed out. Verify that
notebook downloads match their tagged source files and the documentation logs
show fresh execution. A successful local build is not a public-access check.

## Configure PyPI Trusted Publishing

Use the designated maintainer's authorized PyPI account; the GitHub repository
owner is not a prescribed PyPI username. Confirm that `survjax` is available
before using the account's
[Publishing page](https://pypi.org/manage/account/publishing/).
Register a pending GitHub publisher with these exact fields:

| Field | Value |
| --- | --- |
| PyPI project name | `survjax` |
| GitHub owner | `survjax` |
| GitHub repository | `survjax` |
| Workflow filename | `release.yml` |
| GitHub environment | `pypi` |

The workflow is `.github/workflows/release.yml`; enter only its filename in
PyPI's workflow field. A pending publisher creates the project on first use;
it does not reserve the name. If the project already exists, its authorized
owner must add the publisher to that project instead.

Before releasing, create the GitHub repository environment `pypi`, require a
trusted reviewer to approve every publication, and restrict deployments to the
intended release tags. Confirm these protection features are available for
the actual account plan/repository visibility. Do not rely on automatic
creation of an unprotected environment. The workflow's publisher job alone has
`id-token: write`; no long-lived PyPI token or TestPyPI publisher is needed.
Account setup and environment protection remain manual operator actions.

## Prepare, tag and publish the chosen version

1. Complete the prerequisites above. In a reviewed release commit, change
   `pyproject.toml` to the selected version, run `uv lock`, and verify that both installed-version checks compare
   against project metadata. Do not add release-specific literals to tests.
   Update `CITATION.cff` with the actual released version and release date
   when assigned, and reconcile release notes/version-specific instructions.
   The RTD tag selects the docs source; there is no separate Sphinx release
   version currently configured. Do not change an already published version.
2. From that clean commit, run local packaging and full checks on the exact
   built wheel (commands below), and run `scripts/build_docs.sh` directly.
   Inspect wheel/sdist contents, Markdown description, metadata and runtime
   isolation. Confirm full CPU CI on that same commit and the remaining
   acceptance/demo evidence. Do not substitute a fast gate for release checks.
3. Through the maintainer's separately authorized release procedure, tag that
   reviewed commit `v0.0.1a2` and make the tag available to RTD. Complete the
   tagged hosted/public docs checks before publishing the GitHub Release.
   This page creates no tag, pushes no commit and changes no visibility.
4. Publish a GitHub Release for that exact tag (mark it as a prerelease for
   `0.0.1a2`). This is the publishing workflow's trigger; ordinary CI, a tag
   push alone or saving a draft release does not publish the package.
5. The workflow rebuilds a wheel from its sdist, validates tag/version and
   metadata, and runs its configured full checks on that wheel. It hands the
   tested pair to `python-distributions` (retained for 14 days). Before
   approving the waiting `pypi` environment job, inspect the run/commit and
   downloaded artifacts and record both SHA256 hashes. That job publishes
   this run's artifacts through OIDC. The local rehearsal artifacts are not
   the workflow upload; do not manually replace or rebuild the tested pair.
6. Inspect the workflow's PyPI response and installed public package below.
   Record package URLs, hashes and docs build/commit evidence. If publishing
   fails, diagnose before retrying; `skip-existing` is false, and a successful
   upload cannot be overwritten. Do not move the tag or silently reuse a
   version for different files.

Local packaging rehearsal after the coordinated version change, from the
repository root (replace the candidate version consistently if changed):

```sh
uv sync --locked --group dev
release_check_dir="$(mktemp -d)"
printf 'uv_build==0.11.19\n' > "$release_check_dir/build-constraints.txt"
uv build --build-constraints "$release_check_dir/build-constraints.txt" --out-dir "$release_check_dir/dist"
release_wheel="$(uv run python scripts/validate_release.py "$release_check_dir/dist" v0.0.1a2)"
SURVJAX_WHEEL="$release_wheel" scripts/check.sh
scripts/build_docs.sh
```

The A12 gate matches the current CI and publishing workflows and also includes
the completed feature list. Keep this rehearsal aligned with those workflows
when preparing a later release.

## Check the public package from outside a checkout

After a separately authorized upload, create a fresh Python 3.12 environment
outside the repository and install from the public index, with no editable
checkout or demo groups:

```sh
public_check_dir="$(mktemp -d)"
uv venv --python 3.12 "$public_check_dir/venv"
uv pip install --python "$public_check_dir/venv/bin/python" --index-url https://pypi.org/simple --only-binary :all: 'survjax==0.0.1a2'
"$public_check_dir/venv/bin/python" -I - <<'PY'
import importlib.metadata as metadata
import importlib.util
from pathlib import Path
import sys
import survjax

assert metadata.version("survjax") == "0.0.1a2"
assert Path(sys.prefix) in Path(survjax.__file__).resolve().parents
assert importlib.util.find_spec("numpyro") is None
assert importlib.util.find_spec("matplotlib") is None
print(survjax.Exponential(rate=0.5).survival(1.0))
PY
```

Record the actual Python/JAX/NumPy versions and package metadata. Check the
PyPI-rendered description and links, source archive download, both distribution
hashes/attestations, and signed-out RTD pages/downloads. These public checks have
not been performed by local preparation.

Service interfaces checked against the official
[RTD version guide](https://docs.readthedocs.com/platform/stable/versions.html),
[RTD privacy guide](https://docs.readthedocs.com/platform/stable/commercial/privacy-level.html),
[PyPI pending publisher guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [GitHub environment guide](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments)
on 6 November 2026. Recheck account settings at release time.
