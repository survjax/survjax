---
orphan: true
---

# First release operator handoff

This is the maintainer's procedure for a future public alpha release.
Local preparation has added hosting configuration, package metadata and a
publishing workflow. No account connection, hosted build or upload has been
verified. The preparation baseline is `0.0.1.dev0`; examples below use the
proposed release **`0.0.1a1`**, Git tag **`v0.0.1a1`**. Confirm the version
before following the release sequence. Account ownership is not established by
the repository or documentation URLs.

## Complete the release prerequisites

Before publishing, finish and review the initial changelog/release notes,
citation metadata and contribution guidance (planned as RELEASE.5a–5c).
Confirm license/notices, fixture provenance, package-name availability and the
actually tested Python/JAX configurations. The simulation demonstration,
documentation cleanup and independent A11/A12 acceptance integration have
separate owners: obtain their merged completion evidence without importing
unfinished branches. A passing through-A10 gate does not cover A11/A12 or
certify the whole release. Reconcile the publishing gate with the completed
acceptance integration before launch, through its owner's reviewed change.

Run full CPU CI on the final release commit. If advertising the recorded
NVIDIA T4/CUDA 12 configuration, also obtain the manual GPU check for that
commit; do not infer accelerator support from CPU checks. Record outstanding
limitations in the release notes. Proceed only after these prerequisites and
the hosting checks below have evidence.

## Complete the planned GitHub organization transfer

The maintainer has created the GitHub organization `survjax`. The intended
repository destination is **`https://github.com/survjax/survjax`**, keeping the
name `survjax`. The transfer is still an operator action, not a completed result
of local preparation. Confirm permissions to transfer/create in the receiving
organization and that the destination name is available before proceeding.

After the authorized transfer, confirm the canonical URL and coordinate the
local identity update (RELEASE.5d): package repository/issues URLs, README GPU
badge/links, `CITATION.cff`, release workflow repository guard and metadata-test
expectations. The prepared files still bind `vanAmsterdam/survjax`; the guard
would skip the release build under the new owner until updated. Complete and
review this update before tagging/building release artifacts.

Recheck organization access, `pypi` environment protections, Actions permissions,
and RTD GitHub App/webhook access for the transferred repository. Configure the
PyPI publisher with the actual new owner; GitHub URL redirects do not substitute
for the publisher's repository-owner identity. The table below names the
intended post-transfer publisher, not an account configuration already made.
Update authorized local clones' remotes through the operator procedure.
[GitHub's transfer guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/transferring-a-repository)
describes redirects and organization permissions. No transfer or remote change
is performed by these instructions themselves.

## Connect Read the Docs

An authorized operator must confirm who controls the intended project slug
`survjax` and `https://survjax.readthedocs.io`. If that slug is unavailable or
belongs elsewhere, agree on a replacement and update package/README URLs
before building release artifacts. An HTTP error alone does not identify the
owner or account settings.

In the RTD dashboard, add or inspect the project connected to
`https://github.com/survjax/survjax` after transfer. Confirm the repository and default
branch `main`; install/authorize the Read the Docs GitHub App for this repository
and verify its integration/webhook. Use the root `.readthedocs.yaml`.
[Adding a project](https://docs.readthedocs.com/platform/stable/intro/add-project.html)
can trigger a hosted build immediately. [Git integration](https://docs.readthedocs.com/platform/stable/reference/git-integration.html)
on Community hosting requires a public repository; private-source builds
require suitable Business hosting/access.
Repository visibility and service/account changes require separate operator
authorization; none is part of this local preparation.

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
exact commit. The expected release URL is
`https://survjax.readthedocs.io/en/v0.0.1a1/`. Choose the checked tag version as
the site's default for the launch, keeping `latest` available. For Business
hosting, set the version privacy to **Public**; project privacy separately
controls public dashboard/build visibility. An active or hidden version is not
a substitute for the appropriate privacy setting. Do not rely on the `stable`
alias choosing an alpha.

For this initial alpha, README guide links intentionally remain on
`/en/latest/`: they can change after release. Use the verified tag URL when
citing the release's documentation. Check the root redirect/default, version
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
   `pyproject.toml` to the selected version, run `uv lock`, and update the
   installed-version assertion in `tests/install/test_wheel.py` together.
   Update release notes and any version-specific instructions consistently.
   The RTD tag selects the docs source; there is no separate Sphinx release
   version currently configured. Do not change an already published version.
2. From that clean commit, run local packaging and full checks on the exact
   built wheel (commands below), and run `scripts/build_docs.sh` directly.
   Inspect wheel/sdist contents, Markdown description, metadata and runtime
   isolation. Confirm full CPU CI on that same commit and the remaining
   acceptance/demo evidence. Do not substitute a fast gate for release checks.
3. Through the maintainer's separately authorized release procedure, tag that
   reviewed commit `v0.0.1a1` and make the tag available to RTD. Complete the
   tagged hosted/public docs checks before publishing the GitHub Release.
   This page creates no tag, pushes no commit and changes no visibility.
4. Publish a GitHub Release for that exact tag (mark it as a prerelease for
   `0.0.1a1`). This is the publishing workflow's trigger; ordinary CI, a tag
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
release_wheel="$(uv run python scripts/validate_release.py "$release_check_dir/dist" v0.0.1a1)"
SURVJAX_WHEEL="$release_wheel" scripts/check.sh --through A10
scripts/build_docs.sh
```

The command currently names A10 because that is the prepared workflow's gate.
It is not evidence for forthcoming acceptance steps; use the reviewed gate
selected by the independent acceptance integration before public launch.

## Check the public package from outside a checkout

After a separately authorized upload, create a fresh Python 3.12 environment
outside the repository and install from the public index, with no editable
checkout or demo groups:

```sh
public_check_dir="$(mktemp -d)"
uv venv --python 3.12 "$public_check_dir/venv"
uv pip install --python "$public_check_dir/venv/bin/python" --index-url https://pypi.org/simple --only-binary :all: 'survjax==0.0.1a1'
"$public_check_dir/venv/bin/python" -I - <<'PY'
import importlib.metadata as metadata
import importlib.util
from pathlib import Path
import sys
import survjax

assert metadata.version("survjax") == "0.0.1a1"
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
