Independent review: round 1/3

# RELEASE.1 — Prepare Read the Docs builds

## Read first

1. PLAN.md: bounded step, release sequence and preserved workstream ownership.
2. .readthedocs.yaml: Python/uv bootstrap, locked groups, both-dtype build,
   then float64 site staging into READTHEDOCS_OUTPUT/html.
3. scripts/build_docs.sh and DOCS-BUILD.md (unchanged): environment isolation,
   output layout and failure behavior; docs/conf.py retains forced execution.
4. tests/unit/test_docs_build_script.py (unchanged): environment/output selection
   and stop-on-failure tests. Spec revision 41, sections 10.3–10.5 and 12.2.

## Invariants and scope

- Only local release preparation; no publishing, accounts, tags, remote changes,
  runtime dependency, metadata/version or public API changes.
- No Sphinx top-level RTD key: custom jobs own the environment and build;
  no injected/unlocked RTD Sphinx packages. Bootstrap uv is pinned to 0.11.19.
- Checkout-local .venv (Python 3.12.9 here), uv.lock, docs/examples groups.
  JAX 0.11.2, NumPy 2.5.3, Sphinx 9.1.0. No shared agent environment.
- Existing script runs float32 and float64 separately, fresh execution in both,
  warnings/errors fatal. Stage only after the script succeeds in both modes.
- Copy the contents of the complete float64 output, preserving nested assets,
  notebook/example downloads, figures, relative links and search resources.
  Execution caches outside that HTML tree are not staged.
- Completed DOCS plan archived verbatim in its own commit. Historical unfinished
  records retain their original ownership; docs pages/navigation are unchanged.

## Official interface verification (2026-11-06)

- https://docs.readthedocs.com/platform/stable/config-file/v2.html:
  Ubuntu 24.04 and Python 3.12 supported; custom build.jobs without a sphinx
  key supported. Environment changes do not persist across job commands.
- https://docs.readthedocs.com/platform/stable/build-customization.html:
  custom HTML output goes in READTHEDOCS_OUTPUT/html; nonzero commands fail
  the build; custom jobs avoid replacing the established executable pipeline.
- https://docs.readthedocs.com/platform/stable/builds.html#build-resources:
  Community budget 15 minutes / 7 GB; Business 30 minutes / 7 GB (upgradable).
  Actual account tier/project limits remain unverified.
- https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/:
  pending publisher supports first upload, does not reserve the project name;
  informs RELEASE.3/4 only. No publisher/account configured.

## Checks and outcomes

- uv sync --locked --group docs --group examples: exit 0, 166 packages resolved;
  isolated .venv confirmed by sys.prefix. Lockfile unchanged.
- YAML parsed using existing environment PyYAML; Python setting matches
  .python-version, custom job structure checked. No added dependency.
- Exact YAML mkdir/cp commands executed on nested/hidden fixture files and a
  spaced output directory: byte-identical tree; exit 0.
- uv run --group test pytest -q tests/unit/test_docs_build_script.py:
  7 passed in 2.69s. Tests cover the existing contract used by the RTD job.
- Cold-cache MNIST: existing load_mnist on an empty temporary cache downloaded
  all four files from https://ossci-datasets.s3.amazonaws.com/mnist/ and verified
  existing SHA256 hashes; 11,594,722 compressed bytes; 9.4 seconds;
  pixel shape (70000, 1, 28, 28), labels (70000,). No data committed.
- uv sync --locked --group docs --group examples --python-platform
  x86_64-unknown-linux-gnu --dry-run: exit 0, 166 packages resolved, 27 Linux
  package replacements selected. This is resolution, not Linux execution.
- scripts/check.sh --fast --through A10: exit 0, all gates passed.
  Ruff: 173 files already formatted, all checks passed; mypy: 22 source files.
  Implementer float64: 1059 passed / 8 skipped; float32: 1064 / 3.
  Source doctests: 4 passed each mode.
  Spec through A10 float64: 2149 passed / 27 skipped; float32: 2156 / 20.
  Slow tests intentionally deselected by the fast gate. No A11/A12 acceptance
  claim and no spec-test inspection or edits.

- scripts/build_docs.sh: exit 0; both builds report `build succeeded`, forced
  execution retained. Log span approximately 15:15:39–15:18:25 UTC (~3 min),
  excluding dependency install; local CPU concurrent with fast gate, warm data.
  MNIST page: 8.66 / 10.43 seconds; Bayes: 5.94 / 8.81 (float32 / float64).
- uv run --locked --group docs python /tmp/survjax-release-audit.py:
  both sites checked for local links/fragments/assets, sidebar navigation,
  use-case images and byte-identical downloadable source notebooks.
  Each site: 23 HTML pages, 1,533 local links/assets, 14 downloads, zero
  failures; 18 executed notebooks / 153 code cells (execution tree shared by
  sequential builds; the build logs separately confirm both-mode execution).
  Initial ad hoc audit assumed Furo uses a nav element and Bayes a figure
  element; corrected selectors to actual sidebar links/images, no product edits.
- Exact YAML staging commands on the executed float64 site: byte-identical,
  119 files copied, including all downloads/assets.
- git diff --check: exit 0. No new automated test for this low-impact config;
  focused executable checks exercise the real build and staging commands.

## Limitations / remaining checks

- Host is macOS arm64; docker, podman, orb, colima and limactl executables
  unavailable. Fresh Linux execution and RTD integration/public access are
  unverified; the Linux dependency dry run is not a hosted build.
- Docs use the existing warm MNIST cache; cold fetching verified separately.
  RTD must allow outbound HTTPS for all four files (60-second fetch timeout
  each), plus initial locked package installation. Persistent dataset caching
  is optional; execution must remain live even on a cold worker.
- MyST cell timeout remains 120 seconds. No hosted peak-memory measurement;
  confirm both execution and install fit the actual RTD project limits before
  launch. MathJax still uses the existing external CDN.
- RTD project slug/account ownership and GitHub integration are unverified;
  the README URL alone establishes neither. An operator must connect the
  intended vanAmsterdam/survjax repository with authorized access, select active
  public versions/default version and validate a clean hosted tag build later.
- Remaining steps: RELEASE.2 metadata/version choice (0.0.1a1 proposed),
  RELEASE.3 Trusted Publishing workflow/artifact checks, RELEASE.4 operator docs.
  No uploads or public clean-install claims in this step.

## Commit trace

Previous completed step / clean starting SHA / original worktree base:
3cb4cd7fbba3313f00dfabed0028ee6f404384df (DOCS.6).
Plan-only archive / implementation parent: 2c991d0.
This step is RELEASE.1; independent review follows this commit, then stop.
