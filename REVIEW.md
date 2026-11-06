Independent review: round 1/3

# RELEASE.4 — Release operator handoff

Independent reviewer: Astra, low reasoning, as explicitly requested by the
maintainer for this workstream; no Claude invocation or protected skill edit.

## Read first

1. PLAN.md RELEASE.4: one documentation topic, no navigation/other-agent edits.
2. docs/releasing.md: account actions, prerequisites and release sequence.
3. .readthedocs.yaml, .github/workflows/release.yml and
   scripts/validate_release.py (unchanged): verify the handoff matches reality.
4. Current spec revision 46, 10.1/10.3/10.5, 12.2 and 13. The shared design
   source advanced during this workstream; previous review revision labels
   are historical and have not been rewritten. No numerical scope here.

## Invariants and choices

- Only docs/releasing.md and plan/review records change. No API, dependency,
  runtime/config/version/lockfile, navigation, protected path or spec changes.
- Page is a standalone MyST orphan document, releasing.html, searchable but
  outside the existing toctree. This preserves the docs agent's navigation.
- Candidate 0.0.1a1, tag v0.0.1a1, expected RTD slug v0.0.1a1; operator must
  confirm both selected version and actual RTD slug. Preparation baseline
  0.0.1.dev0 remains unchanged. Version/lock/install-assertion updates occur
  together in the eventual reviewed release commit.
- Initial-alpha README links deliberately remain moving /en/latest/ links;
  release documentation can be cited through its checked version-specific URL.
- Exact pending publisher fields match workflow release.yml / environment
  pypi / repository vanAmsterdam/survjax / project survjax. PyPI account owner
  and RTD project owner are not inferred from these identifiers or URLs.
- Record private-source hosting requirements, active/default/privacy settings,
  fresh Linux build requirement, MNIST cold network/cache, timeout/resource
  budget, both dtype execution and complete-site output/download checks.
- Explicitly require changelog, citation and contribution guidance, provenance,
  license, tested configurations, other agents' demo/docs/acceptance evidence,
  public package/docs checks and coordinated version commit before launch.
  A10 is the current prepared gate, not evidence for A11/A12; independent
  acceptance owner must integrate the appropriate release gate.
- Publishing a GitHub Release is the workflow trigger. Before pypi environment
  approval inspect the exact run's tested artifacts/hashes; local rehearsal
  files are not claimed as workflow upload files. No accounts, tags, release,
  push/upload, visibility change or public-install command is executed here.

## Current official interfaces checked (2026-11-06)

- RTD project connection/import:
  https://docs.readthedocs.com/platform/stable/intro/add-project.html and
  https://docs.readthedocs.com/platform/stable/reference/git-integration.html.
- RTD active/hidden/public/default and prerelease behavior:
  https://docs.readthedocs.com/platform/stable/versions.html and
  https://docs.readthedocs.com/platform/stable/commercial/privacy-level.html.
  Hidden is not private; Business project and version privacy differ.
- RTD documented budget:
  https://docs.readthedocs.com/platform/stable/builds.html#build-resources.
- PyPI pending publisher:
  https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/.
  Pending publisher does not reserve the project name.
- GitHub environment protections and account-plan/visibility availability:
  https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments.

## Verification

- scripts/check.sh --fast --through A10: exit 0, all gates passed.
  Ruff: 176 files already formatted / all checks passed; mypy: 22 files clean.
  Implementer float64: 1066 passed / 8 skipped; float32: 1071 / 3.
  Source doctests: 4 passed each. Spec float64: 2149 passed / 27 skipped;
  float32: 2156 / 20. Slow tests intentionally deselected. No new unit tests
  for prose, and no redundant full numerical suite or install-suite rerun.
- First direct scripts/build_docs.sh run: exit 0; both Sphinx builds succeeded
  with fresh existing notebook execution. Kernel shutdown logged sandbox-denied
  psutil process inspection (sysctl PermissionError); this was outside notebook
  calculations. Rerun directly using approved escalation to verify the final
  text without that sandbox limitation; final evidence follows below.
- Both future shell blocks parsed using bash -n; embedded Python parsed with
  ast.parse; no {code-cell} directives in the new page. Passed. The release
  rehearsal and public-index install examples were not executed by this check.
- Escalated scripts/build_docs.sh rerun: exit 0; float32 and float64 each
  report build succeeded, with forced execution retained. No earlier psutil
  PermissionError tracebacks observed in the escalated run. Existing kernel
  local-TCP notices remain tool output, not Sphinx warnings or execution errors.
- uv run --locked --group docs python /tmp/survjax-release-audit.py:
  each site has 24 HTML pages, 1,579 local links/fragments/assets, 14 downloads,
  zero failures. Existing notebook downloads byte-match sources; use-case
  images and navigation present. 18 notebooks / 153 code cells in the shared
  sequential execution tree; build logs confirm both dtype executions.
- Rendered handoff inspection in both sites: title, five sections, exact five
  publisher fields, two shell blocks, source-copy byte identity, search index
  entry and existing sidebar all passed. No visual browser/layout claim;
  standard Furo styling/configuration remains unchanged.
- git diff --check: passed.

## Limitations and remaining work

No hosted Linux/RTD execution, account ownership/configuration, GitHub artifact
service/OIDC, PyPI upload/rendering, public clean-install or signed-out public
access verified. The new page describes these pending operator actions rather
than reporting them complete. Both docs builds use the existing warm MNIST
cache; RELEASE.1 separately recorded a cold-cache fetch, not hosted evidence.
No tag/version change or publication authorization is inferred from this page.

Remaining docs steps: RELEASE.5a changelog, 5b citation, 5c contribution guidance.
Remaining release gates belong to their existing owners. Keep the public launch
blocked until the handoff's prerequisites are evidenced. The new orphan page
has no sidebar entry; a later navigation decision remains with the docs owner.

## Commit trace

Previous completed step / clean start:
035f4e001f46d999a36e035b9db91f0bd6f2bc95 (RELEASE.3).
Original worktree base: 3cb4cd7fbba3313f00dfabed0028ee6f404384df.
Commit the bounded docs step, independent Astra review, then stop.
