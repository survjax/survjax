Independent review: round 1/3

# RELEASE.6 — Post-transfer link audit and RTD setup

Reviewer Astra/low, as authorized. Maintainer reports completed transfer and
asks links checked; subsequently confirms no RTD project yet and asks how.

## Read first

1. PLAN.md RELEASE.6, current spec revision 47 10.5/13.
2. README/CITING/CONTRIBUTING/CHANGELOG diffs: verified GitHub sources, relative
   docs links and confirmed transfer state; hosted site is not claimed live.
3. docs/{citing,contributing,changelog}.md: relative-docs prefix for includes.
4. docs/releasing.md: completed transfer, concrete RTD setup / visibility /
   actual-domain / remote-config prerequisites. Existing code/metadata unchanged.

## Evidence and decisions

- Authenticated gh repo view/API: nameWithOwner and html_url survjax/survjax,
  default main, visibility PRIVATE, has_issues true. GPU workflow API returns
  active GPU CI at .github/workflows/gpu.yml and canonical new-owner html_url.
- Sanitized origin fetch and push URLs both already github.com/survjax/survjax.git.
  No remote change was made. No credentials printed.
- Tracked active text/JSON/notebook inventory found no old-owner GitHub URL.
  Protected spec/tests/spec and historical plans/reviews excluded from reading /
  rewriting; no independent assertions inspected or changed. Old owner remains
  as transfer-source prose only, not current active URL.
- Authenticated contents API verifies main's PH/RMST/verification source URLs.
  Remote main root .readthedocs.yaml API returns 404 (absent); local config is
  committed on release-prepare, not automatically available to a remote builder.
- Anonymous GET checks with curl: GitHub repo/issues/GPU workflow/badge 404,
  consistent with authenticated private repository. RTD root and PH/RMST /
  verification/releasing URLs all 404. Maintainer confirms no RTD project yet.
- Do not invent a replacement docs hostname or configure accounts/visibility.
  Keep intended package Documentation metadata URL pending actual project/domain;
  README labels the intended Community URL as uncreated hosting, not live docs.
  Its guide links now point to existing canonical GitHub source files.
- Root included guides use local docs/ paths. Official MyST relative-docs: docs/
  preserves these links in repository sources and rewrites them in Sphinx HTML.
  No nav or other-agent page changes. No API/dependency/runtime/config/version,
  lockfile, protected path, spec, remote/account/visibility/push/merge/upload edits.
- RTD setup instructions distinguish public Community vs private-source Business;
  Business has a different default domain, use actual dashboard domain/slug.
  First push reviewed config to the build branch, authorize org GitHub App, Add
  project, select canonical repo, confirm root YAML, inspect fresh build/latest.
  Creating a project can publish docs, so this is instructions-only work.

## Official references checked

- https://docs.readthedocs.com/platform/stable/intro/add-project.html
- https://docs.readthedocs.com/platform/stable/reference/git-integration.html
- https://docs.readthedocs.com/platform/stable/commercial/privacy-level.html
- https://docs.readthedocs.com/platform/stable/versions.html
- https://docs.readthedocs.com/platform/stable/custom-domains.html
- https://myst-parser.readthedocs.io/en/latest/syntax/organising_content.html
  and its FAQ: include relative-docs prefix rewriting.

## Checks

- scripts/check.sh --fast --through A10: exit 0, all gates passed. Ruff 182
  files formatted / all checks passed; mypy 22 files clean. Implementer x64
  1066 passed / 8 skipped, x32 1071 / 3; doctests 4 each; spec x64 2149 / 27,
  x32 2156 / 20. Slow cases excluded; no new tests for prose. Gate builds fresh
  wheel and tests its current README description. No A11/A12 claim.
- scripts/build_docs.sh directly, approved escalation: exit 0; both Sphinx
  dtype builds succeeded with force execution/error-fatal policy retained.
- /tmp/survjax-posttransfer-link-audit.py: full root include prose/source bytes
  match both rendered sites; each root relative href becomes the correct local
  HTML href. Transfer-complete/private/RTD-uncreated status and setup present.
- Root Markdown relative target existence and canonical repo URL checks passed.
- Site audit: each mode 27 HTML pages, 1,710 local links/fragments/assets,
  14 downloads, zero failures. Downloaded notebooks match source bytes;
  use-case images/navigation present. Shared execution tree: 18 notebooks /
  153 code cells; both-mode execution confirmed by logs. No visual-browser QA.
- External rendered-page/README href/src audit: 36 unique HTTP targets,
  30 final 2xx/3xx. The remaining six 404s are this private GitHub repository,
  GPU workflow/badge, two guide source pages and docs tree. Authenticated API
  verifies repo, workflow, source file paths and docs directory all at canonical
  survjax/survjax. Report /tmp/survjax-release6-external-links.txt.
  Uncreated RTD intended URLs checked separately as above, not counted live.
  HEAD used with GET fallback on 405; no public-access claim for private URLs.
- git diff --check: passed.

## Limits

No RTD account/project creation, visibility, GitHub App/webhook/environment,
credential, push/merge, tag/release, PyPI upload or hosted Linux execution.
Intended RTD metadata URL is not live; project slug/domain availability remains
an operator check. GitHub endpoints require access while repo stays private.
Local source/link validation does not prove public access. Docs used warm
MNIST cache and recorded results with live calculations, no new full fit/train.
No browser visual-layout or full slow numerical/install rerun claim. Other
owners' launch gates remain open; this follow-up does not publish the branch.

## Commit trace

Previous step / clean start: 0657d65e852448441748b9f7ba038f0104223359.
Original worktree base: 3cb4cd7fbba3313f00dfabed0028ee6f404384df.
Commit the bounded docs/link step, independent Astra review, then stop/report.
