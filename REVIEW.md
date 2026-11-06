Independent review: round 1/3

# RELEASE.5e — Final pre-transfer docs synchronization

Reviewer Astra/low, authorized. Maintainer asks to complete reviewed local work
up until actual transfer; this is the final local step and stops there.

## Read first

1. PLAN.md RELEASE.5e; current spec revision 47 10.1/10.5/13.
2. CITING.md / CONTRIBUTING.md / CHANGELOG.md diffs and docs/releasing.md.
3. Reviewed 5d metadata/CFF/README/release guard/test URLs (unchanged here).
4. Prior reviews /tmp/survjax-RELEASE.5c-review-round1.md and
   /tmp/survjax-RELEASE.5d-review-round1.md: both merge, no required findings.

## Invariants

- Local package/citation URLs, README badge and release guard now target the
  intended survjax/survjax organization repository. Old-owner references in
  current prose describe the transfer source, not active metadata bindings.
- Actual transfer from vanAmsterdam/survjax is still pending and unverified.
  No Git remote, ownership/access, availability, account/environment, GitHub
  App/webhook, visibility, tag/release/push/upload or publication action occurs.
- Guard accepts new namespace only and blocks old-namespace release builds;
  docs explain that condition accurately. Future publisher owner survjax /
  repository survjax / release.yml / pypi matches the intended prepared flow.
- Contribution/citation guides and draft changelog prepared; no broader
  release certification. Unfinished demo/docs/acceptance and Linux/RTD/public
  account checks retain their owners and launch gates. A10 is not A11/A12.
- Coordinated eventual release commit updates package version, lock, installed
  version assertion, actual citation release version/date and final notes.
  Current dev0 version and author credit stay unchanged; no DOI/date invented.
- Only prose and plan/review records change; no dependency, API, runtime,
  configuration, protected path, shared CI, version/lockfile, spec or navigation
  edits. Root citation/contribution/changelog remain single included sources.

## Checks

- Direct scripts/build_docs.sh with approved escalation: exit 0; float32 and
  float64 both report build succeeded. Forced execution/error-fatal policy
  unchanged; Bayesian and MNIST pages executed. Existing kernel TCP notices.
- /tmp/survjax-pretransfer-docs-audit.py: exit 0. Complete citation/contribution /
  changelog include prose, source bytes and search entries match in both modes.
  Rendered handoff has prepared new identity, old source namespace / blocked
  guard status, final citation version/date sequence and exact future publisher
  table. Stale old-binding descriptions absent. Shell examples pass bash -n;
  embedded Python parses, no operator commands executed.
- /tmp/survjax-release-audit.py: each mode 27 pages, 1,705 local links/assets,
  14 downloads, zero failures; notebook downloads match sources, use-case
  images/navigation present. Shared execution tree 18 notebooks / 153 cells,
  logs show both dtype execution. HTML/content QA, no browser visual claim.
- scripts/check.sh --fast --through A10: exit 0, all gates passed. Ruff 182
  files formatted/all checks passed; mypy 22 files clean. Implementer x64
  1066 passed/8 skipped; x32 1071/3; doctests 4 each; spec x64 2149/27,
  x32 2156/20. Slow cases excluded. No new unit tests for prose,
  no redundant full install/numerical suite; 5d independently verified actual
  artifacts and installed-wheel tests in both dtypes.
- git diff --check: passed.

## Limits and transfer boundary

This finishes the authorized local preparation sequence, not actual transfer
or public launch. No live destination, credentials, rights, publisher identity,
hosted Linux/RTD, public access or clean-index installation verified. Existing
warm MNIST cache/results with live calculations used for docs. No full new
neural training/Bayesian fit, R regeneration or A11/A12 acceptance integration.
No remote/operator examples executed. Future service settings must be checked
under the actual organization after separately authorized transfer.

## Commit trace

This turn began at 67fb851d517a919fbb0d0e6e2f88e56912b609c7.
5c last commit 9bffb535df46cff93511afc9578200ced49788e1, starting at 67fb851d.
Previous completed step / clean start for 5e:
f301172d482e4159f8bb75be85dd6fbf7e83dca5 (5d, started at 9bffb535).
Original worktree base: 3cb4cd7fbba3313f00dfabed0028ee6f404384df.
After final commit and independent review, stop before actual transfer.
