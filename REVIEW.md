Independent review: round 1/3

# RELEASE.5c — Contribution guidance

Reviewer: independent Astra, low reasoning, maintainer-authorized.
Maintainer explicitly authorizes continuing after each reviewed step until the
actual transfer boundary; no need for intervening approval for 5d/5e.

## Read first

1. PLAN.md RELEASE.5c and current spec revision 47, 10.1/10.3/10.5/13.
2. CONTRIBUTING.md / docs/contributing.md: one prose source and orphan include.
3. Existing scripts/check.sh, scripts/run_tests.sh, scripts/build_docs.sh,
   pyproject.toml and .github/workflows/ci.yml (unchanged): verify commands.
4. CHANGELOG.md status update.

## Invariants

- Contribution guidance covers actual setup, test tiers, executable docs,
  public API typing/docstrings, synthetic reproducers and provenance.
- Runtime remains JAX/NumPy only. No new dependency, API, policy agreement,
  contributor identity, configuration, protected path or spec changes.
- Separate dtype processes and touched tests without the fast filter. Current
  A10 gate does not cover forthcoming A11/A12 acceptance. Independent acceptance
  tests/gates stay protected; never weaken checks/tolerances to get a pass.
- No patient data or fabricated fixture values. Existing fixture READMEs own
  regeneration; ordinary tests need neither R nor network. Docs cold MNIST can
  need network, explicitly stated; live execution retained.
- Intended survjax/survjax link is qualified as transfer-pending. No remote,
  transfer, account, tag, push/upload or publication action.
- Root prose included at contributing.html/search, no navigation edits or
  unfinished branches imported. Other agents' launch gates remain open.

## Verification

- bash -n on all contribution shell blocks: passed. Referenced touched test,
  tolerance, gate, runner, build and CI paths exist; checked flags/config.
  No {code-cell} directives added, no future operator commands executed.
- scripts/check.sh --fast --through A10: exit 0, all gates passed.
  Ruff 182 files formatted / checks passed; mypy 22 source files clean.
  Implementer x64 1066 passed / 8 skipped; x32 1071 / 3; source doctests 4 each;
  spec x64 2149 / 27; x32 2156 / 20. Fast gate deselects slow cases, no new unit
  tests for prose and no redundant full numerical/install-suite rerun.
- Direct scripts/build_docs.sh, approved escalation: exit 0; float32 and
  float64 both report build succeeded with force execution/errors fatal.
- Site audit: each mode 27 HTML pages, 1,705 local links/assets,
  14 downloads, zero missing files/fragments/assets. Existing notebooks
  match source, use-case images/navigation present. Shared execution tree:
  18 notebooks / 153 code cells; logs show both-mode fresh execution.
- Root contribution prose equals complete rendered HTML prose in both modes;
  wrapper source bytes and search entries match. Changelog include audit passes.
  Initial ad hoc text audit inserted spaces between syntax-highlighting spans;
  corrected extraction to preserve complete preformatted code text, then passed.
  No source change. Programmatic inspection, no browser visual-layout claim.
- git diff --check: passed.

## Remaining sequence and limits

Continue to separate reviewed code/metadata step RELEASE.5d, then docs
synchronization RELEASE.5e. Code steps do not edit docs/. Stop before actual
transfer, git remote, account, visibility, hosting, tag/release/push or upload.
No hosted Linux/RTD run, publisher identity, public installation or overall
release-readiness claim. Actual transfer and separately owned launch gates
remain pending. Docs use existing warm datasets/results with live calculations.

## Commit trace

Previous step / clean start: 67fb851d517a919fbb0d0e6e2f88e56912b609c7.
Original worktree base: 3cb4cd7fbba3313f00dfabed0028ee6f404384df.
Commit/review this step before starting 5d, as authorized by maintainer.
