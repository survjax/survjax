# PH.7 — Maintainer edits and RMST documentation handoff

Independent review: round 1/3

Step start /previous PH.6 final: 5c2354e3539668a927c5b355c173bffaf0d63c1c.
Original worktree launch: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Current AGENTS.md/CLAUDE.md read before edits. Work stays on release-ph.

## Read first

1. PLAN.md PH.7 and docs/proportional-hazards.md diff.
2. docs/index.md and docs/api.md: guide-description changes only.
3. Spec 4.4,7.3,8.7,10.5. The maintainer explicitly requests removal of RMST
   material from the PH guide because the RMST implementer will document it.

## Exact scope and invariants

94 changed docs lines: guide +6/-82, index +2/-2, API +1/-1.
PLAN/REVIEW excluded from the implementation budget. No library/API/test,
dependency/config/protected/spec/deviation/appendix changes. No public export
or API-reference test weakened. No merge, push or publication.

Compared the complete current-guide/ph2.md diff before editing. The user draft
has exactly the opening prose edits plus a stale orphan marker. Applied both
body edits verbatim: positive eta means higher hazard thus shorter survival;
remove the survival-probability-ratio sentence. Retained current toctree/home
navigation rather than reinstating orphan: true. User-owned untracked ph2.md
is untouched and intentionally outside this commit; SHA256 before/after is
985dc60936bf7504afdfc1e2761c78e97f7d2ec08f3a342a05725ef57e374b82.
It is not needed by the standalone committed docs or reviewer checkout.

Removed the RMST section, closure table, scalar/grid cutoff examples, summary
AD caveats and overflow example/explanation. Removed the dangling RMST-contract
cross-reference; renamed the remaining diagnostics section to Diagnostics.
Kept the scoped direct/likelihood ordinary AD guarantee and revision-40 extreme
parameter-derivative limitation, including its affected parameters and lack of
mapped safe boundary. No remaining guide prose discusses RMST. This is an
ownership handoff, not a change to section 8.7's contract or release obligations.

Both guide descriptions now match its contents. Home capability paragraph still
correctly mentions native Exponential/Weibull RMST and their PH wrappers; API
rmst autodoc remains intact. Both home links and sidebar entry remain. The five
surviving code cells are byte-identical to the prior guide, selected from old
cells 0,1,2,3,6. The same survival figure remains.

## Actual checks and evidence

- scripts/build_docs.sh run DIRECTLY, no redirect/environment assignment/shell
  wrapper: exit 0, fresh executable examples in x32 and x64; 12 source pages,
  warnings fatal. PH notebook execution 4.53s float32 /4.25s float64.
- Parsed BOTH generated PH main articles: exactly five output containers,
  requested shorter-survival prose, no RMST narrative, correct printed dtype,
  batch/grid ((3,2),(3,2),(3,3,2)), NLL/gradient/curvature (3,2,3), and
  baseline.scale diagnostic with one finite/one NaN survival value. These
  demonstrate spec 4.4 broadcasting/eta direction and 7.3 ordinary compiled
  derivatives; the existing unchanged executable assertions check boundaries
  and masks. The loss expectations follow the hand-derived 3*exp(eta)-eta.
- Both home main articles retain two PH targets (prose and toctree), and the
  out-of-article sidebar target. Guide prev/next remain built-in families /
  likelihoods. API link and survjax.ProportionalHazards.rmst anchor retained.
- Single plot image SHA256 in BOTH builds is unchanged from the previously
  visually inspected PH.5 plot:
  90a1e0a54cebfb8edac31f4086b1df2980160d95789160e8d40efbbeb4c6a12e.
  No new plot/layout content. Generated output/cache remain uncommitted.
- uv run ruff format --check .: 137 files already formatted.
- uv run ruff check .: All checks passed!
- uv run mypy src: Success: no issues found in 22 source files.
- uv run pytest -q tests/unit/test_api_reference.py: 1 passed in 0.53s.
- git diff --check: exit 0.
- git diff --exit-code -- src tests pyproject.toml uv.lock AGENTS.md CLAUDE.md
  scripts .agents .workmux.yaml docs/conf.py docs/playground.ipynb appendices:
  exit 0. No numerical/wheel gate rerun for this docs-only step; PH.4 tested
  the same library. No R/fixtures or PDF appendix changes.

## Remaining work and uncertainty

No unresolved implementation question in this docs revision. Separate RMST
narrative, including PH closure, derivative restrictions and numerical failure,
is assigned to its implementer by the maintainer. Final PH evidence/acceptance
audit remains separate; independent PH acceptance tests have not yet been
supplied. Existing deviation statuses remain maintainer-owned. Stop after
independent review; do not begin another step.
