# DOCS.6 — Rebase integration

Read docs/index.md, DOCS-MIGRATION.md and the build section of
docs/technical-notes/verification.md. The maintainer requested resolution
of the in-progress rebase onto main e6b3f9e.

## Resolutions and invariants

- Archive all incoming planning records, including Bayes, in
  plans/pre-docs-restructure.md; preserve the docs plan lifecycle.
- Keep the reorganized navigation and add the accepted Bayesian use case.
- Keep the docs review record for the replayed docs step; main's previous
  review remains in git history.
- Update deferred-Bayes descriptions after integration. Simulation stays
  tentative and excluded. The Bayesian page and example files match main.
- Preserve the maintainer's callout title edit and three untracked input files.
  No API, dependency or numerical changes in this integration follow-up.

## Validation

- scripts/build_docs.sh: exit 0; float32 and float64 builds succeeded,
  including fresh execution of the Bayesian and MNIST pages.
- uv run --group test pytest -q tests/unit/test_docs_build_script.py:
  7 passed in 2.06s.
- /tmp/audit_docs.py: each build has 23 HTML pages and 1,844 local
  links/assets, zero failures; 18 executed notebooks, 153 code cells.
- All rendered Markdown source copies match current sources in both builds.
- git diff --check: clean; no conflict markers in docs or planning records.
- Full numerical gate not rerun: this follow-up changes documentation only.
  Independent review remains waived for this docs overhaul.

## Commit trace and remaining work

Original worktree base: 2093ca2. Rebase target: e6b3f9e.
Previous step / start of follow-up: 3a1c399 (previously f89767b).
All eight commits replayed successfully; this record accompanies the final
integration follow-up. No remaining conflict-resolution substeps.
