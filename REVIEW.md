# B1.d1 — Document interval probabilities

Independent review: round 1/3

Read PLAN.md B1.d1, then the new "Probabilities over intervals" section of
 docs/working-with-distributions.md (including its scope note). Compare spec
4.2, 7.1–7.3 and the implemented interval hooks where needed.
Start commit: 218621e (B2 blocker record); previous implementation step: 535b854;
original worktree base: cc0e30f. D-049 remains open; the maintainer says the
independent spec-test author is working on it. This docs work is independent.

One topic, one docs page, no implementation or API change. Explain interval
mass using S/F; define intersected endpoints separately before the conditional
ratio; explain numerator and denominator; demonstrate interval mass, finite
conditioning and Exponential memorylessness. Distinguish direct-hook empty
intersection (-inf) from invalid observation data. Explicitly state that
likelihood integration is pending. No promise of generic-family tail precision
matching the built-in overrides.

Validation:
- scripts/build_docs.sh directly: first sandboxed run failed (exit 2) because
  the existing Weibull-simulation notebook's sysctl CPU query was denied;
  ipykernel process inspection also reported sandbox errors.
- Reran the identical script with approved escalation: exit 0, both x32 and x64
  Sphinx builds succeeded and all notebooks executed. build-identity.txt records
  success for source at 218621e plus the then-dirty PLAN/docs edits. Subsequent
  changes are review bookkeeping only; the built page source is unchanged.
- Inspected generated HTML text, math delimiters and executed outputs in both
  dtypes. Interval probability ~0.334309; conditional probability ~0.462250;
  memorylessness pair matches (~-1.350226). This is content inspection, not a
  browser rendering check. CUA browser host was unavailable; no screenshot or
  visual-layout verification is claimed. Temporary HTTP preview server stopped.
- scripts/check.sh --fast --through A10: all gates passed. Implementer x64
  1081 passed/8 skipped; x32 1086/3; doctests 4 each; spec x64 2155/27, x32
  2160/22. Ruff/mypy and git diff --check passed. No new tests required for prose
  and executed examples; no full non-fast test gate rerun.

Remaining: independent acceptance-test transition for B2 (D-049), then left/
interval likelihood integration and truncation, B3 full numerical/reference
matrix, B4 pruning and remaining docs. This step does not resolve D-049 or
claim that broader observation likelihoods are available.
