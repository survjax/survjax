# B3.3 — Audit the Slice B likelihood evidence

Feature: observation.censoring and observation.truncation verification.
Base / previous step: ccb885d33231809ae757d75f878cb5ec3181d1ee
([B3.2](../B3.2/PLAN.md), review round 3: merge).
Original worktree base: cc0e30f. Dependencies: [B2.1](../B2.1/PLAN.md),
[B2.2](../B2.2/PLAN.md), [B3.1](../B3.1/PLAN.md), B3.2 and the merged families.
Cumulative gate: A12 with the existing observation feature flags.

The maintainer requested the next step after accepting B3.2. Audit the remaining
mathematical matrix before the B4 pruning work. Write only this PLAN.md and
REVIEW.md: a requirement-to-test map, actual command results and bounded gaps.
No public API, implementation, test, dependency, protected path or docs/ changes.
No new tests: this step establishes which existing tests supply evidence and
which follow-up tests are needed, rather than duplicating acceptance assertions.

Read spec 2, 3.2–3.3, 4.2, 7.1–7.3, 10.4 and 12.1–12.2. Map the four observation
kinds against absent/left/right/two-sided truncation, diagnostics, tails, safe
inactive operands, property identities, transformations and derivatives.
Distinguish Exponential/Weibull evidence from brought-forward LogNormal and
LogLogistic evidence and wrapper coverage. Record metadata pruning separately.

Run all observation censoring/truncation acceptance files plus both brought-
forward families' likelihood and derivative files in both dtype modes, without
the slow marker filter. Enable both observation and both family features for
these focused runs; do not change scripts/spec-features.txt. Run the ordinary
fast A12 gate, commit the evidence, invoke independent review, then stop.
Keep records below about 300 changed lines. No Slice B completion claim.

## Docs to-do

Carry forward B2/B3 likelihood support, diagnostics, truncation and fixture
convention notes to a later docs step. This audit supplies test references
and limitations for that step. Remaining work is determined in REVIEW.md;
B4 metadata pruning/measurements and the docs step remain separate.
