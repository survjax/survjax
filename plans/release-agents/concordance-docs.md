# Agent: remaining concordance teaching material

Read the packet README, spec 6.1, 9.1–9.2, 10.5 and 12.1–12.2, metrics.py,
the A10 tests and the R fixture README. An existing agent is implementing
A10.d1 (docs/concordance.md plus index/API links) in this checkout; do not
duplicate that step. Its PLAN.md reports a through-A10 pass on 7c4f57b.
Treat that as recorded evidence, not a new check performed by this packet.

After that guide merges, own only the remaining concordance-specific
example/check files named in your step plan. No production changes, A9
benchmark/verification page, dependency files or unilateral navigation edits.

1. Review the merged guide against its plan and identify only concrete gaps.
   Coordinate a concordance playground section with A9's notebook owner;
   serialize the notebook edit or let that owner integrate the reviewed cells.
   Cover public calls, score draws and diagnostics without duplicating the guide.
2. Separate held-out synthetic teaching example:
   fixed train/test split, training-only fit or fully specified fixed scoring
   rule, test likelihood/C, explicit cohort sizes and score interpretation.
   Reuse the simulation generator after it lands if needed; do not introduce
   an optimizer/dependency solely for this page or claim predictive learning
   from fixed true parameters. Coordinate the boundary with the MNIST page.

Show the R references' synthetic scope/provenance and actual integration gate.
Execute the page in both dtype docs builds, check examples against analytical
or existing reference cases, and inspect rendered tables, equations and links.
Keep smoke execution comfortably below the 120-second notebook timeout.
Finish with reviewed docs and runnable examples; no full-release claim.
