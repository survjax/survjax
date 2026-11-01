# MNIST.14 — Showcase data
Independent review: round 1/3

Read INSTRUCTION.md sections 2/5, PLAN.md MNIST.14, data.py and its two tests.
Fixed official global ordering, seeds, cohort sizes and float32 outcomes remain
invariant. SHA verifies format before simple parsing. Fixed c=7 replaces
calibration; shared generator unchanged. Tests cover download/cache digest,
parsing, preprocessing, splits, independent uniform algebra and determinism.
No library API or dependencies change. Remaining: model/runner/full results,
then README/docs. Maintainer authorized continuing through all steps.
