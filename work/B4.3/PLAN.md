# B4.3 — Argument-form metadata for traced construction

Feature: observation.traced_metadata, spec rev 62 (decision 0060).
Step starting commit: aaba08f6b42c00433792a5984d53faa85a78d7fd

Base: aaba08f (main merged into li-censoring). Previous branch step:
[B5.FAMILIES](../B5.FAMILIES/PLAN.md), 82c1826; pruning dependency and accepted
previous pruning step: [B4.2](../B4.2/PLAN.md), f568b9b.
Original worktree base: cc0e30f. Cumulative gate: A12 plus existing features and
observation.traced_metadata.

Ran ../scripts/spec-since.sh dev and ../scripts/spec-since.sh dev --mark.
The merge already contains 71cadbf, acknowledging design commit a5c1dea
(rev 62); marking produces no .spec-seen diff to commit.

Resumed after maintainer merge b4b6700 supplied the independent test author's
same-shape correction (3d9704a) and updated review workflow. The saved starting
commit remains aaba08f. Follow the optional workspace diff-display hook after
the review loop, using that starting commit and the final reviewed SHA.

Change src/survjax/data.py to derive traced flags solely from None versus
supplied arguments, exactly as spec 3.2 prescribes. Keep concrete metadata,
normalization, issue bits, slicing and pytree reconstruction unchanged.
Explain the positional flags and why argument presence permits pruning.
No new API, module, dependency, numerical helper or protected-path change.

Update tests/unit/test_data_metadata.py's superseded all-true expectations,
and add compact checks for supplied false indicators/default-valued truncation,
bounds without exact, and compilation reuse across changed values. Preserve
the existing normalized-leaf comparisons. Run the 51 independent acceptance
tests in tests/spec/test_observation_traced_metadata.py unchanged. Enable
observation.traced_metadata in scripts/spec-features.txt in this same step.

Extend scripts/benchmark_likelihood_pruning.py to retain prepared concrete and
all-true controls and add ordinary time/event construction inside jit. Build
the all-true control via traced bounds with exact and both truncations supplied;
the old time/event construction no longer supplies that control. Return data
diagnostics in every route, check objective/gradient/diagnostic agreement,
separate preparation, first-call and warmed timings, and synchronize outputs.
Use fixed-seed Exponential/Weibull shared-parameter workloads in both dtypes;
record metadata and limits so construction overhead is not called pruning
speedup. Keep interleaved timing and compilation reuse checks.

Target under 300 implementation changed lines. Run focused metadata,
construction and pruning tests fully in both dtypes, then the fast A12 gate.
Refresh the slow-test list only if newly added tests exceed one second.
Rerun the benchmark separately per dtype and record results in REVIEW.md.
Commit the handoff, run the independent-review skill, and stop.
Spec coverage: 3.2, 6.2, 7.2, 12.1–12.3.

## Docs to-do

The later likelihood docs must distinguish concrete value-derived metadata
from traced argument-form metadata, describe supplied default values versus
omitted arguments, and update the B4.2 benchmark limitations. Final Slice B
evidence audit and integration remain; reviewer-workflow investigation stays
deferred.
