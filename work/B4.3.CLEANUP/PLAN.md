# B4.3.CLEANUP — Keep pruning benchmarks in the experiments repository

Step starting commit: 200c1f7f4b64d147d5d40cbcba44f97e44d3c1f5
Previous step: [B4.3](../B4.3/PLAN.md), same commit.
Original worktree base: cc0e30f. Cumulative gate: A12, features unchanged.

The maintainer requested removing the pruning benchmark script and raw timing
files from the library, with preservation verified in the active experiments
repository. The existing likelihood-pruning experiment is independently pinned
to 82c1826 and has its own evolved benchmark; do not replace that source, pin,
lockfile or existing runs with the B4.3 variant.

Archive B4.3's exact script, tolerance source, JSONL files and full historical
handoff in a fresh ignored artifact directory in the experiment. Commit a small
manifest/run record there with hashes, original library paths/commit, execution
commands and retrieval instructions; note this historical evidence in its notes
and build those PDFs. No experiment rerun or publication is authorized/needed.
Preserve the unrelated dirty experiments-hub dashboard and report its update.

Then remove scripts/benchmark_likelihood_pruning.py and
work/B4.3/benchmark-float{32,64}.jsonl here. Shorten the B4.3 handoff's benchmark
section to link the experiment record and immutable experiment commit. This
specific earlier-record update is authorized by the maintainer's cleanup request;
leave all other steps' historical records intact. Retain implementation,
correctness tests, tolerance constants and the step plan/review records.

Validation: verify artifact checksums before deletion; verify experiment commit
contents and unchanged active pin/source; build experiment notes; run the fast
A12 gate and git diff --check. This changes no public API or numerical behavior.
It changes evidence ownership for spec 12.3, not performance requirements.
Commit the cleanup and use independent review, then stop. The reviewer checkout
is currently dirty; follow the skill's setup restriction if that remains true.

## Docs to-do and remaining work

No new library docs obligation. Likelihood documentation, final Slice B audit
and integration remain. Pruning measurements now belong to the experiments
project; later library handoffs should link its versioned evidence.
