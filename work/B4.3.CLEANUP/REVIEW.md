# B4.3.CLEANUP — Benchmark ownership handoff

Independent review: round 1/3 pending setup; no round launched yet.

Read PLAN.md, the deletion of scripts/benchmark_likelihood_pruning.py and the
two B4.3 JSONL files, then B4.3/REVIEW.md's shortened benchmark section.
No implementation, correctness tests, tolerance constants, feature flags,
dependencies or protected paths change. Earlier B4.1/B4.2 records remain
historical descriptions of their respective commits.

## Preservation and validation

The existing experiment is ../experiments/projects/likelihood-pruning.
Archive provenance commit: aca9b2300a6216bf47dd24eb0d9c8eb61ffbb7c3.
Its runs/20261112-library-b43-import.{md,json} record exact source paths,
SHA-256 checksums, original commands, library commit 200c1f7 and retrieval.
Five original files were copied from committed Git blobs to the fresh ignored
artifacts/20261112-library-b43-import/ directory: benchmark script, tolerance
source, both JSONL outputs and the complete original B4.3 handoff. Each copied
file and local library original was rehashed before deleting library copies.
The history remains intact; the Git blobs are an additional retrieval source.
No remote archive or public publication is claimed.

The experiment's active source, pin (82c1826), snapshot, lock and tolerance
files were verified unchanged. Its newer concurrent report edits were preserved;
only archive notes, their hashes and the new run record were committed by this
cleanup. Subsequent README.md/INTEGRATION.md changes belong to the other writer
and were left untouched. No benchmark or fixture generation was rerun.
The hub DASHBOARD.md already had unrelated edits and was not changed; its owner
can link the historical import without changing the active experiment run/pin.

- Experiment bash build-notes.sh: exit 0; rebuilt notes/EXPERIMENT.pdf and
  notes/RESULTS.pdf. Report provenance records the new PDF page counts. No
  new visual inspection is claimed.
- Archive checks: five SHA-256 matches; ignored destination verified; active
  experiment source, pin, lock, snapshot and tolerances unchanged.
- scripts/check.sh --fast --through A12: exit 0, all gates passed. Ruff format:
  275 files already formatted; lint passed; mypy: no issues in 25 source files.
- Implementation tests: float64 1212 passed / 8 skipped (13.93s), float32
  1214 passed / 6 skipped (14.82s). Source doctests: 4 passed each.
- Acceptance tests: float64 4654 passed / 46 skipped (42.98s), float32
  4597 passed / 93 skipped (41.63s); two existing NumPy warnings per mode
  at test_a11_ph_direct_methods.py:78.
- git diff --check: clean. No new tests are needed for artifact relocation;
  no full non-fast gate was run.
- An empty untracked nvim.log from the prior diff display was preserved in
  the system temporary directory, not committed or discarded.

## Review setup and remaining work

The existing reviewer checkout is detached at 200c1f7 but still has stale
workflow/spec-test changes from B4.3's review (N-2). The independent-review
skill prohibits repairing it without maintainer authorization. Review setup
must be clean before the pending round can start; no verdict is claimed.

Step starting / previous-step commit:
200c1f7f4b64d147d5d40cbcba44f97e44d3c1f5. Original worktree base: cc0e30f.
Likelihood documentation, final Slice B audit and integration remain. The
experiment owns pruning measurements; library correctness tests stay here.
