# B4.3 deviations

## D-B4.3-001 — Acceptance compilation-reuse check changes shape

- Step: B4.3.
- Spec: rev 62 sections 3.2 and 7.2.
- What the spec says: equal metadata yields equal tree structures; compilation
  reuse requires unchanged shapes, dtypes, PyTree structure and static
  configuration.
- Evidence: tests/spec/test_observation_traced_metadata.py:539–544 constructs
  traced data using four-row TIME/EVENT constants, then concrete data using
  two rows. It asserts a single trace of the same jitted consumer. Both
  float64 and float32 focused runs fail only this test (124 passed, 1 failed
  in each). The datasets have equal tree structures but shapes (4,) and (2,).
  A separate reproduction using traced[:2] and the two-row concrete data
  records one trace; passing the original four-row data adds a second trace.
- Proposal: the independent test author should make the two datasets the
  same shape, preserving the intended traced/concrete metadata comparison.
  No production shape-specialization workaround is proposed. The protected
  test has not been edited, skipped or xfailed.
- Status: open. Await maintainer/test-author resolution before the gate and
  review can establish acceptance of B4.3.
- Follow-up: the maintainer supplied merge b4b6700 containing independent
  test sync 3d9704a (tester 96e2da5), which makes both datasets shape (4,).
  The maintainer instructed work to resume. Retain the historical entry and
  leave formal status disposition to the maintainer; rerun the corrected test.
