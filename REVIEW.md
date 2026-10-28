# A8.d1 — R reference evidence and regeneration

Approved by "reviewer is ok; on next subslice, plan is approved" after
A8.5 e158e1e. Branch: a8-1-r-generator. Stop after this docs commit.

## Read in this order

1. docs/builtin-families.md, Comparisons with R: fixture scope, grid route,
   dtype checks, evidence limits and regeneration pointers. The former
   "checked later" sentence now links to this section.
2. docs/distributions.md: two obsolete likelihood-support statements now
   describe exact/right support without truncation and link to the guide.
3. PLAN.md: this execution and the pending protected-fixture handoff.

Audit of docs/api.md found no obsolete verification claims; it is unchanged.
No new Python examples, source/docstring changes, fixture updates, dependencies,
lockfile changes, tolerance changes or protected-path edits.

## Invariants and evidence

- Spec 10.4/11.1: only committed synthetic observations and R outputs support
  numerical claims. Counts, seeds, fixed vectors, parameter mappings and pinned
  versions match the fixture READMEs, generators and checked reference tests.
- Spec 3.3/12.1: explain active-dtype inputs, preserved float64 expectations,
  central tolerances and eager/jit/vmap comparisons already demonstrated by
  A8.4b/A8.5. This docs step does not rerun those reference suites.
- Spec 5/11.1: stored times, (141, 3) time-first group order, direct psurvreg
  grid route and its cross-check are distinguished from observed-time
  singleton/full-data likelihood checks. Related R paths are not independent
  oracles. Fixed-vector comparisons do not establish fitting accuracy.
- Spec 10.4/10.5: explain manifest-before-parse checks, no-write check-only,
  overwrite defaults, scratch destination, full two-run recipes, 17-digit
  exports and the separate historical exact-double round-trip check.
- Protected fixture delivery remains the maintainer's separate tester-repo
  handoff/sync. No through-A8 or Slice A completion claim.

## Actual checks

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/build_docs.sh`:
final exit 0; build succeeded in both separate dtype processes, fresh execution
with Sphinx warnings fatal. Log: /private/tmp/survjax-a8d1-docs-final.log.
Builtin page execution: 2.70s float32 / 2.51s float64.
Custom-family guide execution: 4.16s float32 / 4.31s float64.
All existing executable docs pages ran; the new shell block is illustrative
and was not executed. No R regeneration, full code gate or new unit tests.

Environment retries: the first sandboxed attempt failed downloading locked
docs dependencies (DNS); the escalated retry installed them and built both
modes. After correcting the extra stale sentence, the sandboxed final rebuild
failed when Jupyter bound a local port; the escalated final rebuild passed.
The logs contain IPKernelApp TCP-without-encryption notices from the local
kernel runtime; Sphinx itself reports no build warnings/errors on the passing
runs. Neither retry changed project dependency declarations or the lockfile.

`python3 /private/tmp/check_a8d1_html.py` after the completed final build:
PASS in both modes: 48 local links/anchors resolve, two new tables, dtype
outputs present, stale claims absent, both downloaded fixture READMEs
byte-identical to committed originals. An earlier check ran before the final
x64 writer completed and saw its stale output; the completed-build rerun above
passes. The docs recipe downloads are packaged by Sphinx, not broken links
outside the HTML tree.

Browser inspection: the IAB/browser connector was unavailable (empty browser
inventory). Safari accessibility inspection of the local x64 page confirmed
new headings, both tables, rendered AFT math, command block, README links,
navigation entries and the pending-handoff paragraph. Screenshot inspection
was interrupted by concurrent user Safari navigation to another checkout;
the next action was rejected as user-changed state. Stopped UI actions to
avoid interfering. Complete visual layout inspection in both modes remains
unverified; both final HTML trees were inspected structurally as above.

Final `uv run ruff format --check .`, `uv run ruff check .` (same tmp cache)
and `git diff --check` pass. No new deviations. Stop for review.
