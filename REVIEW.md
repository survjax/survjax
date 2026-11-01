Independent review: round 1/3

# RMST.integration3 — Reviewed PH/RMST merge

Read first, in order:
1. PLAN.md's active RMST.integration3 section, including the queued docs/API
   follow-up, then plans/release-agents/ph-evidence.md's requirement map.
2. Compare parents 907d5cbaa0feed36f73554a0c483a5a2bffc47b6 (RMST.d2) and
   8acfb9f866e6e530b762724ed4ec4c65d3cf1230 (reviewed PH.8, verdict merge).
3. The conflict resolutions in PLAN.md, REVIEW.md and docs/index.md.
4. Fresh combined-tree checks and artifact evidence below.

The maintainer requested completion of this existing PH merge. Work only on
release-rmst; no separate branch-to-main merge, push or publication. Original
worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d. Step start and
previous step final are the RMST.d2 parent above. This is integration sub-step
1; guide/README/API edits are sub-step 2 and final handoff is sub-step 3.

## Scope and preservation invariants

Only the three records/index paths conflicted. Both complete unfinished
parent plans are retained verbatim, not archived or discarded. Earlier
REVIEW.md records remain at the two parent commits; this file records the
new step. Imported PH implementation, exports, tests, guide, numerical helper
rename, deviation reconciliation and appendices are unchanged from PH.8.
Home-page resolution keeps PH's feature prose/navigation plus RMST's prose
link and toctree entry. No new numerical or teaching example is authored.

Host assertions passed: both complete parent plans are substrings of PLAN.md;
all 30 other imported paths and all 23 source files are byte-identical to
PH.8, including incoming protected CLAUDE.md. README.md, docs/rmst.md and
implementation tolerances are identical to RMST.d2. Index equals PH.8 plus
the two-line RMST prose link and one toctree line. No unmerged index entries
or conflict markers; _numerics.py is absent and _rmst.py is present. No authored
protected-file/test/spec/tolerance/dependency change. Spec alignment remains
765c65bba9efb3af8ef4a13c1ecb8e302a47f346 (revision 40).

Tests for imported contracts are mapped in ph-evidence.md: PH construction,
diagnostics and direct/grid identities (4.1–4.7,5); ordinary likelihood
Hessians and invalid-entry isolation (6–7); RMST closure/independent integrals,
full-range first derivatives, supported Hessians, zero and custom-baseline
failure (8.7,12.1); installed exports/runtime-only package (10.2–10.3).
They run unchanged in the fresh combined gate; no acceptance test is replaced
by an implementer test.

## Fresh checks

- Ruff format: 138 files already formatted; Ruff lint: all checks passed;
  mypy: no issues in 22 source files. Staged whitespace check clean.
- scripts/check_wheel.sh --through A10: exit 0, all available gates passed.
  Implementer tests: float64 1282 passed /8 skipped, 124.92s; float32
  1287 passed /3 skipped, 115.92s. Source doctests: 4 passed per mode,
  1.01s /1.04s. Existing acceptance through A10: float64 2332 passed
  /28 skipped, 65.60s; float32 2322 passed /38 skipped, 60.48s.
  Guard passed. Full log: /private/tmp/rmst-ph-integration-gate.log.
- scripts/build_docs.sh invoked directly, exit 0, both separate dtype modes:
  13 source pages each, forced execution, Sphinx warnings fatal. Nine RMST
  cells and five PH cells execute. RMST execution: float32 7.91s /float64
  8.86s; PH: 4.64s /4.91s. Kernel TCP transport notices are emitted, but no
  Sphinx warning or notebook error. Playground source remains output/count-free.
- HTML host audit: both guide links/toctree entries and local links/anchors
  resolve; PH.rmst autodoc is present; 8 RMST and 5 PH output containers,
  zero error containers. RMST output text/shapes/derivatives and figure assets
  are present. Downloaded Weibull appendix PDF bytes equal appendices/ source.
  No full-browser viewport inspection claimed in this merge-only step.
- Audit gap: expecting all three RMST autodoc entries failed. PH's API entry
  exists, but native Exponential/Weibull member lists omit rmst. RMST.d2 N-3
  remains unfinished. Explicitly queued with docs sub-step 2, along with the
  native-only README and guide's obsolete PH-planned sentence. No claim of a
  finished combined documentation/release gate is made here.

## Retained installation evidence

Direct gate wheel:
/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.4VcAb8/survjax-0.0.1.dev0-py3-none-any.whl

Fresh source archive:
/private/tmp/survjax-rmst-ph-integration.e_hbnrwo/survjax-0.0.1.dev0.tar.gz

Archive-built wheel:
/private/tmp/survjax-rmst-ph-integration.e_hbnrwo/from-sdist/survjax-0.0.1.dev0-py3-none-any.whl

Both wheel SHA256s:
e9aa003d19c07d4290041a300a45288b7b8a942982dae204ceab57b4f19ec46d
Source archive SHA256:
ecce72f145b11658820b863cef1f9473403b432c48f0796d0666fdfb5f121e21

uv build --offline --no-python-downloads --sdist succeeded. Safely unpacked
with tarfile's data filter, then ran uv build --offline --no-python-downloads
--wheel against the unpacked directory, using uv's built-in backend. All
23 packaged source files equal this checkout (including py.typed, _rmst.py
and PH); obsolete _numerics.py absent. Direct and archive-built wheels have
31 identical members and identical complete bytes. Fresh hashes also equal
PH.8's retained artifacts. Manifest is alongside the archive as manifest.json.

SURVJAX_WHEEL set to the archive-built wheel, tests/install/test_wheel.py,
in separate JAX_ENABLE_X64 processes: float64 3 passed in 18.48s; float32
3 passed in 20.01s. These create fresh external environments, install only
locked runtime dependencies, verify exports/type marker/core independence,
evaluate installed PH RMST eager/jit and strictly type-check a custom family.
PH.8's direct isolated archive-to-wheel command required an uncached backend;
that failing route was not retried or claimed to pass here. This fresh build
uses its documented unpacked-directory route, with no added dependency.

Environment: CPU, Darwin 25.6.0 arm64, Python 3.12.9, JAX/jaxlib 0.11.2,
NumPy 2.5.3, pytest 9.1.1, mpmath 1.4.1, Ruff 0.16.9, mypy 2.3.1,
uv 0.11.19. No new platform/version matrix is inferred.

## Remaining and uncertainties

Sub-step 2 finishes executable PH RMST closures, cutoff axes, zero/parameter
derivatives, valid-entry isolation, custom-baseline failure, shape-derivative
limits, README links/availability and both native API member entries. Native
Exponential/Weibull RMST implementations are independent; PH delegates to
equivalent distributions. D-032's unrepresentable closure can yield NaN
with clean original diagnostics outside the accuracy domain; D-031's extreme
direct/likelihood derivative exception does not relax RMST. Zero Weibull
and PH-Weibull guarantees end at value and first derivatives.

Sub-step 3 records final combined evidence and all five decisions D-028,
D-029 (log-parameter error coordinates), D-030, D-031 and D-032; formal
statuses remain maintainer-owned. Independent PH/staged RMST acceptance is
pending maintainer delivery/sync (TESTMAP.md and Q-049); the A10 gate cannot
certify it. tests/spec remains read-only, with no skips/xfails/weakening.
No R fixture regeneration or new CPU benchmark; committed reference tests
run in the gate. No core numerical uncertainty introduced by this resolution.
Commit this existing merge, invoke independent-review exactly, then stop.
