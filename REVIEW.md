# DOCS.3 — Attributed teaching link

Read docs/working-with-distributions.md after its method table. Added the
approved callout naming Wouter van Amsterdam in its title and body, linking
the independent Circle of survival article at the requested personal domain.
No executable cells, API, dependencies or other pages changed.

Starting/previous step: c95fdca; original worktree base: 2093ca2.
Independent review remains waived. Validation complete. No remaining substeps
beyond verification and commit. User review/structure files remain untouched.

Validation: direct escalated `scripts/build_docs.sh` exited 0, with
`build succeeded.` in float32 and float64. `/tmp/check_circle_link.py` verified
both rendered outputs: author/title, exact external URL, placement after the
methods table and before Constructing a model, and byte-identical rendered
source. Executable cells match the previous commit. `git diff --check` clean.
No new unit tests, numerical gate or browser visual inspection for this prose
callout; rendered HTML structure checked programmatically. No remaining work.
