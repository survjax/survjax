Independent review: round 1/3

# BAYES.5 — Posterior predictive trial/KM check

Read PLAN.md BAYES.5, 02-details Posterior predictive/Keys/KM; model.py
predictive_trials; figures.py Figure 5/output metadata; check_figures.py's
independent generation and all-line checks; examples/bayesian_rmst/README.md.
Previous step/head 1472f52; original worktree base b1291a8. Same Bayesian
sub-slice, existing approved modules. 310 changed code/check/operational-doc
lines excluding PLAN/REVIEW: 255 changed Python lines plus 55-line README.
No module, API, dependency, protected file, core, docs/ or spec edit.

Invariants:
- Exactly 100 evenly spaced flattened posterior indices floor(i*S/100).
  From 4000 chain-major draws, indices 0,40,...,3960 cover each chain 25 times.
  This deterministic selection is recorded, not claimed as an independent
  posterior subsample or additional convergence evidence.
- Stream 3 = fold_in(fold_in(key(20261025),2),3); each replicate folds its
  index into stream 3. Shared generator alone owns the event/censor split.
- Preserve observed arm assignment and known original synthetic censor limit.
  Generate both event times and independent censor uniforms afresh. PH uses
  gamma*arm and equivalent scale b0*exp(-gamma*arm/k), as designed.
- Concrete host validation; not an outer jit/gradient API. Selected parameters
  explicitly converted to float64, x64 required. Reject malformed shapes,
  nonfinite parameters, absent/nonbinary arms, invalid c and unrepresentable
  natural shape/scale. No clipping, floors, dropping or reseeding. Latent
  infinite event times can exist from U=0 under the shared generator contract;
  observed times and censoring are finite; exact observed times must be >0.
- JAX vmap generates arrays (100,N), including event/censor latent arrays.
  Figure 5 shows all 100 per-arm KMs through their own last observed follow-up,
  then observed KM. It labels the known synthetic censor simplification.
- Save predictive_trials.npz with exact indices/time/event/latent fields;
  figures.json records key mapping, rule, indices and actual known c. Existing
  output directory remains protected. Renderer emits five SVG/PNG pairs.
- No posterior rerun or new KM estimator: reuse the reviewed KM validated
  against R. The predictive check is visual, not a new threshold or p-value.
- README records runnable commands, artifact contract, four-chain selection,
  RNG/censor assumptions, valid-failure saving vs hard input errors, and limits.
  Priors remain Normal(0,5), D-033 open, not resolved by this feature.

Evidence (2026-11-02):
- `JAX_ENABLE_X64=1 uv run --group examples python -m
  examples.bayesian_rmst.check_figures /private/tmp/bayes3-final-full-20261102`:
  dedicated predictive generation/key/domain checks plus existing/new figure
  checks pass, including final rerun after explicit float64 promotion.
- Independent NumPy inverse-CDF algebra on 100 variable-parameter replicates,
  N=4, with literal root/folds and split U/C keys: compare latent event and
  censor time, min observation and event flag, atol1e-12/rtol0. Exact indices,
  repeated-seed arrays, different fresh censoring, wrong-prior-stream contrast,
  dtype/shape, rejected malformed/log-overflow/NaN cases checked. Float32
  input parameters promoted to float64 and verified.
- Full-artifact checks: independent first/last replicate inverse-CDFs from the
  saved selected posterior parameters and original c; all 200 plotted replicate
  KM time/value arrays and both observed overlays match saved predictive trials;
  all generated arrays reload with allow_pickle=False; metadata indices/key
  folds/censor limit match inputs; all five figure pairs emitted. Existing
  original four-figure truth/panel/interval tests remain active.
- In-memory mutation probes (no source/worktree changes): wrong stream 2,
  first 100 instead of evenly spaced indices, negated treatment eta, renderer
  c=1 instead of original c, and predictive KM panels swapped. All five are
  detected by AssertionError in the dedicated checks.
- `JAX_ENABLE_X64=1 uv run --group examples python -m
  examples.bayesian_rmst.figures /private/tmp/bayes3-final-full-20261102
  /private/tmp/bayes5-figures-20261102`: exit 0. Figure 5 visually inspected:
  readable per-arm labels/legend, 100 light-blue KMs plus black observed KM,
  title explains known synthetic censoring; no clipping. Figures 1–4 are
  unchanged numerically and remain exercised by their checks.
- Independent predictive output reload: (100,300) arrays, indices first
  [0,40,80]/last[3880,3920,3960]; mean censor fraction .5412, replicate range
  .42–.623333. Known c=19.17253543619806 years. These are descriptive only,
  no fitted-data match/convergence/model-correctness claim.
- Ruff format/check pass: 145 files already formatted, All checks passed!
  mypy src: 22 files clean; git diff --check passes. Matplotlib temporary
  font-cache warnings are emitted, not hidden.
- No repeated full core gate, NUTS or R comparison: no core/test/spec changes,
  no change to reviewed actual-trial KM or numerical library. Existing R
  comparison evidence is carried in prior PLAN Docs to-do. No docs build yet.

Artifact output: /private/tmp/bayes5-figures-20261102, five figure pairs,
prior/predictive NPZ and figures.json. Input full run remains BAYES.3's saved
full posterior. Nothing generated is committed with code in this step.

Remaining substeps: commit reproducible full artifacts and executable docs
page; interpret prior curves/maintainer's D-033 choice. The posterior predictive
check is now implemented; no library sampling/clinical data/public KM added.
