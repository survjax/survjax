# Release-demo evidence audit — DEMO.E1

Audited at dispatch main `cc0e30f`, against specification revision 47,
sections 10.3–10.5, 11.2–11.3, 12.1 and 12.2 item 4, and the three
use-case main/details documents. This is a demo evidence index, not release
closure, independent A11/A12 acceptance, or documentation/hosting certification.
All commits below are ancestors of dispatch main. No full study, training or
NUTS run was repeated for this audit; no saved numerical result was changed.

## Scope decisions that still matter

The design files have not incorporated every maintainer decision. Do not
interpret the resulting differences as missing work to restore automatically.
[DEVIATIONS.md](../DEVIATIONS.md) records authorized departures, with formal
statuses still maintainer-owned:

- D-045: separate overlapping failure flags and one unique failed count.
- D-046: explicitly shared same-size scenario timings, not per-scenario times.
- D-047: compact simulation evidence replaces committed successful-fit rows.
- D-039: Bayesian priors are all Normal(0, 5), changed by the maintainer at
  `3c554be`; the narrower use-case priors are not the fitted model.
- D-041: approved lean Bayesian delivery drops prior/posterior predictive
  figures, KM helpers, tail ESS, live sampling smoke and committed figures.
  Figures/table are computed from committed posterior draws on the page.
- [Archived MNIST instruction](../plans/mnist-showcase-instruction.md),
  sections 3–5, explicitly simplifies the runner and metadata. The retained
  finite-gradient check is a separate optional test, not a runner guard.
  D-036 records intercept-fit initialization of both shape and output bias.

Consequently, the current deliveries support the approved demo scope; they do
not literally implement every unchanged use-case requirement. Formal spec and
deviation reconciliation remains a maintainer action, not an inferred waiver
of unrelated requirements.

## Weibull simulation

[Full bundle](weibull_simulation/results/README.md) was generated at clean
`ac0fa01` and committed by `8349e7f`. This is the revision-47 fixed-limit run,
not the older calibrated-censoring study. Later `b757e88` simplified the runner
without replacing these recorded results. `8349e7f:REVIEW.md` records the exact
full command, bundle checks and both-mode docs build; it explicitly says no
independent review ran for that regeneration.

| Requirement | Recorded evidence |
| --- | --- |
| Six scenarios, 1000 replicates, two models, float64 | metadata.json: n=100/300/1000, c=50/20, root 20261025, CPU/x64, solver/settings/keys |
| Coverage at n=1000 | performance.csv: Weibull .9425887265 and .9359504132, respectively .9861 and 1.7853 observed MCSE from .95; both <3 |
| Failures retained | 12,000 attempted, 11,863 accepted, 137 rejected; failures.csv lists all rejected IDs and flags, all gradient-only |
| Sampling and illustration | sampling_histograms.json retains accepted counts; illustration.csv retains both fits for scenario 3, replicate 0 |
| Outputs | performance.csv and three SVG figures; compact storage authorized by D-047 |
| Timing/provenance | metadata.json records Apple M4 Max, Python/packages, source/lock/spec identity, invocation, thread settings, synchronized first/warm shared-batch timing |

The five output SHA-256 values in evidence.json match current files. Its
original raw replicates.csv hash is provenance, not verification of a currently
committed raw file. Successful-fit raw diagnostics cannot be independently
recomputed from histograms. Original reducer source is recoverable at the
recorded commit; it is no longer the current runner. Do not relabel the stored
shared first-call timings as per-scenario compilation measurements.

## Bayesian RMST

`021690a:REVIEW.md` records the full four-chain 1000-warmup/1000-draw run,
including fixed-parameter factor wiring and routes A/B. `55565fd` committed
compact posterior data and the older full metadata/figures. The maintainer's
`add507a` lean rewrite supplies the current [results](bayesian_rmst/results/metadata.json).
The seven retained NPZ arrays are exactly equal to their counterparts at
`55565fd`: trial time/event/arm, tau grid and three (4,1000) parameter arrays.
This links the current draws to the historical run without resampling.

| Requirement | Recorded evidence |
| --- | --- |
| Trial/model | 300 rows, 150 per arm; c=19.1725354362, 164 censored; root 20261025 with use-case/stream folds in model.py; Normal(0,5) priors |
| Factor wiring | run.py check_factor uses fixed parameters, prior plus one summed likelihood, atol=1e-10; historical full-run audit at 021690a |
| RMST routes | recorded max absolute difference 2.8421709430e-14, below 1e-12; saved tau grid has 101 points ending at 13 years |
| Current diagnostic gate | max split R-hat 1.0035545696; min NumPyro ESS 2158.5783; zero divergences |
| Truth intervals | HR [.5941096753,1.1688700067] contains .7; delta RMST [-.0802711574,.2743181956] contains .1701579210 |
| Per-draw summaries | 55565fd metadata records P(delta>0)=.84725 and all five summary means/SDs/intervals; current page computes from the retained draws |
| Environment/timing | current metadata records JAX/jaxlib .11.2, NumPy 2.5.3, NumPyro .22.0; 55565fd metadata records CPU/x64, parallel chains, sampling 2.2486s and postprocessing 5.0849s including compile |

Historical metadata at `55565fd` also records tail ESS minimum 2250.7073 and
zero divergences per chain. NumPyro ESS must not be relabelled as independently
verified rank-normalized bulk ESS. Tail ESS and predictive outputs are absent
from the current delivery intentionally under D-041. `b45f511:REVIEW.md`
records the earlier 100-replicate predictive KM checks and figure inspection;
`55565fd` preserves the historical five figures. Those establish historical
work, not execution of predictive checks by today's lean runner.

The optional `tests/unit/check_bayesian_run.py` resamples and compares saved
draws, but is not collected by the ordinary test filename pattern. It was not
rerun here: the retained arrays and recorded diagnostics provide the needed
bookkeeping evidence. Historical CPU model/source cleanliness are not fully
specified by the current JSON; do not invent them or assign the alpha-2
package version to these development-version runs.

## Survival MNIST

`365c173` contains the earlier full experiment. The current metrics and figures
were regenerated in `36c290e` after the approved showcase rewrite; use that
run, not its predecessor, to explain the displayed results. Its REVIEW.md
records the full command, unchanged fixed settings, no test-based tuning and
28 passing optional checks in 19.71s. Later `7a5670b` shares the compiled
concordance function; it does not replace the saved full results.

| Requirement | Recorded evidence |
| --- | --- |
| Full cohorts | metadata.json: 50,000 train / 10,000 validation / 10,000 official test; 10 epochs in history.csv |
| Validation-only selection | epoch 7 is the minimum validation NLL in history.csv; model.train selects only validation NLL, earlier epoch on ties |
| CNN beats intercept | metrics.csv: test mean LL -1.126855135 > -1.237543344; C .689008653 > .5 |
| Shape | 1.368050814, relative error .866% from 1.38, within 5% |
| Oracle gap | oracle LL -1.110933185, C .697534919; CNN shortfalls .015921950 and .008526266 |
| Finite gradients | check_mnist_training.py differentiates the actual first smoke minibatch at initialization and after one step; 36c290e records its passing run |
| Figures | committed fig1.svg–fig3.svg; first image per digit, predicted scales by digit, epoch curves with selected epoch |
| Runtime | metadata.json records Python 3.12.9, macOS arm64, packages; training 19.2675s, warmed 10k-test C .3479/.3143/.3416s |

Finite-gradient evidence is specifically the real smoke minibatch check, not
a record of every gradient in the full run. Data source, four SHA-256 hashes,
root/folds, float32 scaling, split and outcome rules live in data.py;
architecture/Adam(1e-3)/batch128/dropped partial batches live in model.py.
README gives acquisition-by-run command/cache and MNIST attribution. Missing
copies of those constants in metadata were expressly authorized, not lost
results. The run's JSON identifies CPU only as 'arm'; no precise CPU model,
source-cleanliness or execution-time cache-hash receipt is claimed.

## Audit checks and handoff

This audit checked dispatch ancestry, five simulation manifest hashes,
scenario/failure accounting and coverage arithmetic, Bayesian retained-array
identity and saved diagnostic/truth thresholds. Existing MNIST full-result
acceptance test passes (1 test); no training or download needed. REVIEW.md
records the fast A10 gate outcome separately. These checks validate recorded
evidence, not fresh full-demo execution or independent A11/A12 acceptance.

No missing numerical full run or new core implementation gap was established
within the approved scope. Do not schedule expensive reruns just to restore
intentionally removed metadata or outputs. If literal original Bayesian gates
are reinstated, restore the predictive workflow and specify the required ESS
estimator first; historical draws can support diagnostic recomputation without
new NUTS. Changes to approved demo scope require maintainer direction.

Handoff to final-docs-evidence: verify current rendered pages consume these
specific full bundles and label live computations correctly. Weibull/MNIST live
smoke is not full-run evidence; Bayesian live analysis of saved draws is not
live NUTS. Verify figures, cohort labels, limitations and notebook execution
there. Global closure owner should reconcile approved departures with the
normative text and keep demo evidence separate from alpha-2 publication,
hosting and independent A11/A12 gate completion. This audit does not resolve
open deviations, update the roadmap, or certify those other workstreams.
