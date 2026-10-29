# Agent: Survival MNIST

Read the packet README, spec 9.1–9.2, 11.2, 12.1–12.2, shared conventions
and BOTH 03-survival-mnist files. Own examples/survival_mnist/, dedicated
optional-environment checks and docs/use-cases/survival-mnist.md.
Start after the shared generator/environment merge and A10 acceptance passes.
Needs neither PH nor RMST, so may run alongside simulation and Bayesian work.

1. Loader and splits: verified raw IDX downloads/cache, published first-fetch
   hashes and recorded SHA-256, deterministic official train/validation/test
   boundaries, exact global-index key scheme and stable smoke subsets.
   No MNIST data committed; ordinary tests use tiny synthetic IDX inputs and
   need no network. Include dataset attribution in example README and page.
2. Shared generator outcomes: fixed k=1.38, log b=log 10-0.2*digit, 50%
   censoring calibration based only on training-set digit frequencies.
   Compute calibration/oracle values in float64 and store them; train/evaluate
   the network in float32 in a separate process. Labels only generate outcomes
   and support oracle evaluation/plots; the network input is pixels alone.
3. Exact specified Equinox CNN + scalar learned log k, Optax Adam, minibatch
   data[idx], finite gradients at initialization and after step 1. Follow the
   prescribed final-bias initialization, partial-batch handling and epoch budget.
   Select checkpoints by validation likelihood only, ties to the earlier epoch.
4. Intercept-only fit in float64/evaluation in float32, oracle and public
   Harrell C with score -log_scale. Record cohort sizes and n=10000 C timing.
   CPU full run: CNN beats intercept-only on test likelihood and C; learned k
   within 5% of 1.38. Investigate failures before tuning; never select on test.
5. Commit metrics/history/provenance/figures in reviewable chunks and build the
   executable page: live 5k/1k/2k, one-epoch smoke; labelled committed full
   50k/10k/10k, ten-epoch results. Record warm-cache/network requirements and
   demonstrate smoke fits the docs timeout; inspect rendered HTML.

Verify current download/checksum documentation when implementing. Ask the
foundation owner for optional dependencies; do not change core, lockfiles,
shared docs configuration or the generator. No GPU, IBS/AUC or public KM scope.
