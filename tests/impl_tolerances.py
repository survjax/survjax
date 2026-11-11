"""Implementer test tolerances (spec 12.1)."""

# check_grads uses finite differences with step 1e-4. Second-order differences
# in float32 amplify rounding for O(1) polynomial values/derivatives.
# At |f| ~= 6, eps32 * |f| / 1e-4 ~= 7e-3 is a conservative rounding
# scale; this exactly bilinear test needs only 1e-3 in both derivative modes.
# Grid checks use r=1 and t in {0.5, 1}: smooth values/derivatives are O(1).
# With step 1e-4, first differences of AD derivatives have rounding scale
# eps32/1e-4 ~= 1e-3 (eps64/1e-4 ~= 2e-12), within these existing bounds.
DERIVATIVE_ATOL = {True: 1e-7, False: 1e-3}
DERIVATIVE_RTOL = {True: 1e-7, False: 1e-3}

# Likelihood checks average three contributions at O(1) times/parameters.
# Centered differences have O(h**2) truncation and O(eps/h) rounding error,
# also when differencing the O(1) AD derivatives for the second-order check.
# h=1e-4 in x64 gives scales ~1e-8/2e-12; h=1e-3 in x32 gives ~1e-6/1e-4,
# within the existing derivative tolerances above, which are unchanged.
# B3.4 also averages six moderate truncated LogNormal/LogLogistic rows, with
# shape/scale parameters >=0.75 and positive finite endpoints in [0.25, 5]. The same
# O(h**2) / O(eps/h) finite-difference scales apply on these smooth interiors.
# Its independent 80-digit derivative comparisons use ordinary ATOL/RTOL.
# B3.5 uses moderate wrapper predictors (|eta|<=0.375) and averages 21 such
# rows, keeping the same O(h**2)/O(eps/h) scales for parameter differences.
# Its baseline/eta multiprecision Jacobians and Hessians use ordinary ATOL/RTOL.
LIKELIHOOD_GRAD_EPS = {True: 1e-4, False: 1e-3}

ATOL = {True: 1e-10, False: 1e-5}
RTOL = {True: 1e-8, False: 1e-4}

# D-022: absolute tolerances would accept zero for representable tiny tails.
# For H<=600 (x64), H<=80 (x32), log-H reconstruction contributes roughly
# H*log(H)*eps relative error, below these targets, including AD derivatives.
# A7 likelihoods depend on H itself, not exp(-H): relative power error scales
# as |log H|*eps, even at H=1e32. The selected values/gradients have |log H|<=74
# and no severe term cancellation, within the same 1e-8/1e-4 targets.
TAIL_ATOL = {True: 0.0, False: 0.0}
TAIL_RTOL = {True: 1e-8, False: 1e-4}

# Spec 12.1 RMST exception: compare k*dM/dk and b*dM/db errors against |M|,
# i.e. log-parameter units; cutoff derivative error has absolute scale 1.
# Pre-test 60-digit quadrature probes: 126 full-grid points (six shapes,
# seven ratios, three scales) plus 16 switch neighbors, both dtypes.
# Maximum parameter/cutoff errors were 1.72e-14/1.56e-15 (x64) and
# 1.29e-5/6.45e-7 (x32). Fixed factors allow modest rounding margin;
# these are absolute factors with rtol=0, not relative-to-tiny-derivative tests.
RMST_PARAMETER_ATOL = {True: 2e-13, False: 5e-5}
RMST_CUTOFF_ATOL = {True: 2e-13, False: 5e-6}

# Six binomial/delta-method MCSEs allow this fixed-seed family of empirical checks.
SIMULATION_SIGMAS = 6

# At the two-arm exponential MLE, H^-1=[[.5,-.5],[-.5,1]]. The specified
# score bound 1e-5 implies <=1.5e-5 log-parameter error locally; 2e-5 allows
# the O(score**2) remainder. This is solver accuracy, not value/AD accuracy.
SIMULATION_FIT_ATOL = 2e-5

# Mixed censoring at rate .5 has O(1/r^3) third derivatives. A smaller
# x64 finite-difference step keeps truncation error below 1e-7 in random
# projections; x32 retains 1e-3 to control cancellation at its precision.
CENSORING_GRAD_EPS = {True: 1e-5, False: 1e-3}
