# Agent: staged RMST

Read the packet README and spec 4.5, 7.2–7.3, 8.7, 12.1–12.2 in full.
Own distributions/base.py, exponential.py, weibull.py, RMST-specific tests and
justified additions to tests/impl_tolerances.py until the PH closure handoff.
Do not edit wrappers.py or exports. No new public result/configuration types.

1. Add the staged base surface and Exponential RMST with tests. On unsupported
   families rmst raises NotImplementedError; mean/median/quantile raise on
   every family. No sample, inverse method, rmst_grid or options/info API.
   Exponential uses -expm1(-r*tau)/r with safe inactive/invalid inputs.
   Test tau=0, broadcasting, parameter issues and all-order supported derivatives.
2. Plan and implement Weibull values with explicit underflow/overflow handling;
   split value and derivative evidence if the line budget requires it. The
   incomplete-gamma expression alone is insufficient. Keep base/built-in files
   under the repository size guideline; ask before adding an unnamed helper
   module rather than silently creating another numerical layer.
3. Complete independent mpmath quadrature checks across k in [0.2,50] and
   tau/b in [1e-2,1e5], both dtypes, scale changes, k=1, tau=0 and invalid
   entries. Verify eager/jit/vmap, forward/reverse first derivatives in k,b,tau,
   d RMST/d tau=S(tau), finite supported Hessians and the shape-Hessian
   NotImplementedError contract. Keep shared-parameter derivatives of valid
   entries isolated from invalid entries. Derivative tolerances follow the
   measured, value-scaled exception in 12.1; never loosen existing tolerances.
4. Hand core ownership to PH for closure integration; then add a separate
   executable RMST docs topic with the explicit Weibull shape-derivative limit.

Values: invalid/negative/infinite tau gives NaN; tau=0 gives 0 with parameter
derivatives 0 and tau derivative 1. Parameter issues give NaN. No floors,
epsilons, generic quadrature tangent or custom higher-order promise.
Final evidence includes PH RMST once that separately reviewed integration lands.
