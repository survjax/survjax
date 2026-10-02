# Simulation study guide

This illustrative guide shows one way to compose survjax construction,
diagnostics and likelihoods in a simulation study. It is non-normative and adds
no acceptance requirements. The example has not been validated as an executable
recipe; its solver, tolerances, pilot size and loop are choices to evaluate.
The [specification](../spec/SPECIFICATION.md) defines the library contract.

## Study design and checkpoints

Keep invalid observations, valid data without an estimable target, and failed
optimizer attempts distinct. For the exponential model below, all-censored
observations can be valid while having no finite log-rate MLE (spec section 11.4).

1. Pilot every scenario with a few replicates at the intended dataset size,
   precision, optimizer and settings. Sixteen replicates is only an example.
2. Summarize issues, unestimable targets, attempted fits and failure reasons.
   Investigate before expanding a study whose pilot is entirely invalid,
   unestimable or failing. A successful pilot cannot guarantee later success.
3. Retain failures in study denominators with reasons, scenario settings and
   reproducible keys. Record versions and settings; report total, attempted and
   successful fits separately.
4. Continue with fresh keys in batches and apply a study-specific continuation
   policy at checkpoints. Record design changes if pilot results are retained.
   Compiled stopping based on loop state is another option.

## Illustrative exponential fit

The example uses Slice A likelihood capabilities and JAX sampling. Validate
finite, strictly positive `rate` and `censor_time` settings before launching it.
Optimistix is an optional external dependency, outside the survjax core.
[Newton](https://docs.kidger.site/optimistix/api/root_find/) solves for a zero of
`grad(nll)` using its Jacobian, the objective Hessian. This illustrates second
parameter derivatives; [BFGS](https://docs.kidger.site/optimistix/api/minimise/)
instead updates an approximation from gradients. Neither choice promises
that differentiating through the solver is supported (spec section 7.3).

```python
import jax
import jax.numpy as jnp
import optimistix as optx  # optional, outside the survjax core
import survjax

N = 512  # example intended dataset size; keep it fixed for the pilot


def nll(log_rate, data):
    return -survjax.log_likelihood(
        survjax.Exponential.from_log(log_rate=log_rate), data
    ).sum()


score = jax.grad(nll)


@jax.jit
def replicate(key, rate, censor_time):
    event_time = jax.random.exponential(key, (N,)) / rate
    time = jnp.minimum(event_time, censor_time)
    event = event_time <= censor_time
    data = survjax.SurvivalData(time=time, event=event)  # traced ordinary construction
    issues = data.issues
    # For this model, valid data with any event have a finite log-rate MLE.
    attempt = issues.ok & jnp.any(data.exact)

    def fit(_):
        sol = optx.root_find(
            score,
            optx.Newton(rtol=1e-6, atol=1e-6),
            y0=jnp.zeros((), dtype=data.lower.dtype),
            args=data,
            max_steps=64,
            throw=False,
        )
        solver_ok = sol.result == optx.RESULTS.successful
        fit_ok = jnp.isfinite(sol.value) & (jnp.abs(score(sol.value, data)) < 1e-5)
        reason = jnp.where(solver_ok, jnp.where(fit_ok, 0, 4), 3)
        return sol.value, reason, sol.result

    def skip(_):
        reason = jnp.where(issues.ok, 2, 1)
        # solver_result is meaningful only when attempted is true.
        return (
            jnp.full((), jnp.nan, dtype=data.lower.dtype),
            reason,
            optx.RESULTS.successful,
        )

    estimate, reason, solver_result = jax.lax.cond(attempt, fit, skip, None)
    return dict(
        key=key,
        issues=issues,
        attempted=attempt,
        log_rate=estimate,
        reason=reason,
        solver_result=solver_result,
    )


keys = jax.random.split(jax.random.key(2026), 16)  # example pilot for one scenario
pilot = jax.jit(lambda ks: jax.lax.map(lambda k: replicate(k, 1.0, 2.0), ks))(keys)
pilot = jax.device_get(pilot)  # summarize and apply the study's policy on the host
```

Reason codes are local to this example: 0 accepted fit, 1 invalid observations,
2 no finite log-rate MLE, 3 solver failure, 4 failed post-fit check. Preserve
`solver_result` for attempted fits; consult the
[Optimistix diagnostics](https://docs.kidger.site/optimistix/api/solution/).
`throw=False` returns numerical failures as results, but does not make malformed
programs valid. Newton finds stationary points; this exponential objective has
a unique minimum when an event is present. Other models need their own
estimability and optimum checks. Pilot tolerances and step limits before use.

## Choosing how to process replicates

The unbatched [`lax.map`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html)
keeps the condition scalar per replicate and compiles the mapped body once.
[`vmap` over `lax.cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html)
converts batched conditions to selection and can defeat skipping the fit work.
A [`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
can also process replicates sequentially, and is useful when carrying state
between a fixed number of iterations. It avoids the large unrolled computation
that a Python loop inside `jit` can produce; it is not a universal speed rule.
Measure workload and memory use while preserving the study's control-flow
behavior when choosing a primitive; survjax does not mandate one.
