# Agent: ProportionalHazards

Read the packet README and spec 4.1–4.2, 4.4–4.7, 5–7, 8.7, 10.2 and
12.1–12.2. This is the brought-forward PH release item, not AFT or full Slice D.
Own distributions/wrappers.py, distribution/top-level exports and dedicated
PH tests; reserve related import/install export assertions with the maintainer.

1. Plan the registered frozen PH wrapper: baseline plus log_hazard_ratio,
   nested pytree/broadcast/indexing behavior, real predictor constraint and
   baseline.-prefixed parameter issues. Implement direct methods, grids,
   reference-time/infinity handling and exact/right likelihood composition.
   Use PH formulas for direct methods/likelihood; summaries alone delegate.
   Split construction/diagnostics and numerical coverage if needed.
2. Add transformation, identity and invalid-row isolation evidence in another
   small step. Compare every available direct/grid/likelihood method with
   Exponential(r*exp(eta)) and Weibull(k, b*exp(-eta/k)), in values and first
   derivatives, plus required second-order interior checks. Include scalar,
   observation, draw and (S,1) baseline × (S,2) predictor broadcasting,
   zero/infinite times, custom baseline and mixed invalid parameters.
3. After core RMST merges, implement the private PH closure and delegated RMST.
   PH now owns any necessary built-in closure edits; RMST's agent stops editing
   those files. Test eta/baseline/tau derivatives, tail/boundary behavior and
   NotImplementedError for RMST on a custom baseline without closure.
4. Separate executable PH topic and integration evidence, coordinated with A9
   navigation. State eta>0 means higher hazard. No AFT, generic summaries,
   inverse cumulative hazard or sampling API.

Before adding the public export, resolve the API-reference test requirement:
the maintainer must explicitly authorize the narrow docs/api.md edit alongside
code, or approve another passing split. Do not weaken test_api_reference.py.
Before editing base.py/exponential.py/weibull.py, wait for the RMST ownership
handoff. Use safe inputs and log-space reasoning; naive exp(eta)*0 at boundaries
or saturated baseline hazards needs explicit numerical investigation.
Record and escalate contract questions rather than inventing extra behavior.

Final implementer evidence and remaining integration: [PH.8 audit](ph-evidence.md).
