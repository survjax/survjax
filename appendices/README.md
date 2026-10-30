# Mathematical appendices

These notes contain the mathematics behind specific code paths. They are
separate from user guides and from step plans, reviews and approval reports.
Keep derivations and numerical bounds aligned with the linked implementation;
keep workstream history and empirical check results in PLAN.md / REVIEW.md.

| Appendix | Code | Export |
| --- | --- | --- |
| [Weibull RMST](weibull-rmst.md) | [_weibull_rmst](../src/survjax/_numerics.py) | [PDF](weibull-rmst.pdf) |

Rebuild the companion PDF after changing its mathematical source:

```bash
pandoc appendices/weibull-rmst.md --standalone --number-sections \
  --pdf-engine=xelatex -o appendices/weibull-rmst.pdf
```
