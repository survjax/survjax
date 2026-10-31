# Mathematical appendices

These notes contain the mathematics behind specific code paths. They are
separate from user guides and from step plans, reviews and approval reports.
Keep derivations and numerical bounds aligned with the linked implementation;
keep workstream history and empirical check results in PLAN.md / REVIEW.md.

| Appendix | Code | Export |
| --- | --- | --- |
| [PH log-survival scaling](ph-log-survival.md) | [_scale_log_survival](../src/survjax/distributions/wrappers.py) | [PDF](ph-log-survival.pdf) |
| [Weibull RMST](weibull-rmst.md) | [_weibull_rmst](../src/survjax/_numerics.py) | [PDF](weibull-rmst.pdf) |

Rebuild each companion PDF after changing its mathematical source.

PH log-survival scaling:

```bash
pandoc appendices/ph-log-survival.md --standalone --pdf-engine=xelatex \
  --lua-filter=appendices/ph-log-survival-pages.lua \
  -o appendices/ph-log-survival.pdf
```

Plain Pandoc conversion also compiles; the optional filter puts each main
topic on a fresh page. These appendices are not included in the docs site.

Weibull RMST:

```bash
pandoc appendices/weibull-rmst.md --standalone --number-sections \
  --pdf-engine=xelatex -o appendices/weibull-rmst.pdf
```
