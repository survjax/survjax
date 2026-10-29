# Synthetic Harrell concordance R reference

From the repository root, require R 4.4.2, survival 3.7-0 (R reports 3.7.0) and flexsurv 2.3.2; no additional packages.
Default output overwrites the committed CSV/JSON. Normal tests only read them, without R.

```sh
Rscript --vanilla scripts/fixtures/generate_concordance.R --check-only
Rscript --vanilla scripts/fixtures/generate_concordance.R /tmp/concordance-run1
Rscript --vanilla scripts/fixtures/generate_concordance.R /tmp/concordance-run2
diff -r /tmp/concordance-run1 /tmp/concordance-run2
(cd tests/fixtures/concordance && shasum -a 256 -c SHA256SUMS)
```

Seed 20261023 with Mersenne-Twister/Inversion/Rejection generates 100 heavily tied rows; six fixed leading rows guarantee all time-tie kinds.
Four fixed untied rows have 75% censoring and perfect/reversed/constant scores. All data are synthetic, untruncated; integer/quarter times preserve ties in both dtypes.
CSV records rows and three scores; JSON adds R results/provenance/MD5 hashes. The loader checks both payloads against `SHA256SUMS` before parsing.
The [R manual](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/concordance.html) documents the call. R concordant/discordant/tied.x map directly; their sum is comparable.
tied.y/tied.xy are excluded; event/censor time ties are comparable. The matching JSON for protected `tests/spec/fixtures/` is a maintainer handoff.
