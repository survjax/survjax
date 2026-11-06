# Citing survjax

If survjax contributes to your work, cite the software and identify the exact
version or source commit that produced the results. The repository's
`CITATION.cff` contains machine-readable software metadata; GitHub can offer
citation formats from that file once it is on the default branch.

The author entry follows the package metadata: **Wouter A.C. van Amsterdam**.
The title is **survjax: Survival analysis in JAX**. A text citation can use:

```text
Wouter A.C. van Amsterdam. survjax: Survival analysis in JAX.
Software, version <installed version> or Git commit <full SHA>.
<repository URL for that source>.
```

Replace the angle-bracket fields with your actual source details. In the
Python environment used for the analysis, obtain the distribution version with:

```sh
python -c "from importlib.metadata import version; print(version('survjax'))"
```

For a source checkout, also record `git rev-parse HEAD` and any uncommitted
changes. A development version such as `0.0.1.dev0` can cover many commits;
a branch name or moving `latest` documentation URL alone does not identify
reproducible source. Keep Python/JAX/NumPy versions, seeds and example settings
with the analysis record. Cite other software and datasets used by your work
according to their own guidance as well.

## Repository location and release metadata

The preparation checkout records `https://github.com/vanAmsterdam/survjax`.
The maintainer has created the GitHub organization `survjax` and plans to
transfer the repository to **`https://github.com/survjax/survjax`**. The transfer
has not been performed by this preparation step. Use the canonical repository
URL verified for the source you used, together with its version/commit.
Hosting ownership and software author attribution are separate metadata.

Before publishing, confirm the agreed author list and canonical repository
URL. Update `CITATION.cff` with the actual released version and release date
when assigned, consistently with the package and release notes. DOI and ORCID
fields are currently omitted; add identifiers only when assigned and verified.
A proposed version such as `0.0.1a1` is not a published release or a DOI.

The [operator handoff](https://survjax.readthedocs.io/en/latest/releasing.html)
records the coordinated repository/publisher updates and remaining launch
checks. Citation guidance does not certify those checks or imply publication.
