# ThetaScan paper

This directory contains the **preliminary Versioned Technical Paper / Public
Preview v0.1** (2026-07-24) of *ThetaScan: Scan-Parallel Nonlinear
Memory* in two equivalent release forms:

- [Markdown source](ThetaScan-Scan-Parallel-Nonlinear-Memory.md)
- [rendered PDF](ThetaScan-Scan-Parallel-Nonlinear-Memory.pdf)
- [version record and update policy](VERSIONS.md)

The architecture is still undergoing parameter search. This preview is a
citable research snapshot, not an archival or final paper; later numbered
versions may add experiments, algorithms, collaborators, and corrected claims.

The paper uses **ΘetaScan** as the display spelling, where the Greek capital
theta denotes the parameters of the reference memory. **ThetaScan** is the
deliberate ASCII spelling used by the repository, package, filenames, and
citation metadata.

The collective byline is **The ThetaScan Project**. It is a project byline, not
an anonymous-submission designation. Future versions may name collaborators
who opt in and make substantive research contributions.

## Citation

Until a DOI or archival venue identifier exists, cite the paper title, project
byline, version, year, and the exact repository release used:

```bibtex
@article{thetascan2026,
  title   = {ThetaScan: Scan-Parallel Nonlinear Memory},
  author  = {{The ThetaScan Project}},
  year    = {2026},
  version = {0.1},
  note    = {Preliminary versioned technical paper; public preview v0.1, 2026-07-24; accompanies ThetaScan software release 0.1.0}
}
```

Repository-level citation metadata is also available in
[`CITATION.cff`](../CITATION.cff).

## Rebuild the PDF

Install the paper-only rendering dependencies, then run the checked-in
renderer with a local Chrome, Chromium, or Edge executable:

```bash
python -m pip install -r paper/requirements-render.txt
python paper/render_paper.py
```

The renderer converts the Markdown to A4 HTML, typesets TeX with the
version-pinned MathJax 3.2.2 CDN asset, and stamps stable metadata, headers,
footers, and page numbers. Rendering therefore needs network access unless the
MathJax asset is supplied through an equivalent local build environment.
The PDF should still be rendered to page images and reviewed visually before a
release; the script cannot decide whether a table break is aesthetically good.

## Files kept in `paper/`

Python belongs in this directory only when it deterministically rebuilds a
paper artifact. The checked-in `render_paper.py` rebuilds the PDF from the
Markdown source. Future figure scripts are appropriate only when they recreate
checked-in figures from documented, immutable inputs. Experiment launchers,
training utilities, exploratory notebooks, and one-off data-conversion scripts
belong with the benchmark or source code, not with the paper.

## Copyright and licensing boundary

Copyright 2026 Ultimamind SRL, Belgium. All rights reserved.

The paper is provided for reading and citation. It is not software or
accompanying software documentation licensed under the repository's PolyForm
Small Business terms, and no separate permission to copy, adapt, or redistribute
the paper is granted merely by its inclusion here. The paper also grants no
patent license. See [`LICENSING.md`](../LICENSING.md) and
[`PATENTS.md`](../PATENTS.md) for the repository boundaries.

The paper is kept in the source repository and official source distribution so
the research claims and evidence links remain self-contained. It is excluded
from the installable Python wheel. Inclusion in either source artifact does not
change the copyright or licensing boundary stated above.
