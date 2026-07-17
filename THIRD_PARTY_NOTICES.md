# Third-party notices

This file is included in ThetaScan source distributions and wheels. It records
the repository's direct external licensing boundaries; it is not an exhaustive
software bill of materials for every package that an installer may resolve.
Original ThetaScan code remains under ThetaScan's [LICENSE](LICENSE).

## Runtime and optional dependencies

| Component | Relationship to ThetaScan | Upstream license | Bundled here? |
|---|---|---|---:|
| [PyTorch](https://github.com/pytorch/pytorch) | Required runtime dependency, declared as `torch>=2.4` | [BSD-style license](https://github.com/pytorch/pytorch/blob/main/LICENSE) | No |
| [Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention) | Optional accelerator, declared as `flash-linear-attention`; ThetaScan imports its public Python kernel interfaces | [MIT License](https://github.com/fla-org/flash-linear-attention/blob/main/LICENSE), copyright 2023-2026 Songlin Yang, Yu Zhang, and Zhiyuan Li | No |
| [tomli](https://github.com/hukkin/tomli) | Python 3.10 compatibility extra for the optional benchmark launcher | [MIT License](https://github.com/hukkin/tomli/blob/master/LICENSE) | No |
| [setuptools](https://github.com/pypa/setuptools) | PEP 517 build backend, declared as `setuptools>=77.0.3` | [MIT License](https://github.com/pypa/setuptools/blob/main/LICENSE) | No |

Package installers obtain PyTorch, optional FLA/tomli, setuptools, and their
transitive dependencies as separate distributions. ThetaScan does not copy
their Python, C++, CUDA, or
Triton source and does not include their binaries. Those packages retain their
own licenses, copyright notices, and third-party notices; they are not
relicensed under ThetaScan's [LICENSE](LICENSE).

## Benchmark integration

The integration under `benchmarks/parameter-golf/` works with the MIT-licensed
[`openai/parameter-golf`](https://github.com/openai/parameter-golf) project.
Original ThetaScan adapter and configuration files remain under ThetaScan's
license. Any downloaded, copied, modified, or generated upstream-derived files
remain subject to the upstream MIT license, copyright notices, and transitive
third-party notices.

`benchmarks/parameter-golf/prepare_harness.py` is a mixed-boundary source file:
its original transformation and verification code is under ThetaScan's
license, while embedded fragments derived from the pinned Parameter Golf source
and the upstream-derived files it generates retain the MIT conditions above.
Generated-file license assignments are recorded explicitly in
`benchmarks/parameter-golf/adapter_manifest.json`.

The complete upstream notices retained for the source-transform adapter are in
[`licenses/parameter-golf-LICENSE`](licenses/parameter-golf-LICENSE) and
[`licenses/parameter-golf-THIRD_PARTY_NOTICES.md`](licenses/parameter-golf-THIRD_PARTY_NOTICES.md).

The optional strict Mamba-3 comparison installs
[`state-spaces/mamba`](https://github.com/state-spaces/mamba) separately at the
revision recorded in `benchmarks/parameter-golf/adapter_manifest.json`. Its
upstream code is [Apache-2.0 licensed](https://github.com/state-spaces/mamba/blob/main/LICENSE)
and is not redistributed by ThetaScan.

## Benchmark data

The benchmark downloader can fetch the revision-pinned
[`willdepueoai/parameter-golf`](https://huggingface.co/datasets/willdepueoai/parameter-golf)
FineWeb export and its tokenizer artifacts. They are not stored in this
repository or included in ThetaScan distributions. The dataset card identifies
the export as [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) and
also refers users to the upstream FineWeb attribution and
[Common Crawl Terms of Use](https://commoncrawl.org/terms-of-use). Users who
download, use, or redistribute those artifacts are responsible for preserving
the applicable attribution and terms.

## Paper rendering tools

The optional paper build is separate from the installed ThetaScan runtime. It
uses the following independently installed tools; none of their source or
binaries is bundled in this repository, wheel, or sdist:

| Component | Role | Upstream license |
|---|---|---|
| [Python-Markdown](https://github.com/Python-Markdown/markdown) | Markdown-to-HTML conversion | [BSD 3-Clause](https://github.com/Python-Markdown/markdown/blob/master/LICENSE.md) |
| [pypdf](https://github.com/py-pdf/pypdf) | PDF metadata and page composition | [BSD 3-Clause](https://github.com/py-pdf/pypdf/blob/main/LICENSE) |
| [ReportLab](https://docs.reportlab.com/install/open_source_installation/) | Header, footer, and page-number overlays | [BSD license](https://docs.reportlab.com/developerfaqs/#13-licensing) |
| [MathJax 3.2.2](https://github.com/mathjax/MathJax/tree/3.2.2) | TeX typesetting, fetched from jsDelivr at render time | [Apache License 2.0](https://github.com/mathjax/MathJax/blob/3.2.2/LICENSE) |

`paper/render_paper.py` also invokes a user-installed Chrome, Chromium, or Edge
executable. The browser is not downloaded or redistributed by ThetaScan; the
terms of the user's chosen distribution apply.

## Maintenance rule

This notice records the repository's known external licensing boundaries. It
must be updated whenever third-party source, data, model artifacts, kernels, or
other redistributable material is added. If a future ThetaScan distribution
vendors any third-party component rather than installing it separately, that
distribution must also retain the full license and copyright notices required
by that component.
