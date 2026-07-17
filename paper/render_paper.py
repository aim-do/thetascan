"""Render the release paper from Markdown with a local Chromium browser.

The renderer intentionally lives beside the paper rather than in the runtime
package. It produces an A4 PDF, waits for pinned MathJax rendering, then stamps
stable project headers, footers, page numbers, and PDF metadata.
"""
from __future__ import annotations

import argparse
import html
import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import markdown
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, NameObject
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


PAPER_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = PAPER_DIR / "ThetaScan-Scan-Parallel-Nonlinear-Memory.md"
DEFAULT_OUTPUT = PAPER_DIR / "ThetaScan-Scan-Parallel-Nonlinear-Memory.pdf"
ALLOWED_LINK_SCHEMES = {"http", "https", "mailto"}


CSS = r"""
@page {
  size: A4;
  margin: 19mm 18mm 20mm 18mm;
}
html {
  color: #151515;
  background: #ffffff;
}
body {
  margin: 0;
  font-family: "Times New Roman", "Liberation Serif", Georgia, serif;
  font-size: 9.65pt;
  line-height: 1.31;
  text-align: left;
  text-rendering: optimizeLegibility;
}
a { color: #174f75; text-decoration: none; }
p { margin: 0 0 0.66em; orphans: 3; widows: 3; }
strong { font-weight: 700; }
em { font-style: italic; }
h1, h2, h3 { color: #111; page-break-after: avoid; break-after: avoid-page; }
h1 {
  margin: 9mm 0 2.5mm;
  font-size: 26pt;
  line-height: 1.03;
  letter-spacing: -0.35pt;
}
h1 + h3 {
  margin: 0 0 2.4mm;
  font-size: 13.5pt;
  line-height: 1.15;
  font-weight: 400;
  color: #333;
}
h1 + h3 + p {
  margin-bottom: 4mm;
  font-size: 10.6pt;
}
h2 {
  margin: 5.2mm 0 2.3mm;
  padding-bottom: 1.1mm;
  border-bottom: 0.35pt solid #999;
  font-size: 17pt;
  line-height: 1.08;
}
h3 {
  margin: 3.8mm 0 1.6mm;
  font-size: 12.6pt;
  line-height: 1.12;
}
blockquote {
  margin: 2.5mm 0 4.5mm;
  padding: 3mm 4mm;
  border-left: 1.5pt solid #555;
  background: #f3f3f3;
  page-break-inside: avoid;
}
blockquote p { margin: 0 0 0.65em; }
blockquote p:last-child { margin-bottom: 0; }
hr {
  border: 0;
  border-top: 0.35pt solid #aaa;
  margin: 4.5mm 0;
}
ul, ol { margin: 0.3em 0 0.75em 1.5em; padding: 0; }
li { margin: 0.14em 0; }
table {
  width: 100%;
  margin: 2.2mm 0 3.5mm;
  border-collapse: collapse;
  font-size: 7.75pt;
  line-height: 1.16;
  page-break-inside: avoid;
}
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th, td {
  padding: 1.25mm 1.35mm;
  border: 0.35pt solid #aaa;
  vertical-align: top;
}
th { background: #ededed; font-weight: 700; }
code {
  font-family: "Cascadia Mono", Consolas, "Courier New", monospace;
  font-size: 0.88em;
  color: #202020;
}
pre {
  margin: 2mm 0 3mm;
  padding: 2.5mm 3mm;
  border: 0.35pt solid #c5c5c5;
  background: #f6f6f6;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  page-break-inside: avoid;
}
pre code { font-size: 7.8pt; }
mjx-container[jax="CHTML"] { font-size: 91% !important; }
mjx-container[jax="CHTML"][display="true"] {
  margin: 0.72em 0 !important;
  page-break-inside: avoid;
  overflow: visible !important;
}
h2#references ~ p,
h2#references ~ ol,
h2#references ~ ul {
  font-size: 8.25pt;
  line-height: 1.21;
  margin-bottom: 0.36em;
}
"""


def _find_chrome(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for command in ("chrome", "google-chrome", "chromium", "chromium-browser", "msedge"):
        found = shutil.which(command)
        if found:
            candidates.append(Path(found))
    candidates.extend(
        Path(path)
        for path in (
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Chromium browser not found; pass --chrome PATH")


def _html_document(source: str) -> str:
    # The PDF release uses plain ASCII hyphens for robust font/search behavior.
    source = source.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    # Python-Markdown otherwise interprets TeX underscores and adjacent stars as
    # emphasis. Protect math until the prose and tables have been converted.
    display_math: list[str] = []
    inline_math: list[str] = []

    def protect_display(match: re.Match[str]) -> str:
        token = f"THETASCANMATHBLOCK{len(display_math):04d}END"
        display_math.append(match.group(1))
        return f"\n\n{token}\n\n"

    def protect_inline(match: re.Match[str]) -> str:
        token = f"THETASCANMATHINLINE{len(inline_math):04d}END"
        inline_math.append(match.group(1))
        return token

    source = re.sub(r"\$\$(.*?)\$\$", protect_display, source, flags=re.DOTALL)
    source = re.sub(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$", protect_inline, source)
    body = markdown.markdown(
        source,
        extensions=("extra", "sane_lists", "toc"),
        output_format="html5",
    )
    for index, tex in enumerate(display_math):
        token = f"THETASCANMATHBLOCK{index:04d}END"
        rendered = (
            '<div class="math-display">$$'
            + html.escape(tex.strip(), quote=False)
            + "$$</div>"
        )
        body = body.replace(f"<p>{token}</p>", rendered).replace(token, rendered)
    for index, tex in enumerate(inline_math):
        token = f"THETASCANMATHINLINE{index:04d}END"
        rendered = (
            '<span class="math-inline">$'
            + html.escape(tex, quote=False)
            + "$</span>"
        )
        body = body.replace(token, rendered)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ThetaScan: Scan-Parallel Nonlinear Memory</title>
<style>{CSS}</style>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$']],
    processEscapes: true,
    tags: 'none'
  }},
  chtml: {{ scale: 0.94 }},
  options: {{ enableMenu: false }}
}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"></script>
</head>
<body>
{body}
</body>
</html>
"""


def _print_html(chrome: Path, html_path: Path, raw_pdf: Path, profile: Path) -> None:
    command = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--allow-file-access-from-files",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=20000",
        "--no-pdf-header-footer",
        "--print-to-pdf-no-header",
        f"--user-data-dir={profile}",
        f"--print-to-pdf={raw_pdf}",
        html_path.resolve().as_uri(),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if completed.returncode != 0 or not raw_pdf.is_file():
        raise RuntimeError(
            "Chromium PDF render failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _sanitize_link_annotations(page) -> None:
    """Drop local/relative URI links that Chromium derives from temporary HTML.

    Repository-relative Markdown links remain useful in the Markdown paper, but
    when the HTML is printed from a temporary directory Chromium resolves them
    to private ``file:///...`` paths. Keep normal web/mail links and internal
    PDF destinations; make unresolved repository links non-clickable in PDF.
    """
    annotations = page.get("/Annots")
    if not annotations:
        return
    kept = ArrayObject()
    for reference in annotations:
        annotation = reference.get_object()
        action = annotation.get("/A")
        uri = action.get("/URI") if action is not None else None
        if uri is not None:
            scheme = urlsplit(str(uri)).scheme.lower()
            if scheme not in ALLOWED_LINK_SCHEMES:
                continue
        kept.append(reference)
    if kept:
        page[NameObject("/Annots")] = kept
    else:
        page.pop(NameObject("/Annots"), None)


def _validate_link_annotations(pdf: Path) -> None:
    """Fail the release render if any URI annotation is local or unresolved."""
    invalid: list[str] = []
    for page_number, page in enumerate(PdfReader(pdf).pages, start=1):
        for reference in page.get("/Annots", ()):
            annotation = reference.get_object()
            action = annotation.get("/A")
            uri = action.get("/URI") if action is not None else None
            if uri is None:
                continue
            value = str(uri)
            if urlsplit(value).scheme.lower() not in ALLOWED_LINK_SCHEMES:
                invalid.append(f"page {page_number}: {value}")
    if invalid:
        raise RuntimeError(
            "PDF contains local or unresolved link annotations:\n"
            + "\n".join(invalid)
        )


def _validate_page_stamps(pdf: Path) -> None:
    """Require the complete release header and page number on every page."""
    reader = PdfReader(pdf)
    page_count = len(reader.pages)
    missing: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for expected in (
            "ThetaScan - Public Preview v0.1",
            "The ThetaScan Project",
            f"{page_number} / {page_count}",
        ):
            if expected not in text:
                missing.append(f"page {page_number}: {expected}")
    if missing:
        raise RuntimeError("PDF page stamp validation failed:\n" + "\n".join(missing))


def _stamp_pdf(raw_pdf: Path, output: Path) -> None:
    reader = PdfReader(raw_pdf)
    writer = PdfWriter()
    page_count = len(reader.pages)
    for number, page in enumerate(reader.pages, start=1):
        _sanitize_link_annotations(page)
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        buffer = io.BytesIO()
        stamp = canvas.Canvas(buffer, pagesize=(width, height), pageCompression=1)
        stamp.setFillColorRGB(0.34, 0.34, 0.34)
        stamp.setFont("Times-Roman", 6.8)
        stamp.drawRightString(
            width - 18 * mm,
            height - 9.3 * mm,
            "ThetaScan - Public Preview v0.1",
        )
        stamp.drawString(18 * mm, 7.7 * mm, "The ThetaScan Project")
        stamp.drawRightString(width - 18 * mm, 7.7 * mm, f"{number} / {page_count}")
        stamp.save()
        buffer.seek(0)
        overlay = PdfReader(buffer).pages[0]
        page.merge_page(overlay)
        writer.add_page(page)

    writer.add_metadata(
        {
            "/Title": "ThetaScan: Scan-Parallel Nonlinear Memory",
            "/Subject": "Slow dictionaries, fast nonlinear memories, and associative-scan accumulation",
            "/Author": "The ThetaScan Project",
            "/Creator": "paper/render_paper.py",
            "/Producer": f"pypdf {__import__('pypdf').__version__}",
            "/Keywords": "ThetaScan, nonlinear memory, associative scan, Gauss-Newton, Nadaraya-Watson, random feature expansion, language models",
            "/CreationDate": "D:20260724000000+02'00'",
            "/ModDate": "D:20260724000000+02'00'",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        writer.write(handle)
    os.replace(temporary, output)
    _validate_link_annotations(output)
    _validate_page_stamps(output)


def render(source: Path, output: Path, chrome: Path, keep_html: Path | None) -> None:
    source_text = source.read_text(encoding="utf-8")
    html = _html_document(source_text)
    with tempfile.TemporaryDirectory(prefix="thetascan-paper-") as directory:
        work = Path(directory)
        html_path = work / "paper.html"
        raw_pdf = work / "paper-raw.pdf"
        profile = work / "chrome-profile"
        html_path.write_text(html, encoding="utf-8")
        if keep_html is not None:
            keep_html.parent.mkdir(parents=True, exist_ok=True)
            keep_html.write_text(html, encoding="utf-8")
        _print_html(chrome, html_path, raw_pdf, profile)
        _stamp_pdf(raw_pdf, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chrome", default=None, help="path to Chrome/Chromium/Edge")
    parser.add_argument("--keep-html", type=Path, default=None)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    chrome = _find_chrome(args.chrome)
    render(source, output, chrome, args.keep_html)
    print(f"rendered {output}")


if __name__ == "__main__":
    main()
