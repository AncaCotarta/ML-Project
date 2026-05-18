"""
Generate reports/report.pdf from reports/report.md.

Pipeline:
  1. Preprocess markdown — replace missing images with a styled placeholder.
  2. Python-markdown → self-contained HTML (images embedded as base64).
  3. Chrome / Chromium headless → PDF.

Usage:
    python src/generate_pdf.py
"""

import base64
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "reports" / "report.md"
REPORT_DIR = ROOT / "reports"
REPORT_PDF = ROOT / "reports" / "report.pdf"
REPORT_HTML = ROOT / "reports" / "report.html"

# ── Locate the browser ────────────────────────────────────────────────────────
_BROWSER_CANDIDATES = [
    # macOS — Chrome
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    # macOS — Chromium
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    # macOS — Edge
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
    # Windows — Edge
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    # Windows — Chrome
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    # Linux
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/chromium-browser"),
    Path("/usr/bin/chromium"),
]


def find_browser() -> Path:
    for candidate in _BROWSER_CANDIDATES:
        if candidate.exists():
            return candidate
    # also try PATH
    for name in ("google-chrome", "chromium-browser", "chromium", "msedge"):
        found = shutil.which(name)
        if found:
            return Path(found)
    raise FileNotFoundError(
        "No Chromium-based browser found. Install Google Chrome or Chromium."
    )


# ── CSS stylesheet ─────────────────────────────────────────────────────────────
CSS = """
*, *::before, *::after { box-sizing: border-box; }

body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #1c1c1e;
    max-width: 760px;
    margin: 0 auto;
    padding: 56px 72px;
    background: #ffffff;
}

/* ── Headings — all Times New Roman ── */
h1 {
    font-family: 'Times New Roman', Times, serif;
    font-size: 20pt;
    font-weight: bold;
    color: #0d2237;
    border-bottom: 3px solid #2e7da6;
    padding-bottom: 10px;
    margin: 0 0 4px 0;
}
h2 {
    font-family: 'Times New Roman', Times, serif;
    font-size: 14pt;
    font-weight: bold;
    color: #0d2237;
    border-bottom: 1.5px solid #c8d8e4;
    padding-bottom: 5px;
    margin: 40px 0 12px 0;
}
h3 {
    font-family: 'Times New Roman', Times, serif;
    font-size: 12pt;
    font-weight: bold;
    color: #1a3a54;
    margin: 28px 0 8px 0;
}
h4 {
    font-family: 'Times New Roman', Times, serif;
    font-size: 11pt;
    font-weight: bold;
    font-style: italic;
    color: #2e5472;
    margin: 20px 0 6px 0;
}

/* ── Paragraphs & lists ── */
p { margin: 10px 0; text-align: justify; }
ul, ol { margin: 10px 0 10px 26px; }
li { margin: 5px 0; }
strong { font-weight: bold; color: #0d2237; }
em { font-style: italic; }

/* ── Tables ── */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    font-size: 9.5pt;
    font-family: 'Times New Roman', Times, serif;
}
thead tr {
    background: #2e7da6;
    color: #ffffff;
}
th {
    padding: 8px 12px;
    text-align: left;
    font-weight: bold;
    font-size: 9.5pt;
}
td {
    padding: 7px 12px;
    border-bottom: 1px solid #dde5ec;
    vertical-align: top;
}
tr:nth-child(even) td { background-color: #f4f8fb; }
tr:last-child td { border-bottom: 2px solid #2e7da6; }

/* ── Code blocks — keep monospace ── */
code {
    font-family: 'Courier New', Courier, monospace;
    font-size: 9pt;
    background: #f0f4f7;
    color: #b85c38;
    padding: 2px 5px;
    border-radius: 3px;
}
pre {
    background: #f8f9fa;
    border-left: 4px solid #2e7da6;
    padding: 14px 18px;
    margin: 16px 0;
    border-radius: 0 4px 4px 0;
    overflow-x: auto;
    line-height: 1.5;
}
pre code {
    background: none;
    color: #1c1c1e;
    padding: 0;
    font-size: 9pt;
}

/* ── Block quotes ── */
blockquote {
    border-left: 4px solid #e8a838;
    background: #fffbf0;
    margin: 16px 0;
    padding: 10px 18px;
    border-radius: 0 4px 4px 0;
    color: #6b5320;
    font-style: italic;
    font-size: 10pt;
}

/* ── Horizontal rule ── */
hr {
    border: none;
    border-top: 2px solid #c8d8e4;
    margin: 36px 0;
}

/* ── Images ── */
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 20px auto;
    border: 1px solid #dde5ec;
    border-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

/* ── Links — no underline in print ── */
a { color: #2e7da6; text-decoration: none; }

/* ── Print / PDF ── */
@page {
    size: A4;
    margin: 2.2cm 2.4cm 2.5cm 2.4cm;
}
@media print {
    body { padding: 0; max-width: none; }
    a { color: inherit; }
    h1, h2, h3, h4 { page-break-after: avoid; }
    img        { page-break-inside: avoid; max-width: 100%; }
    table      { page-break-inside: avoid; }
    pre        { page-break-inside: avoid; }
    tr         { page-break-inside: avoid; }
    blockquote { page-break-inside: avoid; }
}
"""

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Online Shoppers Revenue Prediction — Report</title>
  <style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def embed_image(path: Path) -> str:
    """Return an <img> tag with the image embedded as a base64 data URI."""
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
    }.get(suffix, "image/png")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:{mime};base64,{data}" alt="{path.name}">'


def preprocess_markdown(text: str, report_dir: Path) -> str:
    """
    Replace ![alt](path) with either an embedded <img> (if the file exists)
    or a styled blockquote placeholder (if not).
    """
    def replace(m: re.Match) -> str:
        alt = m.group(1)
        rel_path = m.group(2)
        abs_path = (report_dir / rel_path).resolve()
        if abs_path.exists():
            return embed_image(abs_path)
        return (
            f'\n> **[Figure not yet generated: _{alt}_]**  \n'
            f'> Run `python src/eda.py` and `python src/train_models.py` '
            f'to generate all figures.\n'
        )

    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace, text)


def md_to_html(md_text: str) -> str:
    """Convert Markdown to HTML using the `markdown` library."""
    try:
        import markdown
        return markdown.markdown(
            md_text,
            extensions=["tables", "fenced_code", "toc", "attr_list"],
        )
    except ImportError:
        print("  [warn] `markdown` library not found — pip install markdown")
        # Minimal fallback: wrap in <pre>
        return f"<pre>{md_text}</pre>"


def build_html(body: str) -> str:
    return HTML_TEMPLATE.format(css=CSS, body=body)


def html_to_pdf(html_path: Path, pdf_path: Path, browser: Path) -> bool:
    """Use headless Chrome/Edge to print the HTML to PDF."""
    result = subprocess.run(
        [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-extensions",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=5000",
            f"--print-to-pdf={pdf_path}",
            "--print-to-pdf-no-header",   # older Chrome
            "--no-pdf-header-footer",      # newer Chrome 112+
            f"file://{html_path.as_posix()}",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    return pdf_path.exists() and pdf_path.stat().st_size > 5_000


def main() -> None:
    print("=" * 60)
    print("  Report PDF generator")
    print("=" * 60)

    if not REPORT_MD.exists():
        raise FileNotFoundError(f"Report not found: {REPORT_MD}")

    # ── Step 1: preprocess markdown ───────────────────────────────────────────
    print("\n[1/4]  Reading and preprocessing Markdown…")
    raw = REPORT_MD.read_text(encoding="utf-8")
    processed_md = preprocess_markdown(raw, REPORT_DIR)

    total_imgs = len(re.findall(r'!\[', raw))
    missing = processed_md.count("Figure not yet generated")
    print(f"       {total_imgs - missing}/{total_imgs} figures found  |  "
          f"{missing} placeholder(s) inserted")

    # ── Step 2: convert to HTML ───────────────────────────────────────────────
    print("\n[2/4]  Converting Markdown → HTML…")
    body_html = md_to_html(processed_md)
    full_html = build_html(body_html)

    tmp_dir = Path(tempfile.mkdtemp())
    temp_html = tmp_dir / "report.html"
    temp_html.write_text(full_html, encoding="utf-8")
    html_kb = temp_html.stat().st_size // 1024
    print(f"       HTML ready ({html_kb} KB, images embedded as base64)")

    # Also save a permanent HTML copy for browser-based fallback
    REPORT_HTML.write_text(full_html, encoding="utf-8")

    try:
        # ── Step 3: locate browser ────────────────────────────────────────────
        print("\n[3/4]  Locating browser…")
        try:
            browser = find_browser()
            print(f"       Found: {browser}")
        except FileNotFoundError as exc:
            print(f"\n  {exc}")
            print(f"\n  Fallback: HTML report saved to  reports/report.html")
            print(f"  Open it in Chrome and use  File → Print → Save as PDF.")
            return

        # ── Step 4: render PDF ────────────────────────────────────────────────
        print("\n[4/4]  Rendering PDF via headless browser…")
        success = html_to_pdf(temp_html, REPORT_PDF, browser)

        if success:
            size_kb = REPORT_PDF.stat().st_size // 1024
            print(f"\n{'=' * 60}")
            print(f"  PDF generated successfully!")
            print(f"  → reports/report.pdf  ({size_kb} KB)")
            print(f"{'=' * 60}")
        else:
            print(f"\n  Browser did not produce a valid PDF.")
            print(f"  HTML report saved to  reports/report.html")
            print(f"  Open it in Chrome and use  File → Print → Save as PDF.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
