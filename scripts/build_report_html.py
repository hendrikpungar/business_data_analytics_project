"""Render docs/report.md to a self-contained docs/report.html.

Figures are inlined as base64 so the output is a single portable file. Open it
in any browser and use *Print → Save as PDF* for the submission PDF (clean,
faithful rendering without a local LaTeX/pandoc toolchain).

    python scripts/build_report_html.py
"""
import base64
import re
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parents[1] / "docs"
SRC, OUT = DOCS / "report.md", DOCS / "report.html"

CSS = """
@page { size: A4; margin: 18mm; }
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
       max-width: 850px; margin: 24px auto; padding: 0 16px; line-height: 1.5;
       color: #1a1a1a; font-size: 15px; }
h1 { font-size: 26px; color: #08306b; margin-bottom: 4px; }
h2 { font-size: 19px; color: #08519c; border-bottom: 2px solid #c6dbef;
     padding-bottom: 4px; margin-top: 30px; }
h3 { font-size: 15.5px; color: #2171b5; margin-top: 20px; }
p, li { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
th { background: #08519c; color: #fff; padding: 6px 9px; text-align: left; }
td { border-bottom: 1px solid #e0e0e0; padding: 5px 9px; }
tr:nth-child(even) td { background: #f5f8fc; }
img { max-width: 78%; display: block; margin: 14px auto; }
blockquote { color: #555; border-left: 3px solid #c6dbef; padding-left: 12px;
             font-style: italic; background: #fafcff; }
code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }
hr { border: none; border-top: 1px solid #ddd; margin: 24px 0; }
h2 { page-break-after: avoid; } img { page-break-inside: avoid; }
"""


def inline_images(html: str) -> str:
    def repl(m):
        src = m.group(1)
        path = (DOCS / src).resolve()
        if not path.exists():
            return m.group(0)
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}"'
    return re.sub(r'<img src="([^"]+)"', repl, html)


def main():
    body = markdown.markdown(SRC.read_text(encoding="utf-8"),
                             extensions=["tables", "fenced_code", "sane_lists"])
    body = inline_images(body)
    html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>Credit-Card Customer Analytics — Report</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB, images embedded)")


if __name__ == "__main__":
    main()
