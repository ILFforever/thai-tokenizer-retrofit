"""Render results/REPORT.md as a readable single HTML page.

    python src/build_report_page.py [--dashboard-url URL]

Writes results/report.html. The markdown stays the source of truth; this is
only its typeset form, with a contents list and the same visual tokens as
the dashboard.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).parent.parent
SRC = ROOT / "results" / "REPORT.md"
OUT = ROOT / "results" / "report.html"

STYLE = """
:root {
  color-scheme: light;
  --bg: #f4f5f2; --surface: #fcfcfb; --ink: #111512; --ink-2: #4b504c; --muted: #7d8179;
  --rule: #dfe1da; --accent: #2a78d6; --accent-soft: #e6f0fb; --code: #eceee8;
  --display: "Bai Jamjuree", "IBM Plex Sans Thai", system-ui, sans-serif;
  --body: "IBM Plex Sans Thai", "Noto Sans Thai", "Leelawadee UI", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark; --bg: #0e100e; --surface: #181a18; --ink: #f1f3ef; --ink-2: #c3c5bc;
    --muted: #8d9189; --rule: #2c2f2b; --accent: #3987e5; --accent-soft: #16263b; --code: #242723;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark; --bg: #0e100e; --surface: #181a18; --ink: #f1f3ef; --ink-2: #c3c5bc;
  --muted: #8d9189; --rule: #2c2f2b; --accent: #3987e5; --accent-soft: #16263b; --code: #242723;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 400 16px/1.7 var(--body); -webkit-font-smoothing: antialiased; }
.layout { max-width: 1180px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 80px; display: grid; grid-template-columns: 230px minmax(0, 1fr); gap: 48px; }
aside { position: sticky; top: calc(env(safe-area-inset-top, 0px) + 24px); align-self: start; max-height: calc(100vh - 48px); overflow-y: auto; font-size: 13.5px; }
aside .eyebrow { font: 500 11.5px/1.4 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }
aside ol { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }
aside a { display: block; padding: 4px 8px; border-radius: 6px; color: var(--ink-2); text-decoration: none; line-height: 1.35; }
aside a:hover, aside a:focus-visible { background: var(--surface); color: var(--ink); outline: none; }
aside .sub a { padding-left: 20px; font-size: 12.5px; color: var(--muted); }
aside .dash { margin-top: 16px; display: inline-block; font-weight: 500; color: var(--accent); }
main { min-width: 0; max-width: 76ch; }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; letter-spacing: -0.01em; line-height: 1.2; }
h1 { font-size: clamp(30px, 5vw, 44px); margin: 0 0 12px; font-weight: 700; }
h2 { font-size: 26px; margin: 56px 0 14px; padding-top: 16px; border-top: 1px solid var(--rule); font-weight: 600; scroll-margin-top: 20px; }
h3 { font-size: 19px; margin: 36px 0 10px; font-weight: 600; scroll-margin-top: 20px; }
p { margin: 0 0 14px; }
main > p:first-of-type { color: var(--ink-2); }
a { color: var(--accent); }
strong { font-weight: 600; }
code { font-family: var(--mono); font-size: .86em; background: var(--code); padding: 1px 5px; border-radius: 4px; }
ul, ol { padding-left: 22px; margin: 0 0 14px; }
li { margin-bottom: 6px; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 28px 0; }
blockquote { margin: 0 0 14px; padding: 2px 0 2px 16px; border-left: 3px solid var(--accent); color: var(--ink-2); }
.tablewrap { overflow-x: auto; margin: 6px 0 18px; border: 1px solid var(--rule); border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; line-height: 1.45; }
th, td { padding: 8px 12px; border-bottom: 1px solid var(--rule); text-align: left; vertical-align: top; }
tr:last-child td { border-bottom: 0; }
th { font: 500 11.5px/1.3 var(--mono); text-transform: uppercase; letter-spacing: .03em; color: var(--muted); background: color-mix(in srgb, var(--bg) 60%, var(--surface)); }
td[style*="right"], th[style*="right"] { font-variant-numeric: tabular-nums; }
.tldr { background: var(--accent-soft); border-radius: 12px; padding: 18px 22px 6px; margin: 18px 0 8px; }
.tldr h2 { border: 0; margin: 0 0 8px; padding: 0; font-size: 18px; }
.refs p { font-size: 14px; padding-left: 24px; text-indent: -24px; color: var(--ink-2); }
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; gap: 12px; padding-inline: 16px; }
  aside { position: static; max-height: none; border-bottom: 1px solid var(--rule); padding-bottom: 12px; }
  aside ol { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); }
  aside .sub { display: none; }
}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dashboard-url", default="")
    args = ap.parse_args()

    md_text = SRC.read_text(encoding="utf-8")
    md_text = re.sub(r"<!--.*?-->\n?", "", md_text, flags=re.S)  # build markers
    conv = markdown.Markdown(extensions=["tables", "toc", "sane_lists"],
                             extension_configs={"toc": {"toc_depth": "2-3"}})
    body = conv.convert(md_text)
    body = re.sub(r"<table>", '<div class="tablewrap"><table>', body)
    body = body.replace("</table>", "</table></div>")
    # wrap the TL;DR block and the references in their own containers
    body = re.sub(r'(<h2 id="tldr">.*?)(?=<hr />)', r'<div class="tldr">\1</div>', body, flags=re.S)
    body = re.sub(r'(<h2 id="references">References</h2>)(.*)$', r'\1<div class="refs">\2</div>', body, flags=re.S)

    toc_items: list[str] = []

    def walk(tokens):
        for t in tokens:
            if t["level"] == 1:
                yield from walk(t["children"])
                continue
            toc_items.append(f'<li><a href="#{t["id"]}">{t["name"]}</a></li>')
            if t["children"]:
                toc_items.append('<li class="sub"><ol>' + "".join(
                    f'<li><a href="#{c["id"]}">{c["name"]}</a></li>' for c in t["children"]) + "</ol></li>")
    list(walk(conv.toc_tokens))
    dash = f'<a class="dash" href="{args.dashboard_url}">Open the results dashboard →</a>' if args.dashboard_url else ""

    html = f"""<title>Thai Retrofit Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bai+Jamjuree:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Thai:wght@400;500;600&display=swap">
<style>{STYLE}</style>
<div class="layout">
  <aside aria-label="Contents"><div class="eyebrow">Contents</div><ol>{''.join(toc_items)}</ol>{dash}</aside>
  <main>{body}</main>
</div>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
