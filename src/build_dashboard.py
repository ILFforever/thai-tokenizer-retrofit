"""Build the results dashboard: template + results/dashboard_data.json.

    python src/dashboard_data.py && python src/build_dashboard.py

Writes results/dashboard.html, a single self-contained page (fonts from
Google Fonts, everything else inline).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
TEMPLATE = Path(__file__).parent / "dashboard_template.html"
DATA = ROOT / "results" / "dashboard_data.json"
OUT = ROOT / "results" / "dashboard.html"

# Per-passage arrays are for the analysis scripts, not the page.
DROP = {"per_passage_bits", "per_passage_tokens", "per_passage_chars", "by_target", "by_previous_input"}


def _slim(x):
    if isinstance(x, dict):
        return {k: _slim(v) for k, v in x.items() if k not in DROP}
    if isinstance(x, list):
        return [_slim(v) for v in x]
    return x


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--report-url", default="")
    args = ap.parse_args()
    data = _slim(json.loads(DATA.read_text(encoding="utf-8")))
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", blob)
    html = html.replace("__REPORT_URL__", args.report_url)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
