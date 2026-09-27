"""Week 1 baseline: how much does Thai text cost across current tokenizers?

    python src/run_baseline.py --docs 1500 --pairs 3000

Writes results/domain_results.csv, results/parallel_results.csv and
results/BASELINE.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from corpora import describe, load_parallel, load_thai_domain  # noqa: E402
from metrics import (  # noqa: E402
    SEGMENTER_NAME,
    score_domain,
    score_parallel,
    to_rows,
)
from registry import load_all  # noqa: E402

DOMAINS = ["wikipedia", "news", "social"]
RESULTS = Path(__file__).parent.parent / "results"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(
    domain_rows: list[dict],
    parallel_rows: list[dict],
    corpus_notes: list[str],
    skipped: list[str],
) -> str:
    out: list[str] = []
    out.append("# Thai tokenizer baseline\n")
    out.append(
        "Primary metric is **tokens per 100 characters**. Thai has no "
        "whitespace word boundaries, so tokens-per-word depends entirely on "
        "the segmenter and is reported only as a secondary figure "
        f"(segmenter: `{SEGMENTER_NAME}`).\n"
    )

    out.append("\n## Thai penalty on matched content\n")
    out.append(
        "From a parallel corpus, so both sides carry the same meaning. This "
        "is the only table where a Thai-vs-English ratio is meaningful.\n"
    )
    out.append(
        "\n| Tokenizer | Family | Thai tok/100ch | English tok/100ch | "
        "Thai penalty |"
    )
    out.append("| --- | --- | ---: | ---: | ---: |")
    for r in sorted(parallel_rows, key=lambda x: -x["thai_penalty_ratio"]):
        out.append(
            f"| {r['tokenizer']} | {r['family']} | "
            f"{r['thai_tok_per_100_chars']:.1f} | "
            f"{r['eng_tok_per_100_chars']:.1f} | "
            f"**{r['thai_penalty_ratio']:.2f}x** |"
        )

    out.append("\n## Cost by Thai domain\n")
    out.append(
        "Lower is better. `social` is informal Thai, the register a "
        "Wikipedia-trained vocabulary is most likely to under-serve.\n"
    )
    domains = sorted({r["domain"] for r in domain_rows})
    out.append("\n| Tokenizer | " + " | ".join(domains) + " | spread |")
    out.append("| --- |" + " ---: |" * (len(domains) + 1))
    by_tok: dict[str, dict[str, float]] = {}
    for r in domain_rows:
        by_tok.setdefault(r["tokenizer"], {})[r["domain"]] = r[
            "tok_per_100_chars"
        ]
    for tok, vals in sorted(by_tok.items(), key=lambda kv: -max(kv[1].values())):
        cells = [f"{vals.get(d, float('nan')):.1f}" for d in domains]
        present = [v for v in vals.values() if v]
        spread = (max(present) / min(present)) if present else 0.0
        out.append(f"| {tok} | " + " | ".join(cells) + f" | {spread:.2f}x |")
    out.append(
        "\nA large spread means the tokenizer's Thai performance depends "
        "heavily on register - relevant to the fairness question, since a "
        "vocabulary built only from formal text penalises everyone who "
        "writes differently.\n"
    )

    out.append("\n## Pre-tokenization check\n")
    out.append(
        "Share of tokens beginning with a Thai combining mark, meaning the "
        "tokenizer cut between a base consonant and its vowel or tone mark. "
        "A high rate means vocabulary changes alone cannot fix that "
        "tokenizer - merges cannot span a boundary the pre-tokenizer already "
        "imposed.\n"
    )
    out.append("\n| Tokenizer | Mark-split rate (wikipedia) |")
    out.append("| --- | ---: |")
    for r in sorted(
        [r for r in domain_rows if r["domain"] == "wikipedia"],
        key=lambda x: -x["mark_split_rate"],
    ):
        out.append(f"| {r['tokenizer']} | {r['mark_split_rate']:.1%} |")

    out.append("\n## Secondary: tokens per segmented word\n")
    out.append(
        f"Segmenter-dependent (`{SEGMENTER_NAME}`). Reported because readers "
        "expect it, not used as a headline.\n"
    )
    out.append("\n| Tokenizer | " + " | ".join(domains) + " |")
    out.append("| --- |" + " ---: |" * len(domains))
    fert: dict[str, dict[str, float | None]] = {}
    for r in domain_rows:
        fert.setdefault(r["tokenizer"], {})[r["domain"]] = r["tok_per_word"]
    for tok, vals in sorted(fert.items()):
        cells = [
            f"{vals.get(d):.2f}" if vals.get(d) else "-" for d in domains
        ]
        out.append(f"| {tok} | " + " | ".join(cells) + " |")

    out.append("\n## Corpora\n")
    for note in corpus_notes:
        out.append(f"- `{note}`")
    if skipped:
        out.append("\n## Tokenizers not available\n")
        for s in skipped:
            out.append(f"- {s}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=1500)
    ap.add_argument("--pairs", type=int, default=3000)
    ap.add_argument("--no-fertility", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()

    print("Loading tokenizers")
    toks, skipped = load_all()
    if not toks:
        raise SystemExit("no tokenizers available")

    print("\nLoading corpora")
    corpora = []
    notes = []
    for d in DOMAINS:
        c = load_thai_domain(d, n_docs=args.docs)
        corpora.append(c)
        notes.append(describe(c))
        print("  " + describe(c))
    par = load_parallel(n_pairs=args.pairs)
    notes.append(describe(par))
    print("  " + describe(par))

    print("\nScoring")
    domain_results = []
    parallel_results = []
    for tok in toks:
        for c in corpora:
            domain_results.append(
                score_domain(tok, c, with_fertility=not args.no_fertility)
            )
        parallel_results.append(score_parallel(tok, par))
        print(f"  {tok.label}")

    d_rows, p_rows = to_rows(domain_results), to_rows(parallel_results)
    write_csv(RESULTS / "domain_results.csv", d_rows)
    write_csv(RESULTS / "parallel_results.csv", p_rows)
    (RESULTS / "BASELINE.md").write_text(
        markdown_report(d_rows, p_rows, notes, skipped), encoding="utf-8"
    )
    (RESULTS / "run_meta.json").write_text(
        json.dumps(
            {
                "docs_per_domain": args.docs,
                "parallel_pairs": args.pairs,
                "segmenter": SEGMENTER_NAME,
                "tokenizers": [t.label for t in toks],
                "skipped": skipped,
                "seconds": round(time.time() - t0, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nDone in {time.time() - t0:.0f}s -> results/BASELINE.md")


if __name__ == "__main__":
    main()
