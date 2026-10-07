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

import tiktoken

sys.path.insert(0, str(Path(__file__).parent))

from corpora import describe, load_parallel, load_thai_domain  # noqa: E402
from metrics import (  # noqa: E402
    SEGMENTER_NAME,
    score_domain,
    score_parallel,
    to_rows,
)
from registry import load_all  # noqa: E402
from versions import REVISIONS  # noqa: E402

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

    out.append("\n## Cluster breaks\n")
    out.append(
        "Share of Thai clusters (a base character plus its vowel/tone marks) "
        "that the final tokens split, measured on byte offsets over the first "
        "400 documents. This is a *token-level* measure: it does not say "
        "whether the pre-tokenizer or the vocabulary made the cut - "
        "`DECOMPOSITION.md` does. The Week 1 mark-split rate is kept for "
        "comparison; it decoded one token at a time and so missed byte-level "
        "fragments, and it is not a pre-tokenization measure.\n"
    )
    out.append(
        "\n| Tokenizer | " + " | ".join(domains) + " | legacy mark-split (wikipedia) |"
    )
    out.append("| --- |" + " ---: |" * (len(domains) + 1))
    cbr: dict[str, dict[str, float]] = {}
    legacy: dict[str, float] = {}
    for r in domain_rows:
        cbr.setdefault(r["tokenizer"], {})[r["domain"]] = r["cluster_break_rate"]
        if r["domain"] == "wikipedia":
            legacy[r["tokenizer"]] = r["mark_split_rate"]
    for tok, vals in sorted(cbr.items(), key=lambda kv: -max(kv[1].values())):
        cells = [f"{vals.get(d, float('nan')):.1%}" for d in domains]
        out.append(
            f"| {tok} | " + " | ".join(cells)
            + f" | {legacy.get(tok, float('nan')):.1%} |"
        )

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
    ap.add_argument("--retrofits", action="store_true",
                    help="also score this project's retrofitted tokenizers")
    ap.add_argument("--out", default="BASELINE",
                    help="report name; CSVs get the same suffix (e.g. BASELINE_after)")
    args = ap.parse_args()
    suffix = "" if args.out == "BASELINE" else "_" + args.out.split("_", 1)[-1]

    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()

    print("Loading tokenizers")
    toks, skipped = load_all(retrofits=args.retrofits)
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
    write_csv(RESULTS / f"domain_results{suffix}.csv", d_rows)
    write_csv(RESULTS / f"parallel_results{suffix}.csv", p_rows)
    (RESULTS / f"{args.out}.md").write_text(
        markdown_report(d_rows, p_rows, notes, skipped), encoding="utf-8"
    )
    (RESULTS / f"run_meta{suffix}.json").write_text(
        json.dumps(
            {
                "docs_per_domain": args.docs,
                "parallel_pairs": args.pairs,
                "segmenter": SEGMENTER_NAME,
                "tokenizers": [t.label for t in toks],
                "hub_revisions": REVISIONS,
                "tiktoken": tiktoken.__version__,
                "skipped": skipped,
                "seconds": round(time.time() - t0, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nDone in {time.time() - t0:.0f}s -> results/{args.out}.md")


if __name__ == "__main__":
    main()
