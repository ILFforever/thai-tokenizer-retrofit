"""Week 3: rank Thai subword candidates by the tokens they would save.

    python src/candidates.py --target Mistral-7B

Writes results/candidates.csv and results/CANDIDATES.md.

Two segmentation levels are mined and compared:

  word  PyThaiNLP newmm words - the units a reader would recognise
  tcc   Thai Character Clusters - a base character plus its marks, the
        smallest unit a tokenizer should never split

Frequencies come from the *training* slices (documents after the Week 1
evaluation slice), with each domain weighted equally so the ranking is not
tuned to Wikipedia alone. The projection is then measured on the evaluation
slice, per domain.

    savings(u) = freq(u) * (base_tokens(u) - 1)

A unit that already costs one token saves nothing. The projection assumes
each chosen unit becomes exactly one token - an upper bound, because in a
BPE vocabulary a new token also needs its intermediate pieces, each of which
takes a slot. Week 4 builds the real thing and pays that cost.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from corpora import load_thai_domain  # noqa: E402

DOMAINS = ["wikipedia", "news", "social"]
RESULTS = Path(__file__).parent.parent / "results"
THAI_ONLY = re.compile(r"^[฀-๿]+$")

# Evaluation slice is the first 1,200 documents (Week 1). Training takes the
# documents after it, capped so the mining stays a few minutes on a laptop.
EVAL_DOCS = 1200
TRAIN_CHAR_CAP = 8_000_000
GRID = [100, 250, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 20000]


def train_texts(domain: str, cap: int = TRAIN_CHAR_CAP) -> list[str]:
    """Training documents for a domain: after the eval slice, up to `cap`
    characters. Social has ~1.3M characters in total, so it is used whole."""
    corpus = load_thai_domain(domain, n_docs=100_000, skip=EVAL_DOCS)
    out, n = [], 0
    for t in corpus.texts:
        out.append(t)
        n += len(t)
        if n >= cap:
            break
    return out


def _segment_batch(args: tuple[str, list[str]]) -> Counter:
    level, texts = args
    if level == "word":
        from pythainlp.tokenize import word_tokenize

        seg = lambda s: word_tokenize(s, engine="newmm")  # noqa: E731
    else:
        from pythainlp.tokenize import subword_tokenize

        seg = lambda s: subword_tokenize(s, engine="tcc")  # noqa: E731
    c: Counter = Counter()
    for t in texts:
        c.update(u for u in seg(t) if THAI_ONLY.match(u))
    return c


def count_units(level: str, texts: list[str], workers: int = 16) -> Counter:
    batches = [texts[i : i + 50] for i in range(0, len(texts), 50)]
    total: Counter = Counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for c in ex.map(_segment_batch, [(level, b) for b in batches]):
            total.update(c)
    return total


def unit_cost_fn(tok):
    """Tokens a unit costs on its own, without the word-start marker a
    SentencePiece-style tokenizer would prepend to a standalone string.

    For HF tokenizers this runs the model (vocabulary + merges) directly on
    the normalised unit; for tiktoken it is a plain encode.
    """
    if tok.kind == "tiktoken":
        return lambda u: len(tok.raw.encode(u, disallowed_special=()))
    backend = tok.raw.backend_tokenizer
    norm = backend.normalizer
    model = backend.model
    import json

    byte_level = "ByteLevel" in json.dumps(json.loads(backend.to_str()).get("pre_tokenizer"))
    if byte_level:
        from pretok import _bytes_to_unicode

        b2u = _bytes_to_unicode()
        return lambda u: len(model.tokenize("".join(b2u[b] for b in u.encode("utf-8"))))
    return lambda u: len(model.tokenize(norm.normalize_str(u) if norm else u))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Mistral-7B")
    ap.add_argument("--top", type=int, default=20000, help="rows written to the CSV")
    args = ap.parse_args()
    t0 = time.time()

    from registry import load_all

    tok = load_all(verbose=False, only=[args.target])[0][0]
    cost = unit_cost_fn(tok)

    # --- mine frequencies on training slices, per domain -----------------
    train = {d: train_texts(d) for d in DOMAINS}
    train_chars = {d: sum(map(len, ts)) for d, ts in train.items()}
    print("training chars:", train_chars, flush=True)
    per_dom: dict[str, dict[str, Counter]] = {"word": {}, "tcc": {}}
    for level in ("word", "tcc"):
        for d in DOMAINS:
            per_dom[level][d] = count_units(level, train[d])
            print(f"  {level:4} {d:9} {len(per_dom[level][d]):>8,} types", flush=True)

    # Equal domain weight: counts per million characters, averaged.
    rows = []
    for level in ("word", "tcc"):
        units = set().union(*(per_dom[level][d].keys() for d in DOMAINS))
        for u in units:
            pm = {d: per_dom[level][d][u] / train_chars[d] * 1e6 for d in DOMAINS}
            freq = sum(pm.values()) / len(DOMAINS)
            if freq < 0.5:  # below one occurrence per 2M chars: noise
                continue
            c = cost(u)
            rows.append(
                {
                    "unit": u,
                    "level": level,
                    "chars": len(u),
                    "freq_per_M": freq,
                    "freq_wikipedia_per_M": pm["wikipedia"],
                    "freq_news_per_M": pm["news"],
                    "freq_social_per_M": pm["social"],
                    "base_tokens": c,
                    "savings_per_M": freq * (c - 1),
                }
            )
    print(f"scored {len(rows):,} candidates", flush=True)

    ranked = {
        level: sorted(
            (r for r in rows if r["level"] == level),
            key=lambda r: -r["savings_per_M"],
        )
        for level in ("word", "tcc")
    }

    # --- projection on the held-out evaluation slice, per domain ---------
    evals = {d: load_thai_domain(d, n_docs=EVAL_DOCS).texts for d in DOMAINS}
    base_tokens = {d: sum(tok.count(t) for t in ts) for d, ts in evals.items()}
    eval_chars = {d: sum(map(len, ts)) for d, ts in evals.items()}
    proj: dict[str, dict[str, list[float]]] = {"word": {}, "tcc": {}}
    for level in ("word", "tcc"):
        for d in DOMAINS:
            ho = count_units(level, evals[d])
            saved, out, k = 0, [], 0
            for n in GRID:
                for r in ranked[level][k:n]:
                    saved += ho[r["unit"]] * (r["base_tokens"] - 1)
                k = min(n, len(ranked[level]))
                out.append(saved / base_tokens[d])
            proj[level][d] = out
            print(f"  projected {level} {d}: " + ", ".join(f"{g}:{p:.1%}" for g, p in zip(GRID, out)), flush=True)

    # --- write -----------------------------------------------------------
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "candidates.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["rank", *rows[0].keys(), "cum_savings_per_M", "cum_share_of_base"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        # Base Thai cost per million training chars, domain-averaged, as the
        # denominator of the cumulative share.
        base_train = {
            d: sum(tok.count(t) for t in train[d][:400]) / sum(map(len, train[d][:400])) * 1e6
            for d in DOMAINS
        }
        base_pm = sum(base_train.values()) / len(DOMAINS)
        for level in ("word", "tcc"):
            cum = 0.0
            for i, r in enumerate(ranked[level][: args.top], 1):
                cum += r["savings_per_M"]
                w.writerow(
                    {"rank": i, **r, "cum_savings_per_M": cum, "cum_share_of_base": cum / base_pm}
                )

    md = [f"# Thai subword candidates (Week 3) - target: {args.target}\n"]
    md.append(
        "Generated by `src/candidates.py`. Frequencies are mined on the "
        "training slices (documents after the evaluation slice) and weighted "
        "equally across domains; projections are measured on the evaluation "
        "slice (first 1,200 documents per domain, the Week 1 slice).\n"
    )
    md.append("\n| Domain | train chars | eval chars | base tokens (eval) | base tok/100ch |")
    md.append("| --- | ---: | ---: | ---: | ---: |")
    for d in DOMAINS:
        md.append(
            f"| {d} | {train_chars[d]:,} | {eval_chars[d]:,} | {base_tokens[d]:,} | "
            f"{base_tokens[d] / eval_chars[d] * 100:.1f} |"
        )
    for level, name in (("word", "newmm words"), ("tcc", "Thai Character Clusters")):
        md.append(f"\n## Projected token reduction on held-out text: {name}\n")
        md.append(
            "Share of the target's Thai tokens saved if the top-N candidates "
            "each became a single token. Upper bound: ignores the slots "
            "needed for intermediate BPE pieces.\n"
        )
        md.append("| Top N | " + " | ".join(DOMAINS) + " |")
        md.append("| ---: |" + " ---: |" * len(DOMAINS))
        for i, g in enumerate(GRID):
            md.append(f"| {g:,} | " + " | ".join(f"{proj[level][d][i]:.1%}" for d in DOMAINS) + " |")
        md.append("\nMarginal gain per 1,000 extra candidates (domain average):\n")
        prev_g, prev = 0, 0.0
        for i, g in enumerate(GRID):
            avg = sum(proj[level][d][i] for d in DOMAINS) / len(DOMAINS)
            md.append(f"- {prev_g:,}-{g:,}: {(avg - prev) / ((g - prev_g) / 1000):.2%} per 1k")
            prev_g, prev = g, avg
        md.append("\nTop 25 by savings:\n")
        md.append("| # | unit | chars | base tokens | freq / M chars (wiki / news / social) | savings / M |")
        md.append("| ---: | --- | ---: | ---: | --- | ---: |")
        for i, r in enumerate(ranked[level][:25], 1):
            md.append(
                f"| {i} | `{r['unit']}` | {r['chars']} | {r['base_tokens']} | "
                f"{r['freq_wikipedia_per_M']:.0f} / {r['freq_news_per_M']:.0f} / {r['freq_social_per_M']:.0f} | "
                f"{r['savings_per_M']:.0f} |"
            )
    (RESULTS / "CANDIDATES.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
