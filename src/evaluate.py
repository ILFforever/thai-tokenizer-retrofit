"""Week 5: before/after evaluation of the retrofitted tokenizers.

    python src/evaluate.py

Compares base Mistral-7B, this project's fixed-size retrofits at each
budget, and Typhoon-7B (the same base with its vocabulary *expanded*), all on
held-out text:

  Thai      the Week 1 evaluation slice, per domain (never pooled)
  English   OPUS-100 English side (conversational) and English Wikipedia
            documents after the prune corpus (encyclopaedic)

Acceptance (PLAN.md, Week 5): Thai tokens down >= 15% in every domain,
English tokens per 100 characters within 1%, effective context (Thai
characters per 1,000 tokens) up >= 15%.

Confidence intervals are percentile bootstraps over documents (1,000
resamples): a reduction is a ratio of sums, so documents - not tokens - are
the independent unit.
"""

from __future__ import annotations

from versions import rev

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from corpora import load_parallel, load_thai_domain  # noqa: E402
from registry import retrofit_dirs  # noqa: E402

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
TOKS = ROOT / "tokenizers"
DOMAINS = ["wikipedia", "news", "social"]
ENGLISH_EVAL_SKIP = 8000
ENGLISH_EVAL_DOCS = 4000
BOOT = 1000


def _counts(tok, texts: list[str]) -> np.ndarray:
    backend = tok.backend_tokenizer
    out = []
    for i in range(0, len(texts), 256):
        out += [len(e.ids) for e in backend.encode_batch(texts[i : i + 256], add_special_tokens=False)]
    return np.array(out, dtype=np.int64)


def _boot_ratio(num: np.ndarray, den: np.ndarray, rng) -> tuple[float, float]:
    """95% CI of sum(num)/sum(den) with documents resampled."""
    idx = rng.integers(0, len(num), size=(BOOT, len(num)))
    r = num[idx].sum(1) / den[idx].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def tokenizer_set() -> list[tuple[str, str, object, str]]:
    """(label, kind, tokenizer, note) - base first, then retrofits, Typhoon."""
    from transformers import AutoTokenizer

    out = [("Mistral-7B (base)", "base", AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", revision=rev("mistralai/Mistral-7B-v0.1")),
            "32,000 entries")]
    def parse(p: Path) -> tuple[int, str]:
        rest = p.name.split("-th", 1)[1]
        k = int(rest.split("-")[0])
        return k, rest[len(str(k)):]

    for d in sorted(retrofit_dirs("mistral-7b"), key=lambda p: (parse(p)[1] != "", parse(p)[0], parse(p)[1])):
        k, suffix = parse(d)
        out.append((f"Mistral-7B +TH{k}{suffix}", "retrofit", AutoTokenizer.from_pretrained(str(d)),
                    f"{k:,} slots swapped, 32,000 entries"))
    out.append(("Typhoon-7B", "expansion", AutoTokenizer.from_pretrained("scb10x/typhoon-7b", revision=rev("scb10x/typhoon-7b")),
                "3,219 entries added, 35,219 entries"))
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    toks = tokenizer_set()
    thai = {d: load_thai_domain(d, n_docs=1200).texts for d in DOMAINS}
    par = load_parallel(n_pairs=3000)
    en_wiki = load_thai_domain("english_raw", n_docs=ENGLISH_EVAL_DOCS, skip=ENGLISH_EVAL_SKIP).texts
    sets = {**{f"th_{d}": t for d, t in thai.items()},
            "th_parallel": par.thai, "en_parallel": par.english, "en_wikipedia": en_wiki}
    chars = {k: np.array([len(t) for t in v], dtype=np.int64) for k, v in sets.items()}

    counts: dict[str, dict[str, np.ndarray]] = {}
    for label, kind, tok, note in toks:
        counts[label] = {k: _counts(tok, v) for k, v in sets.items()}
        print(f"  scored {label}", flush=True)

    base = toks[0][0]
    rows = []
    for label, kind, tok, note in toks:
        c = counts[label]
        row = {"tokenizer": label, "kind": kind, "note": note, "vocab": len(tok)}
        for k in sets:
            row[f"{k}_tok100"] = c[k].sum() / chars[k].sum() * 100
            row[f"{k}_change"] = c[k].sum() / counts[base][k].sum() - 1
            if k.startswith("th_"):
                lo, hi = _boot_ratio(c[k], counts[base][k], rng)
                row[f"{k}_change_ci"] = (lo - 1, hi - 1)
                row[f"{k}_ctx"] = chars[k].sum() / c[k].sum() * 1000
        row["penalty"] = (c["th_parallel"].sum() / chars["th_parallel"].sum()) / (
            c["en_parallel"].sum() / chars["en_parallel"].sum())
        rows.append(row)

    # acceptance
    for r in rows:
        if r["kind"] == "base":
            continue
        th_ok = all(r[f"th_{d}_change"] <= -0.15 for d in DOMAINS)
        en_ok = all(abs(r[k]) <= 0.01 for k in ("en_parallel_change", "en_wikipedia_change"))
        base_ctx = {d: rows[0][f"th_{d}_ctx"] for d in DOMAINS}
        ctx_ok = all(r[f"th_{d}_ctx"] / base_ctx[d] - 1 >= 0.15 for d in DOMAINS)
        r["accept_thai"], r["accept_english"], r["accept_context"] = th_ok, en_ok, ctx_ok

    RESULTS.mkdir(exist_ok=True)
    flat = []
    for r in rows:
        f = {k: (f"{v[0]:.4f}..{v[1]:.4f}" if isinstance(v, tuple) else v) for k, v in r.items()}
        flat.append(f)
    keys = list(flat[-2].keys())
    with (RESULTS / "evaluation.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(flat)
    (RESULTS / "evaluation.json").write_text(json.dumps(flat, ensure_ascii=False, indent=2, default=float),
                                             encoding="utf-8")

    md = ["# Before / after evaluation (Week 5)\n"]
    md.append(
        "Generated by `src/evaluate.py`. All text is held out: the Thai evaluation "
        "slice (first 1,200 documents per domain, the Week 1 slice), the OPUS-100 "
        "parallel pairs, and 4,000 English Wikipedia documents disjoint from the "
        "prune corpus. Changes are relative to base Mistral-7B; brackets are 95% "
        "bootstrap intervals over documents.\n"
    )
    md.append("\n## Thai tokens per 100 characters, per domain\n")
    md.append("| Tokenizer | Vocab | " + " | ".join(DOMAINS) + " |")
    md.append("| --- | ---: |" + " ---: |" * len(DOMAINS))
    for r in rows:
        cells = []
        for d in DOMAINS:
            v = r[f"th_{d}_tok100"]
            if r["kind"] == "base":
                cells.append(f"{v:.1f}")
            else:
                lo, hi = r[f"th_{d}_change_ci"]
                cells.append(f"{v:.1f} ({r[f'th_{d}_change']:+.1%} [{lo:+.1%}, {hi:+.1%}])")
        md.append(f"| {r['tokenizer']} | {r['vocab']:,} | " + " | ".join(cells) + " |")

    md.append("\n## English cost (must not move)\n")
    md.append("| Tokenizer | OPUS-100 EN tok/100ch | change | EN Wikipedia tok/100ch | change |")
    md.append("| --- | ---: | ---: | ---: | ---: |")
    for r in rows:
        md.append(
            f"| {r['tokenizer']} | {r['en_parallel_tok100']:.2f} | {r['en_parallel_change']:+.3%} | "
            f"{r['en_wikipedia_tok100']:.2f} | {r['en_wikipedia_change']:+.3%} |"
        )

    md.append("\n## Effective context and Thai penalty\n")
    md.append("Thai characters per 1,000 tokens (higher is better) and the Thai/English penalty on matched content.\n")
    md.append("| Tokenizer | " + " | ".join(f"{d} chars/1k tok" for d in DOMAINS) + " | Penalty |")
    md.append("| --- |" + " ---: |" * (len(DOMAINS) + 1))
    for r in rows:
        md.append(
            f"| {r['tokenizer']} | "
            + " | ".join(f"{r[f'th_{d}_ctx']:,.0f} ({r[f'th_{d}_ctx'] / rows[0][f'th_{d}_ctx'] - 1:+.0%})" for d in DOMAINS)
            + f" | {r['penalty']:.2f}x |"
        )

    md.append("\n## Acceptance (PLAN.md Week 5)\n")
    md.append("| Tokenizer | Thai >= 15% fewer, every domain | English within 1% | Context >= +15%, every domain |")
    md.append("| --- | --- | --- | --- |")
    for r in rows[1:]:
        md.append(
            f"| {r['tokenizer']} | {'PASS' if r['accept_thai'] else 'FAIL'} | "
            f"{'PASS' if r['accept_english'] else 'FAIL'} | {'PASS' if r['accept_context'] else 'FAIL'} |"
        )
    (RESULTS / "EVALUATION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
