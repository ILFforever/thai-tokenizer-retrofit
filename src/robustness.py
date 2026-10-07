"""Week 6: robustness checks and the honest cost of the retrofit.

    python src/robustness.py

Writes results/ROBUSTNESS.md and results/robustness.json.

1. Second segmenter. Tokens per word depends on the word segmenter, so the
   secondary fertility figure is recomputed with PyThaiNLP's `longest`
   (dictionary longest-match) next to `newmm`. If the ordering of
   tokenizers or the size of the retrofit gain moves, the segmenter was
   doing the work.
2. Register split. Thai cost for formal (news, Wikipedia) and informal
   (social) text, and the informal/formal spread, before and after - does
   the retrofit widen the gap? Includes a formal-only ablation (new merges
   learned from Wikipedia + news only) if it has been built.
3. Embedding-row accounting. Rows swapped, as a count and a share; whether
   any pruned row is used by *held-out* English (the prune list was built
   on a different English slice); which scripts the freed rows came from.
4. Collateral cost. The freed rows were unused by English and Thai - but
   not necessarily by other languages. Chinese, Korean, Russian and
   Japanese Wikipedia text is tokenized before and after to show who pays.
"""

from __future__ import annotations

from versions import rev

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from corpora import load_thai_domain  # noqa: E402

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
TOKS = ROOT / "tokenizers"
DOMAINS = ["wikipedia", "news", "social"]
SEGMENTERS = ["newmm", "longest"]
SAMPLE_DOCS = 100
SAMPLE_CHARS = 2000
OTHER_LANGS = {"zh": "Chinese", "ko": "Korean", "ru": "Russian", "ja": "Japanese",
               "code": "Python code"}


def _seg_count(args: tuple[str, list[str]]) -> list[list[str]]:
    engine, texts = args
    from pythainlp.tokenize import word_tokenize

    return [[w for w in word_tokenize(t, engine=engine) if w.strip()] for t in texts]


def segment_all(engine: str, texts: list[str]) -> list[list[str]]:
    batches = [texts[i : i + 5] for i in range(0, len(texts), 5)]
    out: list[list[str]] = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for r in ex.map(_seg_count, [(engine, b) for b in batches]):
            out.extend(r)
    return out


def other_language_texts(lang: str, n: int = 200) -> list[str]:
    """First n usable documents of a Wikipedia language, cached like the
    Thai corpora."""
    import os

    from datasets import load_dataset

    from corpora import CACHE, _clean

    path = CACHE / f"wiki_{lang}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if lang == "code":
        # Code keeps its whitespace: indentation is part of what a tokenizer pays for.
        ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True, revision=rev("codeparrot/codeparrot-clean-valid"))
        rows = (r["content"][:3000] for r in ds)
    else:
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True, revision=rev("wikimedia/wikipedia"))
        rows = (_clean(r["text"])[:3000] for r in ds)
    out = []
    for t in rows:
        if len(t) >= 500:
            out.append(t)
        if len(out) >= n:
            break
    os.makedirs(CACHE, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    from transformers import AutoTokenizer

    from registry import load_all, retrofit_dirs

    base_toks, _ = load_all(verbose=False)
    retro = {p.name: AutoTokenizer.from_pretrained(str(p)) for p in retrofit_dirs("mistral-7b")}
    report: dict = {}

    # 1. second segmenter ------------------------------------------------
    samples = {d: [t[:SAMPLE_CHARS] for t in load_thai_domain(d, n_docs=SAMPLE_DOCS).texts] for d in DOMAINS}
    words = {e: {d: segment_all(e, samples[d]) for d in DOMAINS} for e in SEGMENTERS}
    fert: dict[str, dict] = {}
    count_fns = {t.label: t.count for t in base_toks}
    count_fns.update({f"Mistral-7B +TH{n.split('-th')[1]}": (lambda s, _t=t: len(_t.encode(s, add_special_tokens=False)))
                      for n, t in retro.items()})
    for label, count in count_fns.items():
        fert[label] = {}
        for d in DOMAINS:
            n_tok = sum(count(t) for t in samples[d])
            for e in SEGMENTERS:
                n_words = sum(len(w) for w in words[e][d])
                fert[label][f"{d}_{e}"] = n_tok / n_words
    wpc = {e: {d: sum(len(w) for w in words[e][d]) / sum(map(len, samples[d])) * 100 for d in DOMAINS}
           for e in SEGMENTERS}

    def ranks(e: str, d: str) -> list[str]:
        return sorted((l for l in fert if "+TH" not in l), key=lambda l: fert[l][f"{d}_{e}"])

    from itertools import combinations

    agree = {}
    for d in DOMAINS:
        a, b = ranks("newmm", d), ranks("longest", d)
        pos_b = {l: i for i, l in enumerate(b)}
        pairs = list(combinations(a, 2))
        concord = sum(1 for x, y in pairs if pos_b[x] < pos_b[y])
        agree[d] = {"kendall_tau": (2 * concord - len(pairs)) / len(pairs), "newmm_order": a, "longest_order": b}
    report["segmenter"] = {"fertility": fert, "words_per_100_chars": wpc, "rank_agreement": agree}
    print("segmenter done", flush=True)

    # 2. register split -----------------------------------------------------
    ev = json.loads((RESULTS / "evaluation.json").read_text(encoding="utf-8"))
    reg = []
    for r in ev:
        formal = (r["th_wikipedia_tok100"] + r["th_news_tok100"]) / 2
        informal = r["th_social_tok100"]
        reg.append({"tokenizer": r["tokenizer"], "formal_tok100": formal, "informal_tok100": informal,
                    "spread": informal / formal})
    report["register"] = reg

    # 3. embedding-row accounting ------------------------------------------
    base = next(t for t in base_toks if t.label == "Mistral-7B").raw
    en_eval = load_thai_domain("english_raw", n_docs=4000, skip=8000).texts
    en_ids = set()
    for i in range(0, len(en_eval), 256):
        for e in base.backend_tokenizer.encode_batch(en_eval[i : i + 256], add_special_tokens=False):
            en_ids.update(e.ids)
    rows = {}
    for name in retro:
        pruned = []
        with (RESULTS / f"prune_list_{name}.csv").open(encoding="utf-8") as fh:
            pruned = list(csv.DictReader(fh))
        scripts: dict[str, int] = {}
        for p in pruned:
            scripts[p["script"]] = scripts.get(p["script"], 0) + 1
        hit = [p for p in pruned if int(p["id"]) in en_ids]
        rows[name] = {
            "swapped": len(pruned), "share_of_vocab": len(pruned) / len(base),
            "english_uses_in_prune_corpus": sum(int(p["english_uses"]) for p in pruned),
            "pruned_ids_used_by_heldout_english": len(hit),
            "heldout_examples": [p["text"] for p in hit[:10]],
            "scripts": scripts,
        }
    report["rows"] = rows
    print("rows done", flush=True)

    # 4. collateral cost on other languages ----------------------------------
    coll = {}
    for lang, lname in OTHER_LANGS.items():
        texts = other_language_texts(lang)
        n_chars = sum(map(len, texts))
        b = sum(len(base.encode(t, add_special_tokens=False)) for t in texts)
        coll[lname] = {"base_tok100": b / n_chars * 100}
        for name, tok in retro.items():
            n = sum(len(tok.encode(t, add_special_tokens=False)) for t in texts)
            coll[lname][name] = {"tok100": n / n_chars * 100, "change": n / b - 1}
    report["collateral"] = coll
    print("collateral done", flush=True)

    (RESULTS / "robustness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- markdown ----------------------------------------------------------
    md = ["# Robustness and the cost of the retrofit (Week 6)\n",
          "Generated by `src/robustness.py`.\n"]
    md.append("\n## 1. Does the word segmenter change the conclusions?\n")
    md.append(f"Tokens per word on the first {SAMPLE_DOCS} evaluation documents per domain "
              f"(first {SAMPLE_CHARS} characters each), under two PyThaiNLP segmenters.\n")
    md.append("Words per 100 characters: " + "; ".join(
        f"{d}: newmm {wpc['newmm'][d]:.1f}, longest {wpc['longest'][d]:.1f}" for d in DOMAINS) + ".\n")
    md.append("\n| Tokenizer | " + " | ".join(f"{d} newmm | {d} longest" for d in DOMAINS) + " |")
    md.append("| --- |" + " ---: |" * (2 * len(DOMAINS)))
    for label in sorted(fert, key=lambda l: fert[l]["wikipedia_newmm"]):
        md.append(f"| {label} | " + " | ".join(
            f"{fert[label][f'{d}_newmm']:.2f} | {fert[label][f'{d}_longest']:.2f}" for d in DOMAINS) + " |")
    md.append("\nRank agreement of the nine published tokenizers between the two segmenters (Kendall's tau; "
              "1.0 = identical ordering): " + ", ".join(f"{d} {agree[d]['kendall_tau']:.2f}" for d in DOMAINS) + ".\n")

    md.append("\n## 2. Register: formal vs informal Thai\n")
    md.append("Formal = mean of Wikipedia and news tok/100ch; informal = social. Spread = informal / formal.\n")
    md.append("| Tokenizer | formal | informal | spread |")
    md.append("| --- | ---: | ---: | ---: |")
    for r in reg:
        md.append(f"| {r['tokenizer']} | {r['formal_tok100']:.1f} | {r['informal_tok100']:.1f} | {r['spread']:.3f} |")

    md.append("\n## 3. Embedding-row accounting\n")
    md.append("| Retrofit | Rows swapped | Share of 32,000 | English uses (prune corpus) | "
              "Pruned rows used by held-out English | Freed from |")
    md.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for name, r in rows.items():
        md.append(f"| {name} | {r['swapped']:,} | {r['share_of_vocab']:.1%} | {r['english_uses_in_prune_corpus']} | "
                  f"{r['pruned_ids_used_by_heldout_english']}"
                  + (f" ({', '.join(r['heldout_examples'][:5])})" if r['heldout_examples'] else "")
                  + " | " + ", ".join(f"{k} {v}" for k, v in sorted(r["scripts"].items(), key=lambda kv: -kv[1])) + " |")

    md.append("\n## 4. Who pays: other languages and code\n")
    md.append("Tokens per 100 characters on 200 documents per language (first 3,000 characters; Wikipedia, "
              "or codeparrot-clean-valid for Python), base Mistral-7B vs each retrofit.\n")
    names = list(retro)
    md.append("| Language | base | " + " | ".join(names) + " |")
    md.append("| --- | ---: |" + " ---: |" * len(names))
    for lname, c in coll.items():
        md.append(f"| {lname} | {c['base_tok100']:.1f} | " + " | ".join(
            f"{c[n]['tok100']:.1f} ({c[n]['change']:+.1%})" for n in names) + " |")
    (RESULTS / "ROBUSTNESS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
