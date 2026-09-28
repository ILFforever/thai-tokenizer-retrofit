"""Collect everything the explainer video shows into video/data.json.

    python src/video_data.py

Numbers come from results/; tokens, the vocabulary map and the merge
sequence come from the real tokenizers, so the animation shows what actually
happens rather than an illustration of it.
"""

from __future__ import annotations

import csv
import heapq
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "video" / "data.json"
TH_SENT = "เมื่อวันที่ 6 ม.ค.60 ที่ทำเนียบรัฐบาล นายวิษณุ เครืองาม รองนายกรัฐมนตรี กล่าวถึงกรณี ที่"
EN_SENT = ("On 6 January 2017 at Government House, Mr. Wissanu Krea-ngam, Deputy Prime Minister, "
           "spoke about the case that")


def pieces(tok, text: str) -> list[str]:
    """Token strings as readable text; byte-fallback pieces of one character
    are shown as that character once, marked partial."""
    from dashboard_data import _pieces

    ids = tok.encode(text, add_special_tokens=False)
    return [("~" if p["partial"] else "") + p["t"] for p in _pieces(tok, ids, "hf")]


def merge_trace(tok_dir: Path, word: str) -> list[list[str]]:
    """Every intermediate segmentation of `word` under the tokenizer's merges,
    in the order Hugging Face tokenizers applies them (lowest rank first)."""
    j = json.loads((tok_dir / "tokenizer.json").read_text(encoding="utf-8"))
    merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in j["model"]["merges"]]
    rank = {m: i for i, m in enumerate(merges)}
    sym = list(word.replace(" ", "▁"))
    states = [list(sym)]
    while True:
        best = None
        for i in range(len(sym) - 1):
            r = rank.get((sym[i], sym[i + 1]))
            if r is not None and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            break
        _, i = best
        sym = sym[:i] + [sym[i] + sym[i + 1]] + sym[i + 2:]
        states.append(list(sym))
    return [[s.replace("▁", " ") for s in st] for st in states]


def two_stages() -> dict:
    """Worked example for the two-stage scene: Typhoon-7B's tokenizer on a short
    phrase. SentencePiece cuts at whitespace first; each piece is then merged
    in the tokenizer's own merge order. Checked against its real output."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    text = "ภาษาไทย ที่ดิน"
    d = Path(hf_hub_download("scb10x/typhoon-7b", "tokenizer.json")).parent
    traces = [merge_trace(d, " " + w) for w in text.split(" ")]
    real = [t.replace("▁", " ") for t in AutoTokenizer.from_pretrained("scb10x/typhoon-7b").tokenize(text)]
    assert [p for tr in traces for p in tr[-1]] == real, (traces, real)
    return {"text": text, "tokenizer": "Typhoon-7B", "traces": traces}


def vocab_map() -> dict:
    """Category of each of Mistral-7B's 32,000 entries, as the retrofit saw it."""
    from candidates import train_texts
    from corpora import load_thai_domain
    from registry import load_all
    from retrofit import load_bpe, usage_counts

    tok = load_all(verbose=False, only=["Mistral-7B"])[0][0]
    bpe = load_bpe(tok)
    en = usage_counts(tok, load_thai_domain("english_raw", n_docs=8000).texts)
    code = usage_counts(tok, load_thai_domain("code", n_docs=5000).texts)
    th = usage_counts(tok, [t for d in ("wikipedia", "news", "social") for t in train_texts(d)])
    freed = set()
    with (RESULTS / "prune_list_mistral-7b-th2483.csv").open(encoding="utf-8", newline="") as fh:
        freed = {int(r["id"]) for r in csv.DictReader(fh)}
    cat = []
    for i in range(len(tok.raw)):
        if i in freed:
            cat.append(4)
        elif en[i]:
            cat.append(0)
        elif code[i]:
            cat.append(1)
        elif th[i]:
            cat.append(2)
        else:
            cat.append(3)  # unused, but another kept entry is built from it (or special/byte)
    counts = {k: cat.count(v) for k, v in (("english", 0), ("code", 1), ("thai", 2), ("locked", 3), ("freed", 4))}
    return {"cells": "".join(map(str, cat)), "counts": counts}


def main() -> None:
    from transformers import AutoTokenizer

    if "--patch" in sys.argv:  # recompute only the cheap parts into the existing file
        D = json.loads(OUT.read_text(encoding="utf-8"))
        D["stages"] = two_stages()
        OUT.write_text(json.dumps(D, ensure_ascii=False), encoding="utf-8")
        print(f"patched {OUT.relative_to(ROOT)}: stages {D['stages']['traces']}")
        return
    D: dict = {}
    D["stages"] = two_stages()
    mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    retro = AutoTokenizer.from_pretrained(str(ROOT / "tokenizers" / "mistral-7b-th2483"))
    typhoon = AutoTokenizer.from_pretrained("scb10x/typhoon-7b")
    D["sentence"] = {
        "th": TH_SENT, "en": EN_SENT,
        "en_pieces": pieces(mistral, EN_SENT),
        "mistral": pieces(mistral, TH_SENT), "retrofit": pieces(retro, TH_SENT), "typhoon": pieces(typhoon, TH_SENT),
    }

    D["penalty"] = [{"name": r["tokenizer"], "x": float(r["thai_penalty_ratio"])}
                    for r in csv.DictReader((RESULTS / "parallel_results.csv").open(encoding="utf-8"))]
    D["penalty"].sort(key=lambda r: -r["x"])

    dec = [r for r in csv.DictReader((RESULTS / "decomposition.csv").open(encoding="utf-8")) if r["domain"] == "parallel_th"]
    pen = lambda r, k: (float(r[k]) / float(r["n_chars"])) / (float(r["T_english"]) / float(r["english_chars"]))
    D["decomp"] = sorted([{"name": r["tokenizer"], "shipped": pen(r, "T_act"), "regex": pen(r, "T_regex"),
                           "vocab": pen(r, "T_vocab"), "cut": float(r["pretok_cut_rate"])} for r in dec],
                         key=lambda r: -r["shipped"])

    from pretok import pretoken_spans
    from registry import load_all

    toks = {t.label: t for t in load_all(verbose=False, only=["GPT-4 (cl100k_base)", "Qwen2.5", "Mistral-7B"])[0]}
    word = "ที่ดิน"
    D["cuts"] = {k: [word[a:b] for a, b in pretoken_spans(t, word)] for k, t in toks.items()}

    q = json.loads((RESULTS / "qwen_unreachable.json").read_text(encoding="utf-8"))
    D["qwen"] = {"thai": 2570, "dead": q["rows"]["unreachable_thai"]["n"],
                 "norm_dead": q["rows"]["unreachable_thai"]["norm"]["median"],
                 "norm_live": q["rows"]["reachable_thai"]["norm"]["median"],
                 "examples": ["ประวัติศาสตร์", "อย่างต่อเนื่อง", "มีประสิทธิภาพ", "นักท่องเที่ยว", "ประชาสัมพันธ์"]}

    print("vocabulary map (encodes the protect corpora, ~2 min)...", flush=True)
    D["vocab"] = vocab_map()
    freed_rows = list(csv.DictReader((RESULTS / "prune_list_mistral-7b-th2483.csv").open(encoding="utf-8", newline="")))
    by = {}
    for r in freed_rows:
        t = r["text"].strip()
        if t and r["script"] in ("cyrillic", "cjk", "hangul", "other") and len(by.get(r["script"], [])) < 8:
            by.setdefault(r["script"], []).append(t)
    D["freed_examples"] = by
    added = list(csv.DictReader((RESULTS / "added_tokens_mistral-7b-th2483.csv").open(encoding="utf-8", newline="")))
    D["added"] = [r["text"] for r in added]
    meta = json.loads((RESULTS / "retrofit_mistral-7b_meta.json").read_text(encoding="utf-8"))
    D["atoms"] = meta["atoms"]

    cand = [r for r in csv.DictReader((RESULTS / "candidates.csv").open(encoding="utf-8")) if r["level"] == "word"][:12]
    D["candidates"] = [{"u": r["unit"], "cost": int(r["base_tokens"]), "save": float(r["savings_per_M"])} for r in cand]
    D["merge"] = merge_trace(ROOT / "tokenizers" / "mistral-7b-th2483", " เมื่อวันที่")

    ev = {r["tokenizer"]: r for r in json.loads((RESULTS / "evaluation.json").read_text(encoding="utf-8"))}
    pick = lambda r: {k: r[k] for k in r if k.endswith(("_tok100", "_change")) or k in ("penalty", "vocab")}
    D["eval"] = {k: pick(v) for k, v in ev.items()}
    D["bootstrap"] = json.loads((RESULTS / "vs_typhoon_bootstrap.json").read_text(encoding="utf-8"))
    rob = json.loads((RESULTS / "robustness.json").read_text(encoding="utf-8"))
    D["register"] = rob["register"]
    D["collateral"] = {k: {"base": v["base_tok100"], "retrofit": v["mistral-7b-th2483"]["change"]}
                       for k, v in rob["collateral"].items()}
    D["bloom"] = json.loads((RESULTS / "embed_adapt_bloom-th5000.json").read_text(encoding="utf-8"))["results"]
    D["mistral_gpu"] = {}
    for p in sorted(RESULTS.glob("mistral_adapt_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        D["mistral_gpu"][p.stem.replace("mistral_adapt_", "")] = {"meta": {k: d["meta"][k] for k in ("steps", "train_tokens_seen")},
                                                                 "results": d["results"]}
    show = json.loads((RESULTS / "qwen_showcase.json").read_text(encoding="utf-8"))
    D["stream"] = {"prompt": show["prompts"][2]["prompt"],
                   "answers": {a["model"]: {k: a[k] for k in ("text", "tokens", "chars", "seconds", "chars_per_s")}
                               for a in show["prompts"][2]["answers"]}}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(D, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB); vocab counts {D['vocab']['counts']}; "
          f"merge steps {len(D['merge'])}")


if __name__ == "__main__":
    main()
