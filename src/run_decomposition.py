"""Week 2: split each tokenizer's Thai excess into pre-tokenization and
vocabulary.

    python src/run_decomposition.py --docs 1200 --pairs 3000

Writes results/decomposition.csv and results/DECOMPOSITION.md.

Three counterfactual token counts per tokenizer and domain:

  T_act     the tokenizer as shipped
  T_regex   same vocabulary and merges, pre-tokenizer made mark-aware
            (the "regex fix" intervention on its own)
  T_vocab   a strong Thai vocabulary (XGLM's) applied inside this tokenizer's
            own pre-token pieces (the best a vocabulary retrofit could hope
            for while the pre-tokenizer stays as it is)
  T_ref     that same strong vocabulary with no pre-tokenizer constraint

The gap T_act - T_ref is the tokenizer's excess over a good Thai tokenizer.
Of that gap, T_vocab - T_ref is lost to the pre-tokenizer - no vocabulary,
however good, recovers it - and T_act - T_vocab is recoverable by vocabulary.
"""

from __future__ import annotations

from versions import rev

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from corpora import load_parallel, load_thai_domain  # noqa: E402

DOMAINS = ["wikipedia", "news", "social"]
RESULTS = Path(__file__).parent.parent / "results"
REF_REPO = "facebook/xglm-564M"
REF_LABEL = "XGLM"
MARKER = "<!-- analysis below is hand-written and preserved on re-run -->"


def _ref_counter():
    """Count tokens under the reference vocabulary for arbitrary segments.

    Uses XGLM's normalizer and Unigram model, but a Metaspace pre-tokenizer
    that does not prepend a word-start marker: a segment that is the middle
    of a word (as cl100k pieces often are) must not be charged for a word
    start it does not have.
    """
    from tokenizers import Tokenizer, pre_tokenizers
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(REF_REPO, revision=rev(REF_REPO))
    ref = Tokenizer.from_str(hf.backend_tokenizer.to_str())
    ref.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁", prepend_scheme="never", split=True
    )
    ref.post_processor = None
    cache: dict[str, int] = {}

    def count(segments: list[str]) -> int:
        todo = [s for s in set(segments) if s not in cache]
        if todo:
            for s, enc in zip(todo, ref.encode_batch(todo, add_special_tokens=False)):
                cache[s] = len(enc.ids)
        return sum(cache[s] for s in segments)

    return count


def _score_one(label: str, domain_texts: dict[str, list[str]], par) -> list[dict]:
    from pretok import cut_stats, pretoken_spans, with_mark_aware_pretok
    from registry import load_all

    toks, skipped = load_all(verbose=False, only=[label])
    if not toks:
        return [{"tokenizer": label, "skipped": skipped}]
    tok = toks[0]
    from pretok import unreachable_thai_tokens

    unreach = unreachable_thai_tokens(tok)
    fixed = with_mark_aware_pretok(tok)
    ref_count = _ref_counter()

    rows = []
    sets = dict(domain_texts)
    sets["parallel_th"] = par["thai"]
    for dom, texts in sets.items():
        n_chars = sum(len(t) for t in texts)
        t_act = sum(tok.count(t) for t in texts)
        t_regex = sum(fixed.count(t) for t in texts) if fixed else t_act
        pieces_all = []
        for t in texts:
            pieces_all.extend(t[a:b] for a, b in pretoken_spans(tok, t))
        t_vocab = ref_count(pieces_all)
        t_ref = ref_count(texts)
        st = cut_stats(tok, texts)
        rows.append(
            {
                "tokenizer": label,
                "family": tok.family,
                "domain": dom,
                "n_chars": n_chars,
                "T_act": t_act,
                "T_regex": t_regex,
                "T_vocab": t_vocab,
                "T_ref": t_ref,
                "pretok_pieces": st.pretok_pieces,
                "clusters": st.clusters,
                "junctions": st.junctions,
                "base_junctions": st.base_junctions,
                "pretok_junction_cuts": st.pretok_junction_cuts,
                "token_broken_clusters": st.token_broken_clusters,
                "docs_skipped_cuts": st.docs_skipped,
                "regex_fix_applicable": fixed is not None,
                "vocab_thai_tokens": unreach["thai_tokens"],
                "vocab_unreachable_thai": unreach["unreachable_thai_tokens"],
                "unreachable_examples": " | ".join(unreach["examples"]),
            }
        )
    if par is not None:
        en = sum(tok.count(t) for t in par["english"])
        for r in rows:
            if r["domain"] == "parallel_th":
                r["T_english"] = en
                r["english_chars"] = sum(len(t) for t in par["english"])
    return rows


def _per100(n, chars):
    return n / chars * 100 if chars else float("nan")


def _derive(r: dict) -> dict:
    c = r["n_chars"]
    d = {
        "tok100": _per100(r["T_act"], c),
        "tok100_regex": _per100(r["T_regex"], c),
        "tok100_vocab": _per100(r["T_vocab"], c),
        "tok100_ref": _per100(r["T_ref"], c),
        "pieces100": _per100(r["pretok_pieces"], c),
        "shatter_ratio": r["pretok_pieces"] / r["clusters"] if r["clusters"] else float("nan"),
        "pretok_cut_rate": r["pretok_junction_cuts"] / r["base_junctions"] if r["base_junctions"] else 0.0,
        "cluster_break_rate": r["token_broken_clusters"] / r["clusters"] if r["clusters"] else 0.0,
        "regex_gain": 1 - r["T_regex"] / r["T_act"],
        "vocab_ceiling_gain": 1 - r["T_vocab"] / r["T_act"],
    }
    gap = r["T_act"] - r["T_ref"]
    # Shares of a near-zero gap are noise (mT5 sits within a few percent of
    # the reference), so they are only reported for a gap over 10%.
    if gap > 0.10 * r["T_ref"]:
        d["pretok_share"] = (r["T_vocab"] - r["T_ref"]) / gap
        d["vocab_share"] = 1 - d["pretok_share"]
    else:
        d["pretok_share"] = d["vocab_share"] = float("nan")
    return d


def _flag(d: dict, penalty: float | None) -> str:
    if d["pretok_cut_rate"] > 0.01:
        return "blocked - pre-tokenizer cuts clusters; regex fix needed first"
    if penalty is not None and penalty < 1.5:
        return "already efficient - little to gain"
    return "retrofit target - damage is in the vocabulary"


def markdown(rows: list[dict], meta: dict) -> str:
    by = {(r["tokenizer"], r["domain"]): (r, _derive(r)) for r in rows}
    toks = sorted({r["tokenizer"] for r in rows})
    par = {t: by[(t, "parallel_th")] for t in toks if (t, "parallel_th") in by}

    def penalty(t, key="T_act"):
        r, _ = par[t]
        return (r[key] / r["n_chars"]) / (r["T_english"] / r["english_chars"])

    order = sorted(toks, key=lambda t: -penalty(t))
    out = ["# Penalty decomposition (Week 2)\n"]
    out.append(
        "Generated by `src/run_decomposition.py`. Reference vocabulary for the "
        f"counterfactuals: {REF_LABEL} (`{REF_REPO}`). Evaluation slice: the "
        f"first {meta['docs']} documents per domain, the same slice as "
        "`BASELINE.md`, plus the OPUS-100 parallel corpus.\n"
    )
    out.append(
        "\n**Columns.** *Pre-tok cuts* = share of consonant|mark junctions the "
        "pre-tokenizer cuts (a retrofit cannot repair these). *Cluster breaks* "
        "= share of Thai clusters (base + marks) split by the final tokens, "
        "byte-aware - replaces Week 1's mark-split rate. *Regex fix* = tok/100ch "
        "with the same vocabulary behind a mark-aware regex. *Best vocab* = "
        "tok/100ch if a strong Thai vocabulary were used inside this "
        "tokenizer's own pre-token pieces. *Pre-tok share* = fraction of the "
        "excess over the reference that no vocabulary can recover.\n"
    )

    out.append("\n## Thai penalty on matched content, and what each fix buys\n")
    out.append(
        "Penalty = Thai tok/100ch ÷ English tok/100ch on OPUS-100 pairs. "
        "`regex fix` and `best vocab` recompute the Thai side only.\n"
    )
    out.append(
        "\n| Tokenizer | Penalty | with regex fix | with best vocab (own pre-tok) "
        "| Pre-tok cuts | Pre-tok share | Vocab share | Flag |"
    )
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for t in order:
        r, d = par[t]
        p = penalty(t)
        out.append(
            f"| {t} | **{p:.2f}x** | {penalty(t, 'T_regex'):.2f}x | "
            f"{penalty(t, 'T_vocab'):.2f}x | {d['pretok_cut_rate']:.0%} | "
            f"{_pct(d['pretok_share'])} | {_pct(d['vocab_share'])} | {_flag(d, p)} |"
        )

    out.append("\n## Thai vocabulary the pre-tokenizer cannot reach\n")
    out.append(
        "Vocabulary entries containing Thai whose own string the shipped "
        "pre-tokenizer would cut apart - they can never be emitted, so the "
        "model never trained on them.\n"
    )
    out.append("\n| Tokenizer | Thai entries | Unreachable | Share | Examples |")
    out.append("| --- | ---: | ---: | ---: | --- |")
    for t in order:
        r, _ = par[t]
        n, u = r["vocab_thai_tokens"], r["vocab_unreachable_thai"]
        ex = r["unreachable_examples"].split(" | ")[:5] if u else []
        out.append(
            f"| {t} | {n:,} | {u:,} | {u / n if n else 0:.0%} | "
            + ", ".join(f"`{e}`" for e in ex)
            + " |"
        )

    for dom in DOMAINS:
        out.append(f"\n## {dom}\n")
        out.append(
            "| Tokenizer | tok/100ch | regex fix | best vocab | ref | "
            "pieces/100ch | shatter | Pre-tok cuts | Cluster breaks | "
            "Pre-tok share | Vocab share |"
        )
        out.append("| --- |" + " ---: |" * 10)
        for t in order:
            if (t, dom) not in by:
                continue
            r, d = by[(t, dom)]
            out.append(
                f"| {t} | {d['tok100']:.1f} | {d['tok100_regex']:.1f} | "
                f"{d['tok100_vocab']:.1f} | {d['tok100_ref']:.1f} | "
                f"{d['pieces100']:.1f} | {d['shatter_ratio']:.2f} | "
                f"{d['pretok_cut_rate']:.0%} | {d['cluster_break_rate']:.1%} | "
                f"{_pct(d['pretok_share'])} | {_pct(d['vocab_share'])} |"
            )
    return "\n".join(out) + "\n"


def _pct(x):
    return "n/a" if x != x else f"{x:.0%}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=1200)
    ap.add_argument("--pairs", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()
    t0 = time.time()

    from registry import HF_SPECS, TIKTOKEN_SPECS

    labels = [s[0] for s in TIKTOKEN_SPECS] + [s[0] for s in HF_SPECS]
    domain_texts = {d: load_thai_domain(d, n_docs=args.docs).texts for d in DOMAINS}
    p = load_parallel(n_pairs=args.pairs)
    par = {"thai": p.thai, "english": p.english}

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_score_one, l, domain_texts, par): l for l in labels}
        for f in futs:
            got = f.result()
            print(f"  {futs[f]}: {len(got)} rows", flush=True)
            rows.extend(r for r in got if "skipped" not in r)

    RESULTS.mkdir(exist_ok=True)
    flat = [{**r, **_derive(r)} for r in rows]
    keys = sorted({k for r in flat for k in r}, key=lambda k: list(flat[0]).index(k) if k in flat[0] else 999)
    with (RESULTS / "decomposition.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)

    md_path = RESULTS / "DECOMPOSITION.md"
    tail = ""
    if md_path.exists():
        old = md_path.read_text(encoding="utf-8")
        if MARKER in old:
            tail = old[old.index(MARKER) :]
    body = markdown(rows, {"docs": args.docs})
    md_path.write_text(body + ("\n" + tail if tail else "\n" + MARKER + "\n"), encoding="utf-8")
    (RESULTS / "decomposition_meta.json").write_text(
        json.dumps(
            {"docs_per_domain": args.docs, "parallel_pairs": args.pairs,
             "reference": REF_REPO, "seconds": round(time.time() - t0, 1)},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
