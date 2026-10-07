"""Are Qwen2.5's unreachable Thai tokens under-trained, and does the "free"
regex fix help or hurt the model without retraining?

    python src/qwen_unreachable.py                 # full run, CPU only
    python src/qwen_unreachable.py --docs 3        # smoke test

Writes results/qwen_unreachable.json and results/QWEN_UNREACHABLE.md.

Why this exists. DECOMPOSITION.md found that 1,578 of Qwen2.5's 2,570 Thai
vocabulary entries can never be emitted by its shipped pre-tokenizer: the
regex matches letters with `\\p{L}+`, which excludes Thai vowel and tone marks,
so any entry that joins a consonant to its mark is cut apart before the
vocabulary is consulted. A mark-aware regex makes those entries reachable and
cuts Thai cost from 56.6 to 36.3 tokens per 100 characters. But a token that
is never emitted never appears in the training stream, so its embedding row
gets no input gradient. If those rows are untrained, the regex fix feeds the
model garbage vectors and the saving is not free.

Two tests on Qwen/Qwen2.5-0.5B (the smallest checkpoint; every Qwen2.5 size
ships the same tokenizer), CPU only:

  1. Embedding rows. Land & Bartolo (2024, "Fishing for Magikarp") find that
     under-trained tokens sit in a tight cluster with atypical norms, close
     to rows that no token maps to. Qwen's matrix has 271 such padding rows
     (ids from len(tokenizer) up to the matrix size) - a built-in control
     for "never trained". Each row is compared with that padding centroid,
     and with the mean of all real-token rows.
  2. Language modelling. Bits per character on real Thai passages, under the
     shipped tokenizer and under the mark-aware one. Bits per character is
     total negative log-likelihood divided by the characters scored, so it
     does not depend on how many tokens the text was cut into and the two
     tokenizations are directly comparable. Both see the same one-token
     prefix, <|endoftext|> (Qwen's document separator; the base model has no
     BOS), and only the passage tokens are scored.

Results are reported per domain, never pooled.
"""

from __future__ import annotations

from versions import rev

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from corpora import load_thai_domain  # noqa: E402
from pretok import MARKS, pretoken_spans, thai_token_ids, with_mark_aware_pretok  # noqa: E402
from registry import Tok, load_all  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B"
DOMAINS = ["wikipedia", "news", "social"]
RESULTS = Path(__file__).parent.parent / "results"
MARKER = "<!-- generated below this line by src/qwen_unreachable.py; text above is hand-written and preserved on re-run -->"

# A row counts as "collapsed onto the padding vector" when its distance to
# the padding centroid is under 1% of that centroid's norm. The threshold
# sits in an empty gap: unreachable Thai rows are all below 0.1%, the
# closest reachable Thai row is at 10%.
COLLAPSE = 0.01
USAGE_DOCS = 1200  # the BASELINE / DECOMPOSITION evaluation slice
GROUPS = ["unreachable_thai", "reachable_thai", "reachable_thai_seen", "other", "special", "padding"]
GROUP_LABEL = {
    "unreachable_thai": "(a) unreachable Thai",
    "reachable_thai": "(b) reachable Thai",
    "reachable_thai_seen": "(b') reachable Thai, seen in eval slice",
    "other": "(c) all other tokens",
    "special": "special tokens",
    "padding": "padding rows (no token; never trained)",
}
_MARK = re.compile(f"[{MARKS}]")
_LEADING_VOWEL = re.compile("[เ-ไ]")


# --- token groups ------------------------------------------------------------


def token_groups(tok: Tok, n_rows: int) -> dict[str, np.ndarray]:
    """Embedding-row ids for each group. (b') is filled in later from usage."""
    thai = thai_token_ids(tok)
    unreach = np.array(sorted(thai["unreachable"]))
    reach = np.array(sorted(thai["reachable"]))
    special = np.array(sorted(tok.raw.added_tokens_decoder))
    n_tok = len(tok.raw)
    other = np.setdiff1d(np.arange(n_tok), np.concatenate([unreach, reach, special]))
    return {
        "unreachable_thai": unreach,
        "reachable_thai": reach,
        "other": other,
        "special": special,
        "padding": np.arange(n_tok, n_rows),
    }


def group_of_ids(groups: dict[str, np.ndarray], n_rows: int) -> np.ndarray:
    """Lookup table id -> 0 other, 1 reachable Thai, 2 unreachable Thai, 3 special/padding."""
    g = np.zeros(n_rows, dtype=np.int8)
    g[groups["reachable_thai"]] = 1
    g[groups["unreachable_thai"]] = 2
    g[groups["special"]] = 3
    g[groups["padding"]] = 3
    return g


# --- test 1: embedding rows ---------------------------------------------------


def row_indicators(W: torch.Tensor, groups: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-row L2 norm, cosine to the mean real-token row, and distance to
    the padding centroid relative to that centroid's norm."""
    regular = np.concatenate([groups["unreachable_thai"], groups["reachable_thai"], groups["other"]])
    mean = W[regular].mean(0)
    pad_c = W[groups["padding"]].mean(0)
    return {
        "norm": W.norm(dim=1).numpy(),
        "cos_to_mean": torch.nn.functional.cosine_similarity(W, mean[None], dim=1).numpy(),
        "dist_to_padding": ((W - pad_c).norm(dim=1) / pad_c.norm()).numpy(),
    }


def _quartiles(x: np.ndarray) -> dict[str, float]:
    q = np.percentile(x, [0, 25, 50, 75, 100])
    return {"mean": float(x.mean()), "min": float(q[0]), "q1": float(q[1]),
            "median": float(q[2]), "q3": float(q[3]), "max": float(q[4])}


def row_stats(ind: dict[str, np.ndarray], groups: dict[str, np.ndarray]) -> dict:
    out = {}
    for name in GROUPS:
        ids = groups[name]
        out[name] = {"n": int(len(ids))}
        for key, vals in ind.items():
            out[name][key] = _quartiles(vals[ids])
        out[name]["collapsed"] = int((ind["dist_to_padding"][ids] < COLLAPSE).sum())
    return out


def whole_vocab_crosscheck(tok: Tok, dist: np.ndarray, groups: dict[str, np.ndarray]) -> dict:
    """Is 'unreachable' the same as 'collapsed' outside Thai too?

    Every regular vocabulary entry that decodes to whole characters is
    tested for reachability exactly as the Thai ones are (the shipped
    pre-tokenizer cuts its own string apart). Byte fragments of a
    multi-byte character cannot be tested this way and are reported
    separately.
    """
    regular = np.concatenate([groups["unreachable_thai"], groups["reachable_thai"], groups["other"]])
    pieces = tok.raw.convert_ids_to_tokens(regular.tolist())
    cells = {"unreachable": [0, 0], "reachable": [0, 0], "byte_fragment": [0, 0]}
    examples: dict[str, list[str]] = {"unreachable_non_thai": [], "collapsed_reachable": []}
    thai_ids = set(groups["unreachable_thai"]) | set(groups["reachable_thai"])
    for idx, piece in zip(regular, pieces):
        s = tok.raw.convert_tokens_to_string([piece])
        collapsed = bool(dist[idx] < COLLAPSE)
        if "�" in s:
            kind = "byte_fragment"
        elif len(pretoken_spans(tok, s)) > 1:
            kind = "unreachable"
            if idx not in thai_ids and len(examples["unreachable_non_thai"]) < 12:
                examples["unreachable_non_thai"].append(s)
        else:
            kind = "reachable"
            if collapsed and len(examples["collapsed_reachable"]) < 12:
                examples["collapsed_reachable"].append(s)
        cells[kind][int(collapsed)] += 1
    return {
        k: {"not_collapsed": v[0], "collapsed": v[1]} for k, v in cells.items()
    } | {"examples": examples}


# --- usage evidence -------------------------------------------------------------


def usage(tok: Tok, fixed: Tok, gid: np.ndarray, texts: list[str]) -> tuple[dict, set[int]]:
    """How often each group is emitted on real Thai, by both tokenizers."""
    row: dict = {"docs": len(texts), "chars": sum(len(t) for t in texts)}
    seen_b: set[int] = set()
    for name, t in [("shipped", tok), ("mark_aware", fixed)]:
        counts = np.zeros(4, dtype=np.int64)
        distinct = [set(), set(), set(), set()]
        for text in texts:
            ids = np.array(t.encode(text))
            g = gid[ids]
            counts += np.bincount(g, minlength=4)
            for k in (1, 2):
                distinct[k].update(ids[g == k].tolist())
        row[name] = {
            "tokens": int(counts.sum()),
            "reachable_thai": int(counts[1]),
            "unreachable_thai": int(counts[2]),
            "distinct_reachable_thai": len(distinct[1]),
            "distinct_unreachable_thai": len(distinct[2]),
        }
        if name == "shipped":
            seen_b = distinct[1]
    return row, seen_b


# --- test 2: language modelling ----------------------------------------------------


def truncate(text: str, max_chars: int) -> str:
    """Cut to about `max_chars` without splitting a Thai cluster.

    Prefer the last space in the final quarter (Thai puts spaces between
    phrases); otherwise back off until the cut neither separates a
    consonant from its following mark nor a leading vowel (เ แ โ ใ ไ) from
    the consonant it is written before.
    """
    if len(text) <= max_chars:
        return text
    sp = text.rfind(" ", 0, max_chars + 1)
    if sp >= 0.75 * max_chars:
        return text[:sp]
    cut = max_chars
    while cut > 0 and (_MARK.match(text[cut]) or _LEADING_VOWEL.match(text[cut - 1])):
        cut -= 1
    return text[:cut]


@torch.inference_mode()
def score(model, prefix: int, ids: list[int], unreach: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Bits for each passage token given the prefix and the tokens before
    it, and the probability mass the model puts on the unreachable-Thai
    group at each of those positions."""
    x = torch.tensor([[prefix] + ids])
    logp = torch.log_softmax(model(x).logits[0, :-1].float(), dim=-1)
    tgt = x[0, 1:]
    bits = -logp.gather(1, tgt[:, None])[:, 0] / math.log(2)
    mass = logp[:, unreach].exp().sum(-1)
    return bits.numpy(), mass.numpy()


def lm_eval(model, toks: dict[str, Tok], passages: list[str], prefix: int,
            gid: np.ndarray, unreach: torch.Tensor, log) -> dict:
    """Score every passage under each tokenizer; per-passage totals plus
    where the bits go (by the group of the target, and of the token read
    just before it)."""
    out: dict = {}
    for name, t in toks.items():
        per_bits, per_toks = [], []
        by_target = np.zeros((4, 2))  # [group] -> (bits, count)
        by_prev = np.zeros((4, 2))
        mass_sum = 0.0
        t0 = time.time()
        for i, p in enumerate(passages):
            ids = t.encode(p)
            assert t.decode(ids) == p, f"{name}: passage {i} does not round-trip"
            bits, mass = score(model, prefix, ids, unreach)
            g = gid[np.array(ids)]
            np.add.at(by_target, (g, 0), bits)
            np.add.at(by_target, (g, 1), 1)
            np.add.at(by_prev, (g[:-1], 0), bits[1:])
            np.add.at(by_prev, (g[:-1], 1), 1)
            mass_sum += float(mass.sum())
            per_bits.append(float(bits.sum()))
            per_toks.append(len(ids))
        log(f"    {name:<10} {len(passages)} passages, {sum(per_toks)} tokens, {time.time() - t0:.0f}s")
        out[name] = {
            "tokens": int(sum(per_toks)),
            "bits": float(sum(per_bits)),
            "per_passage_bits": per_bits,
            "per_passage_tokens": per_toks,
            "mean_mass_on_unreachable": mass_sum / sum(per_toks),
            "by_target": _group_table(by_target),
            "by_previous_input": _group_table(by_prev),
        }
    return out


def _group_table(arr: np.ndarray) -> dict:
    names = ["other", "reachable_thai", "unreachable_thai", "special"]
    return {
        n: {"bits": float(arr[k, 0]), "count": int(arr[k, 1]),
            "bits_per_token": float(arr[k, 0] / arr[k, 1]) if arr[k, 1] else None}
        for k, n in enumerate(names)
    }


def paired_change(base: list[float], new: list[float], chars: list[int],
                  seed: int = 0, n_boot: int = 2000) -> dict:
    """Bits per character before and after on the same passages, the change,
    and a bootstrap 95% interval over passages for that change."""
    c = np.array(chars, dtype=float)
    a, b = np.array(base), np.array(new)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), size=(n_boot, len(c)))
    diffs = (b[idx].sum(1) - a[idx].sum(1)) / c[idx].sum(1)
    total = c.sum()
    return {
        "chars": int(total),
        "bpc_base": float(a.sum() / total),
        "bpc_new": float(b.sum() / total),
        "bpc_change": float((b.sum() - a.sum()) / total),
        "bpc_change_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "passages_better": int((b < a).sum()),
    }


def paired_summary(res: dict, chars: list[int]) -> dict:
    """Shipped vs mark-aware tokenizer on the same passages."""
    p = paired_change(res["shipped"]["per_passage_bits"], res["mark_aware"]["per_passage_bits"], chars)
    total = p["chars"]
    return {
        "chars": total,
        "bpc_shipped": p["bpc_base"],
        "bpc_mark_aware": p["bpc_new"],
        "bpc_change": p["bpc_change"],
        "bpc_change_ci95": p["bpc_change_ci95"],
        "passages_mark_aware_better": p["passages_better"],
        "tok100_shipped": 100 * res["shipped"]["tokens"] / total,
        "tok100_mark_aware": 100 * res["mark_aware"]["tokens"] / total,
    }


# --- report -----------------------------------------------------------------------------


def markdown(r: dict) -> str:
    out = [MARKER, ""]
    m = r["meta"]
    out.append(
        f"Model `{m['model']}` ({m['dtype']}, CPU, {m['threads']} threads). "
        f"Input embeddings and output head tied: **{m['tied_embeddings']}** - "
        "so each row below is both how the model reads a token and how it "
        f"predicts it. Run time {m['seconds'] / 60:.1f} min.\n"
    )

    out.append("## Test 1 - embedding rows\n")
    out.append(
        "*dist to padding* = distance from the row to the centroid of the "
        f"{r['rows']['padding']['n']} padding rows, as a share of that "
        "centroid's norm. *Collapsed* = rows under "
        f"{COLLAPSE:.0%} of it, i.e. indistinguishable from a row no token "
        "uses. Medians with interquartile range.\n"
    )
    out.append("| Group | Rows | L2 norm | cos to mean row | dist to padding | Collapsed |")
    out.append("| --- | ---: | --- | --- | --- | ---: |")
    for g in GROUPS:
        s = r["rows"][g]
        out.append(
            f"| {GROUP_LABEL[g]} | {s['n']:,} | {_iqr(s['norm'])} | "
            f"{_iqr(s['cos_to_mean'])} | {_iqr(s['dist_to_padding'], 4)} | "
            f"{s['collapsed']:,} ({s['collapsed'] / s['n']:.0%}) |"
        )

    x = r["crosscheck"]
    out.append("\n**Cross-check over the whole vocabulary** (every script, not just Thai):\n")
    out.append("| Entry | Collapsed | Not collapsed |")
    out.append("| --- | ---: | ---: |")
    for k, label in [("unreachable", "unreachable under shipped pre-tokenizer"),
                     ("reachable", "reachable"),
                     ("byte_fragment", "byte fragment (cannot be tested)")]:
        out.append(f"| {label} | {x[k]['collapsed']:,} | {x[k]['not_collapsed']:,} |")
    ex = x["examples"]
    out.append(
        "\nUnreachable non-Thai examples: "
        + ", ".join(_code(e) for e in ex["unreachable_non_thai"])
        + ".  \nCollapsed but reachable examples: "
        + ", ".join(_code(e) for e in ex["collapsed_reachable"])
        + ".\n"
    )

    out.append(f"\n## Usage on the {USAGE_DOCS}-document evaluation slice\n")
    out.append("Token counts from encoding the same slice as `BASELINE.md`.\n")
    out.append(
        "| Domain | chars | shipped tok/100ch | (a) emitted, shipped | (b) emitted, shipped "
        "| distinct (b) seen | mark-aware tok/100ch | (a) share of mark-aware tokens | distinct (a) used |"
    )
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for d in DOMAINS:
        u = r["usage"][d]
        s, f = u["shipped"], u["mark_aware"]
        out.append(
            f"| {d} | {u['chars']:,} | {100 * s['tokens'] / u['chars']:.1f} | "
            f"{s['unreachable_thai']:,} | {s['reachable_thai']:,} "
            f"({s['reachable_thai'] / s['tokens']:.0%}) | {s['distinct_reachable_thai']:,} | "
            f"{100 * f['tokens'] / u['chars']:.1f} | {f['unreachable_thai'] / f['tokens']:.1%} | "
            f"{f['distinct_unreachable_thai']:,} |"
        )

    out.append("\n## Test 2 - language modelling on real Thai\n")
    out.append(
        f"First {m['docs']} documents per domain, each cut to about "
        f"{m['max_chars']} characters without splitting a Thai cluster. "
        "Bits per character (lower is better) is comparable across the two "
        "tokenizers because both score exactly the same characters. 95% "
        "interval is a bootstrap over passages.\n"
    )
    out.append(
        "| Domain | passages | chars | tok/100ch shipped | tok/100ch mark-aware | "
        "bpc shipped | bpc mark-aware | change | 95% CI | passages better with mark-aware |"
    )
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for d in DOMAINS:
        s = r["lm"][d]["summary"]
        rel = s["bpc_change"] / s["bpc_shipped"]
        lo, hi = s["bpc_change_ci95"]
        out.append(
            f"| {d} | {r['lm'][d]['passages']} | {s['chars']:,} | {s['tok100_shipped']:.1f} | "
            f"{s['tok100_mark_aware']:.1f} | {s['bpc_shipped']:.3f} | {s['bpc_mark_aware']:.3f} | "
            f"**{s['bpc_change']:+.3f} ({rel:+.0%})** | {lo:+.3f} to {hi:+.3f} | "
            f"{s['passages_mark_aware_better']}/{r['lm'][d]['passages']} |"
        )

    out.append("\n### Where the extra bits go (mark-aware tokenizer)\n")
    out.append(
        "Mean bits per predicted token, split by the group of the token being "
        "predicted (*output side*) and by the group of the token read just "
        "before it (*input side*). *Mass on (a)* = average probability the "
        "model puts on the whole unreachable group at each position.\n"
    )
    out.append(
        "| Domain | predicting (a) | predicting (b) | predicting other | after reading (a) "
        "| after reading (b) | mass on (a), shipped | mass on (a), mark-aware |"
    )
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for d in DOMAINS:
        f = r["lm"][d]["mark_aware"]
        sh = r["lm"][d]["shipped"]
        bt, bp = f["by_target"], f["by_previous_input"]
        out.append(
            f"| {d} | {_bpt(bt['unreachable_thai'])} | {_bpt(bt['reachable_thai'])} | "
            f"{_bpt(bt['other'])} | {_bpt(bp['unreachable_thai'])} | {_bpt(bp['reachable_thai'])} | "
            f"{_prob(sh['mean_mass_on_unreachable'])} | {_prob(f['mean_mass_on_unreachable'])} |"
        )
    out.append(
        "\nFor comparison, the shipped tokenizer's mean bits per token: "
        + ", ".join(
            f"{d} {r['lm'][d]['shipped']['bits'] / r['lm'][d]['shipped']['tokens']:.2f}"
            for d in DOMAINS
        )
        + ".\n"
    )
    return "\n".join(out) + "\n"


def _iqr(s: dict, nd: int = 3) -> str:
    return f"{s['median']:.{nd}f} ({s['q1']:.{nd}f}-{s['q3']:.{nd}f})"


def _code(s: str) -> str:
    """A vocabulary entry as inline code; whitespace-only entries via repr."""
    s = s.replace("`", "'").replace("|", "\\|")
    return f"`{s.strip()}`" if s.strip() else f"`{s!r}`"


def _prob(p: float) -> str:
    return f"{p:.2%}" if p >= 1e-4 else f"{p:.1e}"


def _bpt(cell: dict) -> str:
    v = cell["bits_per_token"]
    return "-" if v is None else f"{v:.2f} (n={cell['count']:,})"


def write_report(r: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "qwen_unreachable.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    md = RESULTS / "QWEN_UNREACHABLE.md"
    head = "# Qwen2.5: are the unreachable Thai tokens trained?\n\n(Conclusion not written yet.)\n\n"
    if md.exists():
        old = md.read_text(encoding="utf-8")
        if MARKER in old:
            head = old[: old.index(MARKER)]
    md.write_text(head + markdown(r), encoding="utf-8")


# --- main ---------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=50, help="passages per domain")
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--no-write", action="store_true", help="print only (smoke test)")
    args = ap.parse_args()
    t_start = time.time()
    torch.set_num_threads(args.threads)
    log = lambda s: print(s, flush=True)

    from transformers import AutoModelForCausalLM

    tok = load_all(verbose=False, only=["Qwen2.5"])[0][0]
    fixed = with_mark_aware_pretok(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, revision=rev(MODEL)).eval()
    W = model.get_input_embeddings().weight.detach()
    tied = model.get_output_embeddings().weight.data_ptr() == W.data_ptr()
    n_rows = W.shape[0]
    log(f"model loaded: {n_rows} rows, tied={tied}")

    groups = token_groups(tok, n_rows)
    gid = group_of_ids(groups, n_rows)
    log(f"groups: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))

    log("usage on the evaluation slice")
    use, seen_b = {}, set()
    for d in DOMAINS:
        texts = load_thai_domain(d, n_docs=USAGE_DOCS).texts
        use[d], seen = usage(tok, fixed, gid, texts)
        seen_b |= seen
        log(f"  {d}: (a) emitted by shipped = {use[d]['shipped']['unreachable_thai']}")
    groups["reachable_thai_seen"] = np.array(sorted(seen_b))

    log("embedding rows")
    ind = row_indicators(W, groups)
    rows = row_stats(ind, groups)
    cross = whole_vocab_crosscheck(tok, ind["dist_to_padding"], groups)

    log("language modelling")
    prefix = tok.raw.convert_tokens_to_ids("<|endoftext|>")
    unreach = torch.tensor(groups["unreachable_thai"])
    lm = {}
    for d in DOMAINS:
        docs = load_thai_domain(d, n_docs=args.docs).texts
        passages = [truncate(t, args.max_chars) for t in docs]
        log(f"  {d}: {len(passages)} passages, {sum(map(len, passages)):,} chars")
        res = lm_eval(model, {"shipped": tok, "mark_aware": fixed}, passages, prefix, gid, unreach, log)
        res["passages"] = len(passages)
        res["per_passage_chars"] = [len(p) for p in passages]
        res["summary"] = paired_summary(res, res["per_passage_chars"])
        s = res["summary"]
        log(f"    bpc shipped {s['bpc_shipped']:.3f}  mark-aware {s['bpc_mark_aware']:.3f}  "
            f"change {s['bpc_change']:+.3f} {s['bpc_change_ci95']}")
        lm[d] = res

    result = {
        "meta": {
            "model": MODEL, "dtype": "float32", "threads": args.threads,
            "tied_embeddings": tied, "embedding_rows": n_rows,
            "tokenizer_len": len(tok.raw), "docs": args.docs, "max_chars": args.max_chars,
            "usage_docs": USAGE_DOCS, "collapse_threshold": COLLAPSE,
            "prefix": "<|endoftext|>", "torch": torch.__version__,
            "seconds": round(time.time() - t_start, 1),
        },
        "rows": rows,
        "crosscheck": cross,
        "usage": use,
        "lm": lm,
    }
    if args.no_write:
        print(markdown(result))
    else:
        write_report(result)
        log(f"wrote {RESULTS / 'QWEN_UNREACHABLE.md'}")
    log(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
