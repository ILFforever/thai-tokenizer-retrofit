"""Week 6: what do the swapped embedding rows actually need?

    python src/embed_adapt.py --tokenizer tokenizers/bloom-th5000 --steps 300

A fixed-size retrofit keeps the embedding matrix's shape, but each swapped
row still holds the vector learned for the token it used to represent. This
script measures, on a real model (BLOOM-560m, CPU), what that costs and what
fixes it. Conditions:

  base          original tokenizer and model - the reference
  stale         retrofit tokenizer, swapped rows left as they are
  random        swapped rows re-drawn from the model's init distribution
                (what Csaki et al. 2023 did)
  mean          each swapped row set to the mean of the embeddings of the
                base tokens its string used to be split into (FVT, Gee et
                al. 2022)
  mean+tune     `mean`, then only the swapped rows trained on Thai text;
                every other parameter frozen
  random+tune   `random`, then the same training

The metric is bits per character (BPC): total negative log-likelihood in
bits divided by characters. It does not depend on how the text was split,
so a tokenizer that uses fewer tokens is compared fairly. English BPC is
reported too: BLOOM ties input and output embeddings, so new rows also
compete in the output softmax and could, in principle, disturb English.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from corpora import load_parallel, load_thai_domain  # noqa: E402

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
DOMAINS = ["wikipedia", "news", "social"]
MODEL = "bigscience/bloom-560m"


def passages(texts: list[str], n: int, size: int, skip_docs: int = 0) -> list[str]:
    """n passages of about `size` characters, one per document, cut at a
    space so no word is split."""
    out = []
    for t in texts[skip_docs:]:
        if len(t) < size // 2:
            continue
        p = t[:size]
        j = p.rfind(" ")
        if len(t) > size and j > size // 2:
            p = p[:j]
        out.append(p.strip())
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def bpc(model, tok, texts: list[str], bos: int) -> tuple[float, float]:
    """(bits per character, tokens per 100 characters) over `texts`."""
    nll_bits, chars, ntok = 0.0, 0, 0
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        x = torch.tensor([[bos] + ids])
        logits = model(x).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1)
        nll = -lp[torch.arange(len(ids)), torch.tensor(ids)].sum().item()
        nll_bits += nll / math.log(2)
        chars += len(t)
        ntok += len(ids)
    return nll_bits / chars, ntok / chars * 100


def swapped_rows(name: str) -> tuple[list[int], list[str]]:
    ids, toks = [], []
    with (RESULTS / f"added_tokens_{name}.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ids.append(int(r["id"]))
            toks.append(r["token"])
    return ids, toks


def mean_init(emb: torch.Tensor, base_tok, toks: list[str], ids: list[int]) -> None:
    """Set each swapped row to the mean of its constituent base embeddings.

    `toks` are strings in the model alphabet (byte-level for BLOOM), so they
    go straight through the base BPE model with no normalisation.
    """
    model = base_tok.backend_tokenizer.model
    for i, t in zip(ids, toks):
        parts = [p.id for p in model.tokenize(t)]
        emb[i] = emb[parts].mean(0)


def train_rows(model, tok, rows: list[int], texts: list[str], steps: int, lr: float,
               seq: int, batch: int, bos: int, log_every: int, eval_fn) -> list[dict]:
    """Train only `rows` of the (tied) embedding matrix on `texts`."""
    emb = model.get_input_embeddings().weight
    for p in model.parameters():
        p.requires_grad_(False)
    emb.requires_grad_(True)
    mask = torch.zeros(emb.shape[0], 1)
    mask[rows] = 1.0
    hook = emb.register_hook(lambda g: g * mask)
    opt = torch.optim.Adam([emb], lr=lr)

    # one long token stream, cut into fixed windows
    stream: list[int] = []
    for t in texts:
        stream += [bos] + tok.encode(t, add_special_tokens=False)
    n_win = len(stream) // seq
    rng = np.random.default_rng(0)
    log = []
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        starts = rng.integers(0, n_win, size=batch) * seq
        x = torch.tensor([stream[s : s + seq] for s in starts])
        out = model(x, labels=x)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps:
            model.eval()
            ev = eval_fn()
            model.train()
            log.append({"step": step, "train_loss": out.loss.item(), **ev,
                        "seconds": round(time.time() - t0, 1)})
            print(f"    step {step}: loss {out.loss.item():.3f} {ev} ({time.time() - t0:.0f}s)", flush=True)
    hook.remove()
    emb.requires_grad_(False)
    model.eval()
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizers/bloom-th5000")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=60, help="passages per domain")
    ap.add_argument("--size", type=int, default=500, help="characters per passage")
    args = ap.parse_args()
    torch.set_num_threads(16)
    torch.manual_seed(0)
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_tok = AutoTokenizer.from_pretrained(MODEL)
    new_tok = AutoTokenizer.from_pretrained(str(ROOT / args.tokenizer))
    name = Path(args.tokenizer).name
    rows, toks = swapped_rows(name)
    bos = base_tok.bos_token_id
    print(f"{name}: {len(rows)} swapped rows ({len(rows) / len(base_tok):.2%} of {len(base_tok):,})", flush=True)

    ev_th = {d: passages(load_thai_domain(d, n_docs=1200).texts, args.n_eval, args.size) for d in DOMAINS}
    ev_en = passages(load_thai_domain("english_raw", n_docs=200, skip=8000).texts, args.n_eval // 2, args.size) + \
        passages(load_parallel(n_pairs=3000).english, args.n_eval // 2, args.size)
    from candidates import train_texts

    train = []
    for d in DOMAINS:  # equal characters per domain
        n = 0
        for t in train_texts(d):
            train.append(t)
            n += len(t)
            if n >= 1_200_000:
                break

    def load_model():
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
        m.eval()
        return m

    def evaluate(model, tok, quick: bool = False) -> dict:
        res = {}
        for d in DOMAINS:
            texts = ev_th[d][: 15 if quick else None]
            b, tpc = bpc(model, tok, texts, bos)
            res[f"{d}_bpc"], res[f"{d}_tok100"] = round(b, 4), round(tpc, 2)
        if not quick:
            b, tpc = bpc(model, tok, ev_en, bos)
            res["english_bpc"], res["english_tok100"] = round(b, 4), round(tpc, 2)
        return res

    results: dict[str, dict] = {}
    logs: dict[str, list] = {}

    model = load_model()
    results["base"] = evaluate(model, base_tok)
    print("  base", results["base"], flush=True)
    emb = model.get_input_embeddings().weight
    orig = emb.detach().clone()

    with torch.no_grad():
        results["stale"] = evaluate(model, new_tok)
    print("  stale", results["stale"], flush=True)

    std = float(orig.std())
    for init in ("random", "mean"):
        with torch.no_grad():
            emb.copy_(orig)
            if init == "random":
                emb[rows] = torch.randn(len(rows), emb.shape[1]) * std + orig.mean(0)
            else:
                mean_init(emb, base_tok, toks, rows)
        results[init] = evaluate(model, new_tok)
        print(f"  {init}", results[init], flush=True)
        if args.steps:
            logs[f"{init}+tune"] = train_rows(
                model, new_tok, rows, train, args.steps, args.lr, args.seq, args.batch, bos,
                log_every=max(1, args.steps // 6), eval_fn=lambda: evaluate(model, new_tok, quick=True))
            results[f"{init}+tune"] = evaluate(model, new_tok)
            print(f"  {init}+tune", results[f"{init}+tune"], flush=True)
            # confirm nothing but the swapped rows moved
            with torch.no_grad():
                moved = (emb != orig).any(1).nonzero().flatten().tolist()
            results[f"{init}+tune"]["rows_changed"] = len(moved)
            results[f"{init}+tune"]["rows_changed_outside_swap"] = len(set(moved) - set(rows))

    meta = {"tokenizer": name, "model": MODEL, "swapped_rows": len(rows), "vocab": len(base_tok),
            "steps": args.steps, "lr": args.lr, "seq": args.seq, "batch": args.batch,
            "train_tokens_seen": args.steps * args.seq * args.batch,
            "eval_passages_per_domain": args.n_eval, "passage_chars": args.size,
            "seconds": round(time.time() - t0, 1)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"embed_adapt_{name}.json").write_text(
        json.dumps({"meta": meta, "results": results, "logs": logs}, indent=2), encoding="utf-8")

    md = [f"# Embedding adaptation on BLOOM-560m ({name})\n"]
    md.append(
        f"Generated by `src/embed_adapt.py`. {len(rows):,} of {len(base_tok):,} embedding rows "
        f"({len(rows) / len(base_tok):.2%}) were swapped. Bits per character (lower is better) on "
        f"held-out text, {args.n_eval} passages of ~{args.size} characters per domain. Tuning: "
        f"{args.steps} steps x {args.batch} x {args.seq} tokens "
        f"({meta['train_tokens_seen']:,} tokens), only the swapped rows trainable, Adam lr {args.lr}.\n"
    )
    md.append("\n| Condition | " + " | ".join(f"{d} BPC" for d in DOMAINS) + " | English BPC | " +
              " | ".join(f"{d} tok/100ch" for d in DOMAINS) + " |")
    md.append("| --- |" + " ---: |" * (2 * len(DOMAINS) + 1))
    for c, r in results.items():
        md.append(f"| {c} | " + " | ".join(f"{r[f'{d}_bpc']:.3f}" for d in DOMAINS) +
                  f" | {r['english_bpc']:.3f} | " + " | ".join(f"{r[f'{d}_tok100']:.1f}" for d in DOMAINS) + " |")
    (RESULTS / f"EMBED_ADAPT_{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
