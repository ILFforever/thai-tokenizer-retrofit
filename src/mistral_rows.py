"""Do the swapped rows work at the target scale? Mistral-7B, no training.

    python src/mistral_rows.py --tokenizer tokenizers/mistral-7b-th2483

The BLOOM-560m experiment (embed_adapt.py) shows that setting each swapped
row to the mean of the embeddings of the pieces it replaces recovers most of
the loss without training. Mistral-7B cannot be tuned on this machine, but it
can be *evaluated* in bfloat16 on a CPU. This script scores a small held-out
sample under three conditions:

  base   original tokenizer, original model
  stale  retrofit tokenizer, swapped rows left as they were
  mean   retrofit tokenizer, each swapped row - in the input embedding and,
         separately, in the output head (Mistral does not tie them) - set to
         the mean of the rows of the base tokens its string used to be split
         into

Bits per character is the metric: it does not depend on the tokenizer.
"""

from __future__ import annotations

from versions import rev

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from corpora import load_parallel, load_thai_domain  # noqa: E402
from embed_adapt import passages  # noqa: E402

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
DOMAINS = ["wikipedia", "news", "social"]
MODEL = "mistralai/Mistral-7B-v0.1"


@torch.no_grad()
def bpc(model, tok, texts: list[str], bos: int) -> tuple[float, float, list[float]]:
    """(bits per character, tokens per 100 characters, per-passage bits)."""
    bits, chars, ntok, per = 0.0, 0, 0, []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        x = torch.tensor([[bos] + ids])
        logits = model(x).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1)
        b = -lp[torch.arange(len(ids)), torch.tensor(ids)].sum().item() / math.log(2)
        per.append(b)
        bits += b
        chars += len(t)
        ntok += len(ids)
    return bits / chars, ntok / chars * 100, per


def tune_rows(model, tok, rows: list[int], bos: int, args, tied: bool, t0: float, quick) -> list[dict]:
    """Train only the swapped rows of the input embedding and output head.

    The model stays in bfloat16, but the swapped rows are optimised as
    float32 master copies and written back after every step: an Adam step of
    lr 1e-4 on bfloat16 weights of this size would partly round away. One
    window per batch is English (protect-corpus documents, never the
    evaluation text) so the new output rows learn to stay quiet in English -
    Qwen2.5's Thai-only repair leaked into English.
    """
    import numpy as np

    from candidates import train_texts

    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    for p in model.parameters():
        p.requires_grad_(False)
    params = [emb] if tied else [emb, head]
    for p in params:
        p.requires_grad_(True)
    idx = torch.tensor(rows)
    masters = [p.detach()[idx].float().clone().requires_grad_(True) for p in params]
    opt = torch.optim.Adam(masters, lr=args.lr)

    def stream(texts: list[str], cap: int) -> list[int]:
        out: list[int] = []
        for t in texts:
            out += [bos] + tok.encode(t, add_special_tokens=False)
            if len(out) >= cap:
                break
        return out

    need = args.tune_steps * args.batch * args.seq * 2
    th = []
    for d in DOMAINS:  # equal share per domain
        th += stream(train_texts(d), need // len(DOMAINS))
    en = stream(load_thai_domain("english_raw", n_docs=300).texts, need) if args.english_windows else []
    rng = np.random.default_rng(0)
    n_th, n_en = len(th) // args.seq, max(1, len(en) // args.seq)
    log = [{"step": 0, **quick(), "seconds": round(time.time() - t0, 1)}]
    print(f"    step 0: {log[-1]}", flush=True)
    model.train()
    every = max(1, args.tune_steps // 4)
    for step in range(1, args.tune_steps + 1):
        wins = [th[s * args.seq:(s + 1) * args.seq] for s in rng.integers(0, n_th, args.batch - args.english_windows)]
        wins += [en[s * args.seq:(s + 1) * args.seq] for s in rng.integers(0, n_en, args.english_windows)]
        x = torch.tensor(wins)
        loss = model(x, labels=x).loss
        loss.backward()
        for m, p in zip(masters, params):
            m.grad = p.grad[idx].float()
            p.grad = None
        torch.nn.utils.clip_grad_norm_(masters, 1.0)
        opt.step()
        opt.zero_grad()
        with torch.no_grad():
            for m, p in zip(masters, params):
                p[idx] = m.to(p.dtype)
        if step % every == 0 or step == args.tune_steps:
            model.eval()
            log.append({"step": step, "train_loss": round(loss.item(), 4), **quick(),
                        "seconds": round(time.time() - t0, 1)})
            model.train()
            print(f"    step {step}: {log[-1]}", flush=True)
    model.eval()
    for p in params:
        p.requires_grad_(False)
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizers/mistral-7b-th2483")
    ap.add_argument("--n-eval", type=int, default=20, help="passages per domain")
    ap.add_argument("--size", type=int, default=400, help="characters per passage")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--tune-steps", type=int, default=0,
                    help="after mean init, train only the swapped rows (reuses saved base/stale/mean)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--english-windows", type=int, default=1,
                    help="windows per batch from English protect-corpus text (not eval)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_tok = AutoTokenizer.from_pretrained(MODEL, revision=rev(MODEL))
    new_tok = AutoTokenizer.from_pretrained(str(ROOT / args.tokenizer))
    name = Path(args.tokenizer).name
    rows, toks = [], []
    with (RESULTS / f"added_tokens_{name}.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(int(r["id"]))
            toks.append(r["token"])
    bos = base_tok.bos_token_id

    ev = {d: passages(load_thai_domain(d, n_docs=1200).texts, args.n_eval, args.size) for d in DOMAINS}
    ev["english"] = passages(load_thai_domain("english_raw", n_docs=100, skip=8000).texts,
                             args.n_eval // 2, args.size) + \
        passages(load_parallel(n_pairs=3000).english, args.n_eval // 2, args.size)

    print("loading model (bfloat16)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, revision=rev(MODEL))
    model.eval()
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    results: dict[str, dict] = {}

    def run(label: str, tok) -> None:
        r = {}
        for k, texts in ev.items():
            b, tpc, _ = bpc(model, tok, texts, bos)
            r[f"{k}_bpc"], r[f"{k}_tok100"] = round(b, 4), round(tpc, 2)
            print(f"    {label} {k}: {b:.3f} bpc, {tpc:.1f} tok/100ch ({time.time() - t0:.0f}s)", flush=True)
        results[label] = r

    out_json = RESULTS / f"mistral_rows_{name}.json"
    saved = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else None
    reuse = bool(args.tune_steps and saved)
    if reuse:
        results.update(saved["results"])
        print("  base/stale/mean reused from", out_json.name, flush=True)
    else:
        run("base", base_tok)
        run("stale", new_tok)

    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    tied = emb.data_ptr() == head.data_ptr()
    bmodel = base_tok.backend_tokenizer.model
    with torch.no_grad():
        for i, t in zip(rows, toks):
            parts = [p.id for p in bmodel.tokenize(t)]
            emb[i] = emb[parts].float().mean(0).to(emb.dtype)
            if not tied:
                head[i] = head[parts].float().mean(0).to(head.dtype)
    if not reuse:
        run("mean", new_tok)

    log = []
    if args.tune_steps:
        log = tune_rows(model, new_tok, rows, bos, args, tied, t0,
                        quick=lambda: {d: round(bpc(model, new_tok, ev[d][:10], bos)[0], 4) for d in DOMAINS})
        run("mean+tune", new_tok)

    meta = {"model": MODEL, "dtype": "bfloat16", "tokenizer": name, "swapped_rows": len(rows),
            "tied_embeddings": tied, "passages_per_domain": args.n_eval, "passage_chars": args.size,
            "english_passages": len(ev["english"]), "seconds": round(time.time() - t0, 1)}
    if args.tune_steps:
        meta["tune"] = {"steps": args.tune_steps, "lr": args.lr, "batch": args.batch, "seq": args.seq,
                        "english_windows_per_batch": args.english_windows,
                        "train_tokens_seen": args.tune_steps * args.batch * args.seq, "log": log}
    (RESULTS / f"mistral_rows_{name}.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")

    keys = DOMAINS + ["english"]
    md = [f"# Swapped rows at the target scale: Mistral-7B ({name})\n",
          f"Generated by `src/mistral_rows.py`. {len(rows):,} swapped rows; bfloat16 on CPU; "
          f"{args.n_eval} held-out passages of ~{args.size} characters per Thai domain and "
          f"{len(ev['english'])} English passages. Bits per character (lower is better); input "
          f"and output embeddings {'tied' if tied else 'separate, both mean-initialised'}.\n",
          "| Condition | " + " | ".join(f"{k} BPC" for k in keys) + " | " +
          " | ".join(f"{k} tok/100ch" for k in DOMAINS) + " |",
          "| --- |" + " ---: |" * (len(keys) + len(DOMAINS))]
    for c, r in results.items():
        md.append(f"| {c} | " + " | ".join(f"{r[f'{k}_bpc']:.3f}" for k in keys) + " | " +
                  " | ".join(f"{r[f'{k}_tok100']:.1f}" for k in DOMAINS) + " |")
    (RESULTS / f"MISTRAL_ROWS_{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
