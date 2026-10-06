"""Train the Mistral-7B retrofit properly, on a laptop GPU (optional extra).

    python src/mistral_adapt_gpu.py --arm control
    python src/mistral_adapt_gpu.py --arm retrofit --tokenizer tokenizers/mistral-7b-th2483

Everything else in this project runs on a CPU. This script needs an NVIDIA
GPU with ~5 GB free and exists because the CPU could only give Mistral-7B's
swapped rows 160 steps (MISTRAL_ROWS_*.md), which recovered about half the
loss. Same design as qwen_adapt.py:

  control   shipped tokenizer; LoRA (rank 16) on every attention and MLP
            projection
  retrofit  2,483-slot retrofit; the swapped rows of the input embedding and
            of the (separate) output head start at the mean of their pieces
            and train alongside the same LoRA

The model is loaded in 4-bit NF4 (QLoRA) to fit a laptop GPU; the embedding
and output head stay in bfloat16 and the swapped rows are optimised as
float32 master copies. Every batch is 6 Thai windows (three domains) and 2
English windows. Quality is bits per character on held-out text, measured
under the same 4-bit weights for every condition.

--kl-english W adds W x KL(original || adapted) per token on the English
windows: the teacher is the same model with the LoRA switched off and the
swapped rows put back to their original values, i.e. the untrained Mistral.
It tests whether anchoring English output stops the English drift.
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
from embed_adapt import passages  # noqa: E402
from paths import MODELS  # noqa: E402

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
DOMAINS = ["wikipedia", "news", "social"]
BASE = "mistralai/Mistral-7B-v0.1"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@torch.no_grad()
def bpc(model, tok, texts: list[str], bos: int, dev) -> tuple[float, float]:
    bits, chars, ntok = 0.0, 0, 0
    model.eval()
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        x = torch.tensor([[bos] + ids], device=dev)
        lp = torch.log_softmax(model(input_ids=x).logits[0, :-1].float(), -1)
        bits += -lp[torch.arange(len(ids), device=dev), torch.tensor(ids, device=dev)].sum().item() / math.log(2)
        chars += len(t)
        ntok += len(ids)
    return bits / chars, ntok / chars * 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["control", "retrofit"], required=True)
    ap.add_argument("--tokenizer", default="tokenizers/mistral-7b-th2483")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--lr-rows", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--english-windows", type=int, default=2)
    ap.add_argument("--thai-windows", type=int, default=None,
                    help="Thai windows per step (default: batch - english); fewer gives a retrofit the same Thai *text* per step as the control")
    ap.add_argument("--kl-english", type=float, default=0.0,
                    help="weight of a KL penalty to the original model on the English windows (0 = off)")
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--tag", default="", help="suffix for a variant run (keeps earlier results)")
    args = ap.parse_args()
    torch.manual_seed(0)
    t0 = time.time()
    dev = torch.device("cuda")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    shipped = AutoTokenizer.from_pretrained(BASE)
    tok = shipped if args.arm == "control" else AutoTokenizer.from_pretrained(str(ROOT / args.tokenizer))
    bos = shipped.bos_token_id
    tag = (args.arm if args.arm == "control" else f"retrofit-{Path(args.tokenizer).name}") + args.tag

    ev = {d: passages(load_thai_domain(d, n_docs=1200).texts, args.n_eval, args.size) for d in DOMAINS}
    ev["english"] = passages(load_thai_domain("english_raw", n_docs=100, skip=8000).texts, args.n_eval // 2, args.size) + \
        passages(load_parallel(n_pairs=3000).english, args.n_eval // 2, args.size)

    from candidates import train_texts

    need = args.steps * args.batch * args.seq * 2
    # Thai windows per step. The loss is always divided by --batch, so with fewer
    # Thai windows each window keeps the same weight as in the control run.
    n_th = args.thai_windows if args.thai_windows is not None else args.batch - args.english_windows

    def stream(texts, cap):
        out = []
        for t in texts:
            out += [bos] + tok.encode(t, add_special_tokens=False)
            if len(out) >= cap:
                break
        return out

    th = []
    for d in DOMAINS:
        th += stream(train_texts(d), need * n_th // args.batch // len(DOMAINS) + args.seq)
    en = stream(load_thai_domain("english_raw", n_docs=800).texts, need * args.english_windows // args.batch + args.seq)
    print(f"  streams: Thai {len(th):,}, English {len(en):,} tokens ({time.time() - t0:.0f}s)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE, device_map={"": 0}, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    print(f"  loaded 4-bit: {torch.cuda.memory_allocated() / 1e9:.1f} GB on GPU ({time.time() - t0:.0f}s)", flush=True)
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight

    def evaluate(label: str) -> dict:
        r = {}
        for k, texts in ev.items():
            b, tpc = bpc(model, tok, texts, bos, dev)
            r[f"{k}_bpc"], r[f"{k}_tok100"] = round(b, 4), round(tpc, 2)
        print(f"  {label}: " + ", ".join(f"{k} {r[f'{k}_bpc']:.3f}" for k in ev) + f" ({time.time() - t0:.0f}s)", flush=True)
        return r

    results: dict[str, dict] = {}
    rows: list[int] = []
    orig_rows = None  # the swapped rows as shipped, for the KL teacher
    if args.arm == "control":
        results["original"] = evaluate("original (shipped tokenizer, no training)")
    else:
        texts = []
        with (RESULTS / f"added_tokens_{Path(args.tokenizer).name}.csv").open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(int(r["id"]))
                texts.append(r["text"])
        with torch.no_grad():
            orig_rows = [emb[rows].clone(), head[rows].clone()]
            for i, t in zip(rows, texts):
                parts = shipped.encode(t, add_special_tokens=False)
                emb[i] = emb[parts].float().mean(0).to(emb.dtype)
                head[i] = head[parts].float().mean(0).to(head.dtype)
        results["mean_init"] = evaluate("retrofit, mean-initialised rows, no training")

    # Not prepare_model_for_kbit_training: it would cast the embedding and
    # output head to float32 (+2 GB with gradients), more than a laptop GPU
    # has free. Checkpointing and input gradients are enough for QLoRA.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    lora = [p for n, p in model.named_parameters() if "lora_" in n]
    groups = [{"params": lora, "lr": args.lr_lora}]
    masters, mats = [], []
    if rows:
        idx = torch.tensor(rows, device=dev)
        mats = [model.get_input_embeddings().weight, model.get_output_embeddings().weight]
        for p in mats:
            p.requires_grad_(True)
        masters = [p.detach()[idx].float().clone().requires_grad_(True) for p in mats]
        groups.append({"params": masters, "lr": args.lr_rows})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    print(f"  training: LoRA r={args.rank} + {len(rows):,} rows x2; "
          f"{torch.cuda.memory_allocated() / 1e9:.1f} GB ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(0)
    log = []
    model.train()
    for step in range(1, args.steps + 1):
        wins = [th[w * args.seq:(w + 1) * args.seq] for w in rng.integers(0, len(th) // args.seq, n_th)]
        wins += [en[w * args.seq:(w + 1) * args.seq] for w in rng.integers(0, len(en) // args.seq, args.english_windows)]
        total, kl_total = 0.0, 0.0
        for j, w in enumerate(wins):  # one window per pass keeps memory low
            x = torch.tensor([w], device=dev)
            if args.kl_english and j >= n_th:  # an English window: anchor it to the original model
                with torch.no_grad(), model.disable_adapter():
                    if rows:
                        for p, o in zip(mats, orig_rows):
                            p[idx] = o
                    teacher = torch.log_softmax(model(input_ids=x).logits[0, :-1].float(), -1)
                    if rows:
                        for m, p in zip(masters, mats):
                            p[idx] = m.to(p.dtype)
                out = model(input_ids=x, labels=x)
                student = torch.log_softmax(out.logits[0, :-1].float(), -1)
                kl = torch.nn.functional.kl_div(student, teacher, log_target=True, reduction="sum") / student.shape[0]
                loss = (out.loss + args.kl_english * kl) / args.batch
                kl_total += kl.item() / args.english_windows
                del teacher, student
            else:
                loss = model(input_ids=x, labels=x).loss / args.batch
            loss.backward()
            total += loss.item()
        for m, p in zip(masters, mats):
            m.grad = p.grad[idx].float()
            p.grad = None
        torch.nn.utils.clip_grad_norm_(lora + masters, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for m, p in zip(masters, mats):
                p[idx] = m.to(p.dtype)
        if step % 25 == 0 or step == 1:
            log.append({"step": step, "loss": round(total, 4), "kl_english": round(kl_total, 4), "seconds": round(time.time() - t0, 1),
                        "gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
            print(f"    step {step}: loss {total:.3f}, English KL {kl_total:.4f} ({time.time() - t0:.0f}s, "
                  f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)", flush=True)
    model.eval()
    results["trained"] = evaluate(f"{args.arm} after {args.steps} steps")

    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / f"mistral-7b-qlora-{tag}"
    model.save_pretrained(str(out))  # LoRA adapter
    if rows:
        torch.save({"ids": rows, "embed": mats[0].detach()[idx].cpu(), "head": mats[1].detach()[idx].cpu()},
                   out / "swapped_rows.pt")
    tok.save_pretrained(str(out))
    meta = {"base": BASE, "quantization": "4-bit NF4, double quant, bf16 compute", "arm": args.arm,
            "tokenizer": args.tokenizer if rows else BASE, "steps": args.steps, "batch": args.batch,
            "seq": args.seq, "english_windows_per_batch": args.english_windows,
            "thai_windows_per_step": n_th, "kl_english": args.kl_english, "train_tokens_seen": args.steps * (n_th + args.english_windows) * args.seq, "lora_rank": args.rank,
            "lr_lora": args.lr_lora, "lr_rows": args.lr_rows, "rows_learned": len(rows),
            "gpu": torch.cuda.get_device_name(0), "log": log, "saved_to": f"models/{out.name}",
            "seconds": round(time.time() - t0, 1)}
    (RESULTS / f"mistral_adapt_{tag}.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2),
                                                      encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
