"""Turn the tokenizer retrofit into a working model: Qwen2.5-0.5B-Instruct.

    python src/qwen_adapt.py --arm control
    python src/qwen_adapt.py --arm retrofit --tokenizer tokenizers/qwen2-5-instruct-ma-th3000

Two arms, identical data, steps and LoRA settings, so any difference between
them comes from the tokenizer and not from the extra training:

  control   the shipped tokenizer; LoRA on every attention and MLP projection
  retrofit  the retrofitted tokenizer (mark-aware regex + Thai merges in freed
            slots); every row the model must learn - the new Thai tokens and
            the 1,578 Thai tokens the shipped regex never let it see - starts
            at the mean of the embeddings of the pieces the shipped tokenizer
            cuts its string into, then those rows train alongside the LoRA

Each batch mixes Thai text (three domains, equal share), English text, and
Thai instructions in the model's chat format, so neither English nor the
ability to follow instructions is trained away. Evaluation is bits per
character on held-out text, which does not depend on the tokenizer.

The adapted model is merged and saved to models/ (bfloat16).
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

from hf_tok import load_tokenizer  # noqa: E402

from corpora import load_parallel, load_thai_domain  # noqa: E402
from embed_adapt import passages  # noqa: E402

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
DOMAINS = ["wikipedia", "news", "social"]
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def instruction_texts(tok, n: int, split: str = "train") -> list[str]:
    """Thai instructions from WangchanThaiInstruct (CC BY-SA 4.0), rendered in
    the model's own chat template."""
    from datasets import load_dataset

    ds = load_dataset("airesearch/WangchanThaiInstruct", split=split, streaming=True)
    out = []
    for r in ds:
        user = r["Instruction"] if r["Input"] in (None, "", "None") else f"{r['Instruction']}\n\n{r['Input']}"
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": r["Output"]}]
        out.append(tok.apply_chat_template(msgs, tokenize=False))
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def bpc(model, tok, texts: list[str], prefix: int, dev=torch.device("cpu")) -> tuple[float, float]:
    bits, chars, ntok = 0.0, 0, 0
    model.eval()
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        x = torch.tensor([[prefix] + ids], device=dev)
        tgt = torch.tensor(ids, device=dev)
        # project to the 152k vocabulary in slices of 256 positions: full-passage
        # logits alone are ~2 GB in float32, more than an 8 GB laptop GPU has free
        hidden = model.get_decoder()(input_ids=x).last_hidden_state[0, :-1]
        head = model.get_output_embeddings()
        for i in range(0, len(ids), 256):
            lp = torch.log_softmax(head(hidden[i:i + 256]).float(), -1)
            bits += -lp.gather(1, tgt[i:i + 256, None]).sum().item() / math.log(2)
        del hidden
        chars += len(t)
        ntok += len(ids)
    return bits / chars, ntok / chars * 100


def rows_to_learn(new_tok, shipped) -> tuple[list[int], list[str]]:
    """Ids (and their text) of every row the retrofit model has not seen:
    tokens added by the retrofit plus Thai tokens the shipped pre-tokenizer
    could never emit."""
    import re

    added = {}
    name = Path(new_tok.name_or_path).name
    with (RESULTS / f"added_tokens_{name}.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            added[int(r["id"])] = r["text"]
    # previously unreachable Thai entries: their string is cut apart by the
    # shipped tokenizer but is a single token under the new one
    thai = re.compile(r"[฀-๿]")
    dead = {}
    inv = {v: k for k, v in new_tok.get_vocab().items()}
    for i in range(len(new_tok)):
        if i in added or i not in inv:
            continue
        text = new_tok.convert_tokens_to_string([inv[i]])
        if thai.search(text) and len(shipped.encode(text, add_special_tokens=False)) > 1 \
                and new_tok.encode(text, add_special_tokens=False) == [i]:
            dead[i] = text
    ids = sorted(added) + sorted(dead)
    texts = [added.get(i, dead.get(i)) for i in ids]
    print(f"  rows to learn: {len(added):,} added + {len(dead):,} previously unreachable", flush=True)
    return ids, texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["control", "retrofit"], required=True)
    ap.add_argument("--tokenizer", default=None, help="retrofit arm: the retrofitted tokenizer folder")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--micro", type=int, default=2, help="windows per forward pass (memory)")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--lr-rows", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--mix", default="4,2,2", help="windows per batch: Thai text, English, Thai instructions")
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--size", type=int, default=600)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="cpu (default, the reproducible path) or cuda")
    ap.add_argument("--tag", default="", help="suffix for a variant run (keeps earlier results)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    torch.manual_seed(0)
    t0 = time.time()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    shipped = AutoTokenizer.from_pretrained(BASE)
    tok = shipped if args.arm == "control" else load_tokenizer(ROOT / args.tokenizer)
    prefix = shipped.convert_tokens_to_ids("<|endoftext|>")
    tag = (args.arm if args.arm == "control" else f"retrofit-{Path(args.tokenizer).name}") + args.tag

    # --- data ---------------------------------------------------------------
    ev = {d: passages(load_thai_domain(d, n_docs=1200).texts, args.n_eval, args.size) for d in DOMAINS}
    ev["english"] = passages(load_thai_domain("english_raw", n_docs=200, skip=8000).texts, args.n_eval // 2, args.size) + \
        passages(load_parallel(n_pairs=3000).english, args.n_eval // 2, args.size)
    ev["instructions"] = [t for t in instruction_texts(shipped, 40, split="test")]

    from candidates import train_texts

    n_th, n_en, n_in = (int(x) for x in args.mix.split(","))
    assert n_th + n_en + n_in == args.batch
    need = args.steps * args.batch * args.seq * 2  # every window seen about once, with margin

    def stream(texts, cap):
        out = []
        for t in texts:
            out += [prefix] + tok.encode(t, add_special_tokens=False)
            if len(out) >= cap:
                break
        return out

    th = []
    for d in DOMAINS:
        th += stream(train_texts(d), need * n_th // args.batch // len(DOMAINS) + args.seq)
    en = stream(load_thai_domain("english_raw", n_docs=600).texts, need * n_en // args.batch + args.seq)
    ins = stream(instruction_texts(shipped, 6000), need * n_in // args.batch + args.seq)
    streams = [(th, n_th), (en, n_en), (ins, n_in)]
    print(f"  streams: Thai {len(th):,}, English {len(en):,}, instructions {len(ins):,} tokens "
          f"({time.time() - t0:.0f}s)", flush=True)

    # --- model --------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).to(dev)
    emb = model.get_input_embeddings().weight
    tied = emb.data_ptr() == model.get_output_embeddings().weight.data_ptr()

    def evaluate(label: str) -> dict:
        # On a GPU run, evaluate on the CPU: the longest evaluation texts need
        # more attention memory in float32 than an 8 GB laptop GPU has, and the
        # CPU gives numbers directly comparable with the CPU-only runs.
        # Module.to() keeps the same Parameter objects, so the optimizer is unaffected.
        cpu = torch.device("cpu")
        if dev.type == "cuda":
            model.to(cpu)
            torch.cuda.empty_cache()
        r = {}
        for k, texts in ev.items():
            b, tpc = bpc(model, tok, texts, prefix, cpu)
            r[f"{k}_bpc"], r[f"{k}_tok100"] = round(b, 4), round(tpc, 2)
        if dev.type == "cuda":
            model.to(dev)
        print(f"  {label}: " + ", ".join(f"{k} {r[f'{k}_bpc']:.3f}" for k in ev), flush=True)
        return r

    results: dict[str, dict] = {}
    rows: list[int] = []
    if args.arm == "control":
        results["original"] = evaluate("original (shipped tokenizer, no training)")
    else:
        rows, texts = rows_to_learn(tok, shipped)
        with torch.no_grad():
            for i, t in zip(rows, texts):
                parts = shipped.encode(t, add_special_tokens=False)
                emb[i] = emb[parts].mean(0)
        results["mean_init"] = evaluate("retrofit, mean-initialised rows, no training")

    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    groups = [{"params": [p for n, p in model.named_parameters() if "lora_" in n], "lr": args.lr_lora}]
    if rows:
        e = model.get_input_embeddings().weight
        e.requires_grad_(True)
        mask = torch.zeros(e.shape[0], 1, device=dev)
        mask[rows] = 1.0
        e.register_hook(lambda g: g * mask)
        groups.append({"params": [e], "lr": args.lr_rows})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    print(f"  trainable: LoRA r={args.rank} + {len(rows):,} embedding rows ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(0)
    log = []
    model.train()
    for step in range(1, args.steps + 1):
        wins = []
        for s, n in streams:
            nw = len(s) // args.seq
            wins += [s[w * args.seq:(w + 1) * args.seq] for w in rng.integers(0, nw, n)]
        opt.zero_grad()
        total = 0.0
        for i in range(0, len(wins), args.micro):
            x = torch.tensor(wins[i:i + args.micro], device=dev)
            loss = model(input_ids=x, labels=x).loss * len(x) / len(wins)
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            log.append({"step": step, "loss": round(total, 4), "seconds": round(time.time() - t0, 1)})
            print(f"    step {step}: loss {total:.3f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    model = model.merge_and_unload()
    results["trained"] = evaluate(f"{args.arm} after {args.steps} steps")

    if rows:
        base_emb = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).get_input_embeddings().weight
        with torch.no_grad():
            moved = (model.get_input_embeddings().weight.cpu() != base_emb).any(1).nonzero().flatten().tolist()
        results["trained"]["rows_changed"] = len(moved)
        results["trained"]["rows_changed_outside_learned"] = len(set(moved) - set(rows))

    MODELS.mkdir(exist_ok=True)
    out = MODELS / f"qwen2.5-0.5b-instruct-{tag}"
    model.to(torch.bfloat16).save_pretrained(str(out))
    tok.save_pretrained(str(out))
    meta = {"base": BASE, "arm": args.arm, "device": args.device, "tag": args.tag, "tokenizer": args.tokenizer or BASE, "tied_embeddings": tied,
            "steps": args.steps, "batch": args.batch, "seq": args.seq, "mix_thai_english_instr": args.mix,
            "train_tokens_seen": args.steps * args.batch * args.seq, "lora_rank": args.rank,
            "lr_lora": args.lr_lora, "lr_rows": args.lr_rows, "rows_learned": len(rows),
            "eval_passages": {k: len(v) for k, v in ev.items()}, "saved_to": str(out.relative_to(ROOT)),
            "log": log, "seconds": round(time.time() - t0, 1)}
    (RESULTS / f"qwen_adapt_{tag}.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2),
                                                   encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
