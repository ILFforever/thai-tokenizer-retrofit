"""Does the retrofit keep what the model can *do*? Belebele reading comprehension.

    python src/belebele_eval.py --kind mistral                      # original
    python src/belebele_eval.py --kind mistral --adapter models/mistral-7b-qlora-control_s1000
    python src/belebele_eval.py --kind mistral --adapter models/mistral-7b-qlora-retrofit-mistral-7b-th2483_s1000
    python src/belebele_eval.py --kind qwen --model models/qwen2.5-0.5b-instruct-control
    python src/belebele_eval.py --kind qwen --base Qwen/Qwen2.5-3B-Instruct --adapter models/qwen2.5-3b-instruct-control_gpu2000

Belebele (Bandarkar et al., 2024; CC BY-SA 4.0) gives a FLORES passage, a
question and four answers, in parallel across languages. Each answer is
scored by the model's log-likelihood of the answer text after the passage and
question, divided by the answer's length in characters (so a tokenizer that
spends fewer tokens is not favoured or penalised), and the highest-scoring
answer is the model's choice. Thai and English are both run: the retrofit
should raise Thai efficiency without costing English ability.

Accuracy carries a 95% Wilson interval; chance is 25%.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
MISTRAL = "mistralai/Mistral-7B-v0.1"
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
TEMPLATE = {
    "tha_Thai": "{p}\n\nคำถาม: {q}\nคำตอบ:",
    "eng_Latn": "{p}\n\nQuestion: {q}\nAnswer:",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(kind: str, model_path: str | None, adapter: str | None, dev, base: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if kind == "mistral" or base:  # 4-bit base + optional LoRA adapter (+ learned rows)
        from transformers import BitsAndBytesConfig

        from hf_tok import load_tokenizer

        name = base or MISTRAL
        model = AutoModelForCausalLM.from_pretrained(
            name, device_map={"": 0}, dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                                   bnb_4bit_use_double_quant=True))
        # Qwen: hf_tok, because AutoTokenizer would drop a retrofit's saved regex
        tok = AutoTokenizer.from_pretrained(name) if kind == "mistral" else load_tokenizer(name)
        if adapter:
            ad = ROOT / adapter
            rows = ad / "swapped_rows.pt"
            if rows.exists():  # retrofit arm: its tokenizer and trained rows
                tok = AutoTokenizer.from_pretrained(str(ad)) if kind == "mistral" else load_tokenizer(ad)
                r = torch.load(rows)
                idx = torch.tensor(r["ids"], device=dev)
                with torch.no_grad():
                    model.get_input_embeddings().weight[idx] = r["embed"].to(dev, torch.bfloat16)
                    if not r.get("tied"):
                        model.get_output_embeddings().weight[idx] = r["head"].to(dev, torch.bfloat16)
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(ad))
        bos = tok.bos_token_id if kind == "mistral" else tok.convert_tokens_to_ids("<|endoftext|>")
    else:
        path = str(ROOT / model_path) if model_path else QWEN
        from hf_tok import load_tokenizer  # AutoTokenizer would drop a retrofit's saved regex

        tok = load_tokenizer(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32)
        model.to(dev)
        bos = tok.convert_tokens_to_ids("<|endoftext|>")
    model.eval()
    return model, tok, bos


@torch.no_grad()
def score(model, tok, bos: int, ctx: str, answer: str, dev) -> float:
    """Log-likelihood of `answer` after `ctx`, per character of the answer."""
    ctx_ids = tok.encode(ctx, add_special_tokens=False)
    full = tok.encode(ctx + " " + answer, add_special_tokens=False)
    if full[: len(ctx_ids)] != ctx_ids:  # the join merged across the boundary
        full = ctx_ids + tok.encode(" " + answer, add_special_tokens=False)
    ans = full[len(ctx_ids):]
    if not ans:
        return -1e9
    x = torch.tensor([[bos] + full], device=dev)
    lp = torch.log_softmax(model(input_ids=x).logits[0, :-1].float(), -1)
    pos = torch.arange(len(ctx_ids), len(full), device=dev)
    return lp[pos, torch.tensor(ans, device=dev)].sum().item() / max(1, len(answer))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["mistral", "qwen"], required=True)
    ap.add_argument("--model", default=None, help="qwen: local model folder (default: the original)")
    ap.add_argument("--adapter", default=None, help="LoRA folder from mistral_adapt_gpu.py, or qwen_adapt.py --quant4")
    ap.add_argument("--base", default=None, help="qwen: a larger Qwen2.5, loaded in 4-bit (e.g. Qwen/Qwen2.5-3B-Instruct)")
    ap.add_argument("--langs", nargs="+", default=["tha_Thai", "eng_Latn"])
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--name", required=True, help="label for the results file")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if dev.type == "cpu":
        torch.set_num_threads(16)
    t0 = time.time()

    from datasets import load_dataset

    model, tok, bos = load(args.kind, args.model, args.adapter, dev, args.base)
    out = {"name": args.name, "kind": args.kind, "model": args.model, "base": args.base, "adapter": args.adapter,
           "device": str(dev), "langs": {}}
    for lang in args.langs:
        ds = load_dataset("facebook/belebele", lang, split="test")
        n = min(args.n, len(ds))
        right, ntok, nchar = 0, 0, 0
        for i in range(n):
            r = ds[i]
            ctx = TEMPLATE[lang].format(p=r["flores_passage"], q=r["question"])
            opts = [r[f"mc_answer{k}"] for k in range(1, 5)]
            s = [score(model, tok, bos, ctx, o, dev) for o in opts]
            pick = max(range(4), key=lambda k: s[k]) + 1
            right += int(pick == int(r["correct_answer_num"]))
            ntok += len(tok.encode(r["flores_passage"], add_special_tokens=False))
            nchar += len(r["flores_passage"])
        lo, hi = wilson(right, n)
        out["langs"][lang] = {"n": n, "correct": right, "accuracy": right / n, "ci95": [lo, hi],
                              "passage_tok100": ntok / nchar * 100}
        print(f"  {args.name} {lang}: {right}/{n} = {right / n:.1%} [{lo:.1%}, {hi:.1%}], "
              f"passages {ntok / nchar * 100:.1f} tok/100ch ({time.time() - t0:.0f}s)", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    (RESULTS / f"belebele_{args.name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
