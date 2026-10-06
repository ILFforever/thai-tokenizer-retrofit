"""Side-by-side answers for the Thai Token Meter page.

    python src/qwen_showcase.py --retrofit models/qwen2.5-0.5b-instruct-retrofit-qwen2-5-instruct-ma-th3000

Loads three models - the original Qwen2.5-0.5B-Instruct, the control (same
LoRA training, shipped tokenizer) and the retrofit - and answers the same
held-out Thai instructions (WangchanThaiInstruct test split, never trained
on) with greedy decoding. Records tokens, seconds and characters per second
on this CPU, and writes showcase/data.json together with the quality table
from results/qwen_adapt_*.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from hf_tok import load_tokenizer  # noqa: E402
from paths import model_dir  # noqa: E402

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
TASKS = ["Open QA", "Brainstorming", "Summarization", "Classification", "Creative writing", "Closed QA"]


def prompts(n: int) -> list[dict]:
    """Held-out instructions, one per task type where possible, short enough
    to read on a page."""
    from datasets import load_dataset

    ds = load_dataset("airesearch/WangchanThaiInstruct", split="test", streaming=True)
    # the split is sorted by domain: take one prompt per domain, preferring a
    # task type not used yet
    by_domain: dict[str, list[dict]] = {}
    for r in ds:
        user = r["Instruction"] if r["Input"] in (None, "", "None") else f"{r['Instruction']}\n\n{r['Input']}"
        if 20 <= len(user) <= 420 and r["Task_type"] != "Multiple choice":
            by_domain.setdefault(r["Domain"], []).append(
                {"label": f"{r['Domain']} · {r['Task_type']}", "prompt": user, "task": r["Task_type"]})
    picked, used = [], set()
    for dom, items in sorted(by_domain.items()):
        item = next((x for x in items if x["task"] not in used), items[0])
        used.add(item["task"])
        picked.append(item)
        if len(picked) >= n:
            return picked
    # fewer domains than prompts wanted: add unused task types from any domain
    for items in by_domain.values():
        for x in items:
            if len(picked) >= n:
                return picked
            if x["task"] not in used:
                used.add(x["task"])
                picked.append(x)
    return picked


@torch.no_grad()
def answer(model, tok, text: str, max_new: int) -> dict:
    msgs = [{"role": "user", "content": text}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):  # some versions return a BatchEncoding
        ids = ids["input_ids"]
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.1,
                         eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                         pad_token_id=tok.convert_tokens_to_ids("<|endoftext|>"))
    dt = time.time() - t0
    new = out[0, ids.shape[1]:].tolist()
    reply = tok.decode(new, skip_special_tokens=True).strip()
    return {"text": reply, "tokens": len(new), "prompt_tokens": int(ids.shape[1]), "seconds": round(dt, 2),
            "chars": len(reply), "chars_per_s": round(len(reply) / dt, 2) if dt else 0.0,
            "tokens_per_s": round(len(new) / dt, 2) if dt else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="models/qwen2.5-0.5b-instruct-control")
    ap.add_argument("--retrofit", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    arms = [
        ("original", BASE, BASE, False),
        ("control", str(model_dir(args.control)), BASE, False),
        ("retrofit", str(model_dir(args.retrofit)), str(model_dir(args.retrofit)), True),
    ]
    ps = prompts(args.n)
    ps.append({"label": "English", "prompt": "In two sentences, why do some languages cost more tokens than English?",
               "domain": "control check"})
    for arm, mpath, tpath, me in arms:
        tok = load_tokenizer(tpath)  # AutoTokenizer would drop a retrofit's saved regex
        model = AutoModelForCausalLM.from_pretrained(mpath, dtype=torch.float32)
        model.eval()
        for p in ps:
            a = answer(model, tok, p["prompt"], args.max_new)
            p.setdefault("answers", []).append({"model": arm, "me": me, **a})
            print(f"  {arm:9} {p['label']:16} {a['tokens']:4} tok {a['chars']:5} chars "
                  f"{a['seconds']:6.1f}s {a['chars_per_s']:6.1f} ch/s", flush=True)
        del model

    # quality table from the training runs
    q = {}
    for f in RESULTS.glob("qwen_adapt_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        q[d["meta"]["arm"]] = d
    cols = [("wikipedia", "Wikipedia"), ("news", "News"), ("social", "Social"),
            ("instructions", "Thai chat"), ("english", "English")]
    rows = []
    ctl, ret = q.get("control"), q.get("retrofit")
    if ctl:
        for key, name in (("original", "Original Qwen2.5-0.5B-Instruct"), ("trained", "Control: same training, original tokenizer")):
            r = ctl["results"][key]
            rows.append({"name": name, **{c: f"{r[f'{c}_bpc']:.3f}" for c, _ in cols},
                         "tok": " / ".join(f"{r[f'{c}_tok100']:.0f}" for c in ("wikipedia", "news", "social"))})
    if ret:
        r = ret["results"]["mean_init"]
        rows.append({"name": "Retrofit, rows mean-initialised, untrained",
                     **{c: f"{r[f'{c}_bpc']:.3f}" for c, _ in cols},
                     "tok": " / ".join(f"{r[f'{c}_tok100']:.0f}" for c in ("wikipedia", "news", "social"))})
        r = ret["results"]["trained"]
        rows.append({"name": "Retrofit: same training, retrofitted tokenizer", "me": True,
                     **{c: f"{r[f'{c}_bpc']:.3f}" for c, _ in cols},
                     "tok": " / ".join(f"{r[f'{c}_tok100']:.0f}" for c in ("wikipedia", "news", "social"))})

    def mean(arm, k):
        v = [a[k] for p in ps for a in p["answers"] if a["model"] == arm and p["label"] != "English"]
        return sum(v) / len(v) if v else 0.0

    speed = {arm: {"chars_per_s": round(mean(arm, "chars_per_s"), 1), "tokens_per_s": round(mean(arm, "tokens_per_s"), 1)}
             for arm, *_ in arms}
    data = {
        "title": "The retrofit in a working model: Qwen2.5-0.5B-Instruct",
        "lede": ("Two copies of the model got identical training on a laptop CPU - same data, same steps, same "
                 "LoRA - one with the original tokenizer, one with this project's retrofit. Quality is bits per "
                 "character on held-out text (lower is better); it does not depend on how the text is split."),
        "quality_columns": [{"key": c, "label": f"{n} bits/char"} for c, n in cols] +
                           [{"key": "tok", "label": "Thai tokens per 100 chars"}],
        "quality": rows,
        "speed": speed,
        "note": (f"<b>Thai text per second on this CPU:</b> original {speed['original']['chars_per_s']}, "
                 f"control {speed['control']['chars_per_s']}, retrofit {speed['retrofit']['chars_per_s']} "
                 f"characters per second at about the same tokens per second "
                 f"({speed['original']['tokens_per_s']} / {speed['control']['tokens_per_s']} / "
                 f"{speed['retrofit']['tokens_per_s']}) - each retrofit token carries more Thai. "
                 f"Answers are greedy and capped at {args.max_new} new tokens; prompts come from the "
                 f"WangchanThaiInstruct test split (CC BY-SA 4.0), which no model here was trained on. "
                 f"<b>Read the answers as a demo of tokens, not of quality</b>: a 0.5B model answers roughly "
                 f"in any form, and quality is the bits-per-character table above."),
        "prompts": [{"label": p["label"], "prompt": p["prompt"], "answers": p["answers"]} for p in ps],
    }
    (ROOT / "showcase" / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS / "qwen_showcase.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote showcase/data.json")


if __name__ == "__main__":
    main()
