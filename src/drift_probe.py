"""Where does the retrofit's extra English drift come from? (Mistral-7B, GPU)

    python src/drift_probe.py control_s1000
    python src/drift_probe.py retrofit_s1000

After 1,000 paired QLoRA steps English bits/char is 0.765 for the control and
0.818 for the retrofit (report §8.5). English text is tokenized identically
in both arms and never uses the 2,483 swapped ids, so one candidate is that
the retrained output-head rows of those ids take probability in English
contexts. This measures it directly on the same English passages the
training script evaluates:

  plain    bits/char as trained (should reproduce the training script)
  mass     mean probability the model puts on the 2,483 swapped ids at each
           English position
  masked   bits/char with those ids' logits set to -inf (renormalised over
           the rest); since no English target is a swapped id, masked =
           plain - mean(-log2(1 - mass)) per character

If masking brings the retrofit back to the control, the head rows explain the
drift; if not, the drift is in the shared weights (LoRA), e.g. from the
retrofit seeing more Thai text per step.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from belebele_eval import load  # noqa: E402
from corpora import load_parallel, load_thai_domain  # noqa: E402
from embed_adapt import passages  # noqa: E402

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
ARMS = {"control_s1000": "models/mistral-7b-qlora-control_s1000",
        "retrofit_s1000_charbal": "models/mistral-7b-qlora-retrofit-mistral-7b-th2483_s1000_charbal",
        "retrofit_s1000": "models/mistral-7b-qlora-retrofit-mistral-7b-th2483_s1000"}


@torch.no_grad()
def probe(model, tok, texts, bos, ids, dev) -> dict:
    bits = bits_m = mass = 0.0
    chars = npos = 0
    idx = torch.tensor(ids, device=dev)
    for t in texts:
        y = tok.encode(t, add_special_tokens=False)
        x = torch.tensor([[bos] + y], device=dev)
        logits = model(input_ids=x).logits[0, :-1].float()
        tgt = torch.tensor(y, device=dev)
        lp = torch.log_softmax(logits, -1)
        bits += -lp.gather(1, tgt[:, None]).sum().item() / math.log(2)
        mass += lp[:, idx].exp().sum(-1).sum().item()
        logits[:, idx] = float("-inf")
        bits_m += -torch.log_softmax(logits, -1).gather(1, tgt[:, None]).sum().item() / math.log(2)
        chars += len(t)
        npos += len(y)
    return {"plain_bpc": round(bits / chars, 4), "masked_bpc": round(bits_m / chars, 4),
            "mean_mass_on_swapped": round(mass / npos, 5), "positions": npos}


def main() -> None:
    dev = torch.device("cuda")
    # the same English passages as mistral_adapt_gpu.py (--n-eval 40, --size 400)
    texts = passages(load_thai_domain("english_raw", n_docs=100, skip=8000).texts, 20, 400) + \
        passages(load_parallel(n_pairs=3000).english, 20, 400)
    with (RESULTS / "added_tokens_mistral-7b-th2483.csv").open(encoding="utf-8", newline="") as fh:
        swapped = [int(r["id"]) for r in csv.DictReader(fh)]
    # one arm per process: a 4-bit 7B model plus a second one do not fit in 8 GB
    name = sys.argv[1] if len(sys.argv) > 1 else "retrofit_s1000"
    path = RESULTS / "drift_probe_mistral-7b.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    out.update({"n_passages": len(texts), "n_swapped_ids": len(swapped)})
    model, tok, bos = load("mistral", None, ARMS[name], dev)
    r = probe(model, tok, texts, bos, swapped, dev)
    out.setdefault("arms", {})[name] = r
    print(f"  {name}: {r}", flush=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
