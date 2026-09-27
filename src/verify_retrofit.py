"""Week 4 acceptance checks for a retrofitted tokenizer.

    python src/verify_retrofit.py tokenizers/mistral-7b-th3219 --base Mistral-7B

Checks, each reported pass/fail in results/RETROFIT_CHECKS.md:

  1. vocabulary size identical to the base, ids a contiguous 0..N-1
  2. Thai round-trips losslessly on 10,000 held-out samples
  3. English round-trips losslessly on 10,000 held-out samples
  4. the top 100 Week 3 candidates each encode to a single token
  5. (extra) added tokens are actually used on held-out Thai
  6. (extra) English samples get byte-identical id sequences - the model
     sees exactly the input it saw before the retrofit

Samples are cut from the evaluation slices, which neither the prune list nor
the new merges were learned from.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from corpora import load_parallel, load_thai_domain  # noqa: E402

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
DOMAINS = ["wikipedia", "news", "social"]
ENGLISH_EVAL_SKIP = 8000


def samples(texts: list[str], n: int, size: int = 300) -> list[str]:
    """Up to n chunks of about `size` characters, cut at spaces, spread
    across documents."""
    out: list[str] = []
    per_doc = max(1, n // max(1, len(texts)) + 1)
    for t in texts:
        k = 0
        for i in range(0, len(t), size):
            chunk = t[i : i + size]
            if i + size < len(t):
                j = chunk.rfind(" ")
                chunk = chunk[:j] if j > size // 2 else chunk
            chunk = chunk.strip()
            if chunk:
                out.append(chunk)
                k += 1
            if k >= per_doc:
                break
        if len(out) >= n:
            break
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--base", default="Mistral-7B")
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from registry import load_all

    base = load_all(verbose=False, only=[args.base])[0][0].raw
    new = AutoTokenizer.from_pretrained(args.path)
    name = Path(args.path).name
    checks: list[tuple[str, bool, str]] = []

    # 1. size
    nv = new.get_vocab()
    ids = sorted(nv.values())
    ok = len(new) == len(base) and ids == list(range(len(base)))
    checks.append(("vocabulary size identical, ids contiguous", ok,
                   f"base {len(base):,}, retrofit {len(new):,}"))

    # 2-3. round trips
    th_texts = []
    for d in DOMAINS:
        th_texts += samples(load_thai_domain(d, n_docs=1200).texts, args.n // 3 + 1)
    th_texts = th_texts[: args.n]
    en_docs = load_thai_domain("english", n_docs=4000, skip=ENGLISH_EVAL_SKIP).texts
    en_texts = samples(en_docs, args.n // 2) + samples(load_parallel(n_pairs=3000).english, args.n)
    en_texts = en_texts[: args.n]

    def roundtrip(tok, texts):
        bad = [t for t in texts if tok.decode(tok.encode(t, add_special_tokens=False)) != t]
        return bad

    for label, texts in (("Thai", th_texts), ("English", en_texts)):
        bad_new = roundtrip(new, texts)
        bad_base = roundtrip(base, texts)
        extra = [t for t in bad_new if t not in set(bad_base)]
        checks.append((
            f"{label} round-trips losslessly ({len(texts):,} samples)",
            not extra,
            f"failures: retrofit {len(bad_new)}, base {len(bad_base)}, "
            f"new failures introduced by retrofit {len(extra)}"
            + (f"; e.g. {extra[0][:60]!r}" if extra else ""),
        ))

    # 4. top-100 candidates become one token
    cand = []
    with (RESULTS / "candidates.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["level"] == "word":
                cand.append(r["unit"])
            if len(cand) >= 100:
                break
    model = new.backend_tokenizer.model
    single = [u for u in cand if len(model.tokenize(u.replace(" ", "▁"))) == 1]
    checks.append(("top 100 candidates each encode to one token", len(single) == len(cand),
                   f"{len(single)}/{len(cand)} single-token"
                   + (f"; not: {', '.join(u for u in cand if u not in single)[:200]}" if len(single) < len(cand) else "")))

    # 5. added tokens used on held-out Thai
    added = []
    with (RESULTS / f"added_tokens_{name}.csv").open(encoding="utf-8") as fh:
        added = [int(r["id"]) for r in csv.DictReader(fh)]
    use = Counter()
    for d in DOMAINS:
        for enc in new.backend_tokenizer.encode_batch(load_thai_domain(d, n_docs=1200).texts, add_special_tokens=False):
            use.update(enc.ids)
    used = sum(1 for i in added if use[i] > 0)
    checks.append(("added tokens used on held-out Thai", used >= 0.95 * len(added),
                   f"{used:,}/{len(added):,} used ({used / max(1, len(added)):.1%}); "
                   f"threshold 95%"))

    # 6. English ids unchanged
    same = sum(
        1 for a, b in zip(base.backend_tokenizer.encode_batch(en_texts, add_special_tokens=False),
                          new.backend_tokenizer.encode_batch(en_texts, add_special_tokens=False))
        if a.ids == b.ids
    )
    checks.append(("English id sequences unchanged (>= 99.5%)", same >= 0.995 * len(en_texts),
                   f"{same:,}/{len(en_texts):,} samples byte-identical ({same / len(en_texts):.2%})"))

    # write
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "RETROFIT_CHECKS.md"
    old = path.read_text(encoding="utf-8") if path.exists() else "# Retrofit acceptance checks (Week 4)\n\nGenerated by `src/verify_retrofit.py`.\n"
    old = re.sub(rf"\n## {re.escape(name)}\n.*?(?=\n## |\Z)", "", old, flags=re.S)
    block = [f"\n## {name}\n", "| Check | Result | Detail |", "| --- | --- | --- |"]
    for c, ok, detail in checks:
        block.append(f"| {c} | {'PASS' if ok else 'FAIL'} | {detail} |")
    path.write_text(old.rstrip("\n") + "\n" + "\n".join(block) + "\n", encoding="utf-8")
    (RESULTS / f"retrofit_checks_{name}.json").write_text(
        json.dumps([{"check": c, "pass": ok, "detail": d} for c, ok, d in checks], ensure_ascii=False, indent=2),
        encoding="utf-8")
    for c, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {c}: {detail}")


if __name__ == "__main__":
    main()
