"""Export tokenizers in a compact form for the in-browser token meter.

    python src/export_web_tokenizers.py

Writes showcase/tok/<name>.json: the vocabulary as an id-ordered list, the
merges as "a b" strings in priority order, and the normalizer and
pre-tokenizer settings - exactly what showcase/bpe.js needs to reproduce the
Rust tokenizer. Also writes showcase/tok/samples.json, real held-out texts
used by the Node test that checks the JavaScript output id-for-id against
the Python tokenizers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent
OUT = ROOT / "showcase" / "tok"

TOKENIZERS = {
    "mistral": ("Mistral-7B", "mistralai/Mistral-7B-v0.1"),
    "mistral_th": ("Mistral-7B + 2,483 Thai (this project)", str(ROOT / "tokenizers" / "mistral-7b-th2483")),
    "typhoon": ("Typhoon-7B (expanded)", "scb10x/typhoon-7b"),
    "qwen": ("Qwen2.5", "Qwen/Qwen2.5-0.5B-Instruct"),
    "qwen_th": ("Qwen2.5 + regex fix + 3,000 Thai (this project)",
                str(ROOT / "tokenizers" / "qwen2-5-instruct-ma-th3000")),
}


def _dump(data: dict) -> str:
    """Compact JSON. U+FFFD is a real vocabulary entry in some tokenizers;
    write it as an escape so the file never contains the raw character."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("�", "\\ufffd")


def export(key: str, label: str, path: str) -> dict:
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(path)
    j = json.loads(hf.backend_tokenizer.to_str())
    m = j["model"]
    assert m["type"] == "BPE", key
    n = max(m["vocab"].values()) + 1
    vocab = [None] * n
    for t, i in m["vocab"].items():
        vocab[i] = t
    merges = [" ".join(x) if isinstance(x, list) else x for x in m["merges"]]
    data = {
        "key": key,
        "label": label,
        "vocab": vocab,
        "merges": merges,
        "byte_fallback": bool(m.get("byte_fallback")),
        "unk": m.get("unk_token"),
        "normalizer": j.get("normalizer"),
        "pre_tokenizer": j.get("pre_tokenizer"),
        "vocab_size": len(hf),
    }
    (OUT / f"{key}.json").write_text(_dump(data), encoding="utf-8")
    return {"key": key, "label": label, "path": path}


TIKTOKEN = {
    "gpt4": ("GPT-4 (cl100k)", "cl100k_base"),
    "gpt4o": ("GPT-4o (o200k)", "o200k_base"),
}


def export_tiktoken(key: str, label: str, name: str) -> dict:
    """tiktoken ranks as a byte-level vocabulary; bpe.js merges by rank."""
    import tiktoken

    from pretok import _bytes_to_unicode

    enc = tiktoken.get_encoding(name)
    b2u = _bytes_to_unicode()
    vocab = [None] * enc.n_vocab
    for tok_bytes, rank in enc._mergeable_ranks.items():
        vocab[rank] = "".join(b2u[b] for b in tok_bytes)
    data = {
        "key": key, "label": label, "vocab": vocab, "merges": [], "merge_by_rank": True,
        "byte_fallback": False, "unk": None, "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": enc._pat_str}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel"}]},
        "vocab_size": enc.n_vocab,
    }
    (OUT / f"{key}.json").write_text(_dump(data), encoding="utf-8")
    return {"key": key, "label": label, "path": name}


def samples() -> dict:
    """Held-out texts for the equivalence test, trimmed to typical meter input."""
    from corpora import load_parallel, load_thai_domain

    out = []
    for d in ("wikipedia", "news", "social"):
        for t in load_thai_domain(d, n_docs=400).texts:
            out.append(t[:700])
    out += [t[:700] for t in load_thai_domain("english_raw", n_docs=300, skip=8000).texts]
    par = load_parallel(n_pairs=600)
    out += par.thai[:300] + par.english[:300]
    out += ["", " ", "ภาษาไทย", " นำ้ ", "ๆๆ ๆ", "ฮา ฮา 555", "Hello, สวัสดีครับ! 123", "\n\nทดสอบ\tแท็บ",
            "น้ำ่่ ่", "เเละ", "ก็ได้ ำํ", "é café", "😀 อิโมจิ", "  สองช่อง  "]
    return {"texts": out}


def main() -> None:
    from transformers import AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    import tiktoken

    manifest = [export(k, label, path) for k, (label, path) in TOKENIZERS.items()]
    manifest += [export_tiktoken(k, label, name) for k, (label, name) in TIKTOKEN.items()]
    s = samples()
    # expected ids from the real (Rust) tokenizers
    for k, (_, path) in TOKENIZERS.items():
        hf = AutoTokenizer.from_pretrained(path)
        s[k] = [hf.encode(t, add_special_tokens=False) for t in s["texts"]]
    for k, (_, name) in TIKTOKEN.items():
        enc = tiktoken.get_encoding(name)
        s[k] = [enc.encode(t, disallowed_special=()) for t in s["texts"]]
    (OUT / "samples.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    for f in sorted(OUT.glob("*.json")):
        print(f"  {f.name:24} {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
