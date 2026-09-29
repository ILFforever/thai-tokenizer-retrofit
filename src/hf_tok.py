"""Load a tokenizer exactly as its tokenizer.json defines it.

transformers' AutoTokenizer, given a *model* folder (one with config.json),
maps the model type to a tokenizer class. For Qwen2 that class rebuilds the
pre-tokenizer with Qwen's default `\\p{L}+` pattern and ignores the one saved
in tokenizer.json - which silently undoes the mark-aware regex fix of a
retrofitted Qwen tokenizer (23 tokens become 40 on a Thai probe). Loading
through TokenizersBackend keeps the saved pipeline; `load_tokenizer` also
checks it against the `tokenizers` library on a probe and fails loudly if they
ever disagree.
"""

from __future__ import annotations

from pathlib import Path

PROBE = "การประมวลผลสัญญาณเสียง เป็นกระบวนการเกี่ยวกับ วิทยาการคอมพิวเตอร์ Hello world, the government said."


def load_tokenizer(path: str | Path):
    from transformers import AutoTokenizer

    p = Path(path)
    if not (p / "tokenizer.json").exists():  # a hub name or a slow-only folder
        return AutoTokenizer.from_pretrained(str(path))
    from tokenizers import Tokenizer
    from transformers import TokenizersBackend

    tok = TokenizersBackend.from_pretrained(str(p))
    want = Tokenizer.from_file(str(p / "tokenizer.json")).encode(PROBE, add_special_tokens=False).ids
    got = tok.encode(PROBE, add_special_tokens=False)
    if got != want:
        raise RuntimeError(f"{p}: transformers tokenizes differently from tokenizer.json ({len(got)} vs {len(want)} ids)")
    return tok
