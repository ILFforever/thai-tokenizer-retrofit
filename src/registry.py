"""One interface over tiktoken and Hugging Face tokenizers.

Any tokenizer that cannot be fetched (gated, renamed, offline) is skipped and
reported rather than aborting the run, so a baseline still lands when one of
the models is unavailable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import tiktoken
from transformers import AutoTokenizer

# Thai combining marks: vowels above/below and the tone marks. A pre-tokenizer
# that treats these as word boundaries shatters Thai before the vocabulary is
# ever consulted, which no amount of vocabulary can repair.
THAI_MARKS = re.compile(r"[ัิ-ฺ็-๎]")


@dataclass
class Tok:
    """A tokenizer plus the metadata we report alongside its numbers."""

    label: str
    family: str  # "closed" or "open-weight"
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]
    vocab_size: int
    note: str = ""
    # The underlying tiktoken Encoding or HF tokenizer, for code that needs
    # more than encode/decode (pre-tokenizer introspection, offsets).
    raw: Any = None
    kind: str = ""  # "tiktoken" or "hf"

    def count(self, text: str) -> int:
        return len(self.encode(text))


# (label, tiktoken encoding name, note)
TIKTOKEN_SPECS = [
    ("GPT-4 (cl100k_base)", "cl100k_base", "OpenAI, pre-2024 generation"),
    ("GPT-4o (o200k_base)", "o200k_base", "OpenAI, current generation"),
]

# (label, HF repo id, note). Gated repos are expected to fail and are skipped.
HF_SPECS = [
    ("Typhoon-7B", "scb10x/typhoon-7b", "Thai-adapted, +5k Thai tokens"),
    ("Qwen2.5", "Qwen/Qwen2.5-0.5B", "common self-hosted base"),
    ("BLOOM", "bigscience/bloom-560m", "multilingual by design"),
    ("mT5", "google/mt5-base", "multilingual by design"),
    ("XGLM", "facebook/xglm-564M", "multilingual by design"),
    ("WangchanBERTa", "airesearch/wangchanberta-base-att-spm-uncased",
     "Thai-only encoder"),
    ("Mistral-7B", "mistralai/Mistral-7B-v0.1", "common self-hosted base"),
]


def load_all(
    verbose: bool = True, only: list[str] | None = None
) -> tuple[list[Tok], list[str]]:
    """Return every tokenizer that loaded, plus labels for those that did not.

    `only` restricts loading to the given labels (worker processes load just
    the tokenizer they score).
    """
    toks: list[Tok] = []
    skipped: list[str] = []
    want = (lambda label: label in only) if only else (lambda label: True)

    for label, enc_name, note in TIKTOKEN_SPECS:
        if not want(label):
            continue
        try:
            enc = tiktoken.get_encoding(enc_name)
            toks.append(
                Tok(
                    label=label,
                    family="closed",
                    encode=enc.encode,
                    decode=enc.decode,
                    vocab_size=enc.n_vocab,
                    note=note,
                    raw=enc,
                    kind="tiktoken",
                )
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            skipped.append(f"{label} ({type(exc).__name__})")

    for label, repo, note in HF_SPECS:
        if not want(label):
            continue
        try:
            hf = AutoTokenizer.from_pretrained(repo)
            toks.append(
                Tok(
                    label=label,
                    family="open-weight",
                    encode=lambda s, _t=hf: _t.encode(s, add_special_tokens=False),
                    decode=lambda ids, _t=hf: _t.decode(ids),
                    vocab_size=len(hf),
                    note=note,
                    raw=hf,
                    kind="hf",
                )
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{label} ({type(exc).__name__})")

    if verbose:
        for t in toks:
            print(f"  loaded  {t.label:<22} vocab={t.vocab_size:>7,}  {t.note}")
        for s in skipped:
            print(f"  skipped {s}")

    return toks, skipped


def mark_split_rate(tok: Tok, samples: list[str], limit: int = 400) -> float:
    """Share of sampled tokens that BEGIN with a Thai combining mark.

    A token starting with a vowel sign or tone mark means the tokenizer cut
    between a base consonant and its mark - the cluster was broken apart. This
    is the practical signature of the pre-tokenization defect that the shatter
    ratio describes, and it is measurable without knowing the tokenizer's
    internal regex.

    Returns 0.0 when nothing could be decoded (some tokenizers do not round
    trip single ids cleanly), so treat 0.0 as "no evidence", not "clean".
    """
    starts, total = 0, 0
    for text in samples[:limit]:
        try:
            ids = tok.encode(text)
        except Exception:  # noqa: BLE001
            continue
        for tid in ids:
            try:
                piece = tok.decode([tid])
            except Exception:  # noqa: BLE001
                continue
            if not piece:
                continue
            total += 1
            if THAI_MARKS.match(piece[0]):
                starts += 1
    return starts / total if total else 0.0
