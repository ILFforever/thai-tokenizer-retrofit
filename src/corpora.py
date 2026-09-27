"""Thai and English corpus loading for the tokenizer baseline.

Four sources, all openly licensed and streamed so nothing large is stored:

  wikipedia  - encyclopaedic Thai (wikimedia/wikipedia, 20231101.th)
  news       - formal Thai news prose (pythainlp/thaisum)
  social     - informal Thai, the register a Wikipedia-trained vocabulary
               is most likely to under-serve (pythainlp/wisesight_sentiment)
  parallel   - matched Thai/English sentence pairs (Helsinki-NLP/opus-100),
               the only source where a Thai-vs-English ratio is meaningful

Only `parallel` supports a like-for-like language comparison: the Thai and
English sides carry the same meaning. Comparing Wikipedia Thai against some
unrelated English text would measure the topics, not the tokenizers.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from datasets import load_dataset

# Thai block plus the combining marks that belong to it.
THAI_RE = re.compile(r"[฀-๿]")


@dataclass
class Corpus:
    """A slice of text from one domain."""

    name: str
    texts: list[str] = field(default_factory=list)

    @property
    def n_chars(self) -> int:
        return sum(len(t) for t in self.texts)

    @property
    def n_docs(self) -> int:
        return len(self.texts)

    def thai_fraction(self) -> float:
        """Share of characters in the Thai block. Guards against a slice
        that is mostly English, numerals or markup."""
        if not self.n_chars:
            return 0.0
        thai = sum(len(THAI_RE.findall(t)) for t in self.texts)
        return thai / self.n_chars


@dataclass
class ParallelCorpus:
    """Matched Thai/English pairs, which is what a ratio needs."""

    name: str
    thai: list[str] = field(default_factory=list)
    english: list[str] = field(default_factory=list)

    @property
    def n_pairs(self) -> int:
        return len(self.thai)

    @property
    def thai_chars(self) -> int:
        return sum(len(t) for t in self.thai)

    @property
    def english_chars(self) -> int:
        return sum(len(t) for t in self.english)


def _clean(text: str) -> str:
    """Normalise whitespace. Deliberately does NOT apply Unicode NFC/NFD
    normalisation: how a tokenizer copes with Thai as it is actually written
    is part of what we are measuring, so re-composing the text first would
    hide the very defect we are looking for."""
    return re.sub(r"\s+", " ", text).strip()


def _is_usable(text: str, min_chars: int, min_thai: float) -> bool:
    if len(text) < min_chars:
        return False
    thai = len(THAI_RE.findall(text))
    return thai / len(text) >= min_thai


def load_thai_domain(
    domain: str,
    n_docs: int = 2000,
    min_chars: int = 120,
    min_thai: float = 0.55,
) -> Corpus:
    """Stream one Thai domain and keep the first `n_docs` usable documents.

    Documents are deduplicated on their first 200 characters, which catches
    the boilerplate headers that repeat across news articles without the cost
    of comparing full texts.
    """
    if domain == "wikipedia":
        ds = load_dataset(
            "wikimedia/wikipedia", "20231101.th", split="train", streaming=True
        )
        field_of = lambda r: r["text"]
    elif domain == "news":
        ds = load_dataset("pythainlp/thaisum", split="train", streaming=True)
        field_of = lambda r: r["body"]
    elif domain == "social":
        ds = load_dataset(
            "pythainlp/wisesight_sentiment", split="train", streaming=True
        )
        field_of = lambda r: r["texts"]
        min_chars = 40  # social posts are legitimately short
    else:
        raise ValueError(f"unknown domain: {domain!r}")

    texts: list[str] = []
    seen: set[str] = set()
    for row in ds:
        text = _clean(field_of(row))
        if not _is_usable(text, min_chars, min_thai):
            continue
        key = text[:200]
        if key in seen:
            continue
        seen.add(key)
        texts.append(text)
        if len(texts) >= n_docs:
            break

    return Corpus(name=domain, texts=texts)


def load_parallel(n_pairs: int = 4000, min_chars: int = 40) -> ParallelCorpus:
    """Matched Thai/English sentence pairs from OPUS-100.

    Short pairs are dropped: on a five-word sentence the per-100-character
    figures are dominated by rounding.
    """
    ds = load_dataset(
        "Helsinki-NLP/opus-100", "en-th", split="train", streaming=True
    )
    thai: list[str] = []
    english: list[str] = []
    seen: set[str] = set()

    for row in ds:
        pair = row["translation"]
        th, en = _clean(pair.get("th", "")), _clean(pair.get("en", ""))
        if len(th) < min_chars or len(en) < min_chars:
            continue
        if not _is_usable(th, min_chars, 0.55):
            continue
        # English side should be mostly Latin, or the pair is mislabelled.
        latin = sum(1 for c in en if c.isascii() and c.isalpha())
        if latin / max(len(en), 1) < 0.5:
            continue
        if th in seen:
            continue
        seen.add(th)
        thai.append(th)
        english.append(en)
        if len(thai) >= n_pairs:
            break

    return ParallelCorpus(name="parallel", thai=thai, english=english)


def describe(corpus: Corpus | ParallelCorpus) -> str:
    if isinstance(corpus, ParallelCorpus):
        return (
            f"{corpus.name:<12} {corpus.n_pairs:>6} pairs  "
            f"TH {corpus.thai_chars:>9,} chars  "
            f"EN {corpus.english_chars:>9,} chars"
        )
    return (
        f"{corpus.name:<12} {corpus.n_docs:>6} docs   "
        f"{corpus.n_chars:>9,} chars   "
        f"Thai {corpus.thai_fraction():.0%}"
    )
