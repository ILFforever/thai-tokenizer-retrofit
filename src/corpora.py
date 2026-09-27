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

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset

# Thai block plus the combining marks that belong to it.
THAI_RE = re.compile(r"[฀-๿]")

# Streamed documents are cached here so later stages (candidate mining,
# pruning) can take large slices without re-streaming. Outside the repo on
# purpose: the project folder is synced and the cache runs to hundreds of MB.
CACHE = Path(os.environ.get("THAITOK_CACHE", Path.home() / ".cache" / "thaitok"))


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


def _stream_domain(domain: str):
    """Yield usable, deduplicated documents for one domain, in stream order."""
    min_chars, min_thai = 120, 0.55
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
    elif domain == "english":
        # English prose for the non-regression side of the retrofit: the
        # prune list is built from one slice, English cost is measured on a
        # later, disjoint slice.
        ds = load_dataset(
            "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
        )
        field_of = lambda r: r["text"]
    else:
        raise ValueError(f"unknown domain: {domain!r}")

    seen: set[str] = set()
    for row in ds:
        text = _clean(field_of(row))
        if domain == "english":
            if len(text) < min_chars:
                continue
            latin = sum(1 for c in text if c.isascii())
            if latin / len(text) < 0.9:
                continue
        elif not _is_usable(text, min_chars, min_thai):
            continue
        key = text[:200]
        if key in seen:
            continue
        seen.add(key)
        yield text


def _cached_docs(domain: str, needed: int) -> list[str]:
    """The first `needed` usable documents of a domain, from disk if cached.

    The cache is a prefix of the stream, so any slice of it is identical to
    what streaming would have produced - the Week 1 numbers stay
    reproducible from the cache.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{domain}.jsonl"
    docs: list[str] = []
    exhausted = False
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if "__exhausted__" in rec:
                    exhausted = True
                    continue
                docs.append(rec["text"])
    if len(docs) >= needed or exhausted:
        return docs[:needed]

    docs = []
    for text in _stream_domain(domain):
        docs.append(text)
        if len(docs) >= needed:
            break
    with path.open("w", encoding="utf-8") as fh:
        for text in docs:
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        if len(docs) < needed:
            fh.write(json.dumps({"__exhausted__": True}) + "\n")
    return docs


def load_thai_domain(domain: str, n_docs: int = 2000, skip: int = 0) -> Corpus:
    """Usable documents [skip, skip + n_docs) of one domain.

    Documents are deduplicated on their first 200 characters, which catches
    the boilerplate headers that repeat across news articles without the cost
    of comparing full texts.

    Evaluation always uses skip=0 (the Week 1 slice). Anything that learns
    from text - candidate mining, prune lists - must use skip >= the
    evaluation size, so nothing is tuned on the text it is scored on.
    """
    docs = _cached_docs(domain, skip + n_docs)
    return Corpus(name=domain, texts=docs[skip : skip + n_docs])


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
