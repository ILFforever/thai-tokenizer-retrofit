"""Metrics for the Thai tokenizer baseline.

The primary unit is tokens per 100 characters, not tokens per word.

Thai is written without spaces between words, so "tokens per word" is only
defined relative to a word segmenter, and different segmenters disagree.
Published work on scripts like Thai reports per-character figures for exactly
this reason and leaves per-word fertility undefined rather than quoting a
number that invites a comparison it cannot support.

Segmenter-based fertility is still computed, as a clearly-labelled secondary
figure with the segmenter named, because it is the number most readers will
expect to see.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from corpora import Corpus, ParallelCorpus
from registry import Tok, mark_split_rate


@dataclass
class DomainResult:
    tokenizer: str
    family: str
    domain: str
    n_docs: int
    n_chars: int
    n_tokens: int
    tok_per_100_chars: float
    chars_per_1k_tokens: float
    tok_per_word: float | None
    mark_split_rate: float


@dataclass
class ParallelResult:
    tokenizer: str
    family: str
    thai_tok_per_100_chars: float
    eng_tok_per_100_chars: float
    thai_penalty_ratio: float
    thai_tokens: int
    eng_tokens: int


def _segmenter():
    """PyThaiNLP's newmm segmenter, or None if unavailable.

    Named in the output so the secondary fertility figure is reproducible.
    """
    try:
        from pythainlp.tokenize import word_tokenize

        return lambda s: word_tokenize(s, engine="newmm")
    except Exception:  # noqa: BLE001
        return None


SEGMENTER_NAME = "pythainlp/newmm"


def score_domain(
    tok: Tok, corpus: Corpus, with_fertility: bool = True
) -> DomainResult:
    n_tokens = sum(tok.count(t) for t in corpus.texts)
    n_chars = corpus.n_chars

    tok_per_word = None
    if with_fertility:
        seg = _segmenter()
        if seg is not None:
            # A sample is enough: segmentation is the slow step and the mean
            # is stable well before the full corpus.
            sample = corpus.texts[:300]
            words = sum(
                len([w for w in seg(t) if w.strip()]) for t in sample
            )
            toks = sum(tok.count(t) for t in sample)
            tok_per_word = toks / words if words else None

    return DomainResult(
        tokenizer=tok.label,
        family=tok.family,
        domain=corpus.name,
        n_docs=corpus.n_docs,
        n_chars=n_chars,
        n_tokens=n_tokens,
        tok_per_100_chars=n_tokens / n_chars * 100 if n_chars else 0.0,
        chars_per_1k_tokens=n_chars / n_tokens * 1000 if n_tokens else 0.0,
        tok_per_word=tok_per_word,
        mark_split_rate=mark_split_rate(tok, corpus.texts),
    )


def score_parallel(tok: Tok, corpus: ParallelCorpus) -> ParallelResult:
    """The only place a Thai-vs-English ratio is meaningful, because the two
    sides of this corpus carry the same meaning."""
    th_tokens = sum(tok.count(t) for t in corpus.thai)
    en_tokens = sum(tok.count(t) for t in corpus.english)

    th_rate = th_tokens / corpus.thai_chars * 100 if corpus.thai_chars else 0.0
    en_rate = (
        en_tokens / corpus.english_chars * 100 if corpus.english_chars else 0.0
    )

    return ParallelResult(
        tokenizer=tok.label,
        family=tok.family,
        thai_tok_per_100_chars=th_rate,
        eng_tok_per_100_chars=en_rate,
        thai_penalty_ratio=th_rate / en_rate if en_rate else 0.0,
        thai_tokens=th_tokens,
        eng_tokens=en_tokens,
    )


def to_rows(results) -> list[dict]:
    return [asdict(r) for r in results]
