"""Pre-tokenization introspection and the penalty decomposition (Week 2).

A BPE tokenizer works in two stages. The pre-tokenizer cuts text into pieces
with a regex (or on whitespace); the vocabulary then encodes each piece
separately. No merge can span a pre-token boundary, so if the pre-tokenizer
cuts a Thai consonant away from its vowel or tone mark, no vocabulary change
can put them back in one token. This module measures both stages directly.

Why not reuse `registry.mark_split_rate()`: it decodes one token id at a time
and looks at the first character. For byte-level tokenizers (GPT-4, Qwen,
BLOOM) and byte-fallback ones (Mistral), many Thai tokens are fragments of a
3-byte UTF-8 character and decode to U+FFFD, so they are counted in the total
but can never register as a split. It also cannot tell whether a split came
from the pre-tokenizer or from a missing merge. Everything here works on
byte offsets instead, so a cut is a cut regardless of tokenizer family.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import regex
import tiktoken
from tokenizers import Regex, Tokenizer, pre_tokenizers

from registry import Tok

THAI = "฀-๿"
# Combining marks, same set as registry.THAI_MARKS: MAI HAN-AKAT, SARA I..
# PHINTHU, MAITAIKHU..YAMAKKAN.
MARKS = "ัิ-ฺ็-๎"
CLUSTER_RE = re.compile(f"[{THAI}][{MARKS}]*")
_MARK_CH = re.compile(f"[{MARKS}]")
_THAI_CH = re.compile(f"[{THAI}]")


def cluster_pieces(text: str) -> list[str]:
    """Mark-aware Thai clusters: a Thai character plus its following marks.

    A tokenizer that respects these never separates a consonant from its
    vowel or tone mark. It is a lower bound on *boundaries that are safe to
    cut at*, not a lower bound on token count - good Thai vocabularies merge
    many clusters into one token.
    """
    return CLUSTER_RE.findall(text)


def internal_junctions(text: str) -> list[int]:
    """Positions b where cutting between text[b-1] and text[b] would break a
    cluster: text[b] is a combining mark attached to a preceding Thai char."""
    return [
        m.start()
        for m in _MARK_CH.finditer(text)
        if m.start() > 0 and _THAI_CH.match(text[m.start() - 1])
    ]


# --- pre-tokenizer access -------------------------------------------------


def mark_aware(pat: str) -> str:
    """Rewrite a pre-tokenizer regex so combining marks count as letters.

    `\\p{L}` does not include Thai vowel and tone marks (they are `\\p{M}`),
    which is why cl100k-style patterns cut before every mark. Inside a
    character class `\\p{L}` becomes `\\p{L}\\p{M}`; a bare `\\p{L}` becomes
    `[\\p{L}\\p{M}]`. This is the o200k fix, applied to any pattern.
    """
    out: list[str] = []
    i, depth = 0, 0
    while i < len(pat):
        if pat.startswith("\\p{L}", i):
            out.append("\\p{L}\\p{M}" if depth else "[\\p{L}\\p{M}]")
            i += 5
            continue
        c = pat[i]
        if c == "\\":
            out.append(pat[i : i + 2])
            i += 2
            continue
        if c == "[":
            depth += 1
        elif c == "]" and depth:
            depth -= 1
        out.append(c)
        i += 1
    return "".join(out)


def _hf_split_pattern(tok: Tok) -> str | None:
    """The Split regex inside an HF pre-tokenizer, if there is one."""
    import json

    j = json.loads(tok.raw.backend_tokenizer.to_str())
    pt = j.get("pre_tokenizer")
    if not pt:
        return None
    stack = [pt]
    while stack:
        node = stack.pop()
        if node.get("type") == "Split":
            return node["pattern"].get("Regex") or re.escape(
                node["pattern"].get("String", "")
            )
        stack.extend(node.get("pretokenizers", []))
    return None


def pretoken_spans(tok: Tok, text: str) -> list[tuple[int, int]]:
    """Character spans of the pieces the pre-tokenizer produces.

    Unlike the plan's first sketch, SentencePiece-style models are not left
    blank: their pre-tokenizer is in `tokenizer.json` and is either absent
    (Typhoon: the whole text is one piece) or splits on whitespace only
    (Mistral does not split at all; mT5, XGLM, WangchanBERTa split at
    spaces). That is read from the model's own config, not assumed.
    """
    if tok.kind == "tiktoken":
        return [m.span() for m in _tiktoken_re(tok.raw._pat_str).finditer(text)]
    pt = tok.raw.backend_tokenizer.pre_tokenizer
    if pt is None:
        return [(0, len(text))] if text else []
    # Run on the raw text so offsets are in original coordinates. The
    # normalizers involved (NFC; SentencePiece's NFKC-like charsmap) do not
    # move whitespace or regex-letter boundaries in Thai.
    return [span for _, span in pt.pre_tokenize_str(text)]


@lru_cache(maxsize=None)
def _tiktoken_re(pat: str):
    return regex.compile(pat)


# --- token boundaries in character coordinates ----------------------------


def _byte_len_table(tok: Tok) -> list[int]:
    """Bytes of original text each token id stands for."""
    cache = getattr(tok, "_byte_lens", None)
    if cache is not None:
        return cache
    if tok.kind == "tiktoken":
        enc = tok.raw
        table = []
        for i in range(enc.n_vocab):
            try:
                table.append(len(enc.decode_single_token_bytes(i)))
            except KeyError:
                table.append(0)
    else:
        table = _hf_byte_lens(tok.raw)
    tok._byte_lens = table  # type: ignore[attr-defined]
    return table


def _hf_byte_lens(hf) -> list[int]:
    import json

    j = json.loads(hf.backend_tokenizer.to_str())
    byte_level = "ByteLevel" in json.dumps(j.get("pre_tokenizer")) or "ByteLevel" in json.dumps(
        j.get("decoder")
    )
    b2u = _bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}
    n = len(hf)
    table = [0] * n
    for piece, idx in hf.get_vocab().items():
        if idx >= n:
            continue
        if byte_level:
            try:
                table[idx] = len(bytes(u2b[c] for c in piece))
            except KeyError:
                table[idx] = len(piece.encode("utf-8"))
        elif re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece):
            table[idx] = 1
        else:
            table[idx] = len(piece.replace("▁", " ").encode("utf-8"))
    return table


@lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def token_cuts(tok: Tok, text: str) -> tuple[list[int], list[int]] | None:
    """Where the final tokens cut the text.

    Returns (clean, inside): `clean` are character positions b with a token
    boundary between text[b-1] and text[b]; `inside` are character indices c
    with a token boundary *inside* the character's UTF-8 bytes (byte-level
    tokenizers do this to Thai all the time). None if the token bytes do not
    line up with the text (normalising tokenizers such as mT5 rewrite some
    characters), in which case the caller falls back to HF offsets.
    """
    if tok.kind == "tiktoken":
        ids = tok.raw.encode(text, disallowed_special=())
        lens = _byte_len_table(tok)
        return _cuts_from_byte_lengths(text, [lens[i] for i in ids])
    enc = tok.raw(text, add_special_tokens=False, return_offsets_mapping=True)
    return _cuts_from_offsets(text, enc["offset_mapping"])


def _cuts_from_byte_lengths(text: str, lens: list[int]):
    raw = text.encode("utf-8")
    if sum(lens) != len(raw):
        return None
    # byte offset -> (char index, is char start)
    char_of_byte = []
    for ci, ch in enumerate(text):
        n = len(ch.encode("utf-8"))
        char_of_byte.extend([ci] * n)
    starts = set()
    pos = 0
    for ch in text:
        starts.add(pos)
        pos += len(ch.encode("utf-8"))
    clean, inside = [], []
    b = 0
    for n in lens[:-1]:
        b += n
        if b in starts:
            clean.append(char_of_byte[b])
        else:
            inside.append(char_of_byte[b])
    return clean, inside


def _cuts_from_offsets(text: str, offsets):
    """HF offsets: byte-level tokens that split a character share that
    character's span, so an overlap between neighbours marks an inside cut."""
    spans = [(s, e) for s, e in offsets if e > s]
    clean, inside = [], []
    for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
        if s1 < e0:  # overlap: boundary falls inside character s1
            inside.append(s1)
        else:
            clean.append(s1)
    return clean, inside


# --- the measurements ------------------------------------------------------


@dataclass
class CutStats:
    thai_chars: int = 0
    clusters: int = 0
    junctions: int = 0  # cluster-internal positions (base|mark or mark|mark)
    base_junctions: int = 0  # consonant/vowel letter | combining mark
    pretok_pieces: int = 0
    pretok_junction_cuts: int = 0  # counted over base_junctions
    token_broken_clusters: int = 0
    docs_skipped: int = 0


def cut_stats(tok: Tok, texts: list[str]) -> CutStats:
    st = CutStats()
    for text in texts:
        junc = internal_junctions(text)
        jset = set(junc)
        clusters = cluster_pieces(text)
        st.thai_chars += len(_THAI_CH.findall(text))
        st.clusters += len(clusters)
        st.junctions += len(junc)
        # A mark|mark junction (vowel + tone mark) is kept together even by
        # the cl100k regex, so the pre-tokenizer rate is over base|mark only.
        base = {b for b in junc if not _MARK_CH.match(text[b - 1])}
        st.base_junctions += len(base)

        spans = pretoken_spans(tok, text)
        st.pretok_pieces += len(spans)
        st.pretok_junction_cuts += sum(1 for s, _ in spans[1:] if s in base)

        cuts = token_cuts(tok, text)
        if cuts is None:
            st.docs_skipped += 1
            continue
        clean, inside = cuts
        # cluster id of every Thai char; a cluster starts at a Thai char
        # that is not an internal junction.
        cid = {}
        k = -1
        for i, ch in enumerate(text):
            if _THAI_CH.match(ch):
                if i not in jset:
                    k += 1
                cid[i] = k
        broken = {cid[b] for b in clean if b in jset}
        broken |= {cid[c] for c in inside if c in cid}
        st.token_broken_clusters += len(broken)
    return st


def with_mark_aware_pretok(tok: Tok) -> Tok | None:
    """The same vocabulary and merges behind a mark-aware pre-tokenizer.

    Returns None when the pre-tokenizer never cuts at marks in the first
    place (whitespace-only or no pre-tokenizer), because then there is
    nothing to fix and the counterfactual equals the original.
    """
    if tok.kind == "tiktoken":
        pat = tok.raw._pat_str
        fixed = mark_aware(pat)
        if fixed == pat or "\\p{M}" in pat:
            return None
        enc = tiktoken.Encoding(
            name=tok.raw.name + "_markaware",
            pat_str=fixed,
            mergeable_ranks=tok.raw._mergeable_ranks,
            special_tokens=tok.raw._special_tokens,
        )
        return Tok(
            label=tok.label + " [mark-aware regex]",
            family=tok.family,
            encode=lambda s, _e=enc: _e.encode(s, disallowed_special=()),
            decode=enc.decode,
            vocab_size=enc.n_vocab,
            raw=enc,
            kind="tiktoken",
        )

    pat = _hf_split_pattern(tok)
    if pat is None:
        return None
    fixed = mark_aware(pat)
    # A pattern that only splits on whitespace/punctuation (BLOOM) has no
    # \p{L} to rewrite and does not cut at marks.
    if fixed == pat:
        return None
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer.from_str(tok.raw.backend_tokenizer.to_str())
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(fixed), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    hf = PreTrainedTokenizerFast(tokenizer_object=backend)
    return Tok(
        label=tok.label + " [mark-aware regex]",
        family=tok.family,
        encode=lambda s, _t=hf: _t.encode(s, add_special_tokens=False),
        decode=lambda ids, _t=hf: _t.decode(ids),
        vocab_size=len(hf),
        raw=hf,
        kind="hf",
    )


def unreachable_thai_tokens(tok: Tok) -> dict:
    """Vocabulary entries containing Thai that the shipped pre-tokenizer can
    never produce, because it would cut the entry's own string apart.

    If a tokenizer's vocabulary was built under a different pre-tokenizer
    (or inherited), such entries sit in the embedding matrix but are never
    emitted - so the model never trained on them. A regex fix makes them
    reachable, but their embeddings are then effectively untrained.
    """
    strings: list[str] = []
    if tok.kind == "tiktoken":
        for i in range(tok.raw.n_vocab):
            try:
                b = tok.raw.decode_single_token_bytes(i)
            except KeyError:
                continue
            try:
                strings.append(b.decode("utf-8"))
            except UnicodeDecodeError:
                pass
    else:
        import json as _json

        j = _json.loads(tok.raw.backend_tokenizer.to_str())
        byte_level = "ByteLevel" in _json.dumps(j.get("pre_tokenizer"))
        u2b = {v: k for k, v in _bytes_to_unicode().items()}
        for piece in tok.raw.get_vocab():
            if byte_level:
                try:
                    strings.append(bytes(u2b[c] for c in piece).decode("utf-8"))
                except (KeyError, UnicodeDecodeError):
                    pass
            else:
                strings.append(piece.replace("▁", " "))
    thai = [s for s in strings if _THAI_CH.search(s)]
    unreachable = [s for s in thai if len(pretoken_spans(tok, s)) > 1]
    with_junction = [s for s in thai if internal_junctions(s)]
    return {
        "thai_tokens": len(thai),
        "thai_tokens_with_base_mark": len(with_junction),
        "unreachable_thai_tokens": len(unreachable),
        "examples": sorted(unreachable, key=len, reverse=True)[:15],
    }


def thai_token_ids(tok: Tok) -> dict[str, dict[int, str]]:
    """Ids of the Thai vocabulary entries, split by whether the shipped
    pre-tokenizer can emit them.

    Same test as `unreachable_thai_tokens` (an entry is unreachable when the
    pre-tokenizer cuts its own string into more than one piece), but keyed by
    id so a caller can look the rows up in a model's embedding matrix.
    Returns {"reachable": {id: string}, "unreachable": {id: string}}.
    Entries that are not valid UTF-8 on their own (byte fragments of a Thai
    character) are in neither group, exactly as in `unreachable_thai_tokens`.
    """
    decoded: dict[int, str] = {}
    if tok.kind == "tiktoken":
        for i in range(tok.raw.n_vocab):
            try:
                decoded[i] = tok.raw.decode_single_token_bytes(i).decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                pass
    else:
        import json as _json

        j = _json.loads(tok.raw.backend_tokenizer.to_str())
        byte_level = "ByteLevel" in _json.dumps(j.get("pre_tokenizer"))
        u2b = {v: k for k, v in _bytes_to_unicode().items()}
        for piece, idx in tok.raw.get_vocab().items():
            if byte_level:
                try:
                    decoded[idx] = bytes(u2b[c] for c in piece).decode("utf-8")
                except (KeyError, UnicodeDecodeError):
                    pass
            else:
                decoded[idx] = piece.replace("▁", " ")
    out: dict[str, dict[int, str]] = {"reachable": {}, "unreachable": {}}
    for idx, s in decoded.items():
        if not _THAI_CH.search(s):
            continue
        key = "unreachable" if len(pretoken_spans(tok, s)) > 1 else "reachable"
        out[key][idx] = s
    return out
