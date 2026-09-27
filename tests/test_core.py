"""Fast checks of the pieces every result depends on. No network, no models.

    python -m pytest tests
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import regex

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pretok import (  # noqa: E402
    _cuts_from_byte_lengths,
    cluster_pieces,
    internal_junctions,
    mark_aware,
)
from retrofit import BPEModel, build, learn_merges, prune_order  # noqa: E402

CL100K = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""


# --- pretok -----------------------------------------------------------------


def test_cl100k_regex_cuts_thai_marks_and_mark_aware_does_not():
    word = "ที่ดิน"  # 'land': two clusters, both with marks
    assert len(regex.findall(CL100K, word)) > 1
    assert regex.findall(mark_aware(CL100K), word) == [word]


def test_mark_aware_rewrites_bare_and_class_letters():
    assert mark_aware(r"\p{L}+") == r"[\p{L}\p{M}]+"
    assert mark_aware(r"[^\p{L}\p{N}]") == r"[^\p{L}\p{M}\p{N}]"
    assert mark_aware(r"\s+") == r"\s+"


def test_clusters_keep_marks_with_their_base():
    assert cluster_pieces("ที่ดิน") == ["ที่", "ดิ", "น"]
    # positions inside a cluster: after ท (the vowel) and after ี (the tone)
    assert internal_junctions("ที่") == [1, 2]
    assert internal_junctions("abc") == []


def test_byte_level_cut_inside_a_thai_character_is_detected():
    text = "กา"  # two 3-byte characters
    clean, inside = _cuts_from_byte_lengths(text, [3, 3])
    assert clean == [1] and inside == []
    clean, inside = _cuts_from_byte_lengths(text, [2, 4])  # splits ก's bytes
    assert inside == [0]
    assert _cuts_from_byte_lengths(text, [2, 2]) is None  # bytes do not add up


# --- retrofit ---------------------------------------------------------------


def _toy_bpe() -> BPEModel:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE

    vocab = {"<unk>": 0, "a": 1, "b": 2, "c": 3, "ab": 4, "abc": 5, "x": 6, "y": 7, "xy": 8}
    merges = [("a", "b"), ("ab", "c"), ("x", "y")]
    tk = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
    tk.add_special_tokens(["<unk>"])
    config = json.loads(tk.to_str())
    return BPEModel("toy", dict(vocab), merges, False, config, {0})


def test_prune_order_never_frees_a_part_of_a_kept_token():
    bpe = _toy_bpe()
    used = np.zeros(len(bpe.vocab), dtype=int)
    used[[1, 2, 3, 6, 7]] = 1  # characters used; merged tokens unused
    order = prune_order(bpe, used)
    id_to = bpe.id_to_tok
    freed = [id_to[i] for i in order]
    # "abc" must go before "ab" (ab is a part of abc); "xy" is free too
    assert set(freed) == {"ab", "abc", "xy"}
    assert freed.index("abc") < freed.index("ab")


def test_prune_order_keeps_parts_of_used_tokens():
    bpe = _toy_bpe()
    used = np.zeros(len(bpe.vocab), dtype=int)
    used[[1, 2, 3, 5]] = 1  # "abc" is used, so "ab" must stay
    freed = {bpe.id_to_tok[i] for i in prune_order(bpe, used)}
    assert "ab" not in freed and "abc" not in freed


def test_learn_merges_picks_the_most_saving_pair_first():
    runs = {"ab": 10.0, "abab": 1.0, "cd": 3.0}
    merges, took, new, savings = learn_merges(
        lambda s: list(s), runs, budget=2, existing={"a", "b", "c", "d"},
        forbid_mark_start=False, to_text_fn=lambda s: s)
    assert merges[0] == ("a", "b") and savings[0] == 12.0  # 10 + 2 occurrences
    assert new == ["ab", "cd"] and all(took)


def test_build_keeps_size_and_encodes_new_tokens():
    bpe = _toy_bpe()
    used = np.zeros(len(bpe.vocab), dtype=int)
    used[[1, 2, 3, 6, 7]] = 1
    order = prune_order(bpe, used)
    # New merges must be built on the base segmentation: "cab" is "c" + "ab"
    # under the existing merges, so the new merge is ("c", "ab").
    tk, pruned = build(bpe, order, ["cab", "yx"], [("c", "ab"), ("y", "x")])
    vocab = json.loads(tk.to_str())["model"]["vocab"]
    assert len(vocab) == len(bpe.vocab)
    assert sorted(vocab.values()) == list(range(len(bpe.vocab)))
    assert [t.value for t in tk.model.tokenize("cab")] == ["cab"]
    assert [t.value for t in tk.model.tokenize("yx")] == ["yx"]
    assert [t.value for t in tk.model.tokenize("ab")] == ["ab"]  # old merge intact
    assert set(pruned) == {"xy", "abc"}  # highest ids first; "ab" still needed


def test_a_merge_against_the_base_segmentation_never_fires():
    # Why retrofit.py learns merges from the base tokenizer's own output:
    # ("c", "a") is appended after ("a", "b"), so in "cab" the older merge
    # takes the "a" first and the new token can never be produced.
    bpe = _toy_bpe()
    used = np.zeros(len(bpe.vocab), dtype=int)
    used[[1, 2, 3, 6, 7]] = 1
    tk, _ = build(bpe, prune_order(bpe, used), ["ca", "cab"], [("c", "a"), ("ca", "b")])
    assert [t.value for t in tk.model.tokenize("cab")] == ["c", "ab"]
