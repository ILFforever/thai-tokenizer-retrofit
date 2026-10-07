"""Pinned Hugging Face revisions.

The exact commit of every model, tokenizer and dataset repo the results in
results/ were produced with: each was the only snapshot in the local Hub
cache when the runs were made (checked 7 Oct 2026). Every Hub load passes
`revision=rev(repo)`, so a rerun reads the same files even after a repo's
main branch moves; streamed datasets in particular are read from the Hub on
every run. Local folders (tokenizers/, models/) are not pinned: rev() returns
None for them, which means "as found".

tiktoken's cl100k_base and o200k_base are fixed by the tiktoken version
(requirements.txt).
"""

from __future__ import annotations

REVISIONS = {
    # tokenizers and models
    "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    "airesearch/wangchanberta-base-att-spm-uncased": "b81d38df6b4755dbedec0bfea863c9956cbb963e",
    "bigscience/bloom-560m": "ac2ae5fab2ce3f9f40dc79b5ca9f637430d24971",
    "facebook/xglm-564M": "f3059f01b98ccc877c673149e0178c0e957660f9",
    "google/mt5-base": "2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f",
    "mistralai/Mistral-7B-v0.1": "27d67f1b5f57dc0953326b2601d68371d40ea8da",
    "scb10x/typhoon-7b": "546cbbc382830f622661a930c2e1faa87b6873fb",
    # datasets
    "Helsinki-NLP/opus-100": "805090dc28bf78897da9641cdf08b61287580df9",
    "airesearch/WangchanThaiInstruct": "9099b833a1b6b9d471d7e6f4abb22bf88f318bd4",
    "codeparrot/codeparrot-clean": "35a59fb025bc0a102f7d96eac09d145b896d487b",
    "codeparrot/codeparrot-clean-valid": "4db92d2ec0c1b4c41eeb439cfae16854511d9dcd",
    "facebook/belebele": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
    "pythainlp/thaisum": "6c2928cb64cd0d8ec0f3790ce848048fde1911dd",
    "pythainlp/wisesight_sentiment": "85a79ed833429457227182c9f61a83e9127dfa2e",
    "wikimedia/wikipedia": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
}


def rev(repo) -> str | None:
    """The pinned commit for a Hub repo id; None for anything else."""
    return REVISIONS.get(str(repo))
