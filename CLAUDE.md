# Thai-Optimized Tokenizer

Context for Claude Code. Read this first.

## What this is

A university AI-course project (Chulalongkorn, CEDT). Goal: reduce the number
of tokens Thai text costs under existing tokenizers, by retrofitting a Thai
subword vocabulary into an existing tokenizer **at fixed vocabulary size**,
and measure the result against published baselines.

Author: Tanabodhi Mukura (Dean). Solo, part-time, ~8 weeks from 26 Sep 2026.
Mid-term proposal was presented 26 Sep. Final delivery: November 2026.

## Status

**Week 1 is done.** The baseline harness is built and has been run on real
corpora. Results are in `results/`. Nothing else has been built yet.

## The core idea, and how it differs from prior work

Thai has no spaces between words and is under-represented in tokenizer
training data, so Thai text fragments into far more tokens than English of
the same meaning. That raises API cost, shrinks usable context, and slows
inference.

Prior Thai work (**Typhoon**, SCB 10X) *expanded* the vocabulary — added
5,000 Thai tokens — which changes the embedding matrix shape and forces
retraining. That is fine if you are already training your own model.

This project keeps vocabulary size **fixed**: prune entries the deployment
does not need, insert Thai units into the freed slots. The method is proven
on Indic scripts (BrahmicTokenizer-131K, reported 26.7% fewer tokens with
English fertility preserved) but untested on a non-delimited script.

## Measured findings (Week 1, real data)

12.4M chars of Thai across 3 domains + 3,000 parallel Thai/English pairs.
Full tables in `results/BASELINE.md`.

Thai penalty on matched content (tokens/100 chars Thai ÷ English):

| Tokenizer | Penalty | Mark-split rate |
|---|---|---|
| BLOOM | 4.25x | 16.1% |
| GPT-4 (cl100k_base) | 3.94x | 17.3% |
| Mistral-7B | 3.92x | 18.9% |
| Qwen2.5 | 2.38x | 28.8% |
| GPT-4o (o200k_base) | 1.92x | 12.8% |
| Typhoon-7B | 1.26x | 6.7% |
| XGLM | 1.16x | 3.8% |
| mT5 | 1.01x | 2.7% |
| WangchanBERTa | 0.42x | 0.7% |

Four things this establishes:

1. The penalty is real and worst in **open-weight** models — the ones people
   self-host, and therefore the only ones where a tokenizer swap is even
   possible.
2. **Pre-tokenization is a hard ceiling.** "Mark-split rate" is the share of
   tokens beginning with a Thai combining mark, meaning the tokenizer cut
   between a base consonant and its vowel/tone mark. Qwen2.5 does this 28.8%
   of the time. A vocabulary retrofit **cannot** fix those, because BPE
   merges cannot span a boundary the pre-tokenizer already imposed.
3. Good Thai tokenizers already exist (mT5 at 1.01x). The open question is
   not "can Thai be cheap" but "can *this* model be made cheap for Thai
   without retraining it".
4. Every tokenizer costs more on informal/social Thai than on news. The more
   Thai-tuned a tokenizer is, the larger that spread (WangchanBERTa 1.25x).
   This is the fairness concern showing up in data: a vocabulary built only
   from formal text penalises everyone who writes differently.

WangchanBERTa's 0.42x means Thai is cheaper than English *for it* — it is
Thai-only and handles English badly. Do not use it as a target.

## Open decisions (not yet made)

1. **Which tokenizer to retrofit.** Best target has high token cost but LOW
   mark-split rate — damage in the vocabulary, where a retrofit can reach it.
   On this data that points at cl100k (17.3%) or BLOOM (16.1%). Qwen2.5 is
   the most commercially valuable but the most blocked by pre-tokenization.
2. **Whether the project is one intervention or two.** Fixing pre-tokenization
   (a regex change) and fixing the vocabulary (the retrofit) are separate
   interventions with different targets. Quantifying how much of each
   tokenizer's penalty is pre-tokenization vs vocabulary would be a strong
   result in its own right.

## Known issues with the proposal as presented

**The "no retraining" claim is too strong and must be narrowed.** A fixed-size
retrofit keeps the embedding matrix *shape* unchanged, but a swapped row still
holds the vector learned for the token it used to represent. Swapped rows need
re-initialising and in practice light tuning. The honest claim is "no
vocabulary expansion and no full pretraining — only the swapped embedding rows
need adaptation", which is still far cheaper than Typhoon's approach.

Consequences: hosted API models (GPT-4, Claude) cannot use this at all — you
cannot swap a served model's tokenizer. The beneficiary is self-hosted
open-weight deployments: on-device/edge, and corporates who must self-host for
PDPA/data-residency reasons.

## Metric choice

Primary metric is **tokens per 100 characters**, not tokens per word. Thai has
no whitespace word boundaries, so per-word fertility depends entirely on the
segmenter chosen. Segmenter-based fertility is computed as a clearly-labelled
secondary figure using `pythainlp/newmm`.

## Layout

```
src/corpora.py        corpus loading, 3 Thai domains + parallel
src/registry.py       tiktoken + HF tokenizers behind one interface;
                      mark_split_rate() is the pre-tokenization diagnostic
src/metrics.py        tokens/100 chars, penalty ratio, effective context
src/run_baseline.py   entry point
results/              BASELINE.md, CSVs, run_meta.json
```

## Running

```bash
pip install -r requirements.txt
python src/run_baseline.py --docs 1200 --pairs 3000   # ~5 min
python src/run_baseline.py --docs 120 --pairs 250     # smoke test, ~80s
```

Corpora stream from Hugging Face; nothing large is stored. All sources are
openly licensed. Gated repos (Llama, FLORES) are skipped and reported rather
than failing the run.

## Constraints

- Public corpora and public models only. Do not use True IDC or any client
  data — the author interns there, and the proposal commits to public data.
- No GPU required, no API spend. Keep it that way; it is what makes the
  scope achievable solo.
- Report results **per domain**, never pooled — a single average hides a
  vocabulary that works on one register and not others.

## Suggested next step

Either pick the retrofit target and build the candidate generator, or first
decompose each tokenizer's penalty into its pre-tokenization and vocabulary
components. The second is lower risk and tells you where to aim.
