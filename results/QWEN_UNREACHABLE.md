# Qwen2.5: are the unreachable Thai tokens trained?

**TL;DR: No. All 1,578 unreachable Thai tokens are untrained. Switching
Qwen2.5 to the mark-aware regex without retraining cuts Thai tokens by about
a third, but makes the model 2-3x worse at Thai (bits per character on
Wikipedia go from 1.42 to 4.30). The regex fix is not free.**

## What was tested

`DECOMPOSITION.md` found that 1,578 of Qwen2.5's 2,570 Thai vocabulary
entries can never be produced by its shipped pre-tokenizer. The hypothesis:
the model never saw them in training, so their embedding rows are untrained,
and the "free" regex fix would feed the model garbage.

- **Model:** `Qwen/Qwen2.5-0.5B`, on CPU. Its input embeddings and output
  head are one shared matrix, so a single row covers both reading a token
  and predicting it.
- **Test 1 - embedding rows.** Rows of (a) unreachable Thai, (b) reachable
  Thai and (c) everything else were compared with the 271 padding rows that
  no token uses, which serve as a built-in "never trained" control. This is
  the test from Land & Bartolo 2024.
- **Test 2 - language modelling.** Bits per character on 50 real passages
  of about 600 characters per domain, once with the shipped tokenizer and
  once with the mark-aware one. Both runs score exactly the same characters
  after the same `<|endoftext|>` prefix, so the numbers compare directly.
  Lower is better.

## What we found

**1. The unreachable rows are untrained, every one of them.**

- All 1,578 rows sit on the same point as the padding rows, within 0.1%.
  Every one has norm 0.301. Reachable Thai rows have a norm of about 0.45.
  Their cosine to the mean row is 0.94, against 0.37 for reachable Thai.
- No reachable Thai row comes close. The nearest one is 10% away, which is
  100 times farther than any unreachable row.
- The shipped tokenizer emitted these tokens **0 times** in 12.4M characters
  of Thai. It emitted reachable Thai tokens 6.1M times.
- **Cross-check:** we ran the same test over the whole vocabulary, all
  scripts. All 1,673 entries the pre-tokenizer can never produce have
  collapsed onto the padding vector, with no exceptions. That is the 1,578
  Thai entries plus 95 others with the same cause: Arabic letters carrying
  vowel marks, and fullwidth numbers like `１０` that Qwen's one-digit rule
  splits. The reverse does not hold. 254 reachable rows are also collapsed,
  all rare junk strings such as `ForCanBeConverted` (the usual "glitch
  tokens"). So unreachable always means untrained, but untrained does not
  always mean unreachable.

What this means: to the model, all 1,578 Thai words are **the same token**.
It cannot tell `ประวัติศาสตร์` from `นักท่องเที่ยว`, either when reading
them or when predicting them.

**2. The regex fix makes Thai modelling much worse.**

| Domain | Tokens saved | bpc shipped | bpc mark-aware | Worse by |
| --- | ---: | ---: | ---: | ---: |
| wikipedia | 42% | 1.42 | 4.30 | 3.0x |
| news | 34% | 1.78 | 4.69 | 2.6x |
| social | 31% | 2.69 | 5.53 | 2.1x |

The mark-aware tokenizer was worse on **50 of 50 passages in every domain**,
and every 95% interval is far from zero. On the full 1,200-document slice
it saves 36% of tokens on Wikipedia, 39% on news and 34% on social.

It breaks in two ways:

- **Predicting the unreachable tokens.** About 1 in 3 mark-aware tokens
  (31-37%) is one of the 1,578. They share a single vector, so even a model
  that knew a whole-word token came next would still have to guess which of
  the 1,578 it was. That guess alone costs at least 10.6 bits per token.
  The floor adds about 1.3 bits per character, nearly the whole shipped cost
  on Wikipedia. The measured cost is 18.6-20.4 bits per such token. Under
  the shipped tokenizer the model gives this group almost no chance (about
  2 in a million), so it has learned never to predict them.
- **Reading them.** The rest of the text gets harder too. Count only the
  bits spent on the other tokens, and the mark-aware run already costs more
  than the shipped tokenizer does for the whole text: 2.05 vs 1.42 bpc on
  Wikipedia, 2.26 vs 1.78 on news, 2.90 vs 2.69 on social. A third of the
  model's input is now a blank vector. Part of this may also be that the
  remaining tokens now appear in splits the model never saw. This run does
  not separate those two causes.

**Is the result mixed?** No. The regex fix saves tokens, but without
retraining it breaks the model on every domain.

## What it means for the project

- The DECOMPOSITION result "Qwen2.5 regex fix: 2.38x -> 1.42x" is a
  **token-count result only**. Without retraining you cannot use it.
- This is direct evidence for the narrowed claim in `CLAUDE.md`. On a
  shipped model, rows that never appeared in training are unusable as they
  are, and they need re-initialising and tuning. Any retrofit that puts new
  strings into rows will hit the same wall.
- The fix is still attractive if those 1,578 rows can be repaired cheaply.
  Qwen2.5-0.5B makes a good CPU-sized test bed for that step. The broken
  rows and their strings are known exactly, and this script measures the
  recovery. A simple first try: set each row to the average of the rows of
  the pieces the shipped tokenizer splits that word into.

**Limits.** Only the 0.5B checkpoint was tested. Every Qwen2.5 size ships
the same tokenizer, so the same rows are unreachable in all of them, but
the larger models were not checked. There are 50 passages per domain, and
the social ones are short (6.7k characters). The effect is large enough
that neither limit changes the conclusion. Bits per character measures how
well the model predicts text, not task accuracy. Still, no task would
survive a 2-3x jump.

<!-- generated below this line by src/qwen_unreachable.py; text above is hand-written and preserved on re-run -->

Model `Qwen/Qwen2.5-0.5B` (float32, CPU, 16 threads). Input embeddings and output head tied: **True** - so each row below is both how the model reads a token and how it predicts it. Run time 5.2 min.

## Test 1 - embedding rows

*dist to padding* = distance from the row to the centroid of the 271 padding rows, as a share of that centroid's norm. *Collapsed* = rows under 1% of it, i.e. indistinguishable from a row no token uses. Medians with interquartile range.

| Group | Rows | L2 norm | cos to mean row | dist to padding | Collapsed |
| --- | ---: | --- | --- | --- | ---: |
| (a) unreachable Thai | 1,578 | 0.301 (0.301-0.301) | 0.941 (0.941-0.941) | 0.0005 (0.0005-0.0006) | 1,578 (100%) |
| (b) reachable Thai | 992 | 0.447 (0.426-0.477) | 0.374 (0.315-0.434) | 1.4735 (1.3888-1.5745) | 0 (0%) |
| (b') reachable Thai, seen in eval slice | 978 | 0.448 (0.427-0.477) | 0.372 (0.314-0.432) | 1.4760 (1.3933-1.5752) | 0 (0%) |
| (c) all other tokens | 149,073 | 0.462 (0.430-0.501) | 0.376 (0.301-0.451) | 1.5364 (1.4521-1.6236) | 377 (0%) |
| special tokens | 22 | 0.301 (0.301-0.301) | 0.941 (0.941-0.941) | 0.0003 (0.0002-0.0004) | 21 (95%) |
| padding rows (no token; never trained) | 271 | 0.301 (0.301-0.301) | 0.941 (0.941-0.941) | 0.0003 (0.0002-0.0004) | 271 (100%) |

**Cross-check over the whole vocabulary** (every script, not just Thai):

| Entry | Collapsed | Not collapsed |
| --- | ---: | ---: |
| unreachable under shipped pre-tokenizer | 1,673 | 0 |
| reachable | 254 | 148,259 |
| byte fragment (cannot be tested) | 28 | 1,429 |

Unreachable non-Thai examples: `１０`, `２０`, `珊󠄁`, `مْ`, `يّ`, `هُ`, `لّ`, `نّ`, `<unk>`, `<s>`, `</s`, `</s>`.  
Collapsed but reachable examples: `TokenNameIdentifier`, `ForCanBeConverted`, `ForCanBeConvertedToF`, `PostalCodesNL`, `$PostalCodesNL`, `useRal`, `(stypy`, `𫵷`, `𬭯`, `𦒍`, `𨭉`, `פייסב`.


## Usage on the 1200-document evaluation slice

Token counts from encoding the same slice as `BASELINE.md`.

| Domain | chars | shipped tok/100ch | (a) emitted, shipped | (b) emitted, shipped | distinct (b) seen | mark-aware tok/100ch | (a) share of mark-aware tokens | distinct (a) used |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wikipedia | 9,140,915 | 56.6 | 0 | 4,471,788 (86%) | 972 | 36.3 | 32.4% | 1,513 |
| news | 3,028,237 | 57.3 | 0 | 1,556,868 (90%) | 966 | 35.1 | 37.2% | 1,481 |
| social | 189,119 | 58.0 | 0 | 92,700 (84%) | 902 | 38.6 | 32.9% | 1,331 |

## Test 2 - language modelling on real Thai

First 50 documents per domain, each cut to about 600 characters without splitting a Thai cluster. Bits per character (lower is better) is comparable across the two tokenizers because both score exactly the same characters. 95% interval is a bootstrap over passages.

| Domain | passages | chars | tok/100ch shipped | tok/100ch mark-aware | bpc shipped | bpc mark-aware | change | 95% CI | passages better with mark-aware |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| wikipedia | 50 | 28,419 | 55.5 | 32.4 | 1.415 | 4.302 | **+2.888 (+204%)** | +2.798 to +2.977 | 0/50 |
| news | 50 | 29,046 | 59.6 | 39.3 | 1.776 | 4.687 | **+2.911 (+164%)** | +2.814 to +2.997 | 0/50 |
| social | 50 | 6,682 | 59.1 | 40.8 | 2.694 | 5.525 | **+2.831 (+105%)** | +2.633 to +3.022 | 0/50 |

### Where the extra bits go (mark-aware tokenizer)

Mean bits per predicted token, split by the group of the token being predicted (*output side*) and by the group of the token read just before it (*input side*). *Mass on (a)* = average probability the model puts on the whole unreachable group at each position.

| Domain | predicting (a) | predicting (b) | predicting other | after reading (a) | after reading (b) | mass on (a), shipped | mass on (a), mark-aware |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wikipedia | 18.62 (n=3,440) | 10.94 (n=4,228) | 7.77 (n=1,540) | 17.19 (n=3,411) | 11.35 (n=4,215) | 1.7e-06 | 2.31% |
| news | 19.55 (n=3,602) | 9.47 (n=5,537) | 5.85 (n=2,275) | 15.97 (n=3,576) | 10.12 (n=5,516) | 2.4e-06 | 1.73% |
| social | 20.42 (n=859) | 11.47 (n=1,223) | 8.30 (n=645) | 17.59 (n=842) | 12.33 (n=1,207) | 2.8e-06 | 1.50% |

For comparison, the shipped tokenizer's mean bits per token: wikipedia 2.55, news 2.98, social 4.56.

