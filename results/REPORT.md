# Retrofitting Thai into an open-weight tokenizer at fixed vocabulary size

Tanabodhi Mukura (Dean) · CEDT, Chulalongkorn University · AI course project ·
draft generated 28 September 2026 from the numbers in `results/`

## TL;DR

- **Thai costs Mistral-7B 3.92x as many tokens as English** for the same
  meaning, and all of that damage is in its vocabulary: its pre-tokenizer
  never cuts Thai text, so a vocabulary retrofit can reach all of it.
- **Swapping 2,483 of its 32,000 entries** (7.8% - every entry that raw
  English, Python code and Thai never use) for Thai pieces cuts Thai tokens by
  **64% on Wikipedia, 67% on news and 61% on social posts**, takes the penalty
  to **1.33x**, and changes English by **+0.01%**. The vocabulary stays the same
  size. At Typhoon's own budget (3,219) the fixed-size retrofit beats Typhoon's
  vocabulary expansion on all three domains.
- **GPT-4's cl100k and Qwen2.5 cannot be fixed this way.** Their regex matches
  letters with `\p{L}+`, which excludes Thai vowel and tone marks, so every
  mark is cut off its consonant before the vocabulary is consulted. Qwen2.5
  even holds **1,578 Thai words it can never emit**; all are untrained, so the
  one-line regex fix alone makes Qwen2.5-0.5B 2-3x *worse* at Thai.
- **The price is real and stated:** informal Thai gains less than formal
  (spread 1.11, the same as Typhoon; 1.21 if the vocabulary is learned from
  formal text only), and the freed slots come from other languages - Korean
  text costs 39% more tokens, Russian 18%, Chinese 13%, Japanese 10%.
- **The swapped embedding rows must be re-set, but that is cheap.** On
  BLOOM-560m, leaving them as they were nearly doubles Thai bits per
  character and random re-initialisation quadruples it; setting each row to
  the mean of the pieces it replaces recovers most of the loss with no
  training, and 200 steps training only those rows ends *better* than the
  original on Thai (-9 to -14% bits per character, with the caveat in §7.6)
  at a third of the tokens, with English unchanged. **On the target model,
  Mistral-7B, a CPU-sized pass is not enough** - it already reads Thai well -
  but QLoRA on a laptop GPU closes most of the gap: after 1,000 steps the
  retrofitted Mistral-7B beats an equally trained control on news (−4%) and
  is 7-8% behind on Wikipedia and social, at a third of the Thai tokens -
  though its English bits/char drift 7% more than the control's (§8.5). On a
  reading-comprehension benchmark (Belebele, Thai and English) neither
  Mistral-7B nor Qwen2.5-0.5B shows a measurable loss against its control
  (Thai −2.4 and −2.8 points, p ≥ 0.2; English unchanged). "No retraining" is
  narrowed to: no expansion and no pretraining from scratch, but the swapped
  rows must be re-set and trained.
- **Several earlier claims did not survive checking** and are corrected here:
  the Week 1 mark-split metric, the cl100k recommendation, "every tokenizer
  costs more on social text", and the novelty of fixed-size replacement for
  Thai (Csaki et al. did it in 2023).

---

## 1. Problem

Thai is written without spaces between words and is thinly represented in the
data most tokenizers were trained on, so the same meaning costs far more
tokens in Thai than in English. Tokens are what a model is billed, limited and
timed by: a tokenizer that needs four times as many tokens for Thai makes Thai
four times as expensive per request, fits a quarter as much Thai into the
context window, and generates it four times more slowly.

On 3,000 matched Thai/English sentence pairs (OPUS-100), the Thai penalty -
Thai tokens per 100 characters divided by English tokens per 100 characters -
ranges from 4.25x (BLOOM) and 3.92x (Mistral-7B) down to 1.01x (mT5)
(`BASELINE.md`). The penalty is worst in open-weight models, which are the
only ones whose tokenizer a deployment can change at all: a hosted API's
tokenizer is fixed by the provider.

The prior Thai answer, Typhoon (SCB 10X), *expanded* Mistral-7B's vocabulary
with Thai entries and continued pretraining. That grows the embedding matrix
and the output layer, which is reasonable if you are training a model anyway.
This project asks the narrower question a self-hosting team faces: **can an
existing open-weight tokenizer be made cheap for Thai without growing it** -
by pruning entries the deployment never uses and putting Thai units in the
freed slots - and what does that actually cost?

## 2. Research questions

1. How much of each tokenizer's Thai penalty is caused by its pre-tokenizer
   (a regex problem no vocabulary can fix) and how much by its vocabulary
   (the problem a retrofit targets)?
2. For a tokenizer whose damage is in the vocabulary, how much Thai cost does
   a fixed-size retrofit remove, per domain, while leaving English untouched?
3. How does that compare with vocabulary expansion (Typhoon) on the same base?
4. Who pays? Does the retrofit widen the gap between formal and informal
   Thai, and which other languages lose the freed slots?
5. What do the swapped embedding rows need before a model can use them?

## 3. Prior work, and what is new here

Every claim below was checked against the primary source; quotes, page
numbers and the corrections are in `results/REFERENCES.md`.

**Tokenizer unfairness.** Petrov et al. (2023) show the same content can cost
up to 15 times more tokens in some languages, and their Shan example - one
consonant plus three diacritics becoming nine tokens - is the same mechanism
this project finds for Thai. Rust et al. (2021) define the per-word fertility
measure used here as a secondary figure.

**Vocabulary expansion for Thai.** Typhoon (Pipatanakul et al., 2023) trained
a 5,000-piece Thai SentencePiece model and merged it into Mistral-7B's
tokenizer. The released tokenizer grows by **3,219** entries (32,000 → 35,219;
every Mistral id unchanged) - the paper does not explain the gap from 5,000.
Its reported 2.62x efficiency gain is measured against GPT-4's tokenizer, not
against Mistral. The expanded embeddings and output layer were then trained
with LoRA during continued pretraining.

**Fixed-size replacement - including for Thai - already exists.** Csaki et
al. (2023) replaced about 5,000 of the least frequent tokens of a 50k GPT-2
vocabulary with Thai BPE tokens, randomly re-initialised the swapped rows, and
report 73% better Thai fertility with diminishing returns past ~5,000 tokens
(Typhoon's "5,000" cites them). **The project notes' claim that the method is
"untested on a non-delimited script" is therefore wrong** and has been
dropped. BrahmicTokenizer-131K (Shravan, 2026) does the same for Indic
scripts; its 26.7% headline is measured against a *different* tokenizer
(Mistral-Nemo's Tekken) at equal size, not before/after its own retrofit, and
its first stage *shrinks* o200k from 200,019 to 131,072 entries - deleting all
of o200k's Thai tokens along the way.

**Embedding initialisation.** FVT (Gee et al., 2022) initialises a new token
as the mean of the embeddings of the old tokens it used to be split into;
WECHSEL (Minixhofer et al., 2022) and FOCUS (Dobler & de Melo, 2023) use
aligned or auxiliary embedding spaces.

**Pre-tokenization.** Dagan et al. (2024) and Velayuthan & Sarveswaran (2025)
show the pre-tokenizer regex bounds what a vocabulary can do; the latter show
GPT-4-style regexes breaking Tamil, Sinhala and Hindi words at vowel signs.

**What this project adds**, relative to that work:

1. A per-tokenizer **decomposition** of the Thai penalty into
   pre-tokenization and vocabulary shares across nine tokenizers, with the
   specific cause named: `\p{L}+` in the cl100k and Qwen2.5 regexes excludes
   Thai combining marks (`\p{M}`).
2. The finding that **Qwen2.5 already contains 1,578 Thai vocabulary entries
   its own pre-tokenizer can never emit**, that all of them are untrained, and
   what it takes to make them usable.
3. A fixed-size retrofit of a **current public open-weight tokenizer**
   (Mistral-7B) with **English and code non-regression by construction**
   (a protect corpus rather than "least frequent"), a merge-safe prune rule,
   a published prune list, and a like-for-like comparison with Typhoon's
   expansion of the same base.
4. **Per-register** results (encyclopaedic, news, informal social), a
   formal-only ablation, and the **collateral cost** to the languages whose
   slots are freed.
5. A CPU-scale test of what the swapped rows need: stale vs random vs
   mean-of-parts initialisation, and row-only tuning.

## 4. Data

All data is public and openly licensed, streamed from Hugging Face and cached
locally. **Evaluation always uses the first 1,200 usable documents of each
Thai domain** (the Week 1 slice); anything that learns from text uses
documents after them, so nothing is tuned on the text it is scored on.

| Role | Source | Size used |
| --- | --- | --- |
| Thai, encyclopaedic | `wikimedia/wikipedia` 20231101.th | eval 9.1M chars; train 8.0M |
| Thai, formal news | `pythainlp/thaisum` | eval 3.0M chars; train 8.0M |
| Thai, informal social | `pythainlp/wisesight_sentiment` | eval 0.19M chars; train 1.33M (all that remains) |
| Thai/English matched | `Helsinki-NLP/opus-100` en-th | 3,000 pairs (206k / 230k chars) |
| English (protect + eval) | `wikimedia/wikipedia` 20231101.en, raw text | protect: docs 0-7,999 (83.7M chars); eval: docs 8,000-11,999 |
| Code (protect) | `codeparrot/codeparrot-clean` train | 5,000 files (~50M chars) |
| Collateral check | Wikipedia zh / ko / ru / ja; `codeparrot-clean-valid` | 200 documents each |

The social domain is small: WiseSight has only ~9,800 usable posts, so its
evaluation slice is 189k characters. Differences involving social text are
therefore reported with bootstrap confidence intervals.

## 5. Metrics

**Primary: tokens per 100 characters.** Thai has no whitespace word
boundaries, so tokens-per-word depends entirely on which word segmenter you
pick; per-character cost does not. Segmenter-based fertility is reported as a
secondary figure under two segmenters (`newmm`, `longest`) to show how much
the segmenter moves it.

**Thai penalty:** Thai tok/100ch ÷ English tok/100ch on matched pairs (the
same measure as Petrov et al.'s "premium").

**Effective context:** Thai characters per 1,000 tokens.

**Cluster-break rate** (replaces Week 1's mark-split rate): the share of Thai
character clusters (a base character plus its following vowel and tone marks)
that the final tokens split, measured on byte offsets. Week 1's measure
decoded one token at a time, which silently skipped every token that was a
fragment of a 3-byte Thai character.

**Bits per character (BPC):** a model's total negative log-likelihood in bits
divided by characters. Unlike perplexity it does not depend on how the text was
split, so it compares one model under two tokenizers fairly.

## 6. Method

### 6.1 Decomposing the penalty (Week 2)

*Source: `src/pretok.py`, `src/run_decomposition.py`.*

A BPE tokenizer works in two stages: a *pre-tokenizer* cuts the text into
pieces (with a regex, or on whitespace), then the *vocabulary* encodes each
piece on its own. No token can span a pre-token boundary. So for each of nine
tokenizers the pre-tokenizer was read from its own configuration and run
directly, and three counterfactual token counts were computed per domain:

- **T_regex**: the same vocabulary and merges behind a *mark-aware* regex
  (every `\p{L}` rewritten to `[\p{L}\p{M}]`, the change o200k made).
  This is the regex fix on its own.
- **T_vocab**: a strong Thai vocabulary (XGLM's) applied *inside the
  tokenizer's own pre-token pieces*. This is the best any vocabulary retrofit
  could do while the pre-tokenizer stays as it is.
- **T_ref**: that same strong vocabulary with no pre-tokenizer constraint.

The gap between the shipped count and T_ref is the excess over a good Thai
tokenizer. Of that gap, T_vocab − T_ref is lost to the pre-tokenizer (no
vocabulary recovers it) and the rest is recoverable by vocabulary. The plan's
original floor - one token per cluster - was dropped because good Thai
tokenizers already use *fewer* tokens than there are clusters, which made the
plan's share formula negative.

### 6.2 Candidates (Week 3)

*Source: `src/candidates.py`.*

Thai training text was segmented into `newmm` words and Thai Character
Clusters (TCC). Frequencies were computed per million characters in each
domain and averaged, so each domain carries equal weight regardless of size.
Each unit was scored by `savings(u) = freq(u) × (base_tokens(u) − 1)`, and the
projected reduction was measured on the held-out evaluation slice per domain.

### 6.3 The retrofit (Week 4)

*Source: `src/retrofit.py`.*

**Freeing slots.** A vocabulary entry may be freed only if (a) it is never
produced when encoding the protect corpora - raw English Wikipedia (line
breaks kept) and public Python code - nor the Thai training text, (b) it is
not a special, byte-fallback or base-byte token, and (c) every merge that uses
it as a part produces an entry that is also being freed, so no kept entry
loses the path that builds it. Among eligible entries the highest id goes
first (BPE and SentencePiece assign ids in learning order, so late ids are the
rarest units). The prune list is published per budget
(`results/prune_list_*.csv`).

**Filling slots.** Thai characters the base cannot represent at all (26 for
Mistral, including the repetition mark `ๆ` and the tone marks `๊ ๋`, which
Mistral spells as three byte tokens) are added first as single-character
entries. Then BPE training is *continued* on the Thai training text from the
base tokenizer's own segmentation: at each step the adjacent pair whose merge
saves the most tokens (domain-weighted) becomes a new entry, and its merge is
**appended after all existing merges**. This is the Week 3 greedy-by-savings
rule applied one slot at a time. It pays for intermediate pieces out of the
same budget, and it satisfies the plan's requirement that every new token be
producible, because each one is built from pieces that already exist. Because
new merges come last, every existing merge keeps its priority and text with no
Thai in it is segmented exactly as before.

**Rebuild.** The new vocabulary (pruned entries removed, new entries in the
freed ids, lowest id first) and merge list are assembled with the `tokenizers`
library's `BPE` model and saved as a standard Hugging Face tokenizer folder.
Vocabulary size is identical by construction.

**A mistake found and fixed.** The first prune list protected only English
prose (with whitespace collapsed by the corpus loader), so it freed code and
LaTeX pieces such as ` func`, `json`, `Override` and `mathcal` that an
English-speaking deployment still needs. The protect corpora are now raw
English text plus raw Python; this cut Mistral's free slots from 4,471 to
2,483. The same protect set is also offered without code (the "English-only"
profile, up to 4,339 slots) to show what the code protection costs Thai.

### 6.4 Evaluation (Week 5)

*Source: `src/evaluate.py`, `src/verify_retrofit.py`.*

Every tokenizer is scored on held-out text only: the Thai evaluation slice per
domain, the OPUS-100 pairs, and 4,000 raw English Wikipedia documents disjoint
from the protect corpus. Changes carry 95% percentile-bootstrap intervals over
documents (1,000 resamples). Acceptance follows the plan: Thai tokens down at
least 15% in every domain, English within 1%, effective context up at least
15%.

### 6.5 Embedding adaptation (Week 6)

*Source: `src/embed_adapt.py`, `src/qwen_repair.py`.*

On BLOOM-560m (CPU), retrofitted with 5,000 Thai entries, the swapped rows
were set in five ways: left stale; re-drawn at random (as Csaki et al. did);
set to the mean of the base-token embeddings the new token used to be split
into (FVT, Gee et al. 2022); and each of the last two followed by training
*only the swapped rows* on Thai text with every other parameter frozen (200
Adam steps of 4 × 256 tokens, learning rate 1e-4, gradients clipped). Each
condition is scored by bits per character on held-out Thai (60 passages per
domain) and English. The same repair was applied to Qwen2.5-0.5B's untrained
Thai rows (§7.5).

## 7. Results

### 7.1 The baseline reproduces; one of its diagnostics did not

Re-running the Week 1 harness reproduced all 36 result rows exactly
(`BASELINE.md`). Two of Week 1's interpretations did not survive:

- **The "mark-split rate" was not a pre-tokenization measure.** It decoded
  one token id at a time, so byte-level fragments of Thai characters decoded
  to `U+FFFD` and were never counted, and it could not separate a
  pre-tokenizer cut from a missing merge. It led Week 1 to recommend
  cl100k as a retrofit target - the opposite of what §7.2 finds.
- **"Every tokenizer costs more on social than on news" is false** for the
  three most expensive tokenizers (Mistral 96.9 vs 99.9 tok/100ch, BLOOM 93.9
  vs 98.9, GPT-4 89.2 vs 93.7). It holds for every tokenizer that has Thai
  vocabulary to lose, and for the strong-vocabulary reference (32.0 vs 28.6),
  so the informal-Thai premium appears once a tokenizer can compress Thai at
  all; a character-level tokenizer pays the same everywhere.

### 7.2 Where the penalty comes from

*Source: `DECOMPOSITION.md`.*

| Tokenizer | Penalty | Pre-tokenizer cuts consonant\|mark | Regex fix alone | Best vocabulary alone | Pre-tok share of excess |
| --- | ---: | ---: | ---: | ---: | ---: |
| BLOOM | 4.25x | 0% | 4.25x | 1.24x | 0% |
| GPT-4 (cl100k) | 3.94x | **100%** | 3.94x | 2.25x | 38% |
| Mistral-7B | 3.92x | 0% | 3.92x | 1.13x | 0% |
| Qwen2.5 | 2.38x | **100%** | **1.42x** | 2.25x | **89%** |
| GPT-4o (o200k) | 1.92x | 0% | 1.92x | 1.27x | 1% |
| Typhoon-7B | 1.26x | 0% | 1.26x | 1.13x | 0% |
| XGLM / mT5 / WangchanBERTa | 1.16x / 1.01x / 0.42x | 0% | - | - | n/a (already near reference) |

Penalty on 3,000 OPUS-100 pairs; "regex fix" and "best vocabulary" recompute
the Thai side only. Per-domain tables are in `DECOMPOSITION.md` and agree.

- **cl100k and Qwen2.5 cut every consonant away from its vowel or tone mark
  in the pre-tokenizer**, because their regex matches letters with `\p{L}+`
  and Thai marks are `\p{M}`. Every other tokenizer cuts none (o200k added
  `\p{M}` to its regex).
- **cl100k needs both fixes.** Its vocabulary was learned behind the broken
  regex, so it has no entries spanning a consonant and its mark: fixing the
  regex alone changes nothing (92.4 → 92.3 tok/100ch on Wikipedia), and a
  perfect vocabulary alone stops at 2.25x.
- **Qwen2.5 is the opposite:** 89% of its excess is the regex, and the regex
  fix alone takes its token count from 2.38x to 1.42x - because its
  vocabulary already holds Thai words it cannot reach (§7.5).
- **Mistral-7B and BLOOM carry all of their damage in the vocabulary**, where
  a retrofit can reach it: a strong Thai vocabulary inside their own
  pre-token pieces would bring them to 1.13x and 1.24x.

**Target chosen: Mistral-7B.** Its damage is all vocabulary, it is the most
expensive fixable open-weight tokenizer after BLOOM, and Typhoon is Mistral-7B
*expanded* - so a fixed-size retrofit of the same base gives a like-for-like
comparison. BLOOM (250k entries) is the secondary target because its 560M
checkpoint runs on a CPU for the embedding experiment.

### 7.3 What the candidates promise

*Source: `CANDIDATES.md`.*

If each of the top-N `newmm` words became a single token, Mistral's held-out
Thai tokens would fall by about 37% at N = 1,000 and 58% at N = 5,000, with
the marginal gain dropping from 17% per thousand (500-1,000) to 2.4% per
thousand (4,000-5,000) - the diminishing returns Csaki et al. and Typhoon
report. Thai Character Clusters saturate at about 40%: there are only ~5,000
distinct clusters, so they are the right *atom* but too small a *target*. The
ranking is register-sensitive: `ไม่` (not) and `แล้ว` (already) are five times
more frequent in social text than in Wikipedia, and the repetition mark `ๆ` -
which Mistral cannot encode as one token at all - four times.

This projection turned out to be **pessimistic**, not the upper bound it was
first taken for. It assumes each slot buys one whole word; BPE instead builds
pieces that are shared by thousands of words (`ที่`, `การ`) and can span word
boundaries (`เมื่อวันที่`), so the real retrofit beat it at every budget -
at 500 slots, 47% fewer tokens on Wikipedia against a projected 28%.

### 7.4 The retrofit

*Source: `RETROFIT_CHECKS.md`, `EVALUATION.md`.*

**How many slots can Mistral free?** Of its 32,000 entries, raw English
Wikipedia uses 26,741, Python code 23,471 and the Thai training text 13,582;
29,196 are used by at least one. After the merge-safety rule, **2,483 entries
(7.8%) can be freed** without touching anything English, code or Thai uses.
Dropping the code protection raises that to 4,339 - the "English-only"
profile, reported separately because it has a cost (§8.4). Even the strict
profile frees fewer slots than Typhoon added (3,219): a 32k vocabulary is
tight when English and code are protected.

**Acceptance (Week 4).** Every retrofit - 500 to 2,483 slots, both ablations,
the English-only profile and four BLOOM budgets (1,000-10,000) - keeps the
vocabulary size identical, round-trips 10,000 held-out Thai and 10,000
held-out English samples with zero failures, and uses at least 99.3% of its
added tokens on held-out Thai. At 2,483 slots, 98 of the top-100 Week 3
candidates encode to a single token; the two that do not (`เนื่องจาก`,
`การเลือกตั้ง`) cost 2 tokens instead of 9 and 12. This is a near-miss on
the plan's literal criterion, caused by the method: BPE continuation adds
one merge at a time, and the last merge of a long word saves only one token
per occurrence, so it ranks below other merges. At 3,219 slots (English-only)
and on BLOOM from 3,219 up, all 100 are single tokens.

**Thai cost on held-out text** (tokens per 100 characters; change vs base
Mistral with 95% bootstrap interval):

| Tokenizer | Vocabulary | Wikipedia | News | Social | Penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mistral-7B (base) | 32,000 | 96.8 | 99.9 | 96.9 | 3.92x |
| + 500 slots | 32,000 | 51.1 (−47.2%) | 50.1 (−49.9%) | 52.8 (−45.5%) | 1.91x |
| + 1,000 slots | 32,000 | 42.9 (−55.7%) | 41.8 (−58.2%) | 45.6 (−52.9%) | 1.63x |
| + 2,000 slots | 32,000 | 36.3 (−62.4%) | 35.1 (−64.9%) | 39.4 (−59.3%) | 1.39x |
| **+ 2,483 slots (strict)** | **32,000** | **34.5 (−64.4% [−64.7, −64.0])** | **33.4 (−66.6% [−66.8, −66.4])** | **37.7 (−61.1% [−61.6, −60.6])** | **1.33x** |
| + 3,219 slots (English-only) | 32,000 | 32.6 (−66.3%) | 31.4 (−68.6%) | 35.8 (−63.0%) | 1.26x |
| + 4,339 slots (English-only) | 32,000 | 30.7 (−68.3%) | 29.4 (−70.6%) | 33.9 (−65.0%) | 1.19x |
| Typhoon-7B (+3,219, expanded) | 35,219 | 34.0 (−64.9%) | 32.6 (−67.4%) | 37.2 (−61.6%) | 1.26x |

**English cost does not move.** Tokens per 100 characters change by +0.01%
on 4,000 held-out raw English Wikipedia documents and +0.11% on the OPUS-100
English side at 2,483 slots (Typhoon: +0.06% and +0.06%). 99.55% of 10,000
English samples get byte-identical token ids, so for them the model's input
is exactly what it was before.

**Effective context almost triples.** Thai characters per 1,000 tokens go
from about 1,000 to 2,901 (Wikipedia), 2,997 (news) and 2,655 (social) at
2,483 slots.

**Fixed size versus expansion.** At the *same* budget as Typhoon (3,219
slots, English-only profile) the fixed-size retrofit is cheaper than Typhoon
on all three domains (−66.3 / −68.6 / −63.0% vs −64.9 / −67.4 / −61.6%) and
ties it on the matched-content penalty (1.26x), without growing the
embedding matrix. Under the strict profile, with 736 fewer slots, it is
within 0.5-0.8 points of Typhoon. Paired on the same documents
(`vs_typhoon_bootstrap.json`, 2,000 resamples), the 3,219-slot retrofit needs
**3.7-4.2% fewer tokens than Typhoon** in every domain and the strict 2,483-slot
retrofit **1.3-2.5% more**; every one of these intervals excludes zero. The
comparison is of tokenizers only: Typhoon also continued pretraining its
model, which this project does not.

**Week 5 acceptance: all pass.** Every budget clears all three targets in
every domain (Thai ≥15% fewer, English within 1%, context ≥ +15%), by a wide
margin.

**Ablations at 2,483 slots.**

- *Cluster-aware* (never create a token that begins with a vowel or tone
  mark): 0 such tokens instead of 211, at a cost of about 1 point of
  reduction (−63.2 / −65.3 / −59.5%). Worth considering if downstream tools
  care that token boundaries respect clusters.
- *Formal-only* (merges learned from Wikipedia and news only): about 0.5
  points better on those two domains, **3 points worse on social**
  (−58.1% vs −61.1%). Its vocabulary spends slots on royal titles, politics
  and Wikipedia boilerplate (`เกล้าเจ้าอยู่หัว`, `ผู้แทนราษฎร`,
  `แหล่งข้อมูลอื่น`) where the equal-weight vocabulary has everyday words
  (`ไม่ได้`, `วันนี้`, `ตอนนี้`, `แบบนี้`, `ลูกค้า`).

**BLOOM**, the secondary target, has far more room: of its 250,680 entries
only 77,741 are used by English, code or Thai, and 170,347 can be freed. From
`BASELINE_after.md` (the Week 1 harness re-run with every retrofit added):

| Tokenizer | Swapped | Wikipedia | News | Social | Penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| BLOOM (base) | - | 94.9 | 98.9 | 93.9 | 4.25x |
| BLOOM + 1,000 | 0.4% | 44.1 | 44.1 | 46.2 | 1.91x |
| BLOOM + 3,219 | 1.3% | 33.4 | 33.3 | 36.1 | 1.50x |
| BLOOM + 5,000 | 2.0% | 30.4 | 30.1 | 33.2 | 1.38x |
| BLOOM + 10,000 | 4.0% | 26.7 | 26.2 | 29.4 | **1.22x** |

At 10,000 slots - 4% of its vocabulary - BLOOM's Thai penalty falls below
Typhoon's (1.26x). All four BLOOM retrofits pass every Week 4 check,
including all top-100 candidates as single tokens from 3,219 slots up.

**Cluster breaks.** The share of Thai clusters split by the final tokens
falls from 22% to 3.1% (Wikipedia) for Mistral at 2,483 slots - the same as
Typhoon (3.1%) - and from 39% to 3.1% for BLOOM at 10,000.

### 7.5 Qwen2.5's Thai vocabulary it cannot use

*Source: `QWEN_UNREACHABLE.md`.*

Qwen2.5's vocabulary contains 2,570 Thai entries. **1,578 of them (61%) can
never be produced by its own pre-tokenizer**, because each contains a
consonant followed by a mark and the regex always cuts there - whole words
such as `ประวัติศาสตร์` (history) and `อย่างต่อเนื่อง` (continuously). Over
12.4M characters of Thai the shipped tokenizer emits them zero times.

In Qwen2.5-0.5B (input and output embeddings tied) **all 1,578 rows are
untrained**: every one sits on the same vector as the 271 padding rows no
token ever uses (norm 0.301, within 0.1% of the padding centroid), while the
992 reachable Thai rows look like any other trained row (median norm 0.447).
A whole-vocabulary check finds the same for all 1,673 entries the
pre-tokenizer can never produce - Thai, Arabic with vowel marks, full-width
digits - with no exceptions.

So the one-line regex fix from §7.2 is **not free**. Behind a mark-aware
regex, Qwen2.5-0.5B's Thai token count drops by 31-42%, but its bits per
character get 2-3 times *worse* on every domain (Wikipedia 1.42 → 4.30,
news 1.78 → 4.69, social 2.69 → 5.53; worse on 50 of 50 passages in each):
a third of the new tokens are rows the model has never seen, each costing
about 19 bits to predict.

**Can the rows be repaired cheaply? Only partly.** Each of the 1,578 rows was
set to the mean of the embeddings of the pieces the shipped tokenizer splits
that word into, and then only those rows were trained on Thai text (200 steps
of 8 × 256 tokens; every other tensor verified bit-identical afterwards). Two
learning rates were run (`QWEN_UNREACHABLE.md`, `QWEN_REPAIR_lr1e-4.md`):

| Domain | Tokens saved | Shipped | Rows untrained | Mean init | + tuning, lr 2e-3 | + tuning, lr 1e-4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wikipedia | 42% | 1.415 | 4.302 | 2.765 | 2.147 | **1.942** (+37% vs shipped) |
| News | 34% | 1.776 | 4.687 | 3.196 | 2.528 | **2.286** (+29%) |
| Social | 31% | 2.694 | 5.525 | 3.868 | 3.368 | **3.069** (+14%) |
| English, Wikipedia | - | 0.888 | - | 0.888 | 0.956 | **0.888** |
| English, OPUS | - | 1.353 | - | 1.353 | 1.487 | **1.353** |

Mean initialisation wins back about half of the loss for free and leaves
English untouched. Row-only tuning at the gentler learning rate wins back
82-87% and still leaves English exactly unchanged; at the twenty-times larger
rate it recovered less (about 75%) *and* pushed English up 8-10%, because the
repaired rows - shared between reading and predicting - started taking
probability at English positions. So the English leak was the learning rate,
not the method. Even so, Thai stays 14-37% worse than with the shipped
tokenizer: **for Qwen2.5, the regex fix plus row repair is not enough.** The
rest of the network learned Thai in about 1.6 times more pieces and is frozen
here; adapting more than the rows (for example LoRA on the attention layers)
is the obvious next test and was not run.

### 7.6 What the swapped rows need

*Source: `EMBED_ADAPT_bloom-th5000.md`.*

BLOOM-560m was retrofitted with 5,000 Thai entries (2.0% of its 250,680 rows)
and each way of setting the swapped rows was scored in bits per character on
60 held-out passages per domain (lower is better):

| Swapped rows set by | Wikipedia | News | Social | English | Thai tok/100ch |
| --- | ---: | ---: | ---: | ---: | ---: |
| *original tokenizer (reference)* | 3.312 | 3.995 | 3.841 | 1.132 | 97.1 / 98.5 / 93.1 |
| left as they were | 6.128 | 6.966 | 6.814 | 1.132 | 28.9 / 32.2 / 31.9 |
| random re-initialisation | 13.930 | 15.921 | 15.884 | 1.197 | same |
| random + 200 row-only steps | 4.178 | 4.854 | 4.697 | 1.143 | same |
| mean of old pieces, no training | 3.934 | 4.381 | 4.155 | 1.132 | same |
| **mean + 200 row-only steps** | **3.028** | **3.450** | **3.343** | **1.132** | same |

Only the 5,000 swapped rows changed during tuning (verified: 5,000 rows
changed, 0 outside the swap). Tuning used 204,800 tokens of Thai training text
(200 Adam steps of 4 × 256, learning rate 1e-4).

- **Leaving the rows as they were is harmful:** Thai bits per character
  nearly double, because each new Thai token is read and predicted with the
  vector of an unrelated pruned token.
- **Random re-initialisation, as Csaki et al. did, is far worse** (about four
  times the original bits per character) and also hurts English: BLOOM ties
  its input and output embeddings, so 5,000 random rows compete in every
  English prediction. Even after 200 steps of row-only tuning it has not caught
  up with the untrained mean initialisation.
- **The mean of the old pieces' embeddings recovers almost everything with no
  training at all**, and leaves English exactly unchanged, while the model
  reads roughly a third as many Thai tokens.

- **Mean initialisation plus light row-only tuning ends up better than the
  original on Thai** - 8.6% lower bits per character on Wikipedia, 13.6% on
  news, 13.0% on social - while the model reads about a third as many Thai
  tokens, and English is unchanged to four decimal places. One caveat keeps
  this from being a clean win for the tokenizer alone: the reference model got
  no Thai training in this experiment, while the tuned rows saw 204,800 tokens
  of Thai from the same three domains (different documents). The defensible
  claim is that after 200 steps of training only the swapped rows, the
  retrofitted model is *at least as good* at Thai as the original, at about a
  third of the Thai token cost. BLOOM was not pretrained on Thai, so its
  reference is weak.
- **Contrast with Qwen2.5 (§7.5).** There the same recipe, at the same
  learning rate (1e-4), recovers 82-87% of the loss but does not reach the
  shipped tokenizer, with English again unchanged. The difference is the
  reference: Qwen already models Thai well through its shipped tokenizer,
  while BLOOM barely models Thai at all, so BLOOM has far less to match.

**At the target scale: Mistral-7B** (`MISTRAL_ROWS_mistral-7b-th2483.md`).
Mistral-7B was evaluated in bfloat16 on the CPU, and the swapped rows were
tuned briefly (160 steps of 4 × 128 tokens, one English window per batch,
float32 master copies of the rows). With the 2,483-slot retrofit, on
40 held-out passages of ~400 characters per domain:

| Swapped rows set by | Wikipedia | News | Social | English |
| --- | ---: | ---: | ---: | ---: |
| *original tokenizer (reference)* | 1.505 | 2.262 | 2.267 | 0.709 |
| left as they were | 4.431 | 5.134 | 4.973 | 0.709 |
| mean of old pieces, no training | 3.490 | 4.035 | 3.903 | 0.709 |
| mean + 160 row-only steps | 3.100 | 3.543 | 3.477 | 0.709 |

Mistral-7B already models Thai well character by character (1.5 bits per
character on Wikipedia, against BLOOM-560m's 3.3), so it has much more to
lose: mean initialisation recovers only 32-40% of the damage here, against
~80% on BLOOM. English is untouched in every condition. **At the target scale
the swapped rows need training, not just a good starting point.**

A short row-only tuning pass (81,920 tokens, about an hour on the CPU) recovers
45-55% of the damage and was still improving when it stopped (quick-check
Wikipedia 3.54 → 3.18 over 160 steps), with English unchanged. It is nowhere
near the original. At 7B scale the swapped rows need a real training budget:
with LoRA plus the rows on a laptop GPU the gap to an equally trained control
falls to 1-10% after 1.6M tokens and to −4..+8% after 4.1M (§8.5).

**So the honest version of the "no retraining" claim is:** no vocabulary
expansion and no pretraining from scratch, but the swapped rows must be
re-set (the mean of the old pieces is the right start) and trained. For a
model that reads Thai poorly (BLOOM-560m) a few hundred row-only steps on a
CPU are enough; for one that already reads Thai well (Mistral-7B) a CPU-sized
pass recovers about half, and 4.1M tokens of QLoRA on a laptop GPU get within
−4..+8% of an equally trained control (§8.5), at some extra English drift - a
budget far below Typhoon's
continued pretraining, though the two were not compared on equal terms.

## 8. Robustness, fairness and cost

All numbers in this section are from `ROBUSTNESS.md`.

### 8.1 The word segmenter does not drive any conclusion

Tokens per word was recomputed under PyThaiNLP's `longest` segmenter next to
`newmm`. `longest` finds 2-5% more words per 100 characters, so every
fertility figure drops by a similar few percent, but the ordering of the nine
published tokenizers is **identical** under both segmenters in all three
domains (Kendall's τ = 1.00), and the retrofit's relative gain is the same.
Per-character cost remains the headline because it needs no segmenter at all.

### 8.2 Register: the retrofit widens the formal/informal gap, like every Thai-capable tokenizer

Spread = social tok/100ch ÷ mean of Wikipedia and news.

| Tokenizer | Formal | Informal | Spread |
| --- | ---: | ---: | ---: |
| Mistral-7B (base) | 98.4 | 96.9 | 0.985 |
| + 2,483 slots, equal domain weights | 33.9 | 37.7 | 1.110 |
| + 2,483 slots, **formal-only** training | 33.6 | 40.6 | **1.211** |
| + 2,483 slots, cluster-aware | 35.2 | 39.2 | 1.115 |
| + 4,339 slots (English-only) | 30.0 | 33.9 | 1.128 |
| Typhoon-7B | 33.3 | 37.2 | 1.117 |

Base Mistral is character-level for Thai, so it charges every register the
same (social is even 1.5% cheaper). Every vocabulary that compresses Thai
compresses formal Thai more - presumably because formal text is more
standardised, though that was not tested. The retrofit's spread (1.110) is in
line with Typhoon's (1.117). What the data *does* show is that the choice of
training text matters: a vocabulary learned from formal text alone raises the
informal premium from +11% to +21%. Equal domain weighting is not a formality -
it is what keeps the informal gap at Typhoon's level instead of twice it.

### 8.3 Embedding-row accounting

| | 2,483 slots (strict) | 3,219 slots (English-only) | 4,339 slots (English-only) |
| --- | ---: | ---: | ---: |
| Rows swapped | 2,483 (7.8%) | 3,219 (10.1%) | 4,339 (13.6%) |
| Uses in the English + code protect corpora | 0 | 0 | 0 |
| Swapped rows seen at least once in 4,000 *held-out* English documents | 196 | 296 | 372 |
| Freed from | Cyrillic 818, ASCII 675, CJK 473, other 234, Hangul 211 | ASCII 1,133, CJK 936, Cyrillic 590, ... | ASCII 1,909, CJK 936, Cyrillic 922, ... |

"Never used by English" holds on the corpus the prune list was built from,
not everywhere: 196 of the 2,483 freed entries appear somewhere in held-out
English Wikipedia - mostly Cyrillic and CJK fragments inside names quoted in
their original script. That is why the held-out English token count moves by
+0.01% rather than exactly zero. Every swapped row must be re-initialised
(§7.6); no row outside the swap is touched.

### 8.4 Who pays for the freed slots

Tokens per 100 characters on 200 documents per language, base Mistral-7B vs
the retrofit:

| Text | Base | 2,483 (strict) | 3,219 (English-only) | 4,339 (English-only) |
| --- | ---: | ---: | ---: | ---: |
| Korean | 106.5 | 148.1 (**+39.0%**) | 149.8 (+40.7%) | 149.8 (+40.7%) |
| Russian | 39.6 | 46.5 (+17.6%) | 42.8 (+8.2%) | 47.8 (+20.7%) |
| Chinese | 113.5 | 128.1 (+12.9%) | 164.2 (**+44.7%**) | 164.2 (+44.8%) |
| Japanese | 109.2 | 120.3 (+10.2%) | 149.1 (+36.6%) | 149.1 (+36.6%) |
| Python code | 32.7 | 32.7 (+0.1%) | 32.9 (+0.8%) | 33.4 (+2.4%) |

This is the price of fixed size, and it should be stated as plainly as the
gain: **the Thai slots are paid for by Korean, Russian, Chinese and Japanese
users.** For a Thai-and-English deployment that serves none of them, that
price is zero. For a multilingual deployment the prune list must be built
with those languages in the protect set, which leaves fewer slots for Thai.
Protecting code also protects CJK: the strict profile frees 473 CJK entries
against 936 without code protection, which is why Chinese moves +12.9%
instead of +44.7% (the code corpus evidently contains CJK text, most likely
in comments and strings - not checked).

## 8.5 Showcase: the retrofit in a working model

*Source: `qwen_adapt_*.json`, `qwen_showcase.json`; live page: the Thai Token Meter.*

To show the method end to end on hardware a student has, both interventions
were applied to **Qwen2.5-0.5B-Instruct**, a small open model that already
handles Thai: the mark-aware regex (so its 1,578 dead Thai entries become
reachable) plus a fixed-size retrofit of 3,000 Thai entries into slots that
English, code and Thai never use. On held-out text this tokenizer needs
46-51% fewer Thai tokens than the shipped one (penalty 2.38x → 1.22x), with
English unchanged; it passes every Week 4 check (including all 100 top
candidates as single tokens and 99.99% of English samples byte-identical).

**Fair comparison.** Two copies of the model were trained identically on the
CPU - same data (Thai text from all three domains, English text, and Thai
instructions from WangchanThaiInstruct in the model's chat format), same 400
steps of 8 × 256 tokens, same LoRA (rank 16 on every attention and MLP
projection) - one with the shipped tokenizer (*control*), one with the
retrofitted tokenizer, where the 4,578 unseen rows (3,000 added, 1,578
previously unreachable) start at the mean of their pieces and train alongside
the LoRA. Any difference between the two arms is the tokenizer's.

| Qwen2.5-0.5B-Instruct | Wikipedia | News | Social | Thai chat | English | Thai tok/100ch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original (no training) | 1.470 | 1.858 | 2.072 | 1.095 | 0.993 | 55 / 60 / 59 |
| control: 400 steps, shipped tokenizer | 1.441 | 1.788 | 1.951 | 1.022 | 0.990 | same |
| retrofit, rows mean-initialised, untrained | 2.981 | 3.453 | 3.449 | 2.445 | 0.993 | about half |
| **retrofit: 400 steps, retrofitted tokenizer** | **1.647** | **2.042** | **2.112** | **1.171** | **0.988** | about half |

(Bits per character on held-out text, lower is better; 50 passages per
domain, 25 + 25 English passages, 40 held-out chat conversations.)

Training closes most of the gap - from about +100% to +8-14% against the
control - but **after 400 CPU steps the retrofitted model is still 8-14%
behind the control on Thai and 14% on Thai chat**, while reading Thai in about
half as many tokens and leaving English unchanged. The retrofit arm had 4,578
rows to learn from scratch and the control none, so equal training is not
equal starting line; its loss was still falling at the end. In generated
answers to held-out instructions the retrofitted model produces about 1.8
times as much Thai text per token (3.7 characters per token against 2.0;
17-29 characters per second on this CPU against 4-16), but a 0.5B model's
answers are rough in every arm and the retrofit's
are sometimes off-topic: the showcase demonstrates cheaper Thai, not better
answers.

**With 5× the training the gap closes (GPU).** The same two-arm design was
rerun for 2,000 steps on the laptop GPU (evaluation still on the CPU in
float32, so the numbers are directly comparable with the CPU runs;
`qwen_adapt_*_gpu2000.json`):

| Qwen2.5-0.5B-Instruct | Wikipedia | News | Social | Thai chat | English | Thai tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original (no training) | 1.470 | 1.858 | 2.072 | 1.095 | 0.993 | 100% |
| control: 2,000 steps | 1.394 | 1.705 | 1.880 | 0.994 | 1.000 | 100% |
| **retrofit: 2,000 steps** | **1.398** | **1.702** | **1.939** | **1.010** | **1.004** | **~50%** |
| retrofit vs control, 400 steps (CPU) | +14% | +14% | +8% | +14% | −0.2% | |
| **retrofit vs control, 2,000 steps** | **+0.3%** | **−0.2%** | **+3.1%** | **+1.6%** | **+0.4%** | |

After 2,000 steps the retrofitted Qwen2.5-0.5B is **level with the equally
trained control on Wikipedia and news**, 3% behind on social posts and 1.6%
on Thai chat, while reading Thai in about half the tokens; English is
unchanged. Unlike Mistral-7B it shows no extra English drift - consistent with
the output-head explanation there, since Qwen2.5-0.5B ties its output head to
the input embedding and has no separate head rows to retrain. On Belebele
(run on the GPU in bfloat16, so compare within the pair only) the 2,000-step
arms are again not separable: Thai 31.9% retrofit vs 33.8% control (p = 0.39),
English 39.6% vs 37.7% (p = 0.41).

**The target model with a real training budget (optional GPU extra).** The
CPU could give Mistral-7B's swapped rows only 160 steps (§7.6). With the
author's approval, the same two-arm design was run on a laptop GPU (RTX 4070,
8 GB) in 4-bit NF4 (QLoRA): 400 steps of 8 × 512 tokens (1.6M tokens), LoRA
rank 16 on every projection, 6 Thai + 2 English windows per batch; the
retrofit arm also trains the 2,483 swapped rows of the input embedding and
the output head from their mean initialisation (`mistral_adapt_*.json`).
Every condition is measured under the same 4-bit weights, on the §7.6
passages:

| Mistral-7B, 4-bit | Wikipedia | News | Social | English | Thai tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| original (no training) | 1.556 | 2.333 | 2.311 | 0.723 | 100% |
| control: 400 steps, shipped tokenizer | 1.456 | 2.013 | 2.047 | 0.733 | 100% |
| retrofit, rows mean-initialised, untrained | 3.514 | 4.044 | 3.913 | 0.723 | ~35% |
| **retrofit: 400 steps, retrofitted tokenizer** | **1.596** | **2.028** | **2.223** | **0.751** | **~35%** |

With 1.6M tokens of training the retrofitted Mistral-7B **matches the
equally trained control on news (+0.7%) and is ~9% behind on Wikipedia and
social**, while reading Thai in about a third of the tokens; it is already
better than the untrained original on news and social. English rises 2.5%
against the control (both arms drift slightly on English because the tuning
text is three-quarters Thai). Compared with the CPU run's 160 steps (Wikipedia
3.10), the gap closes from ~100% to ~9%, which answers the question §7.6 left
open: at 7B scale the swapped rows can be trained back, and a single laptop
GPU is enough to get most of the way. The loss was still falling at 400 steps.

**A longer paired run: 1,000 steps.** Both arms were trained again for 1,000
steps with more English in each batch (6 Thai + 3 English windows, 4.1M
tokens; `mistral_adapt_*_s1000.json`), same design otherwise.

| Mistral-7B, 4-bit | Wikipedia | News | Social | English |
| --- | ---: | ---: | ---: | ---: |
| original (no training) | 1.556 | 2.333 | 2.311 | 0.723 |
| control: 1,000 steps | 1.381 | 1.904 | 1.996 | 0.765 |
| **retrofit: 1,000 steps** | **1.485** | **1.831** | **2.128** | **0.818** |
| retrofit vs control, 400 steps | +9.6% | +0.7% | +8.6% | +2.5% |
| **retrofit vs control, 1,000 steps** | **+7.5%** | **−3.8%** | **+6.6%** | **+6.9%** |

With 2.5× the training **the Thai gap keeps closing**: the retrofit now
**beats the equally trained control on news** and is 7-8% behind on Wikipedia
and social, at a third of the Thai tokens, and it is better than the untrained
original on every register (−5% / −22% / −8%).

**English drifts in both arms, more in the retrofit.** Against the untrained
model the control's English bits/char rise 5.8% and the retrofit's 13.1%,
despite the extra English windows. The tokenizer cannot cause this (English
text is tokenized identically in both arms), so it comes from training: about
half is what a Thai fine-tune of this size does to any Mistral-7B, and the
retrofit adds about 7% on top. The likely extra source is the 2,483 retrained
output-head rows, which raise the probability of the new Thai tokens in every
context, English included; freezing the head rows after a short warm-up, or
mixing in more English, are the obvious next tests. A deployment should
re-check English after any longer tuning.

**Reading comprehension (Belebele, Thai and English, 900 questions each).**
Scored by per-character log-likelihood of the four answers
(`src/belebele_eval.py`, 95% Wilson intervals). Qwen2.5-0.5B-Instruct is small
for this task (chance is 25%):

| Qwen2.5-0.5B-Instruct | Thai | English | Thai passage tokens / 100 chars |
| --- | ---: | ---: | ---: |
| original | 36.2% [33.1, 39.4] | 39.6% [36.4, 42.8] | 55.7 |
| control: 400 steps | 34.7% [31.6, 37.8] | 40.2% [37.1, 43.5] | 55.7 |
| **retrofit: 400 steps** | **31.9% [28.9, 35.0]** | **40.4% [37.3, 43.7]** | **26.3** |

The retrofitted model reads the Thai passages in 53% fewer tokens. Its Thai
accuracy is 2.8 points below the equally trained control, a difference the
intervals do not separate (two-proportion z = 1.25, p = 0.21); English is
unchanged.

The same test on the target model, after the paired 1,000-step runs (4-bit,
on the GPU):

| Mistral-7B, 4-bit | Thai | English | Thai passage tokens / 100 chars |
| --- | ---: | ---: | ---: |
| original | 36.8% [33.7, 40.0] | 54.4% [51.2, 57.7] | 99.3 |
| control: 1,000 steps | 36.3% [33.3, 39.5] | 53.3% [50.1, 56.6] | 99.3 |
| **retrofit: 1,000 steps** | **33.9% [30.9, 37.0]** | **52.8% [49.5, 56.0]** | **31.6** |

The retrofit reads the Thai passages in **68% fewer tokens**; against the
control its Thai accuracy is 2.4 points lower (p = 0.28) and its English 0.6
points lower (p = 0.81). The +7% English bits/char drift of §8.5 does not show
up as lost English comprehension.

**Both models tell the same story:** no measurable loss of reading
comprehension in either language, at half (Qwen) to a third (Mistral) of the
Thai tokens. Both Thai point estimates lean slightly against the retrofit
(−2.4 and −2.8 points), and 900 questions can only detect differences of about
4 points, so a small real cost cannot be ruled out; a larger model or a
longer benchmark would be needed to settle it.

*A bug caught while running this (fixed):* `transformers.AutoTokenizer`, given
the saved Qwen *model* folder, rebuilds a `Qwen2Tokenizer` with Qwen's default
`\p{L}+` pattern and ignores the mark-aware pattern stored in tokenizer.json
(40 ids instead of 23 on a Thai probe). Training loaded the tokenizer from
`tokenizers/` and was unaffected, but the first retrofit Belebele run and the
showcase answers had loaded it from the model folder, so their prompts were
split the old way (that run's Thai passages cost 51.7 tokens per 100
characters instead of 26.3). Both were redone with `src/hf_tok.py`, which
loads the saved pipeline and checks it against the `tokenizers` library;
every Mistral folder was checked and is unaffected.

The **Thai Token Meter** page runs all seven tokenizers in the browser with a
250-line JavaScript BPE that reproduces the originals exactly (identical ids
on all 2,114 held-out test texts for every tokenizer), so anyone can paste
Thai and see the difference.

## 9. Limitations and ethics

**What "no retraining" can honestly mean.** A fixed-size retrofit keeps the
embedding matrix's shape, so nothing is added and no full pretraining is
needed. It does not mean zero adaptation: each swapped row still holds the
vector of the token it used to be, and §7.6 shows that using the retrofit
without touching those rows is harmful. The defensible claim is: *no
vocabulary expansion and no pretraining from scratch; the swapped rows need
re-initialising and training - lightly for a model weak in Thai, with a real
budget (not measured here) for one that is already strong, such as
Mistral-7B.*

**Who can use it.** Only self-hosted open-weight deployments (on-device and
edge, or organisations that must self-host for PDPA or data-residency
reasons). A hosted API's tokenizer cannot be changed by its users, so GPT-4,
GPT-4o or Claude users gain nothing from this method.

**A retrofit is deployment-specific.** "Unused" means unused by the protect
corpora chosen here (English prose and Python). Entries freed for Thai were
mostly Cyrillic, other-language Latin-script and markup pieces, CJK and
Hangul, so a deployment that also serves those languages would pay for the
Thai gain (§8.4). The prune list is published so a
deployment can check it against its own traffic.

**Scale of the model experiments.** Everything except one clearly marked
extra ran on a CPU. The full initialisation-and-tuning comparison used small
models (BLOOM-560m, Qwen2.5-0.5B); Mistral-7B was evaluated on 40 passages per
domain, tuned for 160 steps on the CPU, and - the extra - trained for 400
QLoRA steps in 4-bit on a laptop GPU (§8.5), so its trained numbers are for
4-bit weights. Bits per character is the only quality measure.
These results show the direction and rough size of the effect, not the
downstream task quality of a properly tuned 7B model.

**Data.** The social domain is small (189k evaluation characters) and comes
from brand-monitoring posts, so brand and product names are over-represented
(`เบียร์ช้าง`, `โตโยต้า` appear among the added tokens); OPUS-100
is partly subtitle text and its alignments are noisy; the English protect
corpus is Wikipedia, so English registers it lacks (chat, legal, informal
web) could use entries it never saw. All Thai results are per domain because a
pooled average would hide a vocabulary that works for one register only.

**Method.** Slots are allocated greedily by savings, one merge at a time; an
integer program over candidate sets (as BrahmicTokenizer frames it) could do
slightly better. Merges are learned from at most 4M characters per domain.

**Ethics.** All data and models are public; no client or employer data was
used. WiseSight posts are user-written: the labels are CC0 but copyright stays
with the posters, so they are used only for measurement and never
redistributed. The fairness question is treated as a result, not a footnote:
§8.2 reports whether the retrofit widens the formal/informal gap, and a
formal-only ablation shows what happens when informal Thai is left out of the
vocabulary's training data.

## 10. Conclusion and next steps

**Answers to the research questions.**

1. *Pre-tokenization vs vocabulary.* It depends entirely on the tokenizer.
   cl100k and Qwen2.5 cut every Thai consonant away from its mark in the
   pre-tokenizer (`\p{L}+` excludes `\p{M}`); for Qwen2.5 that accounts for
   89% of the excess, for cl100k 38%, and cl100k needs both fixes to get
   anywhere. Mistral-7B, BLOOM and o200k carry all or nearly all of their
   damage in the vocabulary.
2. *How much a fixed-size retrofit removes.* On Mistral-7B, swapping 2,483
   entries (7.8%) that English, code and Thai never use removes 64.4% of Thai
   tokens on Wikipedia, 66.6% on news and 61.1% on social text, takes the Thai
   penalty from 3.92x to 1.33x and nearly triples effective context, while
   English moves by +0.01% and 99.55% of English samples get identical token
   ids. This is far above the plan's 15% target. It should not be set
   against the Indic 26.7% that motivated the project: that figure compares
   two different tokenizers, not a tokenizer before and after its retrofit,
   and a character-level Thai baseline leaves much more to gain than o200k
   does for Indic scripts. The closest Thai analogue, Csaki et al.'s 73%
   better fertility on GPT-2, is measured per word.
3. *Against expansion.* At Typhoon's own budget the fixed-size retrofit is
   cheaper than Typhoon on all three domains and ties it on matched content,
   without growing the embedding matrix. Under strict English-and-code
   protection Mistral can only free 2,483 slots, and with those it comes
   within a point of Typhoon.
4. *Who pays.* Informal Thai gains less than formal Thai (an informal
   premium of +11%, the same as Typhoon's +12%), and the premium rises to
   +21% if the vocabulary is learned from formal text only. The freed slots are paid for by
   Korean (+39%), Russian (+18%), Chinese (+13%) and Japanese (+10%) text.
5. *What the rows need.* Re-setting, and a little tuning. Stale rows nearly double
   Thai bits per character on BLOOM-560m and random rows quadruple it; the
   mean of the old pieces recovers most of the loss with no training, and
   200 steps of row-only tuning end at least as good as the original on Thai
   with a third of the tokens and English unchanged. On models that already
   model Thai well the same recipe falls short: Mistral-7B recovers 45-55% of
   the damage after 160 steps and Qwen2.5's dead rows 82-87% after 200, both
   with English unchanged at learning rate 1e-4. Row-only repair is enough for
   a weak-Thai model; for a strong one the rows need a real training budget -
   and 4.1M tokens of QLoRA on a laptop GPU brought Mistral-7B within
   −4..+8% of an equally trained control at a third of the Thai tokens, with
   English drifting 7% more than the control's (§8.5).

**What to do next.**

- Run the embedding repair at the target scale: mean-initialise Mistral-7B's
  swapped rows and tune only those rows on a GPU for a few thousand steps,
  then measure downstream Thai tasks, not only bits per character.
- For Qwen2.5, test the cheapest fix of all - the regex change plus repair of
  the 1,578 untrained rows - on a larger checkpoint.
- Build the prune list from a deployment's real traffic (including any
  Korean, Chinese, Japanese or Russian it serves), and replace greedy
  allocation with an integer program.
- Make the cluster-aware variant the default if downstream tools need token
  boundaries that respect Thai clusters; it costs about one point of
  reduction.

## Appendix: reproducing every number

All scripts are in `src/`, all outputs in `results/`; CPU only, public data
only. Corpora download on first use and are cached in `~/.cache/thaitok`.

| Step | Command | Time on a 16-core laptop |
| --- | --- | --- |
| Baseline (Week 1) | `python src/run_baseline.py --docs 1200 --pairs 3000` | ~5 min |
| Decomposition | `python src/run_decomposition.py` | ~4 min |
| Candidates | `python src/candidates.py --target Mistral-7B` | ~3 min |
| Retrofit, strict | `python src/retrofit.py --target Mistral-7B --k 500 1000 2000 3219` | ~10 min (merges then cached) |
| Retrofit, English-only | `python src/retrofit.py --target Mistral-7B --k 1000 2000 3219 5000 --code-docs 0` | ~13 min |
| Ablations | `... --k 3219 --mode cluster`; `... --k 3219 --train-domains wikipedia news` | ~10 min each |
| BLOOM | `python src/retrofit.py --target BLOOM --k 1000 3219 5000 10000` | ~18 min |
| Acceptance checks | `python src/verify_retrofit.py tokenizers/<name> [--base BLOOM]`, then `--summary` | ~3 min each |
| Evaluation | `python src/evaluate.py`; `python src/run_baseline.py --retrofits --out BASELINE_after` | ~5 / ~35 min |
| Robustness | `python src/robustness.py` | ~5 min |
| Embedding experiment | `python src/embed_adapt.py --tokenizer tokenizers/bloom-th5000` | ~2.5 h |
| Qwen study | `python src/qwen_unreachable.py`; `python src/qwen_repair.py` | ~5 min; ~1 h |
| Tests | `python -m pytest tests` | <1 min |

Budgets above the free-slot cap are clipped to it and named by the real
count (`mistral-7b-th2483`).

## References

Checked against primary sources; see `REFERENCES.md` for quotes and page numbers.

Chumpolsathien, N. (2020). *Using knowledge distillation from keyword
extraction to improve the informativeness of neural cross-lingual
summarization* [Master's thesis, Beijing Institute of Technology].
*(Metadata from the ThaiSum dataset card only; thesis not accessed.)*

Chumpolsathien, N. (2020). *ThaiSum* [Data set]. Hugging Face.
https://huggingface.co/datasets/pythainlp/thaisum

Csaki, Z., Pawakapan, P., Thakker, U., & Xu, Q. (2023). *Efficiently adapting
pretrained language models to new languages* (arXiv:2311.05741). arXiv.
https://doi.org/10.48550/arXiv.2311.05741

Dagan, G., Synnaeve, G., & Rozière, B. (2024). Getting the most out of your
tokenizer for pre-training and domain adaptation. In R. Salakhutdinov, Z.
Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, & F. Berkenkamp (Eds.),
*Proceedings of the 41st International Conference on Machine Learning* (Vol.
235, pp. 9784–9805). PMLR. https://proceedings.mlr.press/v235/dagan24a.html

Dobler, K., & de Melo, G. (2023). FOCUS: Effective embedding initialization
for monolingual specialization of multilingual models. In H. Bouamor, J.
Pino, & K. Bali (Eds.), *Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing* (pp. 13440–13454). Association for
Computational Linguistics. https://doi.org/10.18653/v1/2023.emnlp-main.829

Gee, L., Zugarini, A., Rigutini, L., & Torroni, P. (2022). Fast vocabulary
transfer for language model compression. In Y. Li & A. Lazaridou (Eds.),
*Proceedings of the 2022 Conference on Empirical Methods in Natural Language
Processing: Industry Track* (pp. 409–416). Association for Computational
Linguistics. https://doi.org/10.18653/v1/2022.emnlp-industry.41

Land, S., & Bartolo, M. (2024). Fishing for Magikarp: Automatically detecting under-trained tokens in large language models. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (EMNLP 2024, main track). arXiv:2405.05417. https://doi.org/10.48550/arXiv.2405.05417 *(Venue from the arXiv comments; page range not retrieved.)*

Lowphansirikul, L., Polpanumas, C., Jantrakulchai, N., & Nutanong, S. (2021).
*WangchanBERTa: Pretraining transformer-based Thai language models*
(arXiv:2101.09635). arXiv. https://doi.org/10.48550/arXiv.2101.09635

Minixhofer, B., Paischer, F., & Rekabsaz, N. (2022). WECHSEL: Effective
initialization of subword embeddings for cross-lingual transfer of
monolingual language models. In M. Carpuat, M.-C. de Marneffe, & I. V. Meza
Ruiz (Eds.), *Proceedings of the 2022 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language
Technologies* (pp. 3992–4006). Association for Computational Linguistics.
https://doi.org/10.18653/v1/2022.naacl-main.293

OpenAI. (n.d.). *tiktoken* [Computer software]. GitHub. Retrieved September
28, 2026, from https://github.com/openai/tiktoken

Petrov, A., La Malfa, E., Torr, P. H. S., & Bibi, A. (2023). Language model
tokenizers introduce unfairness between languages. In *Advances in Neural
Information Processing Systems 36 (NeurIPS 2023)*.
https://doi.org/10.52202/075280-1608
*(Editors and page range not retrieved; add them from the NeurIPS
proceedings page if your style guide requires them.)*

Phatthiyaphaibun, W., Chaovavanich, K., Polpanumas, C., Suriyawongkul, A.,
Lowphansirikul, L., Chormai, P., Limkonchotiwat, P., Suntorntip, T., &
Udomcharoenchaikit, C. (2023). PyThaiNLP: Thai natural language processing in
Python. In L. Tan, D. Milajevs, G. Chauhan, J. Gwinnup, & E. Rippeth (Eds.),
*Proceedings of the 3rd Workshop for Natural Language Processing Open Source
Software (NLP-OSS 2023)* (pp. 25–36). Association for Computational
Linguistics. https://doi.org/10.18653/v1/2023.nlposs-1.4

Pipatanakul, K., Jirabovonvisut, P., Manakul, P., Sripaisarnmongkol, S.,
Patomwong, R., Chokchainant, P., & Tharnpipitchai, K. (2023). *Typhoon: Thai
large language models* (arXiv:2312.13951). arXiv.
https://doi.org/10.48550/arXiv.2312.13951

Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2021). How good
is your tokenizer? On the monolingual performance of multilingual language
models. In C. Zong, F. Xia, W. Li, & R. Navigli (Eds.), *Proceedings of the
59th Annual Meeting of the Association for Computational Linguistics and the
11th International Joint Conference on Natural Language Processing (Volume 1:
Long Papers)* (pp. 3118–3135). Association for Computational Linguistics.
https://doi.org/10.18653/v1/2021.acl-long.243

SCB 10X. (2023). *Typhoon-7B* [Large language model]. Hugging Face.
https://huggingface.co/scb10x/typhoon-7b

Schmidt, C. W., Reddy, V., Tanner, C., & Pinter, Y. (2025). *Boundless byte
pair encoding: Breaking the pre-tokenization barrier* (arXiv:2504.00178).
arXiv. https://doi.org/10.48550/arXiv.2504.00178
*(Accepted to COLM 2025 per arXiv comments; cite the COLM proceedings version
if available.)*

Shravan, R. (2026). *BrahmicTokenizer-131K: An Indic-capable drop-in
replacement for o200k_base* (arXiv:2605.29379). arXiv.
https://doi.org/10.48550/arXiv.2605.29379

Suriyawongkul, A., Chuangsuwanich, E., Chormai, P., & Polpanumas, C. (2019).
*PyThaiNLP/wisesight-sentiment: First release* (Version v1.0) [Data set].
Zenodo. https://doi.org/10.5281/zenodo.3457447

Theeramunkong, T., Sornlertlamvanich, V., Tanhermhong, T., & Chinnan, W.
(2000). Character cluster based Thai information retrieval. In *Proceedings of
the Fifth International Workshop on Information Retrieval with Asian
Languages* (pp. 75–80). Association for Computing Machinery.
https://doi.org/10.1145/355214.355225

Velayuthan, M., & Sarveswaran, K. (2025). Egalitarian language representation
in language models: It all begins with tokenizers. In O. Rambow, L. Wanner, M.
Apidianaki, H. Al-Khalifa, B. Di Eugenio, & S. Schockaert (Eds.),
*Proceedings of the 31st International Conference on Computational
Linguistics* (pp. 5987–5996). Association for Computational Linguistics.
https://aclanthology.org/2025.coling-main.400/

Wikimedia Foundation. (2023). *Wikipedia* (Version 20231101.th) [Data set].
Hugging Face. https://huggingface.co/datasets/wikimedia/wikipedia
*(Upstream source: Wikimedia Downloads, https://dumps.wikimedia.org)*

Zhang, B., Williams, P., Titov, I., & Sennrich, R. (2020). Improving
massively multilingual neural machine translation and zero-shot translation.
In D. Jurafsky, J. Chai, N. Schluter, & J. Tetreault (Eds.), *Proceedings of
the 58th Annual Meeting of the Association for Computational Linguistics*
(pp. 1628–1639). Association for Computational Linguistics.
https://doi.org/10.18653/v1/2020.acl-main.148
