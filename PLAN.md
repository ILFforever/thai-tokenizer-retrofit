# Implementation plan, Weeks 2–8

Week 1 is done (`results/BASELINE.md`). This is the rest.

## Progress (28 Sep 2026)

| Week | Status | Acceptance | Where |
| --- | --- | --- | --- |
| 2 Decompose | ✅ done | Every tokenizer decomposed (SentencePiece ones too - their pre-tokenizer is readable after all); target chosen: Mistral-7B | `results/DECOMPOSITION.md` |
| 3 Candidates | ✅ done | Top 5,000 ranked with held-out projection; curve flattens. Note: the projection turned out pessimistic, not an upper bound | `results/CANDIDATES.md`, `candidates.csv` |
| 4 Retrofit | ✅ done | Size identical ✅, Thai round-trip 10k ✅, English round-trip 10k ✅, top-100 single-token ⚠️ 98/100 at 2,483 (100/100 at 3,219+ and on BLOOM) | `results/RETROFIT_CHECKS.md`, `tokenizers/` |
| 5 Evaluate | ✅ done | Thai ≥15% ✅ (61-67%), English ≤1% ✅ (+0.01%), context ≥+15% ✅ (+157-200%) | `results/EVALUATION.md`, `BASELINE_after.md` |
| 6 Robustness | ✅ done | Second segmenter ✅, register split ✅, row accounting ✅, collateral cost (extra) ✅, embedding re-init test ✅ (BLOOM-560m: mean init + 200 row-only steps ≥ original on Thai at 1/3 the tokens; Qwen repair only partial) | `results/ROBUSTNESS.md`, `EMBED_ADAPT_*.md`, `QWEN_UNREACHABLE.md` |
| 7 Write-up | ✅ drafted, published | Report structured as 10 sections (the proposal's ten components were not available, so a standard structure is used); references checked from primary sources | `results/REPORT.md`, dashboard |
| 8 Buffer | - | - | - |

Deviations from the plan, and why:
- Week 2: the plan's floor (one token per cluster) sits above what good Thai
  tokenizers already reach, which makes its share formula negative; the
  decomposition uses a strong Thai vocabulary as the reference instead.
- Week 4: new tokens come from continued BPE (greedy by savings, one merge
  per slot) rather than whole-candidate merge paths, so intermediate pieces
  are paid for from the same budget. The protect corpus is raw English plus
  Python code, not English alone.
- Extra: Qwen2.5 unreachable-token study, BLOOM as a second target,
  English-only protection profile, cluster-aware and formal-only ablations.

Each week has a **deliverable**, **acceptance criteria**, and a **fallback**.
Do not move on until acceptance is met — every week's output is the next
week's input.

---

## Week 2 — Decompose the penalty

**Why first.** Week 1 showed a high mark-split rate for several tokenizers.
A vocabulary retrofit cannot fix damage done by pre-tokenization, because BPE
merges cannot span a boundary the pre-tokenizer already imposed. Before
building anything, find out how much of each tokenizer's excess is
pre-tokenization (a regex problem) versus vocabulary (the problem this project
targets). This also decides the retrofit target.

**Build `src/pretok.py`:**

```python
def pretoken_pieces(tok, text) -> list[str] | None
```
- tiktoken: `enc._pat_str` + the `regex` module (`regex.findall(pat, text)`).
  Verified reachable.
- HF fast BPE (Qwen, BLOOM, XGLM): `tok.backend_tokenizer.pre_tokenizer`
  then `.pre_tokenize_str(text)`. Verified reachable.
- SentencePiece models (Typhoon, mT5, WangchanBERTa): `pre_tokenizer` is
  `None`. Return `None` and fall back to the existing `mark_split_rate()`
  diagnostic in `registry.py`. **Do not fabricate a number for these.**

```python
def cluster_pieces(text) -> list[str]
```
Mark-aware Thai clusters — a base character plus its following combining
marks: `[฀-๿][ัิ-ฺ็-๎]*`. This is the
floor: no tokenizer should need more pieces than there are clusters.

**The decomposition.** For each tokenizer and each domain:

```
pieces_actual  = len(pretoken_pieces(tok, text))
pieces_floor   = len(cluster_pieces(text))
shatter_ratio  = pieces_actual / pieces_floor

tokens_actual  = tok.count(text)
tokens_ideal   = sum(tok.count(p) for p in cluster_joined_words(text))

pretok_share   = (tokens_actual - tokens_ideal) / (tokens_actual - tokens_floor)
vocab_share    = 1 - pretok_share
```
where `tokens_floor` is the cluster count (one token per cluster, the best any
vocabulary could do without merging across clusters).

**Deliverable:** `results/DECOMPOSITION.md` — one row per tokenizer giving
shatter ratio, pre-tokenization share, vocabulary share, and a recommendation
flag.

**Acceptance:** every tokenizer either has a decomposition or is explicitly
marked "SentencePiece — introspection unavailable". A target is chosen and
the reason written down.

**Fallback:** if the decomposition is ambiguous, pick the target with the
highest token cost among those with mark-split under 20%, and say so.

---

## Week 3 — Thai subword candidates

**Build `src/candidates.py`.**

1. Segment a large Thai corpus. Use both levels and compare:
   - words: `pythainlp.tokenize.word_tokenize(engine="newmm")`
   - Thai Character Clusters: `pythainlp.tokenize.subword_tokenize(engine="tcc")`
   TCC units are the linguistically correct atom — a base plus its marks — and
   are what the target vocabulary is missing.
2. Count frequencies across **all three domains**, weighted equally, so the
   vocabulary is not tuned to Wikipedia alone. This is the fairness
   commitment from the proposal, implemented.
3. Score each candidate by tokens saved if it were a single token:

```python
savings(u) = freq(u) * (base_tokenizer.count(u) - 1)
```
   A unit already costing one token saves nothing. A frequent unit currently
   costing six tokens saves five per occurrence.

**Deliverable:** `results/candidates.csv` — ranked, with `freq`,
`base_tokens`, `savings`, cumulative projected savings.

**Acceptance:** top 5,000 candidates listed with a projected total token
reduction on held-out Thai text. Typhoon reports diminishing returns past
~5,000 added Thai tokens, so the curve should visibly flatten — if it does
not, the scoring is wrong.

**Fallback:** if TCC segmentation is unreliable, use character n-grams
(n = 2..6) by frequency. Cruder, still works.

---

## Week 4 — Prune and merge

**Build `src/retrofit.py`.** This is the risky week.

1. **Find prunable entries.** Tokenize a large Thai + English held-out corpus
   with the base tokenizer, count usage per vocabulary id, and rank ascending.
   Entries never used are free to remove.
   - **Never prune** anything used by the English corpus. English
     non-regression is guaranteed by construction, and this is where that
     guarantee lives.
   - Record the prune list — the proposal commits to publishing it.
2. **Allocate the budget.** `K` slots freed = `K` candidates added. Greedy by
   `savings` is fine and defensible; note in the write-up that an LP would be
   the optimal version (BrahmicTokenizer used one).
3. **Rebuild.** Use the `tokenizers` library, not string surgery on the JSON.
   For a BPE model, adding token `u` also requires the merge rules that
   produce `u` from its parts — add merges for each adjacent pair along the
   path that builds `u`, in order, or `u` will sit in the vocabulary and never
   be produced.

**Deliverable:** a saved tokenizer directory, loadable with
`AutoTokenizer.from_pretrained`.

**Acceptance, all four:**
- vocabulary size **identical** to the base
- round-trips Thai losslessly: `decode(encode(x)) == x` on 10,000 samples
- round-trips English losslessly on 10,000 samples
- added tokens are actually produced — every one of the top 100 candidates
  appears in the output when its string is encoded

**Fallback:** if BPE merge reconstruction proves intractable, retrain a BPE
model on the combined corpus with the base vocabulary seeded as initial
tokens. Different method, same claim, and say so in the report.

---

## Week 5 — Evaluate

Add the retrofitted tokenizer to `HF_SPECS` in `registry.py` and rerun:

```bash
python src/run_baseline.py --docs 1200 --pairs 3000
```

**Deliverable:** `results/BASELINE_after.md`, plus a before/after table.

**Acceptance:**
- Thai token reduction ≥ 15% against the base, reported per domain
- English tokens per 100 characters changed by ≤ 1%
- effective context (Thai chars per 1,000 tokens) up by ≥ 15%

**Fallback:** a reduction below target is a result, not a failure. Report it
against the Indic 26.7% and explain what about Thai made the transfer
weaker — that answers the project's actual research question.

---

## Week 6 — Robustness and the embedding question

1. **Second segmenter.** Rerun the secondary fertility metric with a
   different pythainlp engine. If conclusions move, the segmenter was doing
   the work — say so.
2. **Register split.** Report Thai cost separately for formal (news,
   wikipedia) and informal (social). Week 1 found the spread widens as a
   tokenizer becomes more Thai-tuned; check whether the retrofit widens it
   further. If it does, that is a real finding for the ethics section, not
   something to bury.
3. **Embedding-row accounting.** The honest cost of the retrofit:
   - how many rows were swapped, as a count and a share of vocabulary
   - confirm no swapped row was used by English
   - state the adaptation required: swapped rows need re-initialising (mean
     of the constituent subword embeddings is the standard approach) plus
     light tuning. **Do not claim zero adaptation.**

**Deliverable:** `results/ROBUSTNESS.md`.

**Acceptance:** each claim in the final report traces to a number in
`results/`.

---

## Week 7 — Write-up

Report structure follows the ten proposal components. Every number comes from
`results/`; nothing is quoted from memory.

Must appear, because they are the project's honest boundaries:
- the metric note — per 100 characters, not per word, because Thai has no
  whitespace word boundaries
- the embedding-adaptation cost from Week 6
- that hosted API models cannot use this at all; the beneficiary is
  self-hosted open-weight deployments
- per-domain results, never pooled

Complete the APA references from primary sources. Read the BrahmicTokenizer
paper directly — it was reached through a secondary citation record.

---

## Week 8 — Buffer

It will be needed. Week 4 is where slippage happens.

---

## Standing rules

- Public corpora and public models only. No True IDC or client data.
- No GPU, no API spend.
- Every result per domain, never pooled.
- Commit after each week's acceptance criteria pass, so there is always a
  working state to fall back to.
