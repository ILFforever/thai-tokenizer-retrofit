# Thai-Optimized Tokenizer

Context for Claude Code. Read this first.

## What this is

A university AI-course project (Chulalongkorn, CEDT). Goal: reduce the number
of tokens Thai text costs under existing tokenizers, by retrofitting a Thai
subword vocabulary into an existing tokenizer **at fixed vocabulary size**,
and measure the result against published baselines.

Author: Tanabodhi Mukura (Dean). Solo, part-time, ~8 weeks from 26 Sep 2026.
Mid-term proposal was presented 26 Sep. Final delivery: November 2026.

## Status (28 Sep 2026)

**Weeks 2-6 of `PLAN.md` are built, run and committed; the Week 7 report is
written** (`results/REPORT.md`, also published as a page with a results
dashboard; see README). Every number comes from `results/`. Branch `week2`.
Proposal mapping done (`results/PROPOSAL_MAP.md`, from the mid-term slides).
Remaining: final polish, Week 8 buffer.

## The core idea, and how it differs from prior work

Thai has no spaces between words and is under-represented in tokenizer
training data, so Thai text fragments into far more tokens than English of
the same meaning. That raises API cost, shrinks usable context, and slows
inference.

Typhoon (SCB 10X) *expanded* Mistral-7B's vocabulary: the released tokenizer
grows by **3,219** entries (32,000 -> 35,219; the paper trained a 5,000-piece
model and does not explain the gap). This project keeps the size **fixed**:
prune entries the deployment never uses, put Thai units in the freed slots.

**Prior art exists** (verified from primary sources, `results/REFERENCES.md`):
Csaki et al. (2023) already did fixed-size replacement for Thai on a GPT-2
vocabulary. BrahmicTokenizer-131K's 26.7% is against a *different* tokenizer,
not before/after, and its first stage shrinks o200k. So the contribution is:
the pre-tokenization/vocabulary decomposition, the Qwen2.5 unreachable-token
finding, English+code protection by construction, per-register results,
collateral cost, and the embedding-row test.

## Findings

Week 1 baseline (`results/BASELINE.md`, reproduced exactly): Thai penalty on
matched content BLOOM 4.25x, GPT-4 3.94x, Mistral 3.92x, Qwen2.5 2.38x, o200k
1.92x, Typhoon 1.26x, XGLM 1.16x, mT5 1.01x, WangchanBERTa 0.42x (Thai-only;
not a target).

Week 2 decomposition (`results/DECOMPOSITION.md`):
- cl100k and Qwen2.5 regexes use `\p{L}+`; Thai marks are `\p{M}`, so the
  pre-tokenizer cuts **100%** of consonant|mark joins. Every other tokenizer
  cuts 0%.
- cl100k needs both fixes (regex alone changes nothing). Qwen2.5: 89% of its
  excess is the regex. Mistral and BLOOM: 100% vocabulary -> retrofit targets.
- Qwen2.5 has 1,578 Thai vocab entries its own pre-tokenizer can never emit;
  all are **untrained** (identical to padding rows), and the regex fix without
  repair makes Qwen2.5-0.5B's Thai bits/char 2-3x worse
  (`results/QWEN_UNREACHABLE.md`).

Weeks 3-5, target Mistral-7B (`results/EVALUATION.md`):
- Freeable under English+code protection: **2,483** entries (7.8%);
  English-only protection: 4,339.
- At 2,483 slots: Thai tokens -64.4% / -66.6% / -61.1% (wiki/news/social),
  penalty 3.92x -> 1.33x, English +0.01% (held-out Wikipedia). All Week 4 and
  Week 5 acceptance checks pass except "top-100 candidates single-token"
  (98/100; the two misses cost 2 tokens).
- At Typhoon's budget (3,219, English-only) it beats Typhoon on all domains.

Embedding rows (`results/EMBED_ADAPT_bloom-th5000.md`, BLOOM-560m, 5,000
swapped rows): stale rows nearly double Thai bits/char, random quadruples it;
mean-of-old-pieces init recovers most with no training; + 200 row-only steps
ends better than the original on Thai (caveat: the reference got no Thai
training) with English unchanged. Mistral-7B (`MISTRAL_ROWS_*.md`, bf16 on
CPU): mean init recovers only 32-40% and 160 row-only steps 45-55% - at 7B
the rows need a real (GPU) budget. Qwen2.5 dead-row repair at lr 1e-4
recovers 82-87% with English unchanged (lr 2e-3 leaked into English); Thai
still 14-37% worse than shipped.

Showcase (report §8.5): Qwen2.5-0.5B-Instruct with regex fix + 3,000 Thai
slots (Thai tokens -46..-51%), two-arm LoRA training: retrofit 8-14% behind
control after 400 CPU steps; after 2,000 GPU steps +0.3% / -0.2% / +3.1%
(wiki/news/social), chat +1.6%, English +0.4% - effectively level at half the
Thai tokens (GPU runs evaluate on the CPU in fp32, so they compare with the
CPU runs). Mistral-7B
two-arm QLoRA on the laptop GPU (optional extra, `src/mistral_adapt_gpu.py`,
venv `~/.venvs/thaitok-gpu`): retrofit within 1-10% of control at ~1/3 of the
Thai tokens (400 steps). Paired 1,000 steps: retrofit vs control +7.5% / -3.8% /
+6.6% (wiki/news/social); English drifts +5.9% (control) vs +13.2% (retrofit)
against the untrained model - half general fine-tune drift, ~7% retrofit extra.
NOT mainly the output-head rows: masking the swapped ids recovers only ~11% of
it (src/drift_probe.py). Character-balancing the mix (--thai-windows 2) closes only
~24% and costs a lot of Thai (Belebele Thai 29.9% vs 36.3%, p = 0.004) - not a fix;
the rest is most plausibly the cost of re-learning 2,483 tokens. Keep the token mix.
A KL penalty to the untrained model on the English windows (--kl-english 1.0,
--english-windows 3, _s1000_kl1) removes 54% of the extra drift (English +9.3%)
at no Thai cost (+5.9% / -4.2% / +5.0% vs control; Belebele Thai 36.2% vs 36.3%)
- the recommended recipe; one run, control not given the same penalty.
Qwen2.5-3B-Instruct (same tokenizer, 4-bit, `qwen_adapt.py --quant4`, 2,000
steps, results/qwen3b_adapt_*.json): retrofit 11-12% behind control on Thai,
about level with the untrained original; Belebele Thai 37.4 vs control 40.9
(p = 0.13) vs original 45.4. The stronger the model in Thai, the longer the
swapped rows take. Belebele
(Qwen2.5-0.5B, near chance): original 36.2 / control 34.7 / retrofit 31.9% Thai,
not separable (p = 0.21), English unchanged; showcase: retrofit ~1.8x Thai
chars/s. Load retrofitted Qwen tokenizers with `src/hf_tok.py` - AutoTokenizer
on a Qwen model folder silently restores the default `\p{L}+` regex. Trained
models live in ~/thaitok_models (outside OneDrive; `src/paths.py`), still
named `models/<folder>` in commands and results. Live page: Thai Token Meter (`showcase/`, JS BPE verified
id-for-id against 7 real tokenizers by `node tests/web_bpe.test.mjs`).
Typhoon-7B is SCB 10X's model (vocabulary expanded), only used for comparison.

Week 6 (`results/ROBUSTNESS.md`): segmenter does not change rankings
(Kendall tau 1.00); retrofit informal/formal spread 1.110 vs Typhoon 1.117,
formal-only training 1.211; freed slots cost Korean +39%, Russian +18%,
Chinese +13%, Japanese +10%, Python +0.1%.

## Corrections to earlier notes (do not reintroduce)

- The Week 1 "mark-split rate" is **not** a pre-tokenization measure (it
  decodes one token at a time and misses byte fragments). Use
  `pretok.cut_stats` (byte-aware cluster breaks) and the decomposition.
- "Retrofit cl100k because its mark-split rate is low" was backwards.
- "Every tokenizer costs more on social than news" is false for Mistral,
  BLOOM and GPT-4.
- "Untested on a non-delimited script" is false (Csaki et al. 2023).
- The `\p{L}+`-excludes-marks diagnosis is not ours first: Regmi et al.
  (2026, arXiv:2608.26449) published it, Thai worst at 9.02x. Their 37.4 vs
  4.1 per 100 chars are pre-token floors, not token counts. Our decomposition
  confirms it on real tokenizers (`results/PROPOSAL_MAP.md`, component 05).

## Known limits of the claim

**"No retraining" is narrowed to: no vocabulary expansion and no full
pretraining; the swapped embedding rows must be re-initialised (mean of the
old pieces' embeddings) and lightly tuned.** Hosted API models cannot use this
at all; the beneficiary is self-hosted open-weight deployments. A retrofit is
deployment-specific: freed slots come from languages the protect corpus does
not contain.

## Metric choice

Primary metric is **tokens per 100 characters**, not tokens per word. Thai has
no whitespace word boundaries, so per-word fertility depends on the segmenter;
it is a secondary figure under `newmm` and `longest`. Model quality is
compared in **bits per character**, which does not depend on the tokenizer.

## Layout

```
src/corpora.py           corpus loading + cache (~/.cache/thaitok), disjoint
                         train/eval slices; domains wikipedia/news/social,
                         english_raw, code
src/registry.py          tiktoken + HF tokenizers; retrofits read from the
                         retrofit meta files in results/
src/metrics.py           tok/100ch, penalty, effective context, cluster breaks
src/pretok.py            pre-tokenizer introspection, cut stats, mark-aware
                         regex counterfactual, unreachable tokens
src/run_baseline.py      Week 1 (--retrofits --out BASELINE_after for Week 5)
src/run_decomposition.py Week 2
src/candidates.py        Week 3
src/retrofit.py          Week 4 (prune list, BPE continuation, rebuild)
src/verify_retrofit.py   Week 4 acceptance checks (--summary to combine)
src/evaluate.py          Week 5
src/robustness.py        Week 6
src/embed_adapt.py       Week 6 embedding experiment (BLOOM-560m, CPU)
src/qwen_unreachable.py, src/qwen_repair.py   Qwen2.5 side study
src/dashboard_data.py, src/build_dashboard.py, src/dashboard_template.html
tokenizers/              retrofitted tokenizers (BLOOM ones gitignored)
results/                 every number; REPORT.md is the write-up
```

## Running

Use a venv outside OneDrive (the project folder syncs). Evaluation always uses
the first 1,200 documents per domain; anything that learns uses later
documents. Learned merges are cached in `~/.cache/thaitok/merges_*.json`.

```bash
pip install -r requirements.txt
python src/run_baseline.py --docs 120 --pairs 250 --out BASELINE_smoke  # smoke test, ~2 min
python src/retrofit.py --target Mistral-7B --k 500 1000 2000 3219   # ~10 min first time
```

## Constraints

- Public corpora and public models only. Do not use True IDC or any client
  data — the author interns there, and the proposal commits to public data.
- No GPU required, no API spend. Keep it that way; it is what makes the
  scope achievable solo.
- Report results **per domain**, never pooled — a single average hides a
  vocabulary that works on one register and not others.
