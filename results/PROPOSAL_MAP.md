# The mid-term proposal, component by component: what was promised, what was delivered

Maps the ten components of the mid-term proposal (slides presented 26 Sep
2026) onto the report (`REPORT.md`) and the files in `results/`. Every
"delivered" number is from `results/`. Written 5 Oct 2026.

## TL;DR

- **Every numeric target was met, by a wide margin.** Thai tokens fell
  61-67% per domain (target ≥15%), English moved +0.01% (target ≤1%), the
  vocabulary trained without social text still cut social text by 58% (target
  ≥10% out of domain), and effective context rose 157-200% (target ≥15%).
- **The research gap the proposal stated was wrong, and the report says so.**
  Fixed-size replacement for Thai was done by Csaki et al. (2023), and the
  `\p{L}+` regex cause was published by Regmi et al. (2026) - a paper the
  proposal itself cites for its 9.02x figure. The contribution is narrowed
  to what neither paper did (component 05).
- **Three plan items changed:** the base tokenizer is open-weight Mistral-7B,
  not a "commercial" one (users cannot change a hosted tokenizer); the third
  domain is informal social posts, not government and academic text; Thai
  training text is ~17M characters, not ≥100M.
- **The project went past its own scope** (model training, embedding
  adaptation, a downstream benchmark). Those are labelled as extras, and
  they are why "no retraining" is narrowed in the report.
- **Still open:** three cover-slide wordings to correct (component 01-02),
  the final report (milestone M3), and two references to add or drop.

## The ten components at a glance

| # | Component (slide) | Status | Where in the report |
| --- | --- | --- | --- |
| 01-02 | Title, problem and motivation (cover) | Delivered; 2 cover numbers need rewording | §1, `BASELINE.md` |
| 03 | Objectives and research questions | All 3 objectives met; RQs answered and extended | TL;DR, §2, §10 |
| 04 | Scope, users and stakeholders | Met inside scope; extras beyond it are labelled | §9, §8.5 |
| 05 | Related work and the gap | Delivered; **gap restated** after checking sources | §3, `REFERENCES.md` |
| 06 | Approach | Delivered; base tokenizer changed | §6, §7.4 |
| 07 | Data | Delivered; volume and domains changed, versions only partly pinned | §4 |
| 08 | Success metrics | All five targets met | §5, §7.4, `EVALUATION.md` |
| 09 | Responsible AI | Delivered; PII screen done after the fact | §8, §9 |
| 10 | Plan, outcomes, references | M1, M2 done; M3 (final report) in Week 8 | `PLAN.md`, References |

---

## 01-02 Title, problem and motivation

**Proposal.** Thai costs more tokens than English for the same meaning. Three
published numbers: 2.62x (GPT-4 vs Typhoon), 37.4 vs 4.1 tokens per 100
characters, a 9.02x "shatter ratio".

**Delivered.** The penalty measured directly on 3,000 matched Thai/English
pairs for nine tokenizers: BLOOM 4.25x, GPT-4 3.94x, Mistral-7B 3.92x,
Qwen2.5 2.38x, o200k 1.92x, Typhoon 1.26x, XGLM 1.16x, mT5 1.01x (§1,
`BASELINE.md`).

**Corrections to the cover slide** (checked against the papers):

| Cover says | What the source says |
| --- | --- |
| 2.62x more tokens under GPT-4 than under Typhoon | Correct. Note it is against GPT-4's tokenizer, not against Mistral, the model Typhoon extends (`REFERENCES.md` claim 2). |
| "37.4 tokens per 100 Thai characters under a naive pre-tokenizer, against 4.1" | These are **pre-tokens** (regex pieces), not tokens, on FLORES-200: a *floor* on the token count, not the count itself (Regmi et al., 2026, Table 6). |
| 9.02x Thai shatter ratio | Correct as "pre-tokens under `\p{L}+` divided by pre-tokens under `[\p{L}\p{M}]+`" - the highest of the 26 languages they test (Regmi et al., 2026, abstract and Table 4). |

## 03 Objectives and research questions

| Objective | Target | Delivered | Status |
| --- | --- | --- | --- |
| O1 Measure Thai token cost, ≥4 tokenizer families, by W1 | ≥4 families | 9 tokenizers on a parallel corpus, three Thai domains | Met |
| O2 Retrofit onto a base tokenizer, by W4 | ≥15% fewer Thai tokens | −64.4% / −66.6% / −61.1% (wiki / news / social), Mistral-7B, 2,483 slots | Met |
| O3 Evaluate on ≥3 domains, English ≤1%, by W6 | 3 domains; English ≤1% | 3 domains; English +0.01%, 99.55% of English samples get identical ids | Met |

| Proposal RQ | Answered in |
| --- | --- |
| RQ1 How large is the premium, does it vary by domain and register? | §7.1, §7.2; per-domain tables in `BASELINE.md`. Also split into pre-tokenizer vs vocabulary shares (report RQ1). |
| RQ2 Can a retrofit cut Thai tokens with English unchanged? | §7.4 - yes (report RQ2). |
| RQ3 Does the reduction hold outside its training corpus? | §7.4 - every number is on held-out documents; a vocabulary learned without any social text still cuts social text 58.1% (`EVALUATION.md`). |

The report adds three questions the proposal did not ask: expansion vs
retrofit (Typhoon), who pays for the freed slots, and what the swapped
embedding rows need (report RQ3-RQ5).

## 04 Scope, users and stakeholders

| Proposal scope line | What happened |
| --- | --- |
| In: Thai and English plain text | Kept. Korean, Russian, Chinese, Japanese and Python are *measured* for collateral cost only (§8.4). |
| In: vocabulary construction and retrofit | Delivered (§6.3, §7.4). |
| In: fertility, compression, context, cost | Delivered (§5, §7.4). |
| Out: training or fine-tuning any LLM | **Exceeded**, as labelled extras: LoRA/QLoRA runs on Qwen2.5-0.5B, Qwen2.5-3B and Mistral-7B (§8.5). The core deliverable needs none of them. |
| Out: embedding-layer adaptation | **Exceeded**: §6.5, §7.6. It had to be done - using a retrofit with untouched rows is harmful - and it is why "no retraining" is narrowed to "no expansion, no pretraining from scratch". |
| Out: downstream task-quality claims | **Exceeded, cautiously**: Belebele reading comprehension, with no measurable loss (§8.5). Not a quality claim. |

**Stakeholders, narrowed.** "Thai-language end users of AI services" gain
only if the service self-hosts an open-weight model: a hosted API's
tokenizer cannot be changed by its users (§9). Thai NLP researchers get the
nine-tokenizer baseline and the prune lists; the instructor gets a measured
result against a baseline.

## 05 Related work and the gap

**The proposal's gap - "no published work retrofits Thai at a fixed
vocabulary size"; the method is "untested on non-delimited scripts" - is
wrong.** Checking the primary sources (`REFERENCES.md`) found:

| Proposal row | What the source says |
| --- | --- |
| Typhoon adds 5,000 Thai tokens | Trained a 5,000-piece model; the released vocabulary grows by **3,219** (32,000 → 35,219). |
| BrahmicTokenizer: fixed-size prune-and-retrofit onto o200k | Its first stage *shrinks* o200k (200,019 → 131,072), deleting all 1,561 Thai tokens; its 26.7% is against a different tokenizer, not before/after. |
| (missing) | **Csaki et al. (2023) did fixed-size replacement for Thai** on GPT-2 (~5,000 tokens, 73% better fertility). Typhoon's "5,000" comes from them. |
| Vowel Signs (2026): "diagnoses the ceiling; no vocabulary built" | It names the cause (`\p{L}+` excludes combining marks), puts Thai worst at 9.02x, notes o200k already fixes it, and builds a Nepali tokenizer and trains models. **Our regex finding confirms theirs; it is not new.** |
| MEGA (2023) | Evaluates generative models across languages; it does not isolate Thai tokenizer cost. Not used in the report. |
| Measuring the tokenization premium (2026) | Covers Bengali, Hindi, Arabic, Tamil and Yoruba, not Thai. Not used; Petrov et al. (2023) is cited instead. |

**The contribution that stands** (§3): the per-tokenizer split of the Thai
penalty into regex and vocabulary shares, with real vocabularies, across nine
tokenizers; the 1,578 Qwen2.5 Thai entries its own regex can never emit, all
untrained; a retrofit of a current open-weight tokenizer with English and
code protected by construction and a published prune list; a like-for-like
comparison with Typhoon on the same base; per-register results and the
collateral cost; and what the swapped rows need, up to 7B scale.

## 06 Approach

| Proposal | Delivered |
| --- | --- |
| Pipeline: input → segmentation → candidates, prune, budget → retrofitted vocabulary | Same pipeline (§6). Candidates are ranked by projected savings (§6.2); slots go greedily by savings, one BPE merge at a time (§6.3). |
| Retrofit onto "a commercial base tokenizer" | **Changed to Mistral-7B (open weights).** A hosted model's tokenizer cannot be changed, and cl100k's damage is mostly in its regex, which a vocabulary cannot reach (§7.2). |
| English preserved by construction | Delivered: a protect corpus (83.7M chars English, ~50M chars Python) instead of "least frequent"; English +0.01%. |
| Baselines: unmodified base; ≥2 open multilingual; Typhoon as upper bound | Delivered: Mistral-7B; BLOOM, XGLM, mT5, Qwen2.5, WangchanBERTa; Typhoon. At Typhoon's own budget the retrofit beats it on all three domains (§7.4). |
| ~5,000 slots bound from Typhoon | Under strict protection Mistral frees only 2,483 (7.8%); 4,339 with English-only protection. |
| One segmenter, second as a Week 6 check | Delivered: `newmm` vs `longest` give the same rankings, Kendall tau 1.00 (§8.1). |
| CRISP-DM, two-week iterations | Weekly plan in `PLAN.md`; not documented as CRISP-DM phases. |

## 07 Data

| Proposal | Delivered | Status |
| --- | --- | --- |
| Thai Wikipedia; news, government, academic; a parallel corpus | Wikipedia, ThaiSum news, WiseSight social posts, OPUS-100 parallel (§4) | **Changed**: informal social text replaced government and academic text. Register fairness needed an informal domain. |
| ≥100M Thai characters for vocabulary training | ~17M train (8.0M + 8.0M + 1.3M), 12M evaluation; merges learn from ≤4M per domain | **Missed.** Not tested whether more text helps. |
| Disjoint train / in-domain held-out / out-of-domain | First 1,200 documents per domain are evaluation only; learning uses later ones; the wikipedia+news vocabulary scores social text unseen | Met |
| No labels; one fixed segmenter with its error profile | No labels; two segmenters compared | Error profile not reported |
| Deduplicated, script-filtered, domain balance reported | Dedup on first 200 chars; ≥55% Thai characters per document; sizes per domain in §4 | Met |
| Corpora and tokenizer versions pinned | Wikipedia dump pinned (20231101); tokenizers named in `run_meta.json` but their hub revisions are not pinned | Partly met |

## 08 Success metrics

| Obj | Metric | Target | Delivered |
| --- | --- | --- | --- |
| O1 | Tokens per 100 Thai characters | Reference for ≥4 tokenizers | 9 tokenizers, 3 domains + parallel (`BASELINE.md`) |
| O2 | Thai token reduction | ≥15% | 61.1-66.6% per domain |
| O2 | English tokens per 100 characters | ≤1% change | +0.01% |
| O3 | Out-of-domain reduction | ≥10% on ≥3 domains | 58.1% on social text unseen in training; the government/academic domains were not tested |
| O3 | Effective context (Thai chars per 1,000 tokens) | ≥15% more | +181% / +200% / +157% |

Per-character as the primary metric, segmenter fertility as secondary, as
proposed (§5).

## 09 Responsible AI

| Proposal | Delivered |
| --- | --- |
| Privacy: open licences, PII screening, PDPA | Public, openly licensed data only; no client or employer data. No PII screen ran before training. Checked afterwards: none of the 2,483 added tokens contains @, #, a URL or a run of 3+ digits. WiseSight posts are used for measurement only, never redistributed (§9). |
| Fairness: per register, never pooled | Every result is per domain. The retrofit's informal/formal spread is 1.11, the same as Typhoon's; 1.21 with a formal-only vocabulary (§8.2). Dialect and code-mixed Thai were not tested. |
| Safety: publish the prune list, measure what was removed | Prune lists in `results/prune_list_*.csv`; collateral cost: Korean +39%, Russian +18%, Chinese +13%, Japanese +10%, Python +0.1% (§8.4). |
| Transparency: metrics fixed in advance, negative results published | Targets from the proposal, unchanged. Negative results reported: the corrected Week 1 metric, the restated gap, the retrofit's extra English drift, the character-balanced run that failed. |

## 10 Plan, outcomes, references

| Week | Planned deliverable | Delivered |
| --- | --- | --- |
| W1 | Baseline fertility (M1) | `BASELINE.md` - **M1 done** |
| W2 | Thai subword inventory | `DECOMPOSITION.md`, `CANDIDATES.md` |
| W3-W4 | Working tokenizer (M2) | `tokenizers/mistral-7b-th2483` + acceptance checks - **M2 done** |
| W5 | Results tables | `EVALUATION.md`, `BASELINE_after.md` |
| W6 | Robustness results | `ROBUSTNESS.md` |
| W7 | Draft report | `REPORT.md`, dashboard, Token Meter |
| W8 | Report and code (M3) | Code complete and reproduced in a clean room; **final report pending** |

**Outputs promised:** retrofitted tokenizer ✓, evaluation code ✓, Thai
fertility benchmark ✓ (`BASELINE.md`, `parallel_results.csv`), final report
(draft ✓). **Impact claim to soften:** "transfers to Lao, Khmer, Burmese" is
plausible (Regmi et al. measure all three) but untested here.

**References.** Typhoon, BrahmicTokenizer and Vowel Signs should be cited in
the final report (Vowel Signs was added on 5 Oct, see §3). MEGA and the
tokenization-premium paper can be dropped, for the reasons in component 05.
