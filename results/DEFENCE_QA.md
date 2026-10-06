# Defence prep: likely questions, short answers

For the final presentation. Each answer is what to say in two or three
sentences, the number to quote, and where it comes from. Hardest questions
first. Every number is from `results/`. Written 5 Oct 2026.

## Say these before anyone asks

1. **"Fixed-size replacement for Thai was done before, by Csaki et al. in
   2023, and the regex cause was published by Regmi et al. in 2026. My
   proposal's gap statement was wrong, and the report corrects it."** Owning
   it first is far stronger than being caught.
2. **"'Without retraining' means no vocabulary expansion and no pretraining
   from scratch. The swapped embedding rows still have to be reset and
   lightly trained."**
3. **"It only helps self-hosted open-weight models. Users of GPT-4 or Claude
   cannot change the tokenizer."**

---

## A. Novelty and prior work

**Q: Csaki et al. already swapped Thai tokens into a fixed-size vocabulary.
What is new?**
They swapped the *least frequent* tokens of GPT-2 and re-initialised the rows
at random. This project: (1) splits each tokenizer's Thai penalty into a
regex share and a vocabulary share, across nine tokenizers; (2) finds 1,578
Qwen2.5 Thai entries its own regex can never produce, all untrained;
(3) protects English and code *by construction*, with a published prune
list; (4) compares like-for-like with Typhoon on the same base; (5) reports
per register and the cost to other languages. *(§3, `REFERENCES.md`)*

**Q: Your cover slide cites the Vowel Signs paper. Isn't the regex finding
theirs?**
Yes. Regmi et al. name `\p{L}+` dropping combining marks, with Thai worst at
9.02x. My decomposition confirms it on real tokenizers and adds what they
did not measure: how much of each tokenizer's excess it explains (89% for
Qwen2.5, 38% for cl100k, ~0% for Mistral). *(§3, §7.2)*

**Q: Your slide said 37.4 tokens per 100 characters versus 4.1.**
Those are pre-tokens - regex pieces on FLORES-200 - which set a floor on the
token count. They are not token counts. I should have said "at least". *(Regmi
et al., Table 6)*

**Q: BrahmicTokenizer got 26.7%. You got 64%. Why so different?**
The two numbers measure different things. Their 26.7% compares two different
tokenizers of the same size. Mine compares one tokenizer before and after its
retrofit, and Mistral starts from a nearly character-level Thai baseline (97
tokens per 100 characters), so there is much more to remove. *(§10, RQ2)*

**Q: Typhoon says 2.62x. How do you compare?**
That 2.62x is against GPT-4's tokenizer, not Mistral. On the same Mistral
base: Typhoon (vocabulary *grown* to 35,219) gives 34.0 / 32.6 / 37.2
tokens per 100 characters; my retrofit at Typhoon's budget, *without growing*,
gives 32.6 / 31.4 / 35.8 - cheaper on all three domains. *(`EVALUATION.md`)*

## B. The method

**Q: Why Mistral-7B and not a commercial tokenizer as proposed?**
A hosted model's tokenizer cannot be changed by its users. And for GPT-4's
cl100k most of the damage is in the regex, which no vocabulary change can
reach. Mistral's damage is 100% in the vocabulary, so it is exactly what a
retrofit can fix. *(§7.2, `DECOMPOSITION.md`)*

**Q: Which entries did you delete? How do you know they were unused?**
Entries that never appear when tokenizing 83.7M characters of English
Wikipedia, ~50M characters of Python, and Thai - plus a rule that never
removes an entry another merge depends on. That frees 2,483 of 32,000
(7.8%). They are mostly Cyrillic, CJK, Hangul and markup pieces. The list is
published. *(§6.3, `prune_list_mistral-7b-th2483.csv`)*

**Q: Why only 2,483 slots when Typhoon used 5,000?**
That is all Mistral can free while protecting English *and* code. Protecting
English only frees 4,339 and gives −68 to −71%. The cap is a design choice
for safety, not a limit of the method. *(`EVALUATION.md`)*

**Q: Why greedy allocation?**
It is simple and transparent. An integer program could do slightly better;
that is listed as future work. *(§9)*

**Q: How are the new rows initialised?**
Each new token's row starts as the mean of the rows of the old pieces it
used to be split into (FVT, Gee et al. 2022). On BLOOM-560m that recovers
most of the loss with no training; random rows make Thai 4x worse. *(§7.6)*

## C. Does the model still work?

**Q: Fewer tokens is easy. Is the model still good?**
Quality is measured in bits per character, which does not depend on the
tokenizer. Trained the same way as a control, the retrofitted Mistral-7B is
within −4% to +8% on Thai (better on news) at a third of the Thai tokens.
Qwen2.5-0.5B after 2,000 steps is level: +0.3% / −0.2% / +3.1%. *(§8.5)*

**Q: Any task-level evidence?**
Belebele reading comprehension (900 questions per language). Mistral-7B Thai:
control 36.3%, retrofit 33.9%, p = 0.28 - not a measurable difference.
English unchanged. Qwen2.5-0.5B is the same story (p = 0.21). *(§8.5)*

**Q: Qwen2.5-0.5B on Belebele is near chance (25%). Isn't that meaningless?**
For 0.5B, largely yes, and I say so. That is why I also ran Qwen2.5-3B
(45% Thai) and Mistral-7B (37%). At 3B the retrofit is 3.4 points below its
control - not significant (p = 0.13) - but it is the largest gap, and I
report it. *(§8.5)*

**Q: Does it work on a stronger model?**
Less easily. Qwen2.5-3B uses the same tokenizer, so the same retrofit
applies. After 2,000 steps it is 11-12% behind an equally trained control on
Thai bits per character - about where the untrained model started - at half
the Thai tokens. Qwen2.5-0.5B was level after the same training. The pattern
across all three models: the better a model already reads Thai, the longer
the swapped rows take to train back. *(§8.5)*

**Q: English got worse after training. Why?**
Any fine-tune drifts: the control's English bits per character rose 5.9%,
the retrofit's 13.2%. Masking the swapped output rows explains only ~11% of
the extra; feeding the same Thai *text* as the control closes 24% but costs
a lot of Thai. **What works: a KL penalty that keeps English predictions
close to the original model.** It removes 54% of the extra drift (English
+9.3%), costs no Thai, and Belebele Thai is level with the control (36.2% vs
36.3%). One run, one weight - but it is now the recommended recipe.
*(§8.5)*

**Q: Is 13% English drift a problem in practice?**
On Belebele English it does not show: 52.8% vs 53.3% for the control. It is
a cost to state, not a broken model.

**Q: You evaluated 4-bit models. Is that fair?**
Every arm - original, control, retrofit - is scored under the same 4-bit
weights, so the comparison is like-for-like. Absolute numbers would differ
in full precision. *(§9)*

## D. Data and measurement

**Q: Why tokens per 100 characters, not tokens per word?**
Thai has no spaces between words, so "words" depend on which segmenter you
pick. Per character does not. Per-word is reported too, under two
segmenters; the rankings are identical (Kendall tau 1.00). *(§5, §8.1)*

**Q: How do you know English is unaffected?**
On 4,000 held-out English Wikipedia documents that were never used to choose
the prune list, English tokens change by +0.01%, and 99.55% of samples get
exactly the same token ids. *(§7.4)*

**Q: You promised ≥100M Thai characters. You used ~17M.**
True; the proposal's target was missed. The merges learn from at most 4M
characters per domain anyway, and the targets were still exceeded. Whether
more text helps was not tested.

**Q: You promised government and academic text.**
I swapped them for informal social posts, because the fairness question
(formal vs informal Thai) needs an informal domain. The out-of-domain target
was tested by building a vocabulary *without any social text*: it still cuts
social text by 58%. *(`EVALUATION.md`)*

**Q: Your social data is small.**
189k evaluation characters, so social results carry bootstrap confidence
intervals. The −61.1% has an interval of [−61.6%, −60.6%]. *(§4)*

**Q: Is the test text separate from the training text?**
Yes. The first 1,200 documents of each domain are only ever scored; every
step that learns uses later documents. *(§4)*

## E. Fairness, cost and ethics

**Q: Who pays for the freed slots?**
Languages the protect corpus did not contain: Korean text gets 39% more
tokens, Russian 18%, Chinese 13%, Japanese 10%; Python +0.1%. A deployment
that serves those languages should build the prune list from its own
traffic. *(§8.4)*

**Q: Does it favour formal Thai?**
A little, like every Thai-capable tokenizer: informal text gains 11% less
than formal, the same as Typhoon (12%). Learning the vocabulary from formal
text only widens it to 21% - so the informal data matters. *(§8.2)*

**Q: Privacy?**
Only public, openly licensed data; no client or employer data. WiseSight
posts are user-written, so they are used for measurement only and not
redistributed. None of the 2,483 added tokens contains a handle, URL or long
digit string. *(§9)*

**Q: Who benefits?**
Teams that self-host an open-weight model - on-device, or for PDPA and data
residency. For them Thai costs about a third of the tokens: cheaper, ~2.8x
more Thai per context window, and ~1.8x faster Thai output in the showcase.
*(§9, §8.5)*

## F. Qwen2.5

**Q: How do you know the 1,578 Qwen entries are untrained?**
Their embedding rows are all within 0.1% of the rows of the padding tokens,
which no text ever uses. *(`QWEN_UNREACHABLE.md`)*

**Q: Why does fixing Qwen's regex make it worse?**
The fix makes the tokenizer start using those 1,578 untrained rows, so the
model sees garbage vectors. Thai bits per character get 2-3x worse until the
rows are repaired. *(§7.5)*

**Q: You found a tokenizer bug late. Could it affect other results?**
It affected only retrofitted *Qwen* tokenizers loaded by folder through
AutoTokenizer, which silently restored the old regex. Every affected result
was re-run, and a loader now checks every tokenizer against its own
`tokenizer.json` and fails loudly on a mismatch. Mistral results were not
affected. *(§8.5, `src/hf_tok.py`)*

## G. Scope and process

**Q: Your proposal said no model training and no embedding work. You did
both.**
The core deliverable - the tokenizer and its measurements - needs neither.
The model work is a clearly labelled extra. It was needed to answer the
obvious question "does the model still work?", and it is the reason the
"no retraining" claim is narrowed.

**Q: Can someone reproduce this?**
Yes. Public data and models only, CPU only for the core results. A
clean-room run on a fresh environment passed: all modules import, all tests
pass, the smoke test passes, and the browser tokenizers match the real ones
on 2,114 of 2,114 test strings. *(README, appendix)*

**Q: What would you do next?**
Train strong models longer (their retrofit loss was still falling), with the
English KL penalty on by default and also on the control; test downstream
Thai tasks; build the prune list from real deployment traffic; try an
integer program for slot allocation. *(§10)*

**Q: Did you use AI tools?** Have your answer ready in line with the course
policy - which parts you designed and checked yourself, and how the tools
were used.

---

## Results added 6 Oct (overnight GPU runs)

- **Qwen2.5-3B, 2,000 steps:** retrofit 11-12% behind its control on Thai,
  English unchanged; Belebele Thai 37.4% vs control 40.9% (p = 0.13) vs
  untrained 45.4%. Training on this mix cost the control 4.5 points too.
- **Mistral-7B retrofit + English KL penalty, 1,000 steps:** removes 54% of
  the extra English drift at no Thai cost; Belebele Thai 36.2% (control
  36.3%), English 53.9% (control 53.3%).
