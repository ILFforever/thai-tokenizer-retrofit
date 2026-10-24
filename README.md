# Thai Tokenizer Retrofit

**Cut the token cost of Thai text without growing the vocabulary.**
Prune the tokenizer entries a deployment never uses, and put Thai subwords
in the freed slots. The vocabulary keeps its size, so the embedding matrix
keeps its shape; only the swapped rows need re-setting.

[![Live demo](https://img.shields.io/badge/demo-Thai%20Token%20Meter-2ea44f)](https://ilfforever.github.io/thai-tokenizer-retrofit/)
[![Report](https://img.shields.io/badge/read-full%20report-blue)](https://ilfforever.github.io/thai-tokenizer-retrofit/report.html)
[![Dashboard](https://img.shields.io/badge/results-dashboard-orange)](https://ilfforever.github.io/thai-tokenizer-retrofit/dashboard.html)
[![Video](https://img.shields.io/badge/watch-explainer-red)](https://ilfforever.github.io/thai-tokenizer-retrofit/video/)

Chulalongkorn University, CEDT, AI course project, 2026. Tanabodhi Mukura.

## Try it

| | |
| --- | --- |
| **[Thai Token Meter](https://ilfforever.github.io/thai-tokenizer-retrofit/)** | Type or paste Thai and see what 7 tokenizers charge for it, live in the browser: Mistral-7B, Qwen2.5, GPT-4, GPT-4o, Typhoon, and this project's two retrofits. Every token id matches the real tokenizer (`node tests/web_bpe.test.mjs`). |
| **[Report](https://ilfforever.github.io/thai-tokenizer-retrofit/report.html)** | The full write-up (source: [`results/REPORT.md`](results/REPORT.md)). |
| **[Dashboard](https://ilfforever.github.io/thai-tokenizer-retrofit/dashboard.html)** | Every result as charts, per domain. |
| **[Explainer video](https://ilfforever.github.io/thai-tokenizer-retrofit/video/)** | The idea in a few minutes. |

## The problem

Thai has no spaces between words and little weight in tokenizer training
data, so the same meaning costs far more tokens in Thai than in English.
More tokens means higher cost, less usable context and slower generation.

| Tokenizer | Thai tokens per English token (same content) |
| --- | ---: |
| BLOOM | 4.25x |
| GPT-4 (cl100k) | 3.94x |
| **Mistral-7B** | **3.92x** |
| Qwen2.5 | 2.38x |
| GPT-4o (o200k) | 1.92x |
| Typhoon-7B (vocabulary expanded) | 1.26x |

## Headline result (Mistral-7B, vocabulary stays at 32,000)

Swapping **2,483 entries** (7.8%: every entry that English, Python code and
Thai never use) for Thai pieces:

| | Wikipedia | News | Social | Thai penalty | English |
| --- | ---: | ---: | ---: | ---: | ---: |
| Thai tokens per 100 chars, before | 96.8 | 99.9 | 96.9 | 3.92x | |
| **After the retrofit** | **34.5 (−64%)** | **33.4 (−67%)** | **37.7 (−61%)** | **1.33x** | **+0.01%** |
| Typhoon-7B (adds 3,219 entries) | 34.0 (−65%) | 32.6 (−67%) | 37.2 (−62%) | 1.26x | +0.06% |

At Typhoon's own budget (3,219 slots) the fixed-size retrofit beats Typhoon's
vocabulary expansion on all three domains. Held-out text, 95% intervals and
the full sweep are in the [report, §7.4](results/REPORT.md).

## What else we found

- **Some tokenizers cannot be fixed by vocabulary alone.** GPT-4's cl100k
  and Qwen2.5 split letters with `\p{L}+`, which leaves out Thai vowel and
  tone marks, so every mark is cut off its consonant before the vocabulary
  is used. Mistral's and BLOOM's problem is 100% vocabulary.
- **Qwen2.5 holds 1,578 Thai tokens it can never produce.** All of them are
  untrained, so fixing only the regex makes Qwen2.5-0.5B 2–3x *worse* at Thai.
- **The trained model keeps up.** With the swapped rows re-set (mean of the
  pieces they replace) and lightly tuned, Qwen2.5-0.5B is level with an
  equally trained control at about half the Thai tokens. On Belebele reading
  comprehension, no retrofit is measurably worse than its control.
- **The cost is stated, not hidden.** The freed slots come from other
  languages (Korean +39% tokens, Russian +18%, Chinese +13%, Japanese +10%,
  Python +0.1%). Stronger models need more tuning (Qwen2.5-3B is still
  11–12% behind its control). Hosted API models cannot use this; it is for
  self-hosted open-weight models.

Metric: **tokens per 100 characters**, because Thai has no spaces to count
words by. Model quality is compared in **bits per character**, which does not
depend on the tokenizer. Results are always reported per domain, never pooled.

`CLAUDE.md` has the project context and `PLAN.md` the week-by-week plan.

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch is enough
pip install -r requirements.txt
python src/run_baseline.py --docs 120 --pairs 250 --out BASELINE_smoke   # smoke test (~2 min)
python -m pytest tests
```

Everything except one marked extra runs on a CPU. Optional extras:
`requirements-gpu.txt` (Mistral-7B QLoRA on an NVIDIA GPU, separate venv with
CUDA torch) and `requirements-video.txt` (the explainer video; also needs
Google Chrome and ffmpeg). Use a venv outside OneDrive or Dropbox folders.

## Pipeline

| Step | Command | Output |
| --- | --- | --- |
| Week 1 baseline | `python src/run_baseline.py` | `results/BASELINE.md` |
| Week 2 decomposition | `python src/run_decomposition.py` | `results/DECOMPOSITION.md` |
| Week 3 candidates | `python src/candidates.py --target Mistral-7B` | `results/CANDIDATES.md`, `candidates.csv` |
| Week 4 retrofit | `python src/retrofit.py --target Mistral-7B --k 500 1000 2000 3219 5000` | `tokenizers/mistral-7b-th*`, prune/add lists |
| Week 4 checks | `python src/verify_retrofit.py tokenizers/mistral-7b-th2483` | `results/RETROFIT_CHECKS.md` |
| Week 5 evaluation | `python src/evaluate.py`; `python src/run_baseline.py --retrofits --out BASELINE_after` | `results/EVALUATION.md`, `BASELINE_after.md` |
| Week 6 robustness | `python src/robustness.py` | `results/ROBUSTNESS.md` |
| Week 6 embeddings | `python src/embed_adapt.py --tokenizer tokenizers/bloom-th5000` | `results/EMBED_ADAPT_*.md` |
| Qwen side study | `python src/qwen_unreachable.py`; `python src/qwen_repair.py` | `results/QWEN_UNREACHABLE.md` |
| Dashboard | `python src/dashboard_data.py && python src/build_dashboard.py` | `results/dashboard.html` |
| Report page | `python src/build_report_page.py` | `results/report.html` (from `REPORT.md`) |
| Explainer video | `python src/video_data.py && python src/build_video.py --render` | `video/index.html` (playable), `video/thai_token_retrofit.mp4` (gitignored) |
| Tests | `python -m pytest tests` | - |

## Showcase: Thai Token Meter and a retrofitted model

| Step | Command | Output |
| --- | --- | --- |
| Qwen2.5 retrofit (regex fix + Thai slots) | `python src/retrofit.py --target Qwen2.5-Instruct --hf-repo Qwen/Qwen2.5-0.5B-Instruct --mark-aware --k 1000 3000 5000 8000` | `tokenizers/qwen2-5-instruct-ma-th*` |
| Train both arms (same data, same LoRA) | `python src/qwen_adapt.py --arm control`; `python src/qwen_adapt.py --arm retrofit --tokenizer tokenizers/qwen2-5-instruct-ma-th3000` | `models/`, `results/qwen_adapt_*.json` |
| Side-by-side answers | `python src/qwen_showcase.py --retrofit models/qwen2.5-0.5b-instruct-retrofit-qwen2-5-instruct-ma-th3000` | `showcase/data.json` |
| Browser tokenizers | `python src/export_web_tokenizers.py` then `node tests/web_bpe.test.mjs` (must print all identical) | `showcase/tok/` (gitignored, regenerate) |
| Reading comprehension | `python src/belebele_eval.py --kind qwen --cpu --model models/<folder> --name <label>` | `results/belebele_*.json` |

## Optional GPU extra (Mistral-7B, laptop GPU)

Trained models are written to `~/thaitok_models` (set `THAITOK_MODELS` to
change it), outside the project folder because they are large and can be
rebuilt. Commands and results still call them `models/<folder>`;
`src/paths.py` maps that name to the real location.

Every Hugging Face model, tokenizer and dataset is loaded at a pinned commit
(`src/versions.py`): the exact files the results were produced with.

| Step | Command | Output |
| --- | --- | --- |
| Two-arm QLoRA, 1,000 steps | `python src/mistral_adapt_gpu.py --arm control --steps 1000 --english-windows 3 --tag _s1000`; same with `--arm retrofit` | `results/mistral_adapt_*.json` |
| Character-balanced retrofit | add `--thai-windows 2 --tag _s1000_charbal` (same Thai *text* per step as the control) | `results/mistral_adapt_*_charbal.json` |
| Belebele on Mistral | `python src/belebele_eval.py --kind mistral --adapter models/<folder> --name <label>` | `results/belebele_mistral_*.json` |
| Where English drift comes from | `python src/drift_probe.py control_s1000` (one arm per process) | `results/drift_probe_mistral-7b.json` |
| Retrofit with an English KL penalty | add `--kl-english 1.0 --tag _s1000_kl1` to the retrofit command (English windows anchored to the untrained model) | `results/mistral_adapt_*_kl1.json` |
| Qwen arms on the GPU | `python src/qwen_adapt.py --arm control --device cuda --steps 2000 --tag _gpu2000` (evaluation stays on the CPU, so numbers match the CPU runs) | `results/qwen_adapt_*_gpu2000.json` |
| Qwen2.5-3B (same tokenizer, 4-bit) | add `--base Qwen/Qwen2.5-3B-Instruct --quant4` to the line above; Belebele: `python src/belebele_eval.py --kind qwen --base Qwen/Qwen2.5-3B-Instruct --adapter models/<folder> --name <label>` | `results/qwen3b_adapt_*.json` |

**Loading a retrofitted Qwen tokenizer from a model folder:** use
`hf_tok.load_tokenizer(path)`, not `AutoTokenizer.from_pretrained`. Given a
folder with a Qwen2 `config.json`, AutoTokenizer rebuilds Qwen's default
pre-tokenizer pattern and silently drops the mark-aware one the retrofit
saved. `load_tokenizer` loads the saved pipeline and checks it against the
`tokenizers` library.

`showcase/index.html` is the Token Meter page: `bpe.js` runs all seven
tokenizers in the browser (checked id-for-id against the originals), and the
model section appears when `showcase/data.json` exists. Serve the folder over
HTTP to try it locally (`python -m http.server --directory showcase`).

Corpora stream from Hugging Face on first use and are cached in
`~/.cache/thaitok` (override with `THAITOK_CACHE`). Evaluation always uses
the first 1,200 documents per domain; anything that learns from text uses
documents after them.

## Layout

```
src/corpora.py           corpus loading + cache, disjoint train/eval slices
src/registry.py          tiktoken + HF tokenizers behind one interface
src/metrics.py           tokens/100 chars, penalty, effective context, cluster breaks
src/pretok.py            pre-tokenizer introspection, byte-aware cluster cuts,
                         mark-aware regex counterfactual, unreachable tokens
src/run_baseline.py      Week 1 (and Week 5 re-run with --retrofits)
src/run_decomposition.py Week 2
src/candidates.py        Week 3
src/retrofit.py          Week 4: prune list, BPE continuation, rebuild
src/verify_retrofit.py   Week 4 acceptance checks
src/evaluate.py          Week 5 before/after
src/robustness.py        Week 6 segmenter, register, row accounting, collateral
src/embed_adapt.py       Week 6 embedding re-initialisation + row-only tuning
src/qwen_unreachable.py  Qwen2.5's untrained Thai rows
src/qwen_repair.py       re-train those rows (row-only)
src/mistral_rows.py      Mistral-7B swapped rows on the CPU (bf16)
src/qwen_adapt.py        two-arm LoRA training, Qwen2.5-0.5B (CPU or --device cuda)
src/qwen_showcase.py     side-by-side answers for the Token Meter
src/mistral_adapt_gpu.py two-arm QLoRA training, Mistral-7B (GPU extra)
src/belebele_eval.py     Belebele reading comprehension (Thai, English)
src/drift_probe.py       how much English drift the swapped output rows cause
src/hf_tok.py            load a saved tokenizer exactly (see the Qwen note above)
src/export_web_tokenizers.py   tokenizers for the in-browser Token Meter
src/dashboard_data.py, build_dashboard.py, build_report_page.py   the pages
src/video_data.py, build_video.py, video_audio.py                 the video
tokenizers/              retrofitted tokenizers (BLOOM and Qwen ones gitignored)
showcase/                Thai Token Meter page + bpe.js
video/                   explainer video page, storyboard, soundtrack (MP4s gitignored)
results/                 every number in the report
archive/                 the original Week 1 upload, kept as received
```

## Data

All openly licensed, streamed from Hugging Face:
`wikimedia/wikipedia` (th, en; zh/ko/ru/ja for the collateral check),
`pythainlp/thaisum`, `pythainlp/wisesight_sentiment`, `Helsinki-NLP/opus-100`
(en-th). Models: public tokenizers only, plus BLOOM-560m and Qwen2.5-0.5B
weights for the CPU embedding experiments.
