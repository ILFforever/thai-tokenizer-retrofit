# Thai-Optimized Tokenizer

Reducing the token cost of Thai text by retrofitting Thai subwords into an
existing open-weight tokenizer at **fixed vocabulary size**: prune entries
the deployment never uses, put Thai units in the freed slots. No vocabulary
expansion, so the embedding matrix keeps its shape; only the swapped rows
need adapting.

See `results/REPORT.md` for the full write-up, `CLAUDE.md` for project
context, and `PLAN.md` for the week-by-week plan.

## Quick start

```bash
pip install -r requirements.txt            # CPU-only; torch only for Week 6
python src/run_baseline.py --docs 120 --pairs 250    # smoke test (~80s)
```

## Pipeline

| Step | Command | Output |
| --- | --- | --- |
| Week 1 baseline | `python src/run_baseline.py` | `results/BASELINE.md` |
| Week 2 decomposition | `python src/run_decomposition.py` | `results/DECOMPOSITION.md` |
| Week 3 candidates | `python src/candidates.py --target Mistral-7B` | `results/CANDIDATES.md`, `candidates.csv` |
| Week 4 retrofit | `python src/retrofit.py --target Mistral-7B --k 500 1000 2000 3219 5000` | `tokenizers/mistral-7b-th*`, prune/add lists |
| Week 4 checks | `python src/verify_retrofit.py tokenizers/mistral-7b-th3219` | `results/RETROFIT_CHECKS.md` |
| Week 5 evaluation | `python src/evaluate.py`; `python src/run_baseline.py --retrofits --out BASELINE_after` | `results/EVALUATION.md`, `BASELINE_after.md` |
| Week 6 robustness | `python src/robustness.py` | `results/ROBUSTNESS.md` |
| Week 6 embeddings | `python src/embed_adapt.py --tokenizer tokenizers/bloom-th5000` | `results/EMBED_ADAPT_*.md` |
| Qwen side study | `python src/qwen_unreachable.py` | `results/QWEN_UNREACHABLE.md` |

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
tokenizers/              retrofitted tokenizers (AutoTokenizer.from_pretrained)
results/                 every number in the report
```

## Data

All openly licensed, streamed from Hugging Face:
`wikimedia/wikipedia` (th, en; zh/ko/ru/ja for the collateral check),
`pythainlp/thaisum`, `pythainlp/wisesight_sentiment`, `Helsinki-NLP/opus-100`
(en-th). Models: public tokenizers only, plus BLOOM-560m and Qwen2.5-0.5B
weights for the CPU embedding experiments.
