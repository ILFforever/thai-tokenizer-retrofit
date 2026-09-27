# Thai-Optimized Tokenizer

Reducing the token cost of Thai text by retrofitting a Thai subword
vocabulary into an existing tokenizer at **fixed vocabulary size** — no
vocabulary expansion, so no full model retraining.

See `CLAUDE.md` for project context, measured findings and open decisions.
See `results/BASELINE.md` for the Week 1 baseline.

## Quick start

```bash
pip install -r requirements.txt
python src/run_baseline.py --docs 120 --pairs 250   # smoke test (~80s)
python src/run_baseline.py                          # full run (~5 min)
```

## What the baseline measures

- **Thai penalty** — tokens per 100 chars, Thai vs English, on matched
  parallel content (the only place a ratio is meaningful)
- **Cost per domain** — encyclopaedic, news, informal social
- **Mark-split rate** — share of tokens starting with a Thai combining mark,
  i.e. the tokenizer cut between a consonant and its vowel/tone mark. High
  values mean a vocabulary retrofit *cannot* help that tokenizer
- **Effective context** — Thai characters per 1,000 tokens
- **Tokens per segmented word** — secondary, segmenter-dependent

## Data

All openly licensed, streamed from Hugging Face:
`wikimedia/wikipedia` (th), `pythainlp/thaisum`, `pythainlp/wisesight_sentiment`,
`Helsinki-NLP/opus-100` (en-th).
