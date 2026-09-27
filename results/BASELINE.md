# Thai tokenizer baseline

Primary metric is **tokens per 100 characters**. Thai has no whitespace word boundaries, so tokens-per-word depends entirely on the segmenter and is reported only as a secondary figure (segmenter: `pythainlp/newmm`).


## Thai penalty on matched content

From a parallel corpus, so both sides carry the same meaning. This is the only table where a Thai-vs-English ratio is meaningful.


| Tokenizer | Family | Thai tok/100ch | English tok/100ch | Thai penalty |
| --- | --- | ---: | ---: | ---: |
| BLOOM | open-weight | 100.9 | 23.7 | **4.25x** |
| GPT-4 (cl100k_base) | closed | 94.3 | 23.9 | **3.94x** |
| Mistral-7B | open-weight | 101.7 | 25.9 | **3.92x** |
| Qwen2.5 | open-weight | 57.1 | 24.0 | **2.38x** |
| GPT-4o (o200k_base) | closed | 44.6 | 23.2 | **1.92x** |
| Typhoon-7B | open-weight | 32.7 | 25.9 | **1.26x** |
| XGLM | open-weight | 29.7 | 25.7 | **1.16x** |
| mT5 | open-weight | 29.5 | 29.3 | **1.01x** |
| WangchanBERTa | open-weight | 23.0 | 54.1 | **0.42x** |

## Cost by Thai domain

Lower is better. `social` is informal Thai, the register a Wikipedia-trained vocabulary is most likely to under-serve.


| Tokenizer | news | social | wikipedia | spread |
| --- | ---: | ---: | ---: | ---: |
| Mistral-7B | 99.9 | 96.9 | 96.8 | 1.03x |
| BLOOM | 98.9 | 93.9 | 94.9 | 1.05x |
| GPT-4 (cl100k_base) | 93.7 | 89.2 | 92.4 | 1.05x |
| Qwen2.5 | 57.3 | 58.0 | 56.6 | 1.03x |
| GPT-4o (o200k_base) | 42.2 | 44.4 | 42.7 | 1.05x |
| Typhoon-7B | 32.6 | 37.2 | 34.0 | 1.14x |
| XGLM | 28.7 | 32.2 | 29.5 | 1.12x |
| mT5 | 28.5 | 32.0 | 29.7 | 1.13x |
| WangchanBERTa | 23.3 | 29.2 | 26.2 | 1.25x |

A large spread means the tokenizer's Thai performance depends heavily on register - relevant to the fairness question, since a vocabulary built only from formal text penalises everyone who writes differently.


## Pre-tokenization check

Share of tokens beginning with a Thai combining mark, meaning the tokenizer cut between a base consonant and its vowel or tone mark. A high rate means vocabulary changes alone cannot fix that tokenizer - merges cannot span a boundary the pre-tokenizer already imposed.


| Tokenizer | Mark-split rate (wikipedia) |
| --- | ---: |
| Qwen2.5 | 28.8% |
| Mistral-7B | 18.9% |
| GPT-4 (cl100k_base) | 17.3% |
| BLOOM | 16.1% |
| GPT-4o (o200k_base) | 12.8% |
| Typhoon-7B | 6.7% |
| XGLM | 3.8% |
| mT5 | 2.7% |
| WangchanBERTa | 0.7% |

## Secondary: tokens per segmented word

Segmenter-dependent (`pythainlp/newmm`). Reported because readers expect it, not used as a headline.


| Tokenizer | news | social | wikipedia |
| --- | ---: | ---: | ---: |
| BLOOM | 4.69 | 4.06 | 4.90 |
| GPT-4 (cl100k_base) | 4.45 | 3.84 | 4.77 |
| GPT-4o (o200k_base) | 2.01 | 1.90 | 2.17 |
| Mistral-7B | 4.74 | 4.18 | 4.98 |
| Qwen2.5 | 2.73 | 2.49 | 2.88 |
| Typhoon-7B | 1.57 | 1.58 | 1.70 |
| WangchanBERTa | 1.13 | 1.22 | 1.30 |
| XGLM | 1.38 | 1.37 | 1.48 |
| mT5 | 1.37 | 1.35 | 1.48 |

## Corpora

- `wikipedia      1200 docs   9,140,915 chars   Thai 86%`
- `news           1200 docs   3,028,237 chars   Thai 92%`
- `social         1200 docs     189,119 chars   Thai 84%`
- `parallel       3000 pairs  TH   206,189 chars  EN   229,688 chars`
