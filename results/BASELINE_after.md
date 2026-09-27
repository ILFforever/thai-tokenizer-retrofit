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
| BLOOM +TH1000 | retrofit | 45.3 | 23.7 | **1.91x** |
| Mistral-7B +TH500 | retrofit | 49.5 | 25.9 | **1.91x** |
| Mistral-7B +TH1000 | retrofit | 42.4 | 26.0 | **1.63x** |
| BLOOM +TH3219 | retrofit | 35.5 | 23.7 | **1.50x** |
| Mistral-7B +TH2000 | retrofit | 36.0 | 26.0 | **1.39x** |
| BLOOM +TH5000 | retrofit | 32.7 | 23.7 | **1.38x** |
| Mistral-7B +TH2483 | retrofit | 34.6 | 26.0 | **1.33x** |
| Typhoon-7B | open-weight | 32.7 | 25.9 | **1.26x** |
| BLOOM +TH10000 | retrofit | 28.9 | 23.7 | **1.22x** |
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
| Mistral-7B +TH500 | 50.1 | 52.8 | 51.1 | 1.05x |
| BLOOM +TH1000 | 44.1 | 46.2 | 44.1 | 1.05x |
| Mistral-7B +TH1000 | 41.8 | 45.6 | 42.9 | 1.09x |
| GPT-4o (o200k_base) | 42.2 | 44.4 | 42.7 | 1.05x |
| Mistral-7B +TH2000 | 35.1 | 39.4 | 36.3 | 1.12x |
| Mistral-7B +TH2483 | 33.4 | 37.7 | 34.5 | 1.13x |
| Typhoon-7B | 32.6 | 37.2 | 34.0 | 1.14x |
| BLOOM +TH3219 | 33.3 | 36.1 | 33.4 | 1.08x |
| BLOOM +TH5000 | 30.1 | 33.2 | 30.4 | 1.10x |
| XGLM | 28.7 | 32.2 | 29.5 | 1.12x |
| mT5 | 28.5 | 32.0 | 29.7 | 1.13x |
| BLOOM +TH10000 | 26.2 | 29.4 | 26.7 | 1.12x |
| WangchanBERTa | 23.3 | 29.2 | 26.2 | 1.25x |

A large spread means the tokenizer's Thai performance depends heavily on register - relevant to the fairness question, since a vocabulary built only from formal text penalises everyone who writes differently.


## Cluster breaks

Share of Thai clusters (a base character plus its vowel/tone marks) that the final tokens split, measured on byte offsets over the first 400 documents. This is a *token-level* measure: it does not say whether the pre-tokenizer or the vocabulary made the cut - `DECOMPOSITION.md` does. The Week 1 mark-split rate is kept for comparison; it decoded one token at a time and so missed byte-level fragments, and it is not a pre-tokenization measure.


| Tokenizer | news | social | wikipedia | legacy mark-split (wikipedia) |
| --- | ---: | ---: | ---: | ---: |
| BLOOM | 39.2% | 39.5% | 38.6% | 16.1% |
| GPT-4 (cl100k_base) | 25.9% | 27.5% | 26.4% | 17.3% |
| Mistral-7B | 23.3% | 26.0% | 22.2% | 18.9% |
| Qwen2.5 | 22.8% | 24.8% | 21.7% | 28.8% |
| BLOOM +TH1000 | 10.8% | 12.4% | 11.2% | 14.5% |
| Mistral-7B +TH500 | 9.6% | 11.9% | 9.8% | 14.2% |
| GPT-4o (o200k_base) | 7.8% | 9.8% | 7.8% | 12.8% |
| Mistral-7B +TH1000 | 6.3% | 8.5% | 6.3% | 10.9% |
| BLOOM +TH3219 | 5.6% | 6.8% | 5.7% | 9.8% |
| Mistral-7B +TH2000 | 3.7% | 5.5% | 3.7% | 7.6% |
| BLOOM +TH5000 | 4.4% | 5.5% | 4.5% | 8.4% |
| mT5 | 3.8% | 5.1% | 3.9% | 2.7% |
| Typhoon-7B | 3.0% | 5.0% | 3.1% | 6.7% |
| Mistral-7B +TH2483 | 3.1% | 4.7% | 3.1% | 6.7% |
| WangchanBERTa | 3.9% | 4.5% | 3.9% | 0.7% |
| BLOOM +TH10000 | 3.0% | 3.8% | 3.1% | 6.5% |
| XGLM | 1.5% | 2.6% | 1.5% | 3.8% |

## Secondary: tokens per segmented word

Segmenter-dependent (`pythainlp/newmm`). Reported because readers expect it, not used as a headline.


| Tokenizer | news | social | wikipedia |
| --- | ---: | ---: | ---: |
| BLOOM | 4.69 | 4.06 | 4.90 |
| BLOOM +TH1000 | 2.11 | 1.97 | 2.24 |
| BLOOM +TH10000 | 1.26 | 1.25 | 1.33 |
| BLOOM +TH3219 | 1.60 | 1.54 | 1.68 |
| BLOOM +TH5000 | 1.45 | 1.41 | 1.53 |
| GPT-4 (cl100k_base) | 4.45 | 3.84 | 4.77 |
| GPT-4o (o200k_base) | 2.01 | 1.90 | 2.17 |
| Mistral-7B | 4.74 | 4.18 | 4.98 |
| Mistral-7B +TH1000 | 2.01 | 1.94 | 2.17 |
| Mistral-7B +TH2000 | 1.69 | 1.67 | 1.82 |
| Mistral-7B +TH2483 | 1.61 | 1.60 | 1.72 |
| Mistral-7B +TH500 | 2.40 | 2.25 | 2.59 |
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
