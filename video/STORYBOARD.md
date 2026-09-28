# Video storyboard - "Thai Token Retrofit"

About 5 minutes, 1920x1080, captions on screen (no voice). Every number and
token comes from `results/` or the tokenizers themselves (`src/video_data.py`).

## Act 1 - The problem
| # | Scene | What moves | Seconds |
|---|---|---|---|
| 1 | Title | Name and subtitle fade in over a drifting field of Thai characters | 6 |
| 2 | What a token is | An English sentence splits into word-sized chips | 9 |
| 3 | Thai under Mistral-7B | The same meaning in Thai pours out as character chips; counters race: English vs Thai | 12 |
| 4 | The penalty | Bars grow for nine tokenizers, 4.25x down to 1.01x, parity line at 1x | 12 |
| 5 | Why it matters | Three meters: cost x4, context /4, time x4 | 10 |
| 6 | Who can fix it | Hosted API behind a lock; open-weight model on your own machine | 8 |

## Act 2 - Diagnosis
| 7 | Two stages of a tokenizer | Text -> pre-tokenizer cuts -> vocabulary merges, as a pipeline | 10 |
| 8 | The \p{L} vs \p{M} cut | ที่ดิน: scissors fall between each consonant and its marks for GPT-4 and Qwen; nothing for Mistral | 12 |
| 9 | Where the penalty comes from | Dots slide: as shipped -> regex fixed -> best vocabulary, per tokenizer | 12 |
| 10 | Qwen's dead words | 2,570 Thai entries; 1,578 grey out as unreachable and untrained | 10 |
| 11 | The target | Mistral-7B and Typhoon (SCB 10X) share a base: one grows, one keeps its size | 8 |

## Act 3 - Building the retrofit
| 12 | 32,000 slots | Mistral's vocabulary as a 200 x 160 grid, one cell per entry | 8 |
| 13 | Who uses what | Cells light by first user: English, code, Thai; the rest stay dark | 12 |
| 14 | Safe to free | 2,483 cells pulse - unused and not needed by any merge | 8 |
| 15 | What they were | Examples of freed entries drift out: Cyrillic, CJK, Korean pieces | 8 |
| 16 | What Thai needs | Top Thai words ranked by tokens saved, bars grow | 10 |
| 17 | Missing letters | 26 Thai characters Mistral cannot spell drop into the first freed cells | 8 |
| 18 | Merging | Characters merge step by step into เมื่อวันที่ (the real merge order) | 14 |
| 19 | Filling the slots | Freed cells turn blue, each now a Thai piece | 8 |
| 20 | Same size | Counter: 32,000 before, 32,000 after | 6 |

## Act 4 - Results
| 21 | One sentence again | 89 chips collapse to 28; Typhoon 31 beside it | 12 |
| 22 | Per domain | Bars shrink: -64% Wikipedia, -67% news, -61% social; English +0.01% | 10 |
| 23 | More slots, fewer tokens | Curve draws itself across budgets; Typhoon's point appears | 10 |
| 24 | Against Typhoon | At the same budget: 3.7-4.2% fewer tokens than Typhoon | 8 |
| 25 | Fairness | Informal Thai gap: base, retrofit, Typhoon, formal-only | 10 |
| 26 | Who pays | Other languages' extra cost: Korean, Russian, Chinese, Japanese, code | 10 |

## Act 5 - Making a model use it
| 27 | The rows problem | Swapped rows still hold the old tokens' vectors: arrows point to the wrong meaning | 10 |
| 28 | Four ways to set the rows | BLOOM bits per character: stale, random, mean, mean + training | 12 |
| 29 | Mistral-7B on a laptop GPU | Control vs retrofit after identical training | 12 |
| 30 | Twice the Thai per token | Two answers stream side by side; the retrofit writes Thai about twice as fast | 10 |
| 31 | What checking changed | Four corrected claims flip from wrong to fixed | 10 |
| 32 | Try it | The Thai Token Meter, the dashboard and the report | 8 |
