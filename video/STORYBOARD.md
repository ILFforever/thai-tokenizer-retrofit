# Video storyboard - "Thai Token Retrofit" (v2, how-it-works cut)

About 4:40, 1920x1080, captions on screen, synthesised sound (no voice).
Style: a dark stage with a moving camera (pan, zoom, roll), a spotlight that
follows the focus, glow on active items, a blurred parallax layer of Thai
letters, vignette and film grain; chapter cards between steps. Every token,
merge, count and neighbour comes from the real tokenizers, results and model
rows (`src/video_data.py`); each scene uses its own Thai example.

| Part | Scene | What you see |
|---|---|---|
| Intro | The same sentence, twice | A news sentence: 34 English tokens vs 89 Thai; up to 4.25x across 9 tokenizers |
| | Title | Title with a light sweep |
| 1 Learn | Count pairs, merge the winner | Real BPE on 950,848 Thai characters: pair counts, the winner flies to the new vocabulary, the Wikipedia sentence merges |
| 2 Read | Two stages | Typhoon's real tokenizer on ภาษาไทย ที่ดิน: cut at the space, merges inside each piece, 15 -> 5 |
| | A pattern that cannot see Thai marks | น้ำแข็ง ("ice"): GPT-4 and Qwen cut the marks off, Mistral keeps it whole |
| | Which stage is to blame | Fix the pattern, fix the vocabulary, re-measure |
| 3 Find room | Scan with English and code | Tilted 32,000-slot grid; scan beams light what English, code and Thai use |
| 4 Safe to remove | The merge tree | " toget" is unused but " together" is built from it: locked. "лся" is a leaf: freed |
| | 2,483 slots are free | Freed slots pulse; their old contents fly out |
| 5 Fill | Missing letters first | 26 characters Mistral spelled as bytes |
| | Keep merging on Thai | New merges fly into the free slots in their real order |
| | Watch one word get built | มหาวิทยาลัย, 11 merges, the camera following each pair |
| 6 Protect English | Merge order | 56,664 kept merges first, 2,457 new after; "government" vs ความรัก |
| | Same size | 32,000 -> 32,000 |
| 7 Teach the model | A recycled row | โรงพยาบาล's slot used to be 証; its row's neighbours are CJK |
| | Average, then train | Mean of 10 old pieces -> Thai-letter neighbours; after 400 GPU steps "Hospital" is a neighbour |
| Result | Three real sentences | Wikipedia 71->19, news 69->26, social 58->18 (Typhoon 16/23/18) |
| | The catch | Korean +39%, Russian +18%, Chinese +13%, Japanese +10%, code +0.1% |
| | Faster Thai answers | Qwen2.5-0.5B answers streaming side by side |
| | Try it | QR code to the Thai Token Meter (checked to decode from the MP4) |
