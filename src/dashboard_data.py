"""Collect every number the dashboard shows into one JSON file.

    python src/dashboard_data.py

Reads only files in results/ (plus the tokenizers, for the token-chip
examples), so the dashboard can never show a number the pipeline did not
produce. Writes results/dashboard_data.json.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
DOMAINS = ["wikipedia", "news", "social"]


def _csv(name: str) -> list[dict]:
    with (RESULTS / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in str(v) else f
    except (TypeError, ValueError):
        return v


def _md_table(md: str, heading: str) -> list[list[str]]:
    """Rows of the first markdown table after `heading`."""
    i = md.index(heading)
    rows = []
    for line in md[i:].splitlines()[1:]:
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not set(cells[0]) <= set("-: "):
                rows.append(cells)
        elif rows:
            break
    return rows


def token_examples() -> list[dict]:
    """A short real sentence per domain, cut by several tokenizers."""
    from transformers import AutoTokenizer

    from corpora import load_thai_domain
    from registry import load_all, retrofit_dirs

    toks = {t.label: t for t in load_all(verbose=False)[0]}
    retro = {p.name: p for p in retrofit_dirs("mistral-7b")}
    strict = max((p for n, p in retro.items() if n.split("-th")[1].isdigit()),
                 key=lambda p: int(p.name.split("-th")[1]))
    show = [
        ("GPT-4 (cl100k_base)", toks["GPT-4 (cl100k_base)"].raw, "tiktoken"),
        ("Qwen2.5", toks["Qwen2.5"].raw, "hf"),
        ("Mistral-7B", toks["Mistral-7B"].raw, "hf"),
        (f"Mistral-7B +TH{strict.name.split('-th')[1]} (this project)",
         AutoTokenizer.from_pretrained(str(strict)), "hf"),
        ("Typhoon-7B (expanded)", toks["Typhoon-7B"].raw, "hf"),
    ]
    picks: dict[str, str] = {}
    for d in DOMAINS:
        for t in load_thai_domain(d, n_docs=60).texts:
            # a short, almost all-Thai opening, cut at a space
            cand = t[:90]
            j = cand.rfind(" ")
            cand = cand[:j] if j > 40 else cand
            thai = len(re.findall(r"[฀-๿]", cand))
            if 40 <= len(cand) <= 90 and thai / len(cand) > 0.85:
                picks[d] = cand
                break
    out = []
    for d, text in picks.items():
        rows = []
        for label, tok, kind in show:
            ids = tok.encode(text) if kind == "tiktoken" else tok.encode(text, add_special_tokens=False)
            rows.append({"tokenizer": label, "pieces": _pieces(tok, ids, kind), "count": len(ids)})
        out.append({"domain": d, "text": text, "rows": rows})
    return out


def _pieces(tok, ids: list[int], kind: str) -> list[dict]:
    """What each token stands for. A token holding only part of a Thai
    character's three UTF-8 bytes is marked `partial` and labelled with the
    character it helps to spell."""
    if kind == "tiktoken":
        blobs = [tok.decode_single_token_bytes(i) for i in ids]
    else:
        from pretok import _bytes_to_unicode

        u2b = {v: k for k, v in _bytes_to_unicode().items()}
        byte_level = "ByteLevel" in json.dumps(json.loads(tok.backend_tokenizer.to_str()).get("pre_tokenizer"))
        blobs = []
        for piece in tok.convert_ids_to_tokens(ids):
            if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece):
                blobs.append(bytes([int(piece[3:5], 16)]))
            elif byte_level:
                blobs.append(bytes(u2b[c] for c in piece))
            else:
                blobs.append(piece.replace("▁", " ").encode("utf-8"))
    out: list[dict] = []
    pending: list[int] = []
    buf = b""
    for b in blobs:
        buf += b
        out.append({"t": "", "partial": True})
        pending.append(len(out) - 1)
        try:
            s = buf.decode("utf-8")
        except UnicodeDecodeError:
            continue
        whole = len(pending) == 1
        for i in pending:
            out[i] = {"t": s, "partial": not whole}
        pending, buf = [], b""
    return out


def main() -> None:
    data: dict = {}

    data["baseline"] = [{k: _num(v) for k, v in r.items()} for r in _csv("parallel_results.csv")]
    data["baseline_domains"] = [{k: _num(v) for k, v in r.items()} for r in _csv("domain_results.csv")]
    data["decomposition"] = [{k: _num(v) for k, v in r.items()} for r in _csv("decomposition.csv")]

    cand_md = (RESULTS / "CANDIDATES.md").read_text(encoding="utf-8")
    proj = {}
    for level, head in (("word", "## Projected token reduction on held-out text: newmm words"),
                        ("tcc", "## Projected token reduction on held-out text: Thai Character Clusters")):
        rows = _md_table(cand_md, head)
        proj[level] = [{"n": int(r[0].replace(",", "")), **{d: float(r[i + 1].rstrip("%")) / 100
                        for i, d in enumerate(DOMAINS)}} for r in rows]
    data["candidates_projection"] = proj
    data["candidates_top"] = [
        {k: _num(v) for k, v in r.items()} for r in _csv("candidates.csv") if r["level"] == "word"][:40]

    for name in ("evaluation.json", "robustness.json", "qwen_unreachable.json"):
        p = RESULTS / name
        if p.exists():
            data[name.split(".")[0]] = json.loads(p.read_text(encoding="utf-8"))
    for p in sorted(RESULTS.glob("embed_adapt_*.json")):
        data.setdefault("embed_adapt", {})[p.stem.replace("embed_adapt_", "")] = json.loads(
            p.read_text(encoding="utf-8"))
    for p in sorted(RESULTS.glob("qwen_repair*.json")):
        data[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    for p in sorted(RESULTS.glob("retrofit_*_meta.json")):
        data.setdefault("retrofit_meta", {})[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    for p in sorted(RESULTS.glob("retrofit_checks_*.json")):
        data.setdefault("retrofit_checks", {})[p.stem.replace("retrofit_checks_", "")] = json.loads(
            p.read_text(encoding="utf-8"))

    data["examples"] = token_examples()
    (RESULTS / "dashboard_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    print("wrote results/dashboard_data.json", len(json.dumps(data)) // 1024, "KB")


if __name__ == "__main__":
    main()
