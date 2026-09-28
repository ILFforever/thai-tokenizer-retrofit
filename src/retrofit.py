"""Week 4: retrofit Thai units into a BPE tokenizer at fixed vocabulary size.

    python src/retrofit.py --target Mistral-7B --k 1000 2000 3219 5000

For each budget K this frees K vocabulary slots and fills them with Thai
units, then saves a tokenizer loadable with `AutoTokenizer.from_pretrained`
plus the published prune and add lists.

Pruning. A slot may be freed only if its token
  - is never produced when encoding the English prune corpus (English
    non-regression by construction) nor the Thai training corpus,
  - is not a special token, a byte-fallback token or a base byte symbol,
  - and every merge that uses it as a part produces a token that is also
    being pruned (so no kept token loses the path that builds it).
Among eligible tokens, the highest id goes first: SentencePiece and BPE
trainers assign ids in order of learning, so late ids are the rarest units.

Adding. New tokens are learned by continuing BPE on Thai training text from
the base tokenizer's own segmentation: at each step, the adjacent pair whose
merge saves the most tokens (equal weight per domain) becomes a new token,
and its merge is appended after all existing merges. This is Week 3's
greedy-by-savings rule applied one slot at a time. Because every new token
is built from pieces that already exist, the plan's requirement - "add the
merges along the path that builds u, in order" - holds by construction, and
intermediate pieces are paid for out of the same budget.

Appending (rather than prepending, as Csaki et al. 2023 did) means every
existing merge keeps its priority, so text that contains no Thai is
segmented exactly as before.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from corpora import load_thai_domain  # noqa: E402

DOMAINS = ["wikipedia", "news", "social"]
ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "tokenizers"
THAI_RUN = re.compile(r" ?[฀-๿]+")
THAI_CH = re.compile(r"[฀-๿]")
MARK_START = re.compile(r"^[ัิ-ฺ็-๎]")

EVAL_DOCS = 1200
ENGLISH_PRUNE_DOCS = 8000  # english docs [0, 8000) build the prune list
CODE_PRUNE_DOCS = 5000  # Python files whose tokens are protected too
ENGLISH_EVAL_SKIP = 8000  # english docs [8000, ...) are held out for Week 5
BPE_CHAR_CAP = 4_000_000  # Thai training chars per domain for BPE learning


# --- the base tokenizer as data -------------------------------------------


@dataclass
class BPEModel:
    label: str
    vocab: dict[str, int]
    merges: list[tuple[str, str]]
    byte_level: bool
    config: dict  # full tokenizer.json, for everything we do not touch
    special_ids: set[int] = field(default_factory=set)

    @property
    def id_to_tok(self) -> dict[int, str]:
        return {i: t for t, i in self.vocab.items()}


def load_bpe(tok) -> BPEModel:
    j = json.loads(tok.raw.backend_tokenizer.to_str())
    m = j["model"]
    if m["type"] != "BPE":
        raise ValueError(f"{tok.label} is {m['type']}, not BPE")
    merges = [tuple(x) if isinstance(x, list) else tuple(x.split(" ", 1)) for x in m["merges"]]
    byte_level = "ByteLevel" in json.dumps(j.get("pre_tokenizer")) or "ByteLevel" in json.dumps(
        j.get("decoder")
    )
    special = {a["id"] for a in j.get("added_tokens", [])}
    return BPEModel(tok.label, dict(m["vocab"]), merges, byte_level, j, special)


def to_model_str(bpe: BPEModel, text: str) -> str:
    """Text as the BPE model sees it: byte-level alphabet, or `▁` for space."""
    if bpe.byte_level:
        from pretok import _bytes_to_unicode

        b2u = _bytes_to_unicode()
        return "".join(b2u[b] for b in text.encode("utf-8"))
    return text.replace(" ", "▁")


def to_text(bpe: BPEModel, s: str) -> str:
    if bpe.byte_level:
        from pretok import _bytes_to_unicode

        u2b = {v: k for k, v in _bytes_to_unicode().items()}
        return bytes(u2b[c] for c in s).decode("utf-8", errors="replace")
    return s.replace("▁", " ")


# --- usage and the prune list ----------------------------------------------


def usage_counts(tok, texts: list[str], batch: int = 256) -> np.ndarray:
    backend = tok.raw.backend_tokenizer
    counts = np.zeros(len(tok.raw), dtype=np.int64)
    for i in range(0, len(texts), batch):
        for enc in backend.encode_batch(texts[i : i + batch], add_special_tokens=False):
            np.add.at(counts, enc.ids, 1)
    return counts


def _protected(bpe: BPEModel, tokstr: str, tid: int) -> bool:
    if tid in bpe.special_ids:
        return True
    if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", tokstr):  # byte fallback
        return True
    if bpe.byte_level and len(tokstr) == 1:  # base byte alphabet
        return True
    return False


def prune_order(bpe: BPEModel, used: np.ndarray) -> list[int]:
    """Every prunable id, in the order it may be pruned.

    A token becomes available once every token built from it has been
    pruned; among available tokens the highest id goes first.
    """
    parents: dict[str, set[str]] = defaultdict(set)  # part -> results
    for a, b in bpe.merges:
        parents[a].add(a + b)
        parents[b].add(a + b)
    id_to = bpe.id_to_tok
    eligible = {
        t
        for t, i in bpe.vocab.items()
        if used[i] == 0 and not _protected(bpe, t, i)
    }
    # A token with a parent that is not eligible can never be pruned.
    blocked_by = {t: {p for p in parents[t] if p != t} for t in eligible}
    remaining = {t: len(ps) for t, ps in blocked_by.items()}
    children: dict[str, list[str]] = defaultdict(list)  # result -> parts
    for t, ps in blocked_by.items():
        for p in ps:
            children[p].append(t)
    heap = [(-bpe.vocab[t], t) for t, n in remaining.items() if n == 0]
    heapq.heapify(heap)
    order: list[int] = []
    while heap:
        neg, t = heapq.heappop(heap)
        order.append(-neg)
        for c in children.get(t, []):
            if c in remaining:
                remaining[c] -= 1
                if remaining[c] == 0:
                    heapq.heappush(heap, (-bpe.vocab[c], c))
    assert all(id_to[i] in eligible for i in order)
    return order


# --- learning Thai merges ----------------------------------------------------


def thai_runs(bpe: BPEModel, texts_by_domain: dict[str, list[str]]) -> Counter:
    """Weighted counts of Thai runs (with an optional leading space), equal
    total weight per domain, in the model's string alphabet."""
    runs: Counter = Counter()
    for d, texts in texts_by_domain.items():
        c: Counter = Counter()
        n_chars = 0
        for t in texts:
            n_chars += len(t)
            c.update(THAI_RUN.findall(t))
        w = 1e6 / n_chars  # per million characters of this domain
        for r, n in c.items():
            runs[to_model_str(bpe, r)] += n * w / len(texts_by_domain)
    return runs


def learn_merges(
    base_segment,
    runs: Counter,
    budget: int,
    existing: set[str],
    forbid_mark_start: bool,
    to_text_fn,
) -> tuple[list[tuple[str, str]], list[bool], list[str], list[float]]:
    """Continue BPE on the weighted Thai runs.

    Returns (merges, whether each merge took a slot, new token strings,
    savings per step). A merge whose result already exists in the
    vocabulary costs no slot.
    """
    sym_id: dict[str, int] = {}
    sym_str: list[str] = []

    def sid(s: str) -> int:
        if s not in sym_id:
            sym_id[s] = len(sym_str)
            sym_str.append(s)
        return sym_id[s]

    words: list[list[int]] = []
    weights: list[float] = []
    for run, w in runs.items():
        words.append([sid(s) for s in base_segment(run)])
        weights.append(w)

    pair_count: dict[tuple[int, int], float] = defaultdict(float)
    where: dict[tuple[int, int], set[int]] = defaultdict(set)

    def add_word(wi: int, sign: float) -> None:
        syms = words[wi]
        for p in zip(syms, syms[1:]):
            pair_count[p] += sign * weights[wi]
            if sign > 0:
                where[p].add(wi)

    for wi in range(len(words)):
        add_word(wi, 1.0)

    def allowed(p: tuple[int, int]) -> bool:
        if not forbid_mark_start:
            return True
        return not MARK_START.match(to_text_fn(sym_str[p[0]] + sym_str[p[1]]).lstrip(" "))

    heap = [(-c, p) for p, c in pair_count.items() if c > 0]
    heapq.heapify(heap)
    merges: list[tuple[str, str]] = []
    took_slot: list[bool] = []
    new_tokens: list[str] = []
    savings: list[float] = []
    used_slots = 0
    while heap and used_slots < budget:
        negc, p = heapq.heappop(heap)
        c = pair_count.get(p, 0.0)
        if c <= 1e-12 or abs(c + negc) > 1e-9 * max(1.0, c):
            if c > 1e-12:
                heapq.heappush(heap, (-c, p))
            continue
        if not allowed(p):
            continue
        a, b = sym_str[p[0]], sym_str[p[1]]
        new = a + b
        merges.append((a, b))
        savings.append(c)
        took_slot.append(new not in existing)
        if new not in existing:
            new_tokens.append(new)
            existing.add(new)
            used_slots += 1
        nid = sid(new)
        touched: set[tuple[int, int]] = set()
        for wi in list(where.pop(p, ())):
            syms = words[wi]
            if len(syms) < 2:
                continue
            for q in zip(syms, syms[1:]):
                pair_count[q] -= weights[wi]
                touched.add(q)
            out, i = [], 0
            while i < len(syms):
                if i + 1 < len(syms) and syms[i] == p[0] and syms[i + 1] == p[1]:
                    out.append(nid)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            words[wi] = out
            for q in zip(out, out[1:]):
                pair_count[q] += weights[wi]
                where[q].add(wi)
                touched.add(q)
        pair_count.pop(p, None)
        for q in touched:
            if q != p and pair_count.get(q, 0) > 1e-12:
                heapq.heappush(heap, (-pair_count[q], q))
    return merges, took_slot, new_tokens, savings


# --- assembling the tokenizer -------------------------------------------------


def build(bpe: BPEModel, prune_ids: list[int], new_tokens: list[str], new_merges: list[tuple[str, str]]):
    """A new backend Tokenizer with pruned entries removed and new tokens in
    the freed ids (lowest freed id first). Same size as the base."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE

    if len(new_tokens) > len(prune_ids):
        raise ValueError("more new tokens than freed slots")
    id_to = bpe.id_to_tok
    pruned = {id_to[i] for i in prune_ids[: len(new_tokens)]}
    vocab = {t: i for t, i in bpe.vocab.items() if t not in pruned}
    freed = sorted(bpe.vocab[t] for t in pruned)
    for t, i in zip(new_tokens, freed):
        assert t not in vocab, t
        vocab[t] = i
    merges = [
        (a, b)
        for a, b in bpe.merges
        if a not in pruned and b not in pruned and a + b not in pruned
    ] + list(new_merges)
    for a, b in merges:
        assert a in vocab and b in vocab and a + b in vocab, (a, b)
    m = bpe.config["model"]
    opts = {
        k: m[k]
        for k in ("unk_token", "continuing_subword_prefix", "end_of_word_suffix",
                  "fuse_unk", "byte_fallback", "ignore_merges")
        if m.get(k) is not None
    }
    model = BPE(vocab=vocab, merges=merges, **opts)
    tk = Tokenizer.from_str(json.dumps(bpe.config))
    tk.model = model
    return tk, sorted(pruned, key=lambda t: bpe.vocab[t])


def save_hf(tk, base_hf, path: Path) -> None:
    from transformers import PreTrainedTokenizerFast

    hf = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        bos_token=base_hf.bos_token,
        eos_token=base_hf.eos_token,
        unk_token=base_hf.unk_token,
        pad_token=base_hf.pad_token,
    )
    # An instruct model's chat template has to survive the rebuild, or the
    # retrofitted tokenizer can no longer format a conversation.
    if getattr(base_hf, "chat_template", None):
        hf.chat_template = base_hf.chat_template
    path.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(str(path))


def script_of(s: str) -> str:
    s = s.strip()
    if not s:
        return "space"
    o = ord(s[0])
    if o < 128:
        return "ascii"
    if 0x400 <= o < 0x530:
        return "cyrillic"
    if 0x3040 <= o < 0x3100 or 0x4E00 <= o < 0xA000 or 0x3400 <= o < 0x4DC0:
        return "cjk"
    if 0xAC00 <= o < 0xD7B0 or 0x1100 <= o < 0x1200 or 0x3130 <= o < 0x3190:
        return "hangul"
    if 0x0E00 <= o < 0x0E80:
        return "thai"
    if o < 0x250:
        return "latin-ext"
    return "other"


# --- driver ----------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Mistral-7B")
    ap.add_argument("--k", type=int, nargs="+", default=[1000, 2000, 3219, 5000])
    ap.add_argument("--mode", choices=["plain", "cluster"], default="plain",
                    help="cluster: never create a token that starts with a combining mark")
    ap.add_argument("--english-docs", type=int, default=ENGLISH_PRUNE_DOCS)
    ap.add_argument("--code-docs", type=int, default=CODE_PRUNE_DOCS,
                    help="Python files whose tokens are also protected (0 = English only)")
    ap.add_argument("--train-domains", nargs="+", default=DOMAINS,
                    help="Thai domains the new merges are learned from (ablation: "
                         "formal-only = wikipedia news)")
    ap.add_argument("--hf-repo", default=None,
                    help="load the base tokenizer from this Hugging Face repo instead of the "
                         "registry (e.g. Qwen/Qwen2.5-0.5B-Instruct); --target is then its name")
    ap.add_argument("--mark-aware", action="store_true",
                    help="first make the pre-tokenizer mark-aware (the regex fix), so merges "
                         "can span a Thai consonant and its marks")
    args = ap.parse_args()
    # `learn_variant` changes the learned merges; `variant` also names the
    # protection profile, which changes only the prune list.
    learn_variant = ("-cluster" if args.mode == "cluster" else "") + (
        "" if sorted(args.train_domains) == sorted(DOMAINS) else "-" + "+".join(args.train_domains))
    variant = learn_variant + ("-enonly" if args.code_docs == 0 else "")
    t0 = time.time()

    from candidates import train_texts
    from registry import load_all

    if args.hf_repo:
        from transformers import AutoTokenizer

        from registry import Tok

        hf = AutoTokenizer.from_pretrained(args.hf_repo)
        tok = Tok(args.target, "open-weight", lambda s, _t=hf: _t.encode(s, add_special_tokens=False),
                  hf.decode, len(hf), raw=hf, kind="hf")
    else:
        tok = load_all(verbose=False, only=[args.target])[0][0]
    save_base = tok.raw  # special tokens and chat template come from the original
    if args.mark_aware:
        from pretok import with_mark_aware_pretok

        fixed = with_mark_aware_pretok(tok)
        if fixed is None:
            raise SystemExit(f"{args.target}'s pre-tokenizer already keeps marks with their base")
        tok = fixed
    bpe = load_bpe(tok)
    slug = re.sub(r"[^a-z0-9]+", "-", args.target.lower()).strip("-") + ("-ma" if args.mark_aware else "")
    print(f"{args.target}: vocab {len(bpe.vocab):,}, merges {len(bpe.merges):,}, "
          f"byte_level={bpe.byte_level}", flush=True)

    # 1. usage on the protected corpora and on Thai (training slices).
    #    English is raw text (spacing kept) and code is raw Python: an
    #    earlier version protected whitespace-normalised English prose only,
    #    and so freed code and LaTeX pieces (` func`, `json`, `mathcal`) that
    #    an English deployment still needs.
    english = load_thai_domain("english_raw", n_docs=args.english_docs).texts
    code = load_thai_domain("code", n_docs=args.code_docs).texts if args.code_docs else []
    thai_train = {d: train_texts(d) for d in DOMAINS}
    used_en = usage_counts(tok, english)
    used_code = usage_counts(tok, code) if code else np.zeros_like(used_en)
    used_th = usage_counts(tok, [t for ts in thai_train.values() for t in ts])
    used = used_en + used_code + used_th
    order = prune_order(bpe, used)
    print(f"  used by English: {(used_en > 0).sum():,}; by code: {(used_code > 0).sum():,}; "
          f"by Thai: {(used_th > 0).sum():,}; by any: {(used > 0).sum():,}; "
          f"prunable: {len(order):,}  ({time.time() - t0:.0f}s)", flush=True)
    kmax = max(args.k)
    if kmax > len(order):
        print(f"  WARNING: budget {kmax} exceeds prunable {len(order)}; capping", flush=True)

    # 2. Thai atoms: characters the base cannot represent as a single token
    #    (byte fallback) are added first - merges cannot involve them otherwise.
    runs_text = {d: [] for d in args.train_domains}
    for d in args.train_domains:
        n = 0
        for t in thai_train[d]:
            runs_text[d].append(t)
            n += len(t)
            if n >= BPE_CHAR_CAP:
                break
    runs = thai_runs(bpe, runs_text)
    char_freq: Counter = Counter()
    for r, w in runs.items():
        for ch in r:
            char_freq[ch] += w
    atoms = [
        ch for ch, w in char_freq.most_common()
        if ch not in bpe.vocab and not bpe.byte_level and THAI_CH.match(ch) and w >= 0.5
    ]
    print(f"  Thai runs: {len(runs):,} types; atoms to add: {len(atoms)} {''.join(atoms)}", flush=True)

    # base segmentation of a run, with atoms available as single symbols
    from tokenizers.models import BPE as BPEM

    m = bpe.config["model"]
    seg_vocab = dict(bpe.vocab)
    for a in atoms:
        seg_vocab[a] = len(seg_vocab)
    seg_model = BPEM(vocab=seg_vocab, merges=bpe.merges,
                     **{k: m[k] for k in ("unk_token", "fuse_unk", "byte_fallback")
                        if m.get(k) is not None})

    def base_segment(run: str) -> list[str]:
        return [t.value for t in seg_model.tokenize(run)]

    # The learned merges depend only on the Thai text, the base tokenizer
    # and the mode - not on the prune list - so they are cached and reused.
    from corpora import CACHE

    cache = CACHE / f"merges_{slug}{learn_variant}_cap{BPE_CHAR_CAP}.json"
    budget = min(kmax, len(order)) - len(atoms)
    cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
    if cached and cached["atoms"] == atoms and cached["budget"] >= budget:
        merges = [tuple(m_) for m_ in cached["merges"]]
        took_slot, new_tokens, savings = cached["took_slot"], cached["new_tokens"], cached["savings"]
        print(f"  reusing {len(merges):,} cached merges ({cache.name})", flush=True)
    else:
        existing = set(bpe.vocab) | set(atoms)
        merges, took_slot, new_tokens, savings = learn_merges(
            base_segment, runs, budget, existing,
            forbid_mark_start=args.mode == "cluster",
            to_text_fn=lambda s: to_text(bpe, s),
        )
        cache.write_text(json.dumps({"atoms": atoms, "budget": budget, "merges": merges,
                                     "took_slot": took_slot, "new_tokens": new_tokens,
                                     "savings": savings}, ensure_ascii=False), encoding="utf-8")
    print(f"  learned {len(merges):,} merges -> {len(new_tokens):,} new tokens "
          f"({time.time() - t0:.0f}s)", flush=True)

    # A learned merge may land on a string the base vocabulary already has.
    # Such a token must stay, and so must its parts, so mark it used and
    # recompute the prune order.
    reused = [a + b for (a, b), took in zip(merges, took_slot) if not took and (a + b) in bpe.vocab]
    if reused:
        for t in reused:
            used[bpe.vocab[t]] += 1
        order = prune_order(bpe, used)
        print(f"  {len(reused)} merges reuse existing entries; prunable now {len(order):,}", flush=True)

    # 3. one tokenizer per budget
    RESULTS.mkdir(exist_ok=True)
    summary = []
    for k in sorted(args.k):
        k = min(k, len(order))
        n_new = k - len(atoms)
        toks_k = atoms + new_tokens[:n_new]
        # every merge up to (and including) the n_new-th that took a slot
        merges_k, slots = [], 0
        for mg, took in zip(merges, took_slot):
            if slots >= n_new:
                break
            merges_k.append(mg)
            slots += took
        tk, pruned = build(bpe, order, toks_k, merges_k)
        name = f"{slug}-th{k}{variant}"
        save_hf(tk, save_base, OUT / name)
        id_to = bpe.id_to_tok
        with (RESULTS / f"prune_list_{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "token", "text", "script", "english_uses", "code_uses", "thai_uses"])
            for t in pruned:
                i = bpe.vocab[t]
                w.writerow([i, t, to_text(bpe, t), script_of(to_text(bpe, t)), int(used_en[i]), int(used_code[i]), int(used_th[i])])
        new_vocab = json.loads(tk.to_str())["model"]["vocab"]
        with (RESULTS / f"added_tokens_{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "token", "text", "kind"])
            for t in toks_k:
                w.writerow([new_vocab[t], t, to_text(bpe, t), "atom" if t in atoms else "merge"])
        scripts = Counter(script_of(to_text(bpe, t)) for t in pruned)
        summary.append({"name": name, "k": k, "atoms": len(atoms), "merged": n_new,
                        "merges_added": len(merges_k), "pruned_scripts": dict(scripts)})
        print(f"  saved {name}: pruned {len(pruned)} ({dict(scripts.most_common(5))})", flush=True)

    (RESULTS / f"retrofit_{slug}{variant}_meta.json").write_text(
        json.dumps({"target": args.target, "mode": args.mode, "prunable": len(order),
                    "train_domains": args.train_domains,
                    "english_prune_docs": args.english_docs,
                    "english_prune_chars": sum(map(len, english)),
                    "used_by_english": int((used_en > 0).sum()),
                    "code_prune_docs": args.code_docs,
                    "code_prune_chars": sum(map(len, code)),
                    "used_by_code": int((used_code > 0).sum()),
                    "used_by_thai": int((used_th > 0).sum()),
                    "used_by_any": int((used > 0).sum()),
                    "thai_run_types": len(runs), "atoms": atoms,
                    "bpe_char_cap_per_domain": BPE_CHAR_CAP,
                    "runs": summary, "seconds": round(time.time() - t0, 1)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
