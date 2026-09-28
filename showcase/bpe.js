// A small BPE tokenizer that reproduces Hugging Face `tokenizers` output for
// the five tokenizers in the token meter (Mistral-7B and its Thai retrofit,
// Typhoon-7B, Qwen2.5 and its Thai retrofit). It implements only the pieces
// those files use: NFC / Prepend / Replace normalizers, Metaspace / Split /
// ByteLevel pre-tokenizers, and a BPE model with optional byte fallback.
//
// The merge loop follows tokenizers' `Word::merge_all`: a min-heap on
// (merge rank, position), so the lowest-ranked pair merges first and ties go
// to the leftmost pair. tests/web_bpe.test.mjs checks the output id-for-id
// against the Rust tokenizers on held-out text.

const META = "▁";

function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const b2u = new Array(256);
  bs.forEach((b, i) => { b2u[b] = String.fromCodePoint(cs[i]); });
  return b2u;
}
const B2U = bytesToUnicode();
const U2B = new Map(B2U.map((u, b) => [u, b]));
const ENC = new TextEncoder();

// Rust/Oniguruma/fancy-regex pattern -> JavaScript. Two constructs need
// rewriting: inline case-insensitive groups "(?i:...)" and possessive
// quantifiers ("?+", "++", "*+", "{m,n}+", used by tiktoken's patterns), which
// behave the same as greedy ones on these patterns.
function toJsRegex(pat) {
  // possessive -> greedy, only after a real quantifier (not the "}" of \p{L})
  pat = pat.replace(/(\?|\+|\*|\{\d+(?:,\d*)?\})\+/g, "$1");
  let out = "", i = 0;
  while (i < pat.length) {
    if (pat.startsWith("(?i:", i)) {
      let depth = 1, j = i + 4, body = "", inClass = false;
      while (j < pat.length && depth) {
        const c = pat[j];
        if (c === "\\") { body += pat.slice(j, j + 2); j += 2; continue; }
        if (!inClass && c === "(") depth++;
        if (!inClass && c === ")") { depth--; if (!depth) break; }
        if (c === "[") inClass = true;
        if (c === "]") inClass = false;
        if (/[a-zA-Z]/.test(c)) {
          const lo = c.toLowerCase(), up = c.toUpperCase();
          body += inClass ? lo + up : `[${lo}${up}]`;
        } else body += c;
        j++;
      }
      out += "(?:" + body + ")";
      i = j + 1;
      continue;
    }
    out += pat[i]; i++;
  }
  return new RegExp(out, "gu");
}

class MinHeap {
  constructor() { this.a = []; }
  less(x, y) { return x.rank < y.rank || (x.rank === y.rank && x.pos < y.pos); }
  push(v) {
    const a = this.a; a.push(v);
    let i = a.length - 1;
    while (i > 0) { const p = (i - 1) >> 1; if (!this.less(a[i], a[p])) break; [a[i], a[p]] = [a[p], a[i]]; i = p; }
  }
  pop() {
    const a = this.a; if (!a.length) return undefined;
    const top = a[0], last = a.pop();
    if (a.length) {
      a[0] = last; let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1; let m = i;
        if (l < a.length && this.less(a[l], a[m])) m = l;
        if (r < a.length && this.less(a[r], a[m])) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m], a[i]]; i = m;
      }
    }
    return top;
  }
}

export class BPETokenizer {
  constructor(data) {
    this.key = data.key;
    this.label = data.label;
    this.vocab = data.vocab;
    this.ids = new Map();
    data.vocab.forEach((t, i) => { if (t !== null) this.ids.set(t, i); });
    this.byteFallback = data.byte_fallback;
    this.unk = data.unk != null ? this.ids.get(data.unk) : undefined;
    this.normalizer = data.normalizer;
    this.pre = data.pre_tokenizer;
    this.byteLevel = JSON.stringify(this.pre || {}).includes("ByteLevel");
    // tiktoken encodings have no merge list: the pair whose concatenation has
    // the lowest rank (= id) merges first.
    this.byRank = !!data.merge_by_rank;
    // (id_a, id_b) -> [rank, new_id]
    this.merges = new Map();
    (data.merges || []).forEach((m, rank) => {
      const sp = m.indexOf(" ", 1);
      const a = m.slice(0, sp), b = m.slice(sp + 1);
      const ia = this.ids.get(a), ib = this.ids.get(b), ic = this.ids.get(a + b);
      if (ia === undefined || ib === undefined || ic === undefined) return;
      const key = ia * 1048576 + ib;
      if (!this.merges.has(key)) this.merges.set(key, [rank, ic]);
    });
    this.byteIds = [];
    if (this.byteFallback) for (let b = 0; b < 256; b++) {
      this.byteIds[b] = this.ids.get("<0x" + b.toString(16).toUpperCase().padStart(2, "0") + ">");
    }
    this.regexes = new Map();
    this.cache = new Map();
  }

  normalize(text, n = this.normalizer) {
    if (!n) return text;
    switch (n.type) {
      case "NFC": return text.normalize("NFC");
      case "Prepend": return text.length ? n.prepend + text : text;
      case "Replace": return text.split(n.pattern.String).join(n.content);
      case "Sequence": return n.normalizers.reduce((t, m) => this.normalize(t, m), text);
      default: throw new Error("normalizer not supported: " + n.type);
    }
  }

  preTokenize(pieces, p = this.pre) {
    if (!p) return pieces;
    switch (p.type) {
      case "Sequence": return p.pretokenizers.reduce((ps, q) => this.preTokenize(ps, q), pieces);
      case "Metaspace": {
        if (p.split) throw new Error("Metaspace split not supported");
        return pieces.map((s, i) => {
          let t = s.split(" ").join(p.replacement);
          const prepend = p.prepend_scheme === "always" || (p.prepend_scheme === "first" && i === 0);
          if (prepend && t.length && !t.startsWith(p.replacement)) t = p.replacement + t;
          return t;
        }).filter(s => s.length);
      }
      case "Split": {
        if (p.behavior !== "Isolated" || p.invert) throw new Error("Split mode not supported");
        const src = p.pattern.Regex;
        if (!this.regexes.has(src)) this.regexes.set(src, toJsRegex(src));
        const re = this.regexes.get(src);
        const out = [];
        for (const s of pieces) {
          let last = 0;
          re.lastIndex = 0;
          for (const m of s.matchAll(re)) {
            if (m.index > last) out.push(s.slice(last, m.index));
            if (m[0].length) out.push(m[0]);
            last = m.index + m[0].length;
          }
          if (last < s.length) out.push(s.slice(last));
        }
        return out;
      }
      case "ByteLevel":
        return pieces.map(s => Array.from(ENC.encode(s), b => B2U[b]).join(""));
      default: throw new Error("pre-tokenizer not supported: " + p.type);
    }
  }

  bpe(word) {
    const hit = this.cache.get(word);
    if (hit) return hit;
    // symbols as a doubly linked list over an array
    const c = [], prev = [], next = [], alive = [];
    for (const ch of word) {
      const id = this.ids.get(ch);
      if (id !== undefined) c.push(id);
      else if (this.byteFallback) { for (const b of ENC.encode(ch)) c.push(this.byteIds[b]); }
      else if (this.unk !== undefined) c.push(this.unk);
    }
    const n = c.length;
    for (let i = 0; i < n; i++) { prev[i] = i - 1; next[i] = i + 1 < n ? i + 1 : -1; alive[i] = true; }
    const heap = new MinHeap();
    const lookup = (a, b) => {
      if (!this.byRank) return this.merges.get(a * 1048576 + b);
      const id = this.ids.get(this.vocab[a] + this.vocab[b]);
      return id === undefined ? undefined : [id, id];
    };
    const tryPush = (i) => {
      const j = next[i];
      if (j < 0) return;
      const m = lookup(c[i], c[j]);
      if (m) heap.push({ rank: m[0], pos: i, id: m[1] });
    };
    for (let i = 0; i < n - 1; i++) tryPush(i);
    for (let top = heap.pop(); top; top = heap.pop()) {
      const i = top.pos;
      if (!alive[i] || next[i] < 0) continue;
      const j = next[i];
      const m = lookup(c[i], c[j]);
      if (!m || m[1] !== top.id) continue; // stale entry
      c[i] = top.id;
      alive[j] = false;
      next[i] = next[j];
      if (next[j] >= 0) prev[next[j]] = i;
      if (prev[i] >= 0) tryPush(prev[i]);
      tryPush(i);
    }
    const out = [];
    for (let i = 0; i >= 0 && i < n; i = next[i]) if (alive[i]) out.push(c[i]);
    if (this.cache.size > 50000) this.cache.clear();
    this.cache.set(word, out);
    return out;
  }

  encode(text) {
    const norm = this.normalize(text);
    if (!norm.length) return [];
    const ids = [];
    for (const piece of this.preTokenize([norm])) for (const id of this.bpe(piece)) ids.push(id);
    return ids;
  }

  /** Bytes of original text each token stands for (for drawing chips). */
  tokenBytes(id) {
    const t = this.vocab[id];
    if (t === null || t === undefined) return new Uint8Array();
    const fb = /^<0x([0-9A-Fa-f]{2})>$/.exec(t);
    if (fb) return Uint8Array.of(parseInt(fb[1], 16));
    if (this.byteLevel) return Uint8Array.from(Array.from(t, ch => U2B.get(ch) ?? 63));
    return ENC.encode(t.split(META).join(" "));
  }
}
