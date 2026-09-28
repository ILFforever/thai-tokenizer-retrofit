// Checks showcase/bpe.js against the Rust tokenizers, id for id.
//   node tests/web_bpe.test.mjs
// Expected ids come from src/export_web_tokenizers.py (showcase/tok/samples.json).
import { readFileSync } from "node:fs";
import { BPETokenizer } from "../showcase/bpe.js";

const dir = new URL("../showcase/tok/", import.meta.url);
const samples = JSON.parse(readFileSync(new URL("samples.json", dir), "utf8"));
let failed = 0;
for (const key of ["mistral", "mistral_th", "typhoon", "qwen", "qwen_th", "gpt4", "gpt4o"]) {
  const t0 = Date.now();
  const tok = new BPETokenizer(JSON.parse(readFileSync(new URL(`${key}.json`, dir), "utf8")));
  const load = Date.now() - t0;
  let ok = 0, bad = [];
  const t1 = Date.now();
  samples.texts.forEach((text, i) => {
    const got = tok.encode(text), want = samples[key][i];
    if (got.length === want.length && got.every((v, k) => v === want[k])) ok++;
    else bad.push(i);
  });
  const ms = Date.now() - t1;
  console.log(`${key.padEnd(11)} ${ok}/${samples.texts.length} identical  (load ${load} ms, encode ${ms} ms)`);
  for (const i of bad.slice(0, 3)) {
    const got = tok.encode(samples.texts[i]), want = samples[key][i];
    let k = 0; while (k < got.length && got[k] === want[k]) k++;
    console.log(`   #${i} first difference at token ${k}: got ${got.slice(k, k + 5)} want ${want.slice(k, k + 5)}`,
      JSON.stringify(samples.texts[i].slice(0, 60)));
  }
  failed += bad.length;
}
process.exit(failed ? 1 : 0);
