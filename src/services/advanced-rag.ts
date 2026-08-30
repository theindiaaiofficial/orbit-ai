import type { RetrievedChunk } from '../domain/types.js';

/**
 * TypeScript production adapter for the standalone Advanced RAG principles.
 * Retrieval is intentionally split into broad candidate recall and narrow
 * evidence selection. The vector provider remains the durable tenant-scoped
 * store; lexical/BM25 signals are computed over its bounded candidate set.
 */
const STOP = new Set([
  'the','and','for','how','what','when','where','why','does','can','could','you','your','my','do','i','a','an','is','are','to','of','or','in','on','with','me','it','this','that','please','tell','about','have','has','from','our','they','them','their','will','would','should','which','who',
]);
const tokens = (value: string) => [...new Set(value.toLocaleLowerCase().match(/[\p{L}\p{N}]{2,}/gu) ?? [])];
const meaningful = (value: string) => tokens(value).filter((x) => !STOP.has(x));
const normalize = (value: string) => value.replace(/\0/g, ' ').replace(/\s+/g, ' ').trim();
const phrase = (query: string) => normalize(query).toLocaleLowerCase();

function bm25(query: string[], text: string, corpus: RetrievedChunk[]) {
  const words = tokens(text);
  const counts = new Map<string, number>();
  for (const word of words) counts.set(word, (counts.get(word) ?? 0) + 1);
  const avg = Math.max(1, corpus.reduce((n, x) => n + tokens(x.text).length, 0) / Math.max(1, corpus.length));
  const length = words.length;
  return query.reduce((score, term) => {
    const tf = counts.get(term) ?? 0;
    if (!tf) return score;
    const df = corpus.filter((x) => tokens(x.text).includes(term)).length;
    const idf = Math.log(1 + (corpus.length - df + 0.5) / (df + 0.5));
    return score + idf * ((tf * 2.2) / (tf + 1.2 * (0.8 + 0.2 * length / avg)));
  }, 0);
}

function rrf(rank: number, k = 60) { return 1 / (k + rank + 1); }

function answerFragment(text: string, query: string) {
  const clean = normalize(text);
  const faq = [...clean.matchAll(/\*{0,2}Q:\s*(.*?)\*{0,2}\s+\*{0,2}A:\s*(.*?)(?=\s+\*{0,2}Q:|$)/gis)]
    .map((m) => ({ q: m[1]!.trim(), a: m[2]!.trim() }))
    .filter((x) => x.a);
  const qTerms = meaningful(query);
  if (faq.length) {
    const ranked = faq.map((x) => ({ x, hits: meaningful(`${x.q} ${x.a}`).filter((t) => qTerms.includes(t)).length }))
      .filter((x) => x.hits > 0).sort((a, b) => b.hits - a.hits);
    if (ranked.length) return ranked.slice(0, 2).map((x) => x.x.a).join(' ');
  }
  const sentences = clean.split(/(?<=[.!?])\s+|\n+/).map((x) => x.trim()).filter(Boolean);
  const ranked = sentences.map((s) => ({ s, hits: meaningful(s).filter((t) => qTerms.includes(t)).length }))
    .filter((x) => x.hits > 0).sort((a, b) => b.hits - a.hits || a.s.length - b.s.length);
  return ranked.length ? ranked.slice(0, 3).map((x) => x.s).join(' ') : clean;
}

export type EvidenceDecision = {
  evidence: RetrievedChunk[];
  rejected: Array<{ id: string; score: number; reason: string }>;
};

/** Select at most three close-scoring, deduplicated, question-focused chunks. */
export function selectFocusedEvidence(candidates: RetrievedChunk[], query: string, maxChars = 6000): EvidenceDecision {
  const clean = candidates.filter((x) => normalize(x.text).length > 0);
  const q = meaningful(query);
  const qPhrase = phrase(query);
  const ranked = clean.map((x) => {
    const text = normalize(x.text);
    const terms = meaningful(text);
    const lexical = q.filter((t) => terms.includes(t)).length / Math.max(1, q.length);
    const exact = qPhrase.length > 8 && text.toLocaleLowerCase().includes(qPhrase) ? 1 : 0;
    const bm = bm25(q, text, clean);
    return { x, text, lexical, exact, bm };
  });
  const lexicalRank = [...ranked].sort((a, b) => b.lexical - a.lexical || b.x.score - a.x.score);
  const bmRank = [...ranked].sort((a, b) => b.bm - a.bm || b.x.score - a.x.score);
  const vectorRank = [...ranked].sort((a, b) => b.x.score - a.x.score);
  const combined = ranked.map((r) => ({
    ...r,
    fused: rrf(vectorRank.findIndex((x) => x.x.id === r.x.id)) + rrf(lexicalRank.findIndex((x) => x.x.id === r.x.id)) + rrf(bmRank.findIndex((x) => x.x.id === r.x.id)),
  })).sort((a, b) => (b.fused - a.fused) || (b.exact - a.exact) || (b.lexical - a.lexical) || (b.x.score - a.x.score));
  const best = combined[0]?.x.score ?? 0;
  const selected: RetrievedChunk[] = [];
  const seen = new Set<string>();
  let chars = 0;
  for (const item of combined) {
    const focused = answerFragment(item.text, query);
    const key = focused.toLocaleLowerCase().replace(/\s+/g, ' ').trim();
    // A high vector score is sufficient for a semantic paraphrase even when
    // its vocabulary differs; weaker candidates must also have lexical/BM25
    // support so tenant-wide neighbors cannot enter the prompt.
    const relevant = item.exact > 0 || item.lexical >= (q.length <= 2 ? 0.5 : 0.2) || item.bm > 0 || (selected.length === 0 && item.x.score >= 0.78);
    const close = item.x.score >= best - 0.14;
    if (!relevant) continue;
    if (!close) continue;
    if (seen.has(key)) continue;
    if (selected.length >= 3) continue;
    const remaining = maxChars - chars;
    if (remaining <= 80) break;
    const text = focused.length > remaining ? focused.slice(0, remaining).replace(/\s+\S*$/, '').trim() : focused;
    if (!text) continue;
    selected.push({ ...item.x, text });
    seen.add(key); chars += text.length;
  }
  const chosen = new Set(selected.map((x) => x.id));
  const rejected = clean.filter((x) => !chosen.has(x.id)).map((x) => {
    const item = combined.find((r) => r.x.id === x.id)!;
    const reason = !item ? 'invalid-empty-candidate' : item.x.score < best - 0.14 ? 'outside-score-band' : item.lexical === 0 && item.bm === 0 && item.exact === 0 ? 'no-question-term-or-phrase-match' : selected.length >= 3 ? 'evidence-cap' : 'duplicate-or-context-budget';
    return { id: x.id, score: Number(x.score.toFixed(4)), reason };
  });
  return { evidence: selected, rejected };
}

export function cleanAnswerScaffolding(answer: string) {
  const text = answer.replace(/\\n/g, ' ').replace(/```[a-z]*\s*/gi, '').replace(/```/g, '')
    .replace(/(^|\s)#{1,6}\s*/g, '$1')
    .replace(/\*{1,3}([^*]+)\*{1,3}/g, '$1')
    .replace(/(?:^|\s)Q:\s*.*?\s+A:\s*/gis, ' ')
    .replace(/(^|\s)[QA]:\s*/gi, '$1')
    .replace(/\b(?:FAQ|CONVERSATION|ESCALATION\s*\/\s*HUMAN CONTACT|POLICIES)\b\s*:?[\s-]*/gi, ' ')
    .replace(/\s+/g, ' ').trim();
  const seen = new Set<string>();
  return text.split(/(?<=[.!?])\s+/).filter((s) => { const k = s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim(); if (!k || seen.has(k)) return false; seen.add(k); return true; }).join(' ').trim();
}
