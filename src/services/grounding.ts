import type { RetrievedChunk } from '../domain/types.js';

export type GroundingCheck = { ok: boolean; reasons: string[] };

const numberTokens = (text: string) =>
  (text.match(/(?:£|AUD\s*\$|\$|€|\b)\d+(?:[.,]\d+)?(?:\s*%|\s*(?:per|a)\s*(?:month|year|day|week))?/gi) ?? [])
    .map((x) => x.toLowerCase().replace(/\s+/g, '').replace(/,/g, ''));

const words = (text: string) =>
  new Set((text.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []).filter((x) => !['what', 'does', 'have', 'with', 'from', 'that', 'this', 'about', 'are', 'the'].includes(x)));

function overlap(a: Set<string>, b: Set<string>) {
  let n = 0;
  for (const x of a) if (b.has(x)) n++;
  return n;
}

/**
 * Conservative, evidence-based checks for the most dangerous grounding failures.
 * It deliberately avoids trying to judge prose style or paraphrase quality.
 */
export function validateGrounding(question: string, answer: string, evidence: RetrievedChunk[]): GroundingCheck {
  if (!evidence.length) return { ok: true, reasons: [] };
  const source = evidence.map((x) => x.text).join('\n');
  const reasons: string[] = [];
  const answerNumbers = numberTokens(answer);
  const sourceNumbers = new Set(numberTokens(source));
  const unsupported = answerNumbers.filter((x) => !sourceNumbers.has(x));
  if (unsupported.length) reasons.push('unsupported numeric claim');

  const qWords = words(question);
  const unknownLines = source.split(/(?<=[.!?])\s+|\n/).filter((x) => /\bunknown\b|not available|not specified|cannot be verified/i.test(x));
  if (unknownLines.some((line) => overlap(qWords, words(line)) >= 1) && answerNumbers.length && !/\b(unknown|not available|not specified|cannot verify|check the official|support)\b/i.test(answer)) {
    reasons.push('answer supplies a value where relevant evidence is unknown');
  }

  const negativeLines = source.split(/(?<=[.!?])\s+|\n/).filter((x) => /\b(no|not|does not|do not|never|cannot|must not|without)\b/i.test(x));
  const qualifiedLines = source.split(/(?<=[.!?])\s+|\n/).filter((x) => /\b(most|some|vary|varies|depends|subject to|may|can differ)\b/i.test(x));
  const affirmative = /\b(is|are|does|do|offers?|provides?|guarantees?|available|included|open|free)\b/i.test(answer) && !/\b(not|no|cannot|unavailable|unknown|should not)\b/i.test(answer);
  if (affirmative && negativeLines.some((line) => overlap(qWords, words(line)) >= 1)) reasons.push('possible contradiction with negative evidence');
  if (/\b(all|every|always|never)\b/i.test(answer) && qualifiedLines.some((line) => overlap(qWords, words(line)) >= 1)) reasons.push('answer drops an evidence qualification');

  return { ok: reasons.length === 0, reasons };
}

export function groundingCorrection(reasons: string[]) {
  return `Grounding check failed (${reasons.join('; ')}). Re-read TENANT KNOWLEDGE as authoritative. Correct or remove every unsupported claim, number, price, availability statement, guarantee, or policy. If the requested fact is absent or marked UNKNOWN, say it is not available in the current tenant knowledge. Return only the corrected customer-facing answer.`;
}
