import type { RetrievedChunk } from '../domain/types.js';

export type GroundingCheck = {
  ok: boolean;
  reasons: string[];
};

type NumericClaim = { key: string; value: string; currency: string; qualifier: string };

/**
 * Normalize numeric claims for practical grounding. Currency is retained so a
 * supported pound amount cannot silently become an unsupported dollar amount;
 * omitted currency and harmless spacing/unit wording remain compatible.
 */
const numberTokens = (text: string): NumericClaim[] =>
  [...text.matchAll(/(£|AUD\s*\$|\$|€)?\s*(\d+(?:[.,]\d+)?)(?:\s*%|\s*(?:per|a)\s*(?:month|year|day|week))?/gi)]
    .map((match) => {
      const raw = match[0].toLowerCase();
      const currency = (match[1] ?? '').replace(/\s+/g, '').toLowerCase();
      const value = match[2]!.replace(/,/g, '');
      const qualifier = /%/.test(raw) ? '%' : /\b(?:per|a)\s*(month|year|day|week)\b/i.exec(raw)?.[1]?.toLowerCase() ?? '';
      return { key: `${currency}:${value}:${qualifier}`, value, currency, qualifier };
    });

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
export function validateGrounding(
  question: string,
  answer: string,
  evidence: RetrievedChunk[],
): GroundingCheck {
  if (!evidence.length) {
    return {
      ok: false,
      reasons: ['insufficient evidence'],
    };
  }

  const source = evidence.map((x) => x.text).join('\n');
  const reasons: string[] = [];

  // ------------------------------------------------------------
  // 1. Numeric claims must exist in tenant evidence.
  // ------------------------------------------------------------
  const answerNumbers = numberTokens(answer);
  const sourceNumbers = numberTokens(source);

  const unsupportedNumbers = answerNumbers.filter((claim) => {
    return !sourceNumbers.some(
      (supported) =>
        supported.value === claim.value &&
        (!claim.currency ||
          !supported.currency ||
          supported.currency === claim.currency) &&
        (!claim.qualifier ||
          !supported.qualifier ||
          supported.qualifier === claim.qualifier),
    );
  });

  if (unsupportedNumbers.length) {
    reasons.push('unsupported numeric claim');
  }

  // ------------------------------------------------------------
  // 2. UNKNOWN / unavailable evidence.
  // Only reject when the answer actually claims something
  // about the same topic.
  // ------------------------------------------------------------
  const unknownLines = source
    .split(/(?<=[.!?])\s+|\n/)
    .filter((line) =>
      /\bunknown\b|not available|not specified|cannot be verified/i.test(line),
    );

  const qWords = words(question);

  const answerProvidesFact =
    /\b(is|are|has|have|does|do|offers?|provides?|costs?|cost|available|open|free|serves?|supports?)\b/i.test(
      answer,
    ) || answerNumbers.length > 0;

  if (
    answerProvidesFact &&
    unknownLines.some(
      (line) => overlap(qWords, words(line)) >= 2,
    )
  ) {
    reasons.push('answer supplies a value where relevant evidence is unknown');
  }

  // ------------------------------------------------------------
  // 3. Contradiction detection.
  //
  // IMPORTANT:
  // Do NOT use generic "no/not/cannot" lines against every answer.
  // Only consider a contradiction when the evidence line shares
  // meaningful topic words with the question AND answer.
  // ------------------------------------------------------------
  const negativeLines = source
    .split(/(?<=[.!?])\s+|\n/)
    .filter((line) =>
      /\b(no|not|does not|do not|never|cannot|must not|without|unavailable)\b/i.test(
        line,
      ),
    );

  const answerWords = words(answer);

  const contradiction = negativeLines.some((line) => {
    const lineWords = words(line);

    const questionOverlap = overlap(qWords, lineWords);
    const answerOverlap = overlap(answerWords, lineWords);

    return questionOverlap >= 2 && answerOverlap >= 2;
  });

  if (contradiction) {
    reasons.push('possible contradiction with negative evidence');
  }

  // ------------------------------------------------------------
  // 4. Evidence qualifications must not be silently strengthened.
  // ------------------------------------------------------------
  const qualifiedLines = source
    .split(/(?<=[.!?])\s+|\n/)
    .filter((line) =>
      /\b(most|some|usually|typically|often|vary|varies|depends|depending on|subject to|may|can differ|approximately|around|starting from|up to|at least|about)\b/i.test(
        line,
      ),
    );

  const absoluteAnswer =
    /\b(all|every|always|never|exactly|guaranteed|guarantees|fixed)\b/i.test(
      answer,
    );

  if (
    absoluteAnswer &&
    qualifiedLines.some((line) => {
      const lineWords = words(line);
      return overlap(qWords, lineWords) >= 2;
    })
  ) {
    reasons.push('answer drops an evidence qualification');
  }

  return {
    ok: reasons.length === 0,
    reasons,
  };
}
export function groundingCorrection(reasons: string[]) {
  return `Grounding check failed (${reasons.join('; ')}). Re-read TENANT KNOWLEDGE as authoritative. Correct or remove every unsupported claim, number, price, availability statement, guarantee, or policy. If the requested fact is absent or marked UNKNOWN, say it is not available in the current tenant knowledge. Return only the corrected customer-facing answer.`;
}
