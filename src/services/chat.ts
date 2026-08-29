import crypto from 'node:crypto';
import type { Repository } from '../repositories/repository.js';
import type { EmbeddingProvider } from '../providers/embedding.js';
import type { VectorProvider } from '../providers/vector.js';
import type { LlmProvider, LlmInput, ChatDebug } from '../providers/llm.js';
import type { Client, ChatMessage, RetrievedChunk } from '../domain/types.js';
import { AppError } from '../lib/errors.js';
import { groundingCorrection, validateGrounding } from './grounding.js';

const MAX_CANDIDATES = 40;
const MAX_EVIDENCE = 6;

function focusEvidenceText(text: string, query: string) {
  const normalized = text.replace(/\s+/g, ' ').trim();
  const terms = [...new Set(query.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? [])]
    .filter((term) => !new Set(['the', 'and', 'for', 'how', 'what', 'when', 'where', 'does', 'can', 'you', 'my', 'do', 'i', 'a', 'an', 'is', 'are', 'to', 'of', 'or', 'in', 'on', 'with', 'your', 'me', 'it', 'this', 'that']).has(term));
  const faq = [...normalized.matchAll(/\*{0,2}Q:\s*(.*?)\*{0,2}\s+\*{0,2}A:\s*(.*?)(?=\s+\*{0,2}Q:|$)/gis)]
    .map((match) => ({ question: match[1]!.trim(), answer: match[2]!.trim() }))
    .filter((item) => item.answer);
  if (faq.length) {
    const ranked = faq.map((item) => {
      const haystack = `${item.question} ${item.answer}`.toLowerCase();
      return { item, hits: terms.filter((term) => haystack.includes(term)).length };
    }).sort((a, b) => b.hits - a.hits);
    if (ranked[0]!.hits > 0) return ranked.slice(0, 2).filter((x) => x.hits > 0).map((x) => x.item.answer).join(' ');
  }
  if (!terms.length) return normalized;
  const sentences = normalized.split(/(?<=[.!?])\s+|\n+/).map((x) => x.trim()).filter(Boolean);
  const ranked = sentences.map((sentence) => ({ sentence, hits: terms.filter((term) => sentence.toLowerCase().includes(term)).length }))
    .filter((x) => x.hits > 0).sort((a, b) => b.hits - a.hits || a.sentence.length - b.sentence.length);
  return ranked.length ? ranked.slice(0, 3).map((x) => x.sentence).join(' ') : normalized;
}

const DEBUG_QUESTION = 'How much does a PureGym day pass cost, and how long is it valid?';
const DEBUG_GOUSTO_QUESTION = 'how long do gousto ingredients stay fresh?';
const safePreview = (value: string, limit = 800) => value
  .replace(/(?:sk|pk|api[_ -]?key|bearer)[=: ]+[A-Za-z0-9._-]+/gi, '[REDACTED_SECRET]')
  .replace(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi, '[REDACTED_EMAIL]')
  .replace(/(?:\+?\d[\d ()-]{7,}\d)/g, '[REDACTED_PHONE]')
  .slice(0, limit);


type Prepared = { debug?: ChatDebug; client: Client; sid: string; conversationId: string; history: ChatMessage[]; evidence: RetrievedChunk[]; candidates: RetrievedChunk[]; started: number; question: string; fallbackReason?: string };

function professionalFallback(client: Client) {
  const company = client.config.companyName?.trim() || client.name;
  const contact = client.config.teamEmail?.trim();
  return contact
    ? `I don’t have verified information about that in the information provided by ${company}. For the most accurate answer, please contact ${company} directly at ${contact}.`
    : `I don’t have verified information about that in the information provided by ${company}. Please contact ${company} directly through its official support channel for the most accurate answer.`;
}

function isGenericFallback(answer: string) {
  return /\b(?:sorry|apolog(?:y|ize|ise)|don['’]?t|do not)\b[\s,]*(?:have|know|possess)|\b(?:no|not)\s+(?:verified\s+)?information\b/i.test(answer);
}

/** Remove ingestion/prompt scaffolding before any answer is persisted or streamed. */
function cleanCustomerAnswer(answer: string) {
  let text = answer
    .replace(/\\n/g, ' ')
    .replace(/```[a-z]*\s*/gi, '')
    .replace(/```/g, '')
    .replace(/\*{1,3}([^*]+)\*{1,3}/g, '$1')
    // FAQ-shaped generations repeat the source question before the answer;
    // keep only the customer-facing answer portion.
    .replace(/(?:^|\s)Q:\s*.*?\s+A:\s*/gis, ' ')
    .replace(/(^|\s)Q:\s*/gi, '$1')
    .replace(/(^|\s)A:\s*/gi, '$1')
    .replace(/\b(?:CONVERSATION|ESCALATION\s*\/\s*HUMAN CONTACT|POLICIES)\b\s*:?[\s-]*/gi, '');
  // Also handle providers that place markdown markers directly around the FAQ
  // labels or omit the separating whitespace.
  text = text.replace(/^\s*\**Q:\s*.*?\**\s*A:\s*/is, '').replace(/^\s*\**A:\s*/i, '');
  text = text.replace(/(?:^|\s)#{1,6}\s*(?:FAQ|CONVERSATION|ESCALATION\s*\/\s*HUMAN CONTACT|POLICIES)\s*:?[\s-]*/gi, ' ');
  text = text.split(/\r?\n/)
    .filter((line) => !/^\s*(?:#{1,6}\s*)?(?:FAQ|CONVERSATION|ESCALATION\s*\/\s*HUMAN CONTACT|POLICIES)\s*:?[\s-]*$/i.test(line))
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();
  const sentences = text.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean);
  const seen = new Set<string>();
  return sentences.filter((sentence) => {
    const key = sentence.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  }).join(' ').trim();
}

// Retrieval evidence alone is not enough: an answer must also address a
// meaningful part of the current question. This prevents a relevant-to-tenant
// but irrelevant chunk (or a stale session topic) from being presented as the
// answer to a new question.
function questionEvidenceAligned(question: string, answer: string, evidence: RetrievedChunk[]) {
  const stop = new Set(['the', 'and', 'for', 'how', 'what', 'when', 'where', 'does', 'can', 'you', 'my', 'do', 'i', 'a', 'an', 'is', 'are', 'to', 'of', 'or', 'in', 'on', 'with', 'your', 'me', 'it', 'this', 'that', 'gousto', 'stay', 'long', 'much', 'late']);
  const terms = [...new Set(question.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? [])].filter((term) => !stop.has(term));
  if (!terms.length || !evidence.length) return true;
  const answerWords = new Set(answer.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []);
  const evidenceWords = new Set(evidence.flatMap((chunk) => chunk.text.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []));
  // A concise answer may legitimately omit the question's exact nouns (for
  // example, “Call 020 3011 1002” answers a Customer Care question). Require
  // overlap with the selected authoritative evidence as well as either a
  // question-term hit or a sufficiently direct evidence match.
  const answerEvidenceOverlap = [...answerWords].filter((word) => evidenceWords.has(word)).length;
  return terms.some((term) => answerWords.has(term) && evidenceWords.has(term)) || answerEvidenceOverlap >= 2;
}

/** Tenant-scoped conversation orchestration with bounded recall and conservative grounding. */
export class ChatService {
  constructor(private repo: Repository, private embedding: EmbeddingProvider, private vector: VectorProvider, private llm: LlmProvider) {}

  private isKnowledgeQuestion(message: string) {
    const text = message.trim().toLowerCase();
    if (!text) return false;
    if (/^(hi|hello|hey|thanks|thank you|okay|ok|good morning|good afternoon|good evening|who are you|can you help me)[!.? ]*$/i.test(text)) return false;
    // Treat ordinary interrogative requests as knowledge intents, rather than
    // maintaining a brittle list of business nouns. This preserves retrieval
    // for forms such as “how does…”, “how many…”, “which days…”, and “what if…”.
    // Clearly general/conversational prompts remain provider-only.
    if (/\b(?:meaning of life|tell me a joke|write a poem|creative writing|weather today)\b/i.test(text)) return false;
    return /^(?:who|what|when|where|why|how|which|can|could|do|does|is|are|will|may|should|please)\b/i.test(text)
      || /\?$/.test(text);
  }

  private retrievalQuery(message: string, history: ChatMessage[]) {
    const lower = message.toLowerCase();
    // Standalone factual questions must be embedded on their own. Short
    // questions are common in chat, but length alone does not make a request a
    // follow-up; mixing unrelated prior answers into the vector query can push
    // the tenant's relevant chunk out of the semantic result. Only include
    // history when the wording explicitly depends on earlier conversation.
    const needsContext = /\b(?:it|that|this|they|them|there|those|he|she|previous|above|earlier|same|again|just)\b|\bwhat about\b|\bwhat did you\b/i.test(lower);
    if (!needsContext || !history.length) return message;
    return [...history.slice(-6).map((x) => `${x.role}: ${x.content}`), `user: ${message}`].join('\n');
  }

  private selectEvidence(candidates: RetrievedChunk[], query: string) {
    const terms = new Set(query.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []);
    const ranked = [...candidates]
      .filter((x) => x.text.trim().length > 0)
      .map((x) => ({ x, lexical: [...terms].filter((t) => x.text.toLowerCase().includes(t)).length }))
      // Semantic similarity is primary; lexical overlap only breaks near ties.
      .sort((a, b) => (Math.abs(b.x.score - a.x.score) > 0.08 ? b.x.score - a.x.score : (b.lexical - a.lexical) || (b.x.score - a.x.score)));
    const best = ranked[0]?.x.score ?? 0;
    const seen = new Set<string>();
    return ranked
      // Do not send a long tail of merely tenant-related chunks to the model.
      // Keep close-scoring candidates only, while allowing a modest multi-part
      // question set and retaining the 40-candidate retrieval ceiling.
      .filter(({ x }) => x.score >= best - 0.12)
      .slice(0, 3)
      .map(({ x }) => ({ ...x, text: focusEvidenceText(x.text, query) }))
      .filter((x) => {
        const key = x.text.toLowerCase().replace(/\s+/g, ' ').trim();
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
  }

  /** Extract only directly supported sentences when a provider refuses despite evidence. */
  private evidenceBackedAnswer(question: string, evidence: RetrievedChunk[]) {
    const stop = new Set(['the', 'and', 'for', 'how', 'what', 'when', 'where', 'does', 'can', 'you', 'my', 'do', 'i', 'a', 'an', 'is', 'are', 'to', 'of', 'or', 'in', 'on', 'with', 'your', 'me', 'it', 'this', 'that', 'gousto']);
    const terms = [...new Set(question.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? [])].filter((x) => !stop.has(x));
    if (!terms.length) return null;
    const sentences = evidence.flatMap((chunk) => chunk.text.split(/(?<=[.!?])\s+|\n+/).map((text) => text.trim()).filter(Boolean));
    const ranked = sentences.map((text) => {
      const textWords = new Set(text.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []);
      return { text, hits: terms.filter((term) => textWords.has(term)).length };
    })
      .filter((x) => x.hits > 0).sort((a, b) => b.hits - a.hits || a.text.length - b.text.length);
    const distinctive = terms.filter((term) => !['how', 'long', 'much', 'late', 'stay', 'fresh'].includes(term));
    const supported = ranked.filter((x) => x.hits >= 2 || (distinctive.length > 0 && x.hits >= 1 && distinctive.some((term) => new Set(x.text.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []).has(term))));
    if (!supported.length) return null;
    return supported.slice(0, 2).map((x) => x.text).join(' ');
  }

  private isSummaryRequest(message: string) {
    return /\b(?:summari[sz]e|summary|recap|what did we discuss|what have we discussed|summarise this chat)\b/i.test(message);
  }

  private hasSufficientSemanticEvidence(evidence: RetrievedChunk[], threshold: number) {
    if (!evidence.length) return false;
    // The configured threshold is intentionally permissive for candidate
    // recall. A stronger, generic acceptance floor decides whether the first
    // query is trustworthy enough to skip the one bounded recovery pass.
    const floor = Math.max(0.55, threshold + 0.2);
    return evidence[0]!.score >= floor;
  }

  private async prepare(clientId: string, message: string, sessionId?: string): Promise<Prepared> {
    const client = await this.repo.getClient(clientId);
    if (!client?.enabled) throw new AppError(403, 'TENANT_DISABLED', 'Tenant is disabled');
    const started = Date.now();
    const sid = sessionId ?? crypto.randomUUID();
    const debugEnabled = process.env.ORBIT_TEMP_DEBUG_LLM_INPUT === '1'
      || (process.env.ORBIT_TEMP_DEBUG_PUREGYM === '1' && message === DEBUG_QUESTION && process.env.ORBIT_TEMP_DEBUG_CLIENT_ID === clientId)
      || (process.env.ORBIT_TEMP_DEBUG_GOUSTO === '1' && message === DEBUG_GOUSTO_QUESTION);
    const debug: ChatDebug | undefined = debugEnabled
      ? (stage, data) => console.info(JSON.stringify({ event: 'chat.temp-debug', traceId: sid, stage, ...data }))
      : undefined;
    debug?.('USER_INPUT', { chars: message.length, tokensApprox: Math.ceil(message.length / 4), questionPreview: safePreview(message, 240) });
    debug?.('TENANT', { clientId: client.id, clientName: client.name, tenantIsolation: 'repository/vector calls use resolved clientId' });
    const conversationId = await this.repo.createConversation(clientId, sid);
    const prior = (await this.repo.messages(conversationId)).slice(-12);
    await this.repo.addMessage(conversationId, 'user', message);
    const history = prior.slice(-12);
    debug?.('HISTORY_STATUS', { loaded: true, sentToLlm: false, historyCount: 0, historyChars: 0, historyTokensApprox: 0, loadedMessageCount: history.length, loadedChars: history.reduce((total, item) => total + item.content.length, 0) });
    const summaryRequest = this.isSummaryRequest(message);
    const query = this.retrievalQuery(message, history);
    debug?.('RAG_QUERY', { chars: query.length, tokensApprox: Math.ceil(query.length / 4), historyAdded: query !== message, historyMessageCount: history.length, boundedHistoryCount: Math.min(history.length, 6), queryPreview: safePreview(query, 240) });
    debug?.('RETRIEVAL_QUERY', { query: safePreview(query, 2400), queryChars: query.length, queryTokensApprox: Math.ceil(query.length / 4), historyAdded: query !== message, historyMessageCount: history.length, boundedHistoryCount: Math.min(history.length, 6) });
    const k = Math.min(client.config.topK ?? MAX_CANDIDATES, MAX_CANDIDATES);
    const threshold = client.config.minSimilarity ?? 0.05;
    let candidates: RetrievedChunk[] = [];
    let evidenceQuery = query;
    let reformulatedQueries: string[] = [];
    let q: number[] | undefined;
    const knowledgeQuestion = !summaryRequest && this.isKnowledgeQuestion(message);
    if (knowledgeQuestion) {
      [q] = await this.embedding.embed([query]);
      candidates = await this.vector.search(clientId, q!, k, threshold);
      // Bounded semantic recall recovery for interrogative phrasing: remove
      // grammatical scaffolding and retry the same tenant-scoped index. This
      // is generic and bounded; it does not inject tenant facts or bypass the
      // vector provider.
      if (!candidates.length) {
        const contentQuery = message
          .replace(/\b(?:who|what|when|where|why|how|which|can|could|do|does|is|are|will|may|should|please)\b/gi, ' ')
          .replace(/[?!.]/g, ' ').replace(/\s+/g, ' ').trim();
        if (contentQuery && contentQuery !== message) {
          const [contentQ] = await this.embedding.embed([contentQuery]);
          candidates = await this.vector.search(clientId, contentQ!, k, threshold);
          debug?.('RETRIEVAL_RESULTS', { phase: 'content-query', query: safePreview(contentQuery, 240), candidateCount: candidates.length });
        }
      }
      debug?.('RETRIEVAL_RESULTS', { phase: 'initial', candidateCount: candidates.length, results: candidates.slice(0, MAX_CANDIDATES).map((x) => ({ chunkId: x.id, score: Number(x.score.toFixed(4)), excerpt: safePreview(x.text, 360) })) });
      console.info(JSON.stringify({ event: 'retrieval.initial', clientId, query: query.toLowerCase().replace(/\\s+/g, ' ').trim().slice(0, 240), embeddingDimension: q?.length ?? 0, topK: k, minSimilarity: client.config.minSimilarity ?? 0.05, candidateCount: candidates.length, scores: candidates.slice(0, 12).map((x) => Number(x.score.toFixed(4))), chunkIds: candidates.slice(0, 12).map((x) => x.id) }));
      if (!candidates.length && query !== message) {
        const [messageQ] = await this.embedding.embed([message]);
        candidates = await this.vector.search(clientId, messageQ!, k, threshold);
      }
    }
    const initialEvidence = this.selectEvidence(candidates, query);
    // Common words are not enough to establish that a candidate answers the
    // question. Require two meaningful terms, or one distinctive term, before
    // suppressing the one bounded recovery search.
    // A retrieval miss or weak semantic hit means only that this query failed;
    // it does not prove that the tenant lacks the information. Make one provider
    // call that may return up to three concise alternatives, then run each
    // through the same tenant-scoped vector provider. Never loop. Do not use
    // exact-word overlap as the gate: semantically equivalent tenant wording
    // is valid evidence even when vocabulary differs.
    if (knowledgeQuestion && !this.hasSufficientSemanticEvidence(initialEvidence, threshold) && this.llm.reformulate) {
      const reformulated = await this.llm.reformulate({ question: message, prompt: client.prompt, context: initialEvidence, config: client.config, history });
      reformulatedQueries = (reformulated ?? '').split(/\r?\n|\s*\|\s*/).map((x) => x.trim().replace(/^[-*\d.)]+\s*/, '')).filter((x) => x && x.toUpperCase() !== 'NONE').slice(0, 3);
      const retryResults: RetrievedChunk[] = [];
      for (const alternative of reformulatedQueries) {
        const [alternativeQ] = await this.embedding.embed([alternative]);
        retryResults.push(...await this.vector.search(clientId, alternativeQ!, k, threshold));
      }
      if (retryResults.length) {
        candidates = [...candidates, ...retryResults].filter((chunk, index, all) => all.findIndex((x) => x.id === chunk.id) === index);
        evidenceQuery = reformulatedQueries[0] ?? query;
      }
    }
    const evidence = this.selectEvidence(candidates, evidenceQuery);
    debug?.('RAG_RETRIEVAL', { retrievedChunkCount: candidates.length, chunks: candidates.map((x) => ({ id: x.id, source: safePreview(x.source, 240), score: Number(x.score.toFixed(4)), chars: x.text.length, tokensApprox: Math.ceil(x.text.length / 4) })) });
    debug?.('RAG_CHUNKS', { selected: evidence.map((x) => ({ id: x.id, score: Number(x.score.toFixed(4)), source: safePreview(x.source, 240), chars: x.text.length, tokensApprox: Math.ceil(x.text.length / 4), preview: safePreview(x.text, 500) })) });
    debug?.('SELECTED_CHUNKS', { count: evidence.length, chars: evidence.reduce((total, x) => total + x.text.length, 0), tokensApprox: Math.ceil(evidence.reduce((total, x) => total + x.text.length, 0) / 4), ids: evidence.map((x) => x.id) });
    debug?.('RETRIEVAL_RESULTS', { phase: 'final', candidateCount: candidates.length, results: candidates.slice(0, MAX_CANDIDATES).map((x) => ({ chunkId: x.id, score: Number(x.score.toFixed(4)), excerpt: safePreview(x.text, 360) })) });
    debug?.('SELECTED_EVIDENCE', { selected: evidence.map((x) => ({ chunkId: x.id, score: Number(x.score.toFixed(4)), excerpt: safePreview(x.text, 500) })), dayPassInformationPresent: evidence.some((x) => /day pass|1\s*(?:to|-|–)\s*30|30 days/i.test(x.text)) });
    console.info(JSON.stringify({ event: 'retrieval.final', clientId, query: query.toLowerCase().replace(/\\s+/g, ' ').trim().slice(0, 240), reformulatedQueries: reformulatedQueries.map((x) => x.slice(0, 240)), candidateCount: candidates.length, evidenceCount: evidence.length, evidenceIds: evidence.map((x) => x.id), evidenceScores: evidence.map((x) => Number(x.score.toFixed(4))), fallback: knowledgeQuestion && !evidence.length ? 'no-verified-evidence-after-bounded-reformulation' : 'none' }));
    return { debug, client, sid, conversationId, history, candidates, evidence, started, question: message, fallbackReason: knowledgeQuestion && !evidence.length ? 'no-verified-evidence-after-bounded-reformulation' : undefined };
  }

  private input(p: Prepared, prompt = p.client.prompt): LlmInput {
    return { question: p.question, prompt, context: p.evidence, config: p.client.config, history: p.history, debug: p.debug };
  }

  private async groundedAnswer(p: Prepared) {
    const knowledgeQuestion = !this.isSummaryRequest(p.question) && this.isKnowledgeQuestion(p.question);
    if (knowledgeQuestion && !p.evidence.length) {
      p.debug?.('FALLBACK', { called: true, reason: p.fallbackReason ?? 'no-evidence' });
      return professionalFallback(p.client);
    }
    let answer = cleanCustomerAnswer(await this.llm.answer(this.input(p)));
    if (!knowledgeQuestion) return isGenericFallback(answer) ? professionalFallback(p.client) : answer;
    let check = validateGrounding(p.question, answer, p.evidence);
    p.debug?.('VALIDATION', { attempt: 1, grounded: check.ok && !isGenericFallback(answer), reasons: check.reasons, answer: safePreview(answer) });
    if (!check.ok || isGenericFallback(answer) || !questionEvidenceAligned(p.question, answer, p.evidence)) {
      const reasons = check.reasons.length ? check.reasons : ['generic fallback despite tenant evidence'];
      p.debug?.('CORRECTION', { triggered: true, reason: reasons });
      answer = cleanCustomerAnswer(await this.llm.answer(this.input(p, `${p.client.prompt}\n\n${groundingCorrection(reasons)}`)));
      check = validateGrounding(p.question, answer, p.evidence);
      p.debug?.('VALIDATION', { attempt: 2, grounded: check.ok && !isGenericFallback(answer), reasons: check.reasons, answer: safePreview(answer) });
      if (!check.ok || isGenericFallback(answer) || !questionEvidenceAligned(p.question, answer, p.evidence)) {
        p.debug?.('FALLBACK', { called: true, reason: check.reasons.length ? check.reasons : ['generic fallback after correction'] });
        return this.evidenceBackedAnswer(p.question, p.evidence) ?? professionalFallback(p.client);
      }
    }
    return answer;
  }

  private result(p: Prepared, answer: string) {
    return { sessionId: p.sid, conversationId: p.conversationId, answer, sources: p.evidence.map((x) => ({ chunkId: x.id, source: x.source, score: Math.round(x.score * 1000) / 1000 })), leadCollection: p.client.config.collectLead ? { enabled: true as const, fields: ['name', ...(p.client.config.leadFields ?? ['email', 'requirement']).filter((field) => field !== 'name') ] } : undefined };
  }

  async chat(clientId: string, message: string, sessionId?: string) {
    const p = await this.prepare(clientId, message, sessionId);
    const answer = await this.groundedAnswer(p);
    await this.repo.addMessage(p.conversationId, 'assistant', answer);
    await this.repo.usage(clientId, 'chat', message.length + answer.length, Date.now() - p.started);
    p.debug?.('FINAL_RESPONSE', { chars: answer.length, preview: safePreview(answer, 2400) });
    return this.result(p, answer);
  }

  async stream(clientId: string, message: string, sessionId: string | undefined, onToken: (token: string) => void) {
    const p = await this.prepare(clientId, message, sessionId);
    let answer: string;
    const buffered: string[] = [];
    const knowledgeQuestion = !this.isSummaryRequest(message) && this.isKnowledgeQuestion(message);
    if (knowledgeQuestion && !p.evidence.length) {
      p.debug?.('FALLBACK', { called: true, reason: p.fallbackReason ?? 'no-evidence' });
      answer = professionalFallback(p.client);
    }
    else {
      answer = cleanCustomerAnswer(await this.llm.streamAnswer(this.input(p), (token) => buffered.push(token)));
      if (!knowledgeQuestion && isGenericFallback(answer)) answer = professionalFallback(p.client);
      if (knowledgeQuestion) {
        let check = validateGrounding(message, answer, p.evidence);
        p.debug?.('VALIDATION', { attempt: 1, grounded: check.ok && !isGenericFallback(answer), reasons: check.reasons, answer: safePreview(answer) });
        if (!check.ok || isGenericFallback(answer) || !questionEvidenceAligned(p.question, answer, p.evidence)) {
          p.debug?.('CORRECTION', { triggered: true, reason: check.reasons.length ? check.reasons : ['generic fallback despite tenant evidence'] });
          answer = cleanCustomerAnswer(await this.llm.answer(this.input(p, `${p.client.prompt}\n\n${groundingCorrection(check.reasons.length ? check.reasons : ['generic fallback despite tenant evidence'])}`)));
          check = validateGrounding(message, answer, p.evidence);
          p.debug?.('VALIDATION', { attempt: 2, grounded: check.ok && !isGenericFallback(answer), reasons: check.reasons, answer: safePreview(answer) });
          if (!check.ok || isGenericFallback(answer) || !questionEvidenceAligned(p.question, answer, p.evidence)) { p.debug?.('FALLBACK', { called: true, reason: check.reasons.length ? check.reasons : ['generic fallback after correction'] }); answer = this.evidenceBackedAnswer(message, p.evidence) ?? professionalFallback(p.client); }
        }
      }
    }
    // Emit only the accepted answer. This preserves real provider streaming semantics
    // without showing a claim that the final grounding check rejects.
    p.debug?.('FINAL_STREAM', { text: safePreview(answer, 2400), chars: answer.length });
    for (const token of answer.match(/\S+\s*/g) ?? []) onToken(token);
    await this.repo.addMessage(p.conversationId, 'assistant', answer);
    await this.repo.usage(clientId, 'chat', message.length + answer.length, Date.now() - p.started);
    p.debug?.('FINAL_RESPONSE', { chars: answer.length, preview: safePreview(answer, 2400) });
    return this.result(p, answer);
  }
}
