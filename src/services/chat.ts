import crypto from 'node:crypto';
import type { Repository } from '../repositories/repository.js';
import type { EmbeddingProvider } from '../providers/embedding.js';
import type { VectorProvider } from '../providers/vector.js';
import type { LlmProvider, LlmInput } from '../providers/llm.js';
import type { Client, ChatMessage, RetrievedChunk } from '../domain/types.js';
import { AppError } from '../lib/errors.js';
import { groundingCorrection, validateGrounding } from './grounding.js';

const MAX_CANDIDATES = 40;
const MAX_EVIDENCE = 12;

type Prepared = { client: Client; sid: string; conversationId: string; history: ChatMessage[]; evidence: RetrievedChunk[]; candidates: RetrievedChunk[]; started: number; question: string; fallbackReason?: string };

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

/** Tenant-scoped conversation orchestration with bounded recall and conservative grounding. */
export class ChatService {
  constructor(private repo: Repository, private embedding: EmbeddingProvider, private vector: VectorProvider, private llm: LlmProvider) {}

  private isKnowledgeQuestion(message: string) {
    const text = message.trim().toLowerCase();
    if (/^(hi|hello|hey|thanks|thank you|okay|ok|good morning|good afternoon|good evening|who are you|can you help me)[!.? ]*$/i.test(text)) return false;
    return /\b(price|pricing|cost|service|services|offer|provide|policy|policies|hours|open|close|warranty|refund|return|book|booking|schedule|contact|phone|email|address|location|available|availability|emergency|company|product|plan|support|delivery|shipping|membership|rate|fee|how much|where|when|what do you|does .* have|can .* bring|minimum age)\b/i.test(text);
  }

  private retrievalQuery(message: string, history: ChatMessage[]) {
    const lower = message.toLowerCase();
    const needsContext = message.trim().split(/\s+/).length <= 8 || /\b(it|that|this|they|them|there|those|what|how much|summarize)\b/i.test(lower);
    if (!needsContext || !history.length) return message;
    return [...history.slice(-6).map((x) => `${x.role}: ${x.content}`), `user: ${message}`].join('\n');
  }

  private selectEvidence(candidates: RetrievedChunk[], query: string) {
    const terms = new Set(query.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []);
    return [...candidates]
      .map((x) => ({ x, lexical: [...terms].filter((t) => x.text.toLowerCase().includes(t)).length }))
      .sort((a, b) => (b.lexical - a.lexical) || (b.x.score - a.x.score))
      .slice(0, MAX_EVIDENCE)
      .map((x) => x.x);
  }

  private async prepare(clientId: string, message: string, sessionId?: string): Promise<Prepared> {
    const client = await this.repo.getClient(clientId);
    if (!client?.enabled) throw new AppError(403, 'TENANT_DISABLED', 'Tenant is disabled');
    const started = Date.now();
    const sid = sessionId ?? crypto.randomUUID();
    const conversationId = await this.repo.createConversation(clientId, sid);
    const prior = (await this.repo.messages(conversationId)).slice(-12);
    await this.repo.addMessage(conversationId, 'user', message);
    const history = prior.slice(-12);
    const query = this.retrievalQuery(message, history);
    const k = Math.min(client.config.topK ?? MAX_CANDIDATES, MAX_CANDIDATES);
    const threshold = client.config.minSimilarity ?? 0.05;
    const [q] = await this.embedding.embed([query]);
    let candidates = await this.vector.search(clientId, q!, k, threshold);
    let evidenceQuery = query;
    let reformulatedQueries: string[] = [];
    console.info(JSON.stringify({ event: 'retrieval.initial', clientId, query: query.toLowerCase().replace(/\\s+/g, ' ').trim().slice(0, 240), embeddingDimension: q?.length ?? 0, topK: k, minSimilarity: client.config.minSimilarity ?? 0.05, candidateCount: candidates.length, scores: candidates.slice(0, 12).map((x) => Number(x.score.toFixed(4))), chunkIds: candidates.slice(0, 12).map((x) => x.id) }));
    if (!candidates.length && query !== message) {
      const [messageQ] = await this.embedding.embed([message]);
      candidates = await this.vector.search(clientId, messageQ!, k, threshold);
    }
    const knowledgeQuestion = this.isKnowledgeQuestion(message);
    const initialEvidence = this.selectEvidence(candidates, query);
    // Common words are not enough to establish that a candidate answers the
    // question. Require two meaningful terms, or one distinctive term, before
    // suppressing the one bounded recovery search.
    const stopTerms = new Set(['about', 'after', 'also', 'are', 'can', 'does', 'from', 'have', 'help', 'how', 'much', 'the', 'their', 'there', 'these', 'this', 'what', 'when', 'where', 'which', 'with', 'you', 'your']);
    const queryTerms = new Set((message.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []).filter((term) => !stopTerms.has(term)));
    const anchorCounts = initialEvidence.map((chunk) => [...queryTerms].filter((term) => chunk.text.toLowerCase().includes(term)).length);
    const hasSemanticAnchor = anchorCounts.some((count) => count >= 2 || (count === 1 && [...queryTerms].some((term) => term.length >= 8)));
    // A retrieval miss or irrelevant hit means only that this query failed. It
    // does not prove that the tenant lacks the information. Make one provider
    // call that may return up to three concise alternatives, then run each
    // through the same tenant-scoped vector provider. Never loop.
    if (knowledgeQuestion && !hasSemanticAnchor && this.llm.reformulate) {
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
    console.info(JSON.stringify({ event: 'retrieval.final', clientId, query: query.toLowerCase().replace(/\\s+/g, ' ').trim().slice(0, 240), reformulatedQueries: reformulatedQueries.map((x) => x.slice(0, 240)), candidateCount: candidates.length, evidenceCount: evidence.length, evidenceIds: evidence.map((x) => x.id), evidenceScores: evidence.map((x) => Number(x.score.toFixed(4))), fallback: knowledgeQuestion && !evidence.length ? 'no-verified-evidence-after-bounded-reformulation' : 'none' }));
    return { client, sid, conversationId, history, candidates, evidence, started, question: message, fallbackReason: knowledgeQuestion && !evidence.length ? 'no-verified-evidence-after-bounded-reformulation' : undefined };
  }

  private input(p: Prepared, prompt = p.client.prompt): LlmInput {
    return { question: p.question, prompt, context: p.evidence, config: p.client.config, history: p.history };
  }

  private async groundedAnswer(p: Prepared) {
    const knowledgeQuestion = this.isKnowledgeQuestion(p.question);
    if (knowledgeQuestion && !p.evidence.length) return professionalFallback(p.client);
    let answer = await this.llm.answer(this.input(p));
    if (!knowledgeQuestion) return isGenericFallback(answer) ? professionalFallback(p.client) : answer;
    let check = validateGrounding(p.question, answer, p.evidence);
    if (!check.ok || isGenericFallback(answer)) {
      answer = await this.llm.answer(this.input(p, `${p.client.prompt}\n\n${groundingCorrection(check.reasons)}`));
      check = validateGrounding(p.question, answer, p.evidence);
      if (!check.ok || isGenericFallback(answer)) return professionalFallback(p.client);
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
    return this.result(p, answer);
  }

  async stream(clientId: string, message: string, sessionId: string | undefined, onToken: (token: string) => void) {
    const p = await this.prepare(clientId, message, sessionId);
    let answer: string;
    const buffered: string[] = [];
    if (this.isKnowledgeQuestion(message) && !p.evidence.length) answer = professionalFallback(p.client);
    else {
      answer = await this.llm.streamAnswer(this.input(p), (token) => buffered.push(token));
      if (!this.isKnowledgeQuestion(message) && isGenericFallback(answer)) answer = professionalFallback(p.client);
      if (this.isKnowledgeQuestion(message)) {
        let check = validateGrounding(message, answer, p.evidence);
        if (!check.ok || isGenericFallback(answer)) {
          answer = await this.llm.answer(this.input(p, `${p.client.prompt}\n\n${groundingCorrection(check.reasons.length ? check.reasons : ['generic fallback despite tenant evidence'])}`));
          check = validateGrounding(message, answer, p.evidence);
          if (!check.ok || isGenericFallback(answer)) answer = professionalFallback(p.client);
        }
      }
    }
    // Emit only the accepted answer. This preserves real provider streaming semantics
    // without showing a claim that the final grounding check rejects.
    for (const token of answer.match(/\S+\s*/g) ?? []) onToken(token);
    await this.repo.addMessage(p.conversationId, 'assistant', answer);
    await this.repo.usage(clientId, 'chat', message.length + answer.length, Date.now() - p.started);
    return this.result(p, answer);
  }
}
