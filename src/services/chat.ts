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

type Prepared = { client: Client; sid: string; conversationId: string; history: ChatMessage[]; evidence: RetrievedChunk[]; candidates: RetrievedChunk[]; started: number; question: string };

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
    const [q] = await this.embedding.embed([query]);
    let candidates = await this.vector.search(clientId, q!, k, client.config.minSimilarity ?? 0.05);
    if (!candidates.length && query !== message) {
      const [fallbackQ] = await this.embedding.embed([message]);
      candidates = await this.vector.search(clientId, fallbackQ!, k, client.config.minSimilarity ?? 0.05);
    }
    const knowledgeQuestion = this.isKnowledgeQuestion(message);
    const initialEvidence = this.selectEvidence(candidates, query);
    const queryTerms = new Set(message.toLowerCase().match(/[\\p{L}\\p{N}]{3,}/gu) ?? []);
    const hasSemanticAnchor = initialEvidence.some((chunk) => [...queryTerms].some((term) => chunk.text.toLowerCase().includes(term)));
    // A vector hit is not proof that the hit is useful. If the bounded candidate
    // set contains no lexical anchor for a knowledge question, spend one
    // context-aware reformulation pass before falling back. This fixes false
    // fallbacks without raising K, weakening tenant filters, or bypassing RAG.
    if (knowledgeQuestion && !hasSemanticAnchor && this.llm.reformulate) {
      const reformulated = await this.llm.reformulate({ question: message, prompt: client.prompt, context: initialEvidence, config: client.config, history });
      if (reformulated) {
        const [fallbackQ] = await this.embedding.embed([reformulated]);
        candidates = await this.vector.search(clientId, fallbackQ!, k, client.config.minSimilarity ?? 0.05);
      } else if (!hasSemanticAnchor) {
        candidates = [];
      }
    }
    return { client, sid, conversationId, history, candidates, evidence: this.selectEvidence(candidates, query), started, question: message };
  }

  private input(p: Prepared, prompt = p.client.prompt): LlmInput {
    return { question: p.question, prompt, context: p.evidence, config: p.client.config, history: p.history };
  }

  private async groundedAnswer(p: Prepared) {
    const knowledgeQuestion = this.isKnowledgeQuestion(p.question);
    if (knowledgeQuestion && !p.evidence.length) return p.client.config.fallbackMessage;
    let answer = await this.llm.answer(this.input(p));
    if (!knowledgeQuestion) return answer;
    let check = validateGrounding(p.question, answer, p.evidence);
    if (!check.ok) {
      answer = await this.llm.answer(this.input(p, `${p.client.prompt}\n\n${groundingCorrection(check.reasons)}`));
      check = validateGrounding(p.question, answer, p.evidence);
      if (!check.ok) return p.client.config.fallbackMessage;
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
    if (this.isKnowledgeQuestion(message) && !p.evidence.length) answer = p.client.config.fallbackMessage;
    else {
      answer = await this.llm.streamAnswer(this.input(p), (token) => buffered.push(token));
      if (this.isKnowledgeQuestion(message)) {
        let check = validateGrounding(message, answer, p.evidence);
        if (!check.ok) {
          answer = await this.llm.answer(this.input(p, `${p.client.prompt}\n\n${groundingCorrection(check.reasons)}`));
          check = validateGrounding(message, answer, p.evidence);
          if (!check.ok) answer = p.client.config.fallbackMessage;
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
