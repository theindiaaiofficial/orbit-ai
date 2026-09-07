import crypto from 'node:crypto';
import type { Repository } from '../repositories/repository.js';
import type { EmbeddingProvider } from '../providers/embedding.js';
import type { VectorProvider } from '../providers/vector.js';
import type { LlmProvider } from '../providers/llm.js';
import type { Client, ChatMessage, RetrievedChunk } from '../domain/types.js';
import { AppError } from '../lib/errors.js';
import { selectConversationContext, type ConversationMemoryMessage } from './conversation-memory.js';

type ChatResult = {
  sessionId: string;
  conversationId: string;
  answer: string;
  sources: { chunkId: string; source: string; score: number }[];
  leadCollection?: { enabled: true; fields: string[] };
};

/** Tenant-scoped orchestration. Browser memory is optional; repository history is never LLM history. */
export class ChatService {
  constructor(private repo: Repository, private embedding: EmbeddingProvider, private vector: VectorProvider, private llm: LlmProvider) {}

  private async prepare(clientId: string, message: string, sessionId?: string, conversationContext?: ConversationMemoryMessage[]) {
    const client = await this.repo.getClient(clientId);
    if (!client?.enabled) throw new AppError(403, 'TENANT_DISABLED', 'Tenant is disabled');
    const sid = sessionId ?? crypto.randomUUID();
    // These records remain for leads/analytics/audit compatibility only.
    const conversationId = await this.repo.createConversation(clientId, sid);
    await this.repo.addMessage(conversationId, 'user', message);

    const history: ChatMessage[] = conversationContext?.length
      ? selectConversationContext(conversationContext, message)
        .filter((entry) => entry.content !== message)
        .map(({ role, content }) => ({ role, content }))
      : [];
    const query = this.retrievalQuery(message, history);
    const [q] = await this.embedding.embed([query]);
    let found = await this.vector.search(clientId, q!, client.config.topK ?? 20, client.config.minSimilarity ?? 0.05);
    if (!found.length && query !== message) {
      const [fallbackQ] = await this.embedding.embed([message]);
      found = await this.vector.search(clientId, fallbackQ!, client.config.topK ?? 20, client.config.minSimilarity ?? 0.05);
    }
    if (!found.length && this.isKnowledgeQuestion(message)) {
      const reformulated = await this.llm.reformulate?.({ question: message, prompt: client.prompt, context: [], config: client.config, history });
      if (reformulated) {
        const [fallbackQ] = await this.embedding.embed([reformulated]);
        found = await this.vector.search(clientId, fallbackQ!, client.config.topK ?? 20, client.config.minSimilarity ?? 0.05);
      }
    }
    return { client, sid, conversationId, history, found, started: Date.now() };
  }

  private isKnowledgeQuestion(message: string) {
    const text = message.trim().toLowerCase();
    if (/^(hi|hello|hey|thanks|thank you|okay|ok|good morning|good afternoon|good evening|who are you|can you help me)[!.? ]*$/i.test(text)) return false;
    return /\b(price|pricing|cost|service|services|offer|provide|policy|policies|hours|open|close|warranty|refund|return|book|booking|schedule|contact|phone|email|address|location|available|availability|emergency|company|product|plan|support|delivery|shipping|membership|rate|fee|how much|where|when|what do you)\b/i.test(text);
  }

  private retrievalQuery(message: string, history: ChatMessage[]) {
    const lower = message.toLowerCase();
    const needsContext = message.trim().split(/\s+/).length <= 8 || /\b(it|that|this|they|them|he|she|you|there|those|what|how much|summarize)\b/i.test(lower);
    if (!needsContext || !history.length) return message;
    return [...history.slice(-6).map((x) => `${x.role}: ${x.content}`), `user: ${message}`].join('\n');
  }

  private result(client: Client, sid: string, conversationId: string, answer: string, found: RetrievedChunk[]): ChatResult {
    return { sessionId: sid, conversationId, answer, sources: found.map((x) => ({ chunkId: x.id, source: x.source, score: Math.round(x.score * 1000) / 1000 })), leadCollection: client.config.collectLead ? { enabled: true, fields: ['name', ...(client.config.leadFields ?? ['email', 'requirement']).filter((field) => field !== 'name')] } : undefined };
  }

  async chat(clientId: string, message: string, sessionId?: string, conversationContext?: ConversationMemoryMessage[]): Promise<ChatResult> {
    const prepared = await this.prepare(clientId, message, sessionId, conversationContext);
    const answer = await this.llm.answer({ question: message, prompt: prepared.client.prompt, context: prepared.found, config: prepared.client.config, history: prepared.history });
    await this.repo.addMessage(prepared.conversationId, 'assistant', answer);
    await this.repo.usage(clientId, 'chat', message.length + answer.length, Date.now() - prepared.started);
    return this.result(prepared.client, prepared.sid, prepared.conversationId, answer, prepared.found);
  }

  async stream(clientId: string, message: string, sessionId: string | undefined, contextOrToken: ConversationMemoryMessage[] | ((token: string) => void), maybeOnToken?: (token: string) => void): Promise<ChatResult> {
    const conversationContext = Array.isArray(contextOrToken) ? contextOrToken : undefined;
    const onToken = Array.isArray(contextOrToken) ? maybeOnToken! : contextOrToken;
    const prepared = await this.prepare(clientId, message, sessionId, conversationContext);
    const answer = await this.llm.streamAnswer({ question: message, prompt: prepared.client.prompt, context: prepared.found, config: prepared.client.config, history: prepared.history }, onToken);
    await this.repo.addMessage(prepared.conversationId, 'assistant', answer);
    await this.repo.usage(clientId, 'chat', message.length + answer.length, Date.now() - prepared.started);
    return this.result(prepared.client, prepared.sid, prepared.conversationId, answer, prepared.found);
  }
}
