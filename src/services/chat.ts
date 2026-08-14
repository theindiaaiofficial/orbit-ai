import crypto from 'node:crypto';
import type { Repository } from '../repositories/repository.js';
import type { EmbeddingProvider } from '../providers/embedding.js';
import type { VectorProvider } from '../providers/vector.js';
import type { LlmProvider } from '../providers/llm.js';
import { AppError } from '../lib/errors.js';
export class ChatService {
  constructor(
    private repo: Repository,
    private embedding: EmbeddingProvider,
    private vector: VectorProvider,
    private llm: LlmProvider,
  ) {}
  async chat(clientId: string, message: string, sessionId?: string) {
    const client = this.repo.getClient(clientId);
    if (!client?.enabled) throw new AppError(403, 'TENANT_DISABLED', 'Tenant is disabled');
    const started = Date.now();
    const sid = sessionId ?? crypto.randomUUID();
    const cid = this.repo.createConversation(clientId, sid);
    this.repo.addMessage(cid, 'user', message);
    const [q] = await this.embedding.embed([message]);
    const found = await this.vector.search(
      clientId,
      q!,
      client.config.topK ?? 4,
      client.config.minSimilarity ?? 0.05,
    );
    const answer = await this.llm.answer({
      question: message,
      prompt: client.prompt,
      context: found,
      config: client.config,
    });
    this.repo.addMessage(cid, 'assistant', answer);
    this.repo.usage(clientId, 'chat', message.length + answer.length, Date.now() - started);
    return {
      sessionId: sid,
      conversationId: cid,
      answer,
      sources: found.map((x) => ({
        chunkId: x.id,
        source: x.source,
        score: Math.round(x.score * 1000) / 1000,
      })),
      leadCollection: client.config.collectLead
        ? {
            enabled: true,
            fields: [
              'name',
              ...(client.config.leadFields ?? ['email', 'requirement']).filter(
                (field) => field !== 'name',
              ),
            ],
          }
        : undefined,
    };
  }
}
