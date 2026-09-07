import { describe, expect, it, vi } from 'vitest';
import { ChatService } from '../src/services/chat.js';
import type { Client, RetrievedChunk } from '../src/domain/types.js';
import { isCurrentConversation, newConversation, selectConversationContext, type ConversationMemoryMessage } from '../src/services/conversation-memory.js';

const client: Client = { id: 'tenant-a', name: 'Tenant A', slug: 'tenant-a', enabled: true, prompt: 'Be concise.', createdAt: '', updatedAt: '', config: { assistantName: 'A', fallbackMessage: 'UNKNOWN', topK: 4 } };
const chunk: RetrievedChunk = { id: 'knowledge-1', text: 'The answer is in tenant knowledge.', source: 'knowledge.md', score: .9 };

describe('browser conversation memory', () => {
  it('has one non-renewing 24-hour lifetime', () => {
    const record = newConversation(1_000);
    expect(record.expiresAt).toBe(86_401_000);
    expect(isCurrentConversation(record, 86_400_999)).toBe(true);
    expect(isCurrentConversation(record, 86_401_000)).toBe(false);
  });

  it('keeps recent turns, selects relevant older turns, and enforces both limits', () => {
    const messages: ConversationMemoryMessage[] = [
      { role: 'user', content: 'The delivery window is Tuesday.' },
      { role: 'assistant', content: 'Tuesday delivery is available.' },
      ...Array.from({ length: 20 }, (_, i) => ({ role: 'user' as const, content: `unrelated turn ${i}` })),
      { role: 'user', content: 'What was the delivery window?' },
    ];
    const selected = selectConversationContext(messages, 'What was the delivery window?', 12);
    expect(selected.length).toBeLessThanOrEqual(12);
    expect(selected.reduce((n, m) => n + m.content.length, 0)).toBeLessThanOrEqual(12000);
    expect(selected.some((m) => m.content.includes('Tuesday'))).toBe(true);
  });

  it('sanitizes roles and empty content', () => {
    expect(selectConversationContext([{ role: 'user', content: '' }, { role: 'system' as never, content: 'secret' }], 'follow up')).toEqual([]);
  });

  it('uses browser context for continuity without reading durable repository history, while RAG still searches the tenant', async () => {
    const repo = { getClient: vi.fn(async () => client), createConversation: vi.fn(async () => 'db-conversation'), addMessage: vi.fn(async () => undefined), messages: vi.fn(async () => [{ role: 'assistant', content: 'MUST NOT BE USED' }]), usage: vi.fn(async () => undefined) } as any;
    const embedding = { embed: vi.fn(async (texts: string[]) => texts.map((text) => [text.length])) } as any;
    const vector = { search: vi.fn(async () => [chunk]) } as any;
    const llm = { answer: vi.fn(async (input: any) => input.history.map((m: any) => m.content).join('|')), streamAnswer: vi.fn(), health: vi.fn() } as any;
    const service = new ChatService(repo, embedding, vector, llm);
    const result = await service.chat('tenant-a', 'What about Tuesday?', undefined, [{ role: 'user', content: 'Delivery is on Tuesday.' }, { role: 'user', content: 'What about Tuesday?' }]);
    expect(repo.messages).not.toHaveBeenCalled();
    expect(llm.answer).toHaveBeenCalledWith(expect.objectContaining({ history: [{ role: 'user', content: 'Delivery is on Tuesday.' }] }));
    expect(embedding).toHaveProperty('embed');
    expect(vector.search).toHaveBeenCalledWith('tenant-a', expect.any(Array), 4, .05);
    expect(result.sources[0]?.chunkId).toBe('knowledge-1');
  });
});
