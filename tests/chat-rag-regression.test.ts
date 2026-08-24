/* eslint-disable @typescript-eslint/no-explicit-any */
import { describe, expect, it, vi } from 'vitest';
import { ChatService } from '../src/services/chat.js';
import type { Client, RetrievedChunk } from '../src/domain/types.js';

const client: Client = {
  id: 'tenant-a', name: 'Tenant A', slug: 'tenant-a', enabled: true,
  prompt: 'Be concise.', createdAt: '', updatedAt: '',
  config: { assistantName: 'A', companyName: 'Tenant A', fallbackMessage: 'UNKNOWN', topK: 4 },
};
const chunk = (id: string, text: string, score: number): RetrievedChunk => ({ id, text, source: 'knowledge.md', score });

function fixture(opts: { search?: (q: number[]) => Promise<RetrievedChunk[]>; answer?: (input: any) => Promise<string>; reformulate?: () => Promise<string | null> } = {}) {
  const searches: number[][] = [];
  const repo: any = {
    getClient: vi.fn(async () => client), createConversation: vi.fn(async () => 'conversation-1'),
    messages: vi.fn(async () => []), addMessage: vi.fn(async () => undefined), usage: vi.fn(async () => undefined),
  };
  const embedding: any = { embed: vi.fn(async (values: string[]) => values.map((x) => [x.length])) };
  const vector: any = { search: vi.fn(async (_id: string, q: number[]) => { searches.push(q); return opts.search?.(q) ?? []; }) };
  const llm: any = {
    answer: vi.fn(opts.answer ?? (async (input: any) => input.context[0]?.text ?? 'UNKNOWN')),
    streamAnswer: vi.fn(async (input: any) => opts.answer?.(input) ?? input.context[0]?.text ?? 'UNKNOWN'),
    reformulate: opts.reformulate ? vi.fn(opts.reformulate) : undefined,
  };
  return { service: new ChatService(repo, embedding, vector, llm), repo, embedding, vector, llm, searches };
}

describe('generic RAG recovery and grounding path', () => {
  it('keeps semantic similarity ahead of generic lexical overlap', async () => {
    const { service } = fixture();
    const selected = (service as any).selectEvidence([
      chunk('generic', 'Contact our general service team.', 0.58),
      chunk('answer', 'Customer Care can be reached by email at help@example.test.', 0.82),
    ], 'How can I contact Customer Care?');
    expect(selected[0].id).toBe('answer');
  });

  it('recovers after a weak first hit and merges recovered tenant evidence', async () => {
    const relevant = chunk('care', 'For assistance, customers can contact the Customer Care team by email.', 0.79);
    let searchCount = 0;
    const { service, llm, vector } = fixture({
      search: async (_q) => (++searchCount === 1 ? [chunk('weak', 'Our company offers customer services.', 0.31)] : [relevant]),
      reformulate: async () => 'support contact assistance',
    });
    const result = await service.chat('tenant-a', 'How can I contact Customer Care?');
    expect(llm.reformulate).toHaveBeenCalledOnce();
    expect(vector.search).toHaveBeenCalledTimes(2);
    expect(result.sources[0]?.chunkId).toBe('care');
    expect(result.answer).toContain('Customer Care');
  });

  it('uses bounded conversation history for summaries without knowledge retrieval', async () => {
    const { service, repo, embedding, vector, llm } = fixture({ answer: async (input) => input.history.map((x: any) => x.content).join(' | ') });
    repo.messages.mockResolvedValue([{ role: 'user', content: 'We discussed delivery.' }, { role: 'assistant', content: 'Delivery takes two days.' }]);
    const result = await service.chat('tenant-a', 'Please provide a detailed summary for this chat');
    expect(embedding.embed).not.toHaveBeenCalled();
    expect(vector.search).not.toHaveBeenCalled();
    expect(llm.answer).toHaveBeenCalledOnce();
    expect(result.answer).toContain('Delivery takes two days.');
  });
});
