import crypto from 'node:crypto';

export type ConversationMemoryMessage = {
  id?: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp?: number;
  sequence?: number;
};

export const MAX_CONTEXT_MESSAGES = 12;
export const MAX_MESSAGE_CHARS = 4000;
export const MAX_CONTEXT_CHARS = 12000;

const terms = (value: string) => new Set(value.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []);

export type ConversationRecord = {
  conversationId: string;
  startedAt: number;
  expiresAt: number;
  messages: ConversationMemoryMessage[];
};

export function newConversation(now = Date.now()): ConversationRecord {
  return { conversationId: crypto.randomUUID(), startedAt: now, expiresAt: now + 86_400_000, messages: [] };
}

/** Selects bounded browser history; the backend repeats only hard safety limits. */
export function selectConversationContext(
  messages: ConversationMemoryMessage[],
  question: string,
  maxMessages = MAX_CONTEXT_MESSAGES,
): ConversationMemoryMessage[] {
  const limit = Math.min(MAX_CONTEXT_MESSAGES, Math.max(1, Math.floor(maxMessages)));
  const bounded = messages
    .filter((message) => (message.role === 'user' || message.role === 'assistant') && typeof message.content === 'string' && message.content.trim().length > 0)
    .map((message, index) => ({
      role: message.role,
      content: message.content.trim().slice(0, MAX_MESSAGE_CHARS),
      ...(typeof message.id === 'string' ? { id: message.id } : {}),
      ...(typeof message.timestamp === 'number' ? { timestamp: message.timestamp } : {}),
      sequence: typeof message.sequence === 'number' && Number.isSafeInteger(message.sequence) && message.sequence >= 0 ? message.sequence : index,
    }))
    .filter((message) => message.content.length > 0)
    .slice(-limit * 2);
  if (!bounded.length) return [];

  const recent = bounded.slice(-Math.min(8, limit));
  const queryTerms = terms(question);
  const older = bounded.slice(0, -recent.length)
    .map((message) => ({ message, score: [...queryTerms].filter((term) => terms(message.content).has(term)).length }))
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score || (a.message.sequence ?? 0) - (b.message.sequence ?? 0))
    .slice(0, Math.max(0, limit - recent.length))
    .map((entry) => entry.message);

  // Reserve the budget for the newest turns first; older relevance must never
  // crowd out the continuity users expect from the recent exchange.
  const recentKept: ConversationMemoryMessage[] = [];
  let chars = 0;
  for (const message of [...recent].reverse()) {
    if (recentKept.length >= limit || chars + message.content.length > MAX_CONTEXT_CHARS) continue;
    recentKept.push(message);
    chars += message.content.length;
  }
  recentKept.reverse();
  const olderKept: ConversationMemoryMessage[] = [];
  for (const message of older) {
    if (olderKept.length + recentKept.length >= limit || chars + message.content.length > MAX_CONTEXT_CHARS) continue;
    olderKept.push(message);
    chars += message.content.length;
  }
  return [...olderKept, ...recentKept].sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0));
}

export function isCurrentConversation(record: ConversationRecord | null | undefined, now = Date.now()) {
  return Boolean(record && Number.isFinite(record.startedAt) && Number.isFinite(record.expiresAt) && record.expiresAt > now && record.startedAt <= now && new RegExp('^[0-9a-f-]{36}$', 'i').test(record.conversationId));
}
