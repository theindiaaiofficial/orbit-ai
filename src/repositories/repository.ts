import type { Client, ClientConfig, ChatMessage } from '../domain/types.js';
export type CreateClient = {
  id: string;
  name: string;
  slug: string;
  config: ClientConfig;
  prompt: string;
  keyHash: string;
  domains: string[];
};
export interface Repository {
  init(): void;
  close(): void;
  createClient(v: CreateClient): Client;
  listClients(): Client[];
  getClient(id: string): Client | undefined;
  getClientByKeyHash(hash: string): Client | undefined;
  updateClient(
    id: string,
    v: Partial<{ name: string; config: ClientConfig; prompt: string; enabled: boolean }>,
  ): Client | undefined;
  deleteClient(id: string): boolean;
  domains(id: string): string[];
  setDomains(id: string, d: string[]): void;
  rotateKey(id: string, hash: string): void;
  createConversation(clientId: string, sessionId: string): string;
  addMessage(conversationId: string, role: 'user' | 'assistant', content: string): void;
  messages(conversationId: string): ChatMessage[];
  saveLead(
    clientId: string,
    conversationId: string | undefined,
    data: Record<string, string>,
  ): string;
  listLeads(clientId: string): unknown[];
  usage(clientId: string, kind: string, units: number, latencyMs: number): void;
  audit(clientId: string | undefined, action: string, detail: unknown): void;
  stats(clientId: string): unknown;
}
