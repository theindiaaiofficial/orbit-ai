import type { ChatMessage, ClientConfig, ProviderHealth, RetrievedChunk } from '../domain/types.js';
import type { LlmConfig } from '../config/llm.js';
export interface LlmInput { question: string; prompt: string; context: RetrievedChunk[]; config: ClientConfig; history?: ChatMessage[] }
export interface LlmProvider {
  readonly name: string;
  answer(input: LlmInput): Promise<string>;
  streamAnswer(input: LlmInput, onToken: (token: string) => void): Promise<string>;
  reformulate?(input: LlmInput): Promise<string | null>;
  health(): Promise<ProviderHealth>;
}
type Fetch = typeof fetch;
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const retryable = (status: number) => [408, 409, 425, 429].includes(status) || status >= 500;

export class LocalLlm implements LlmProvider {
  readonly name: string;
  constructor(private providerName = 'local') { this.name = providerName === 'local' ? 'local-extractive' : providerName; }
  async answer(i: LlmInput) {
    if (/(summarize|recap|what did i|what you just)/i.test(i.question) && i.history?.length)
      return i.history.slice(-4).map((x) => x.content).join(' ');
    const terms = new Set(i.question.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []);
    const sentences = [...i.context.flatMap((c) => c.text.split(/(?<=[.!?])\s+/)), ...(i.history ?? []).map((x) => x.content)]
      .map((s) => ({ s, hit: [...terms].filter((t) => s.toLowerCase().includes(t)).length }))
      .filter((x) => x.hit > 0).sort((a, b) => b.hit - a.hit).slice(0, 3);
    return sentences.length ? sentences.map((x) => x.s).join(' ') : i.config.fallbackMessage;
  }
  async streamAnswer(i: LlmInput, onToken: (token: string) => void) {
    const answer = await this.answer(i);
    for (const token of answer.match(/\S+\s*/g) ?? []) onToken(token);
    return answer;
  }
  async health(): Promise<ProviderHealth> { return { provider: this.name, type: 'local', connected: true, configured: true, reachable: true, authentication: 'not-required', status: 'ready' }; }
}

export class OpenAICompatibleLlm implements LlmProvider {
  readonly name: string;
  constructor(private config: LlmConfig, private fetchImpl: Fetch = fetch) { this.name = config.providerName; }
  private async request(url: string | URL, init: RequestInit, retries: number): Promise<Response> {
    let last: unknown;
    for (let attempt = 0; attempt <= retries; attempt++) {
      const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), this.config.timeoutMs);
      try {
        const response = await this.fetchImpl(url, { ...init, signal: controller.signal });
        if (!retryable(response.status) || attempt === retries) return response;
        last = new Error(`retryable status ${response.status}`);
      } catch (error) {
        last = error;
        if (attempt === retries) throw new Error(`LLM request failed after ${attempt + 1} attempt(s)`, { cause: error });
      } finally { clearTimeout(timer); }
      await sleep(this.config.retryBaseMs * 2 ** attempt);
    }
    throw last;
  }
  private url(path: string) { const url = new URL(path.slice(1), `${this.config.baseUrl}/`); for (const [name, value] of Object.entries(this.config.queryParams)) url.searchParams.set(name, value); return url; }
  private headers() { const key = this.config.authScheme ? `${this.config.authScheme} ${this.config.apiKey!}` : this.config.apiKey!; return { ...this.config.headers, [this.config.authHeaderName]: key, 'content-type': 'application/json' }; }
  private payload(i: LlmInput, stream: boolean) {
    const context = i.context.map((x, n) => `[${n + 1}] ${x.text}`).join('
');
    const messages = [
      { role: 'system', content: `${i.prompt}
Use tenant knowledge for company-specific facts and answer greetings and general conversation naturally. Use conversation history to resolve references and follow-ups. If a company-specific fact is not supported by the supplied tenant knowledge, respond honestly without inventing it; use this limitation wording when appropriate: ${i.config.fallbackMessage}` },
      ...(i.history ?? []).slice(-12).map((x) => ({ role: x.role, content: x.content })),
      { role: 'user', content: `Tenant knowledge context:
${context || '(none)'}

Question: ${i.question}` },
    ];
    return { model: this.config.model, temperature: this.config.sampling.temperature,
      ...(this.config.sampling.topP === undefined ? {} : { top_p: this.config.sampling.topP }),
      ...(this.config.sampling.maxTokens === undefined ? {} : { max_tokens: this.config.sampling.maxTokens }), stream, messages };
  }
  async answer(i: LlmInput) {
    const r = await this.request(this.url(this.config.chatPath), { method: 'POST', headers: this.headers(), body: JSON.stringify(this.payload(i, false)) }, this.config.maxRetries);
    if (!r.ok) throw new Error(`LLM provider failed (${r.status})`);
    const j = await r.json() as { choices?: { message?: { content?: string } }[] };
    return j.choices?.[0]?.message?.content ?? i.config.fallbackMessage;
  }
  async streamAnswer(i: LlmInput, onToken: (token: string) => void) {
    const r = await this.request(this.url(this.config.chatPath), { method: 'POST', headers: this.headers(), body: JSON.stringify(this.payload(i, true)) }, this.config.maxRetries);
    if (!r.ok) throw new Error(`LLM provider failed (${r.status})`);
    if (!r.body) return this.answer(i);
    const reader = r.body.getReader(); const decoder = new TextDecoder(); let buffer = ''; let answer = '';
    const emit = (line: string) => {
      if (!line.startsWith('data: ') || line === 'data: [DONE]') return;
      try { const token = (JSON.parse(line.slice(6)) as { choices?: { delta?: { content?: string } }[] }).choices?.[0]?.delta?.content ?? ''; if (token) { answer += token; onToken(token); } } catch { /* ignore incomplete provider frames */ }
    };
    while (true) { const { value, done } = await reader.read(); buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done }); const lines = buffer.split('
'); buffer = lines.pop() ?? ''; lines.forEach(emit); if (done) break; }
    if (!answer) { const fallback = i.config.fallbackMessage; onToken(fallback); return fallback; }
    return answer;
  }
  async reformulate(i: LlmInput) {
    const history = (i.history ?? []).slice(-8).map((x) => `${x.role}: ${x.content}`).join('
');
    const payload = {
      model: this.config.model, temperature: 0, max_tokens: 80, stream: false,
      messages: [
        { role: 'system', content: 'Convert the user request into one concise search query for the current tenant knowledge base. Resolve pronouns using the conversation. Return only the query, with no explanation. If this is a greeting or general conversational request, return NONE.' },
        { role: 'user', content: `Conversation:
${history || '(none)'}
Current request: ${i.question}` },
      ],
    };
    const r = await this.request(this.url(this.config.chatPath), { method: 'POST', headers: this.headers(), body: JSON.stringify(payload) }, 0);
    if (!r.ok) return null;
    const j = await r.json() as { choices?: { message?: { content?: string } }[] };
    const query = j.choices?.[0]?.message?.content?.trim().replace(/^['\"]|['\"]$/g, '');
    return query && query.toUpperCase() !== 'NONE' ? query : null;
  }
  async health(): Promise<ProviderHealth> {
    const common = { provider: this.config.providerName, type: this.config.providerType, model: this.config.model, baseUrl: this.config.baseUrl, configured: true };
    try { const r = await this.request(this.url(this.config.healthPath), { method: 'GET', headers: this.headers() }, 0); const authentication = [401, 403].includes(r.status) ? 'rejected' : r.ok ? 'accepted' : 'unknown'; return { ...common, connected: r.ok, reachable: true, authentication, status: r.ok ? 'ready' : authentication === 'rejected' ? 'authentication-failed' : 'upstream-error', detail: `diagnostic HTTP ${r.status}` }; }
    catch (error) { const timeout = error instanceof Error && (error.name === 'AbortError' || String(error.cause).includes('AbortError')); return { ...common, connected: false, reachable: false, authentication: 'unknown', status: timeout ? 'timeout' : 'unreachable', detail: timeout ? `diagnostic timed out after ${this.config.timeoutMs}ms` : 'diagnostic request failed' }; }
  }
}
export function createLlmProvider(config: LlmConfig, fetchImpl: Fetch = fetch): LlmProvider { return config.providerType === 'local' ? new LocalLlm(config.providerName) : new OpenAICompatibleLlm(config, fetchImpl); }
