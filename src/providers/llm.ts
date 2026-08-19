import type { ChatMessage, ClientConfig, ProviderHealth, RetrievedChunk } from '../domain/types.js';
import type { LlmConfig } from '../config/llm.js';
export interface LlmInput { question: string; prompt: string; context: RetrievedChunk[]; config: ClientConfig; history?: ChatMessage[] }
export interface LlmProvider {
  readonly name: string;
  answer(input: LlmInput): Promise<string>;
  streamAnswer(input: LlmInput, onToken: (token: string) => void): Promise<string>;
  /** One bounded, tenant-scoped retrieval-query reformulation after an initial miss. */
  reformulate?(input: LlmInput): Promise<string | null>;
  health(): Promise<ProviderHealth>;
}
type Fetch = typeof fetch;
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const retryable = (status: number) => [408, 409, 425, 429].includes(status) || status >= 500;

/** Customer-facing identity is tenant configuration, never the model/provider. */
export function tenantIdentityPolicy(config: ClientConfig) {
  const assistant = config.assistantName.trim();
  const company = config.companyName?.trim();
  const tenant = company ? `the company ${JSON.stringify(company)}` : 'the current tenant';
  return [
    `You are the customer-facing AI assistant for ${tenant}.`,
    `Your configured assistant identity is ${JSON.stringify(assistant)}.`,
    'Use that tenant identity when describing who you are, what you are, your name, your creator, or the company you belong to.',
    'Do not present the underlying language model, vendor, provider, or implementation as your customer-facing identity.',
    'If asked whether you are a particular AI model or provider, politely clarify that you are the tenant-branded assistant and explain your role instead.',
    'Do not invent a creator or disclose implementation details unless the tenant knowledge explicitly supports a customer-facing answer.',
  ].join(' ');
}

function explicitIdentityQuestion(question: string) {
  const text = question.trim().replace(/[!?.,]+$/g, '');
  return /^(who are you|what are you|what(?:'s| is) your name|tell me about yourself|who created you|who developed you|what company do you belong to|which ai model are you|are you\s+\S+)$/i.test(text);
}

function tenantIdentityAnswer(question: string, config: ClientConfig) {
  if (!explicitIdentityQuestion(question)) return null;
  const assistant = config.assistantName.trim();
  const company = config.companyName?.trim();
  if (company) return `I’m ${company}'s AI assistant, ${assistant}. I’m here to help with ${company}'s products, services, and information.`;
  return `I’m ${assistant}. I’m here to help with this tenant’s products, services, and information.`;
}

export class LocalLlm implements LlmProvider {
  readonly name: string;
  constructor(private providerName = 'local') { this.name = providerName === 'local' ? 'local-extractive' : providerName; }
  async answer(i: LlmInput) {
    const identity = tenantIdentityAnswer(i.question, i.config);
    if (identity) return identity;
    if (/\b(summarize|recap|what did i|what you just)\b/i.test(i.question) && i.history?.length)
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
    const context = i.context.map((x, n) => `[${n + 1}] ${x.text}`).join('\n');
    const messages = [
      { role: 'system', content: `${tenantIdentityPolicy(i.config)}\n${i.prompt}\nGrounding rules: the tenant knowledge block is verified evidence for this tenant. If it contains evidence that answers the question, answer directly using that evidence and preserve precise values, prices, dates, and requirements; do not use the fallback merely because the question is phrased differently from the evidence. Use conversation history to resolve references and follow-ups. For greetings and general conversational requests, respond naturally without requiring tenant knowledge. If a company-specific fact is not supported by the supplied tenant knowledge, say that it is unavailable and do not invent it; use this limitation wording when appropriate: ${i.config.fallbackMessage}` },
      ...(i.history ?? []).slice(-12).map((x) => ({ role: x.role, content: x.content })),
      { role: 'user', content: `Tenant knowledge context:\n${context || '(none)'}\n\nQuestion: ${i.question}` },
    ];
    return { model: this.config.model, temperature: this.config.sampling.temperature,
      ...(this.config.sampling.topP === undefined ? {} : { top_p: this.config.sampling.topP }),
      ...(this.config.sampling.maxTokens === undefined ? {} : { max_tokens: this.config.sampling.maxTokens }), stream, messages };
  }
  async answer(i: LlmInput) {
    const identity = tenantIdentityAnswer(i.question, i.config);
    if (identity) return identity;
    const r = await this.request(this.url(this.config.chatPath), { method: 'POST', headers: this.headers(), body: JSON.stringify(this.payload(i, false)) }, this.config.maxRetries);
    if (!r.ok) throw new Error(`LLM provider failed (${r.status})`);
    const j = await r.json() as { choices?: { message?: { content?: string } }[] };
    return j.choices?.[0]?.message?.content ?? i.config.fallbackMessage;
  }
  async streamAnswer(i: LlmInput, onToken: (token: string) => void) {
    const identity = tenantIdentityAnswer(i.question, i.config);
    if (identity) { for (const token of identity.match(/\S+\s*/g) ?? []) onToken(token); return identity; }
    const r = await this.request(this.url(this.config.chatPath), { method: 'POST', headers: this.headers(), body: JSON.stringify(this.payload(i, true)) }, this.config.maxRetries);
    if (!r.ok) throw new Error(`LLM provider failed (${r.status})`);
    if (!r.body) return this.answer(i);
    const reader = r.body.getReader(); const decoder = new TextDecoder(); let buffer = ''; let answer = '';
    const emit = (line: string) => {
      if (!line.startsWith('data: ') || line === 'data: [DONE]') return;
      try { const token = (JSON.parse(line.slice(6)) as { choices?: { delta?: { content?: string } }[] }).choices?.[0]?.delta?.content ?? ''; if (token) { answer += token; onToken(token); } } catch { /* ignore incomplete provider frames */ }
    };
    while (true) { const { value, done } = await reader.read(); buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done }); const lines = buffer.split('\n'); buffer = lines.pop() ?? ''; lines.forEach(emit); if (done) break; }
    if (!answer) { const fallback = i.config.fallbackMessage; onToken(fallback); return fallback; }
    return answer;
  }
  async reformulate(i: LlmInput) {
    const history = (i.history ?? []).slice(-8).map((x) => `${x.role}: ${x.content}`).join('\n');
    const payload = {
      model: this.config.model, temperature: 0, max_tokens: 80, stream: false,
      messages: [
        { role: 'system', content: 'Convert the user request into one concise search query for the current tenant knowledge base. Resolve pronouns using the conversation. Return only the query, with no explanation. If this is a greeting or general conversational request, return NONE.' },
        { role: 'user', content: `Conversation:\n${history || '(none)'}\nCurrent request: ${i.question}` },
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
