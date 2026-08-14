import type { ClientConfig, ProviderHealth, RetrievedChunk } from '../domain/types.js';
import type { LlmConfig } from '../config/llm.js';
export interface LlmProvider {
  readonly name: string;
  answer(input: {
    question: string;
    prompt: string;
    context: RetrievedChunk[];
    config: ClientConfig;
  }): Promise<string>;
  health(): Promise<ProviderHealth>;
}
type Fetch = typeof fetch;
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const retryable = (status: number) => [408, 409, 425, 429].includes(status) || status >= 500;

export class LocalLlm implements LlmProvider {
  readonly name: string;
  constructor(private providerName = 'local') {
    this.name = providerName === 'local' ? 'local-extractive' : providerName;
  }
  async answer(i: {
    question: string;
    prompt: string;
    context: RetrievedChunk[];
    config: ClientConfig;
  }) {
    if (!i.context.length) return i.config.fallbackMessage;
    const terms = new Set(i.question.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []);
    const sentences = i.context
      .flatMap((c) => c.text.split(/(?<=[.!?])\s+/))
      .map((s) => ({ s, hit: [...terms].filter((t) => s.toLowerCase().includes(t)).length }))
      .filter((x) => x.hit > 0)
      .sort((a, b) => b.hit - a.hit)
      .slice(0, 3);
    return sentences.length ? sentences.map((x) => x.s).join(' ') : i.config.fallbackMessage;
  }
  async health(): Promise<ProviderHealth> {
    return {
      provider: this.name,
      type: 'local',
      connected: true,
      configured: true,
      reachable: true,
      authentication: 'not-required',
      status: 'ready',
    };
  }
}

export class OpenAICompatibleLlm implements LlmProvider {
  readonly name: string;
  constructor(
    private config: LlmConfig,
    private fetchImpl: Fetch = fetch,
  ) {
    this.name = config.providerName;
  }
  private async request(url: string | URL, init: RequestInit, retries: number): Promise<Response> {
    let last: unknown;
    for (let attempt = 0; attempt <= retries; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.config.timeoutMs);
      try {
        const response = await this.fetchImpl(url, { ...init, signal: controller.signal });
        if (!retryable(response.status) || attempt === retries) return response;
        last = new Error(`retryable status ${response.status}`);
      } catch (error) {
        last = error;
        if (attempt === retries)
          throw new Error(`LLM request failed after ${attempt + 1} attempt(s)`, { cause: error });
      } finally {
        clearTimeout(timer);
      }
      await sleep(this.config.retryBaseMs * 2 ** attempt);
    }
    throw last;
  }
  private url(path: string) {
    const url = new URL(path.slice(1), `${this.config.baseUrl}/`);
    for (const [name, value] of Object.entries(this.config.queryParams))
      url.searchParams.set(name, value);
    return url;
  }
  private headers() {
    const key = this.config.authScheme
      ? `${this.config.authScheme} ${this.config.apiKey!}`
      : this.config.apiKey!;
    return {
      ...this.config.headers,
      [this.config.authHeaderName]: key,
      'content-type': 'application/json',
    };
  }
  async answer(i: {
    question: string;
    prompt: string;
    context: RetrievedChunk[];
    config: ClientConfig;
  }) {
    if (!i.context.length) return i.config.fallbackMessage;
    const context = i.context.map((x, n) => `[${n + 1}] ${x.text}`).join('\n');
    const r = await this.request(
      this.url(this.config.chatPath),
      {
        method: 'POST',
        headers: this.headers(),
        body: JSON.stringify({
          model: this.config.model,
          temperature: this.config.sampling.temperature,
          ...(this.config.sampling.topP === undefined ? {} : { top_p: this.config.sampling.topP }),
          ...(this.config.sampling.maxTokens === undefined
            ? {}
            : { max_tokens: this.config.sampling.maxTokens }),
          stream: this.config.stream,
          messages: [
            {
              role: 'system',
              content: `${i.prompt}\nUse ONLY the supplied context. If unsupported, respond exactly: ${i.config.fallbackMessage}`,
            },
            { role: 'user', content: `Context:\n${context}\n\nQuestion: ${i.question}` },
          ],
        }),
      },
      this.config.maxRetries,
    );
    if (!r.ok) throw new Error(`LLM provider failed (${r.status})`);
    if (this.config.stream) {
      const body = await r.text();
      return (
        body
          .split('\n')
          .filter((line) => line.startsWith('data: ') && line !== 'data: [DONE]')
          .map((line) => {
            try {
              return (
                (JSON.parse(line.slice(6)) as { choices?: { delta?: { content?: string } }[] })
                  .choices?.[0]?.delta?.content ?? ''
              );
            } catch {
              return '';
            }
          })
          .join('') || i.config.fallbackMessage
      );
    }
    const j = (await r.json()) as { choices?: { message?: { content?: string } }[] };
    return j.choices?.[0]?.message?.content ?? i.config.fallbackMessage;
  }
  async health(): Promise<ProviderHealth> {
    const common = {
      provider: this.config.providerName,
      type: this.config.providerType,
      model: this.config.model,
      baseUrl: this.config.baseUrl,
      configured: true,
    };
    try {
      const r = await this.request(
        this.url(this.config.healthPath),
        { method: 'GET', headers: this.headers() },
        0,
      );
      const authentication = [401, 403].includes(r.status)
        ? 'rejected'
        : r.ok
          ? 'accepted'
          : 'unknown';
      return {
        ...common,
        connected: r.ok,
        reachable: true,
        authentication,
        status: r.ok
          ? 'ready'
          : authentication === 'rejected'
            ? 'authentication-failed'
            : 'upstream-error',
        detail: `diagnostic HTTP ${r.status}`,
      };
    } catch (error) {
      const timeout =
        error instanceof Error &&
        (error.name === 'AbortError' || String(error.cause).includes('AbortError'));
      return {
        ...common,
        connected: false,
        reachable: false,
        authentication: 'unknown',
        status: timeout ? 'timeout' : 'unreachable',
        detail: timeout
          ? `diagnostic timed out after ${this.config.timeoutMs}ms`
          : 'diagnostic request failed',
      };
    }
  }
}

/** Factory boundary: callers only receive the provider abstraction. */
export function createLlmProvider(config: LlmConfig, fetchImpl: Fetch = fetch): LlmProvider {
  return config.providerType === 'local'
    ? new LocalLlm(config.providerName)
    : new OpenAICompatibleLlm(config, fetchImpl);
}
