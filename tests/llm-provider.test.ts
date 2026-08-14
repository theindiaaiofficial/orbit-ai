import { describe, expect, it, vi } from 'vitest';
import { loadLlmConfig } from '../src/config/llm.js';
import { createLlmProvider } from '../src/providers/llm.js';

const input = {
  LLM_PROVIDER_NAME: 'nvidia',
  LLM_PROVIDER_TYPE: 'openai-compatible',
  LLM_BASE_URL: 'https://integrate.example/v1',
  LLM_API_KEY: 'test-only-key',
  LLM_MODEL: 'vendor/model',
  LLM_TIMEOUT_MS: '100',
  LLM_MAX_RETRIES: '1',
  LLM_RETRY_BASE_MS: '0',
  LLM_HEADERS: '{"x-provider-feature":"enabled"}',
};
const answerInput = {
  question: 'What?',
  prompt: 'safe',
  config: { assistantName: 'a', fallbackMessage: 'no' },
  context: [{ id: 'chunk-1', text: 'Known answer.', source: 'one.txt', score: 1 }],
};
describe('generic LLM provider configuration and adapter', () => {
  it('defaults to deterministic local configuration', async () => {
    const config = loadLlmConfig({});
    expect(config.providerType).toBe('local');
    expect((await createLlmProvider(config).health()).status).toBe('ready');
  });
  it('loads generic compatible settings and custom key reference', () => {
    const config = loadLlmConfig({
      ...input,
      LLM_API_KEY_ENV: 'NVIDIA_TOKEN',
      LLM_API_KEY: undefined,
      NVIDIA_TOKEN: 'secret',
    });
    expect(config).toMatchObject({
      providerName: 'nvidia',
      providerType: 'openai-compatible',
      apiKeyEnv: 'NVIDIA_TOKEN',
      model: 'vendor/model',
    });
    expect(config.headers).toEqual({ 'x-provider-feature': 'enabled' });
  });
  it('supports legacy openai selector while switching through configuration only', () => {
    expect(
      loadLlmConfig({
        LLM_PROVIDER: 'openai',
        OPENAI_BASE_URL: 'https://api.example/v1',
        OPENAI_API_KEY: 'x',
        OPENAI_CHAT_MODEL: 'm',
      }).providerType,
    ).toBe('openai-compatible');
  });
  it('fails fast when non-local configuration is incomplete', () => {
    expect(() => loadLlmConfig({ LLM_PROVIDER_TYPE: 'openai-compatible' })).toThrow(
      /Missing non-local LLM configuration/,
    );
  });
  it('sends compatible requests and retries a transient response', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response('busy', { status: 503 }))
      .mockResolvedValueOnce(Response.json({ choices: [{ message: { content: 'answer' } }] }));
    const p = createLlmProvider(loadLlmConfig(input), fetchMock);
    await expect(p.answer(answerInput)).resolves.toBe('answer');
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const request = fetchMock.mock.calls[1]![1] as RequestInit;
    expect(fetchMock.mock.calls[1]![0]).toEqual(
      new URL('https://integrate.example/v1/chat/completions'),
    );
    expect(request.headers).toMatchObject({
      authorization: 'Bearer test-only-key',
      'x-provider-feature': 'enabled',
    });
  });
  it('configures Azure-style paths, query parameters, and api-key authentication from env only', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ choices: [{ message: { content: 'azure answer' } }] }))
      .mockResolvedValueOnce(new Response('', { status: 200 }));
    const config = loadLlmConfig({
      LLM_PROVIDER_TYPE: 'openai-compatible',
      LLM_PROVIDER_NAME: 'azure-openai',
      LLM_BASE_URL: 'https://example.openai.azure.com',
      LLM_MODEL: 'deployment-name',
      LLM_API_KEY_ENV: 'AZURE_OPENAI_API_KEY',
      AZURE_OPENAI_API_KEY: 'azure-test-key',
      LLM_CHAT_PATH: '/openai/deployments/deployment-name/chat/completions',
      LLM_HEALTH_PATH: '/openai/models',
      LLM_QUERY_PARAMS: '{"api-version":"2024-10-21"}',
      LLM_AUTH_HEADER_NAME: 'api-key',
      LLM_AUTH_SCHEME: '',
      LLM_TIMEOUT_MS: '100',
      LLM_MAX_RETRIES: '0',
      LLM_RETRY_BASE_MS: '0',
    });
    const p = createLlmProvider(config, fetchMock);
    await expect(p.answer(answerInput)).resolves.toBe('azure answer');
    await expect(p.health()).resolves.toMatchObject({
      provider: 'azure-openai',
      baseUrl: 'https://example.openai.azure.com',
      status: 'ready',
    });
    expect(fetchMock.mock.calls[0]![0]).toEqual(
      new URL(
        'https://example.openai.azure.com/openai/deployments/deployment-name/chat/completions?api-version=2024-10-21',
      ),
    );
    expect(fetchMock.mock.calls[1]![0]).toEqual(
      new URL('https://example.openai.azure.com/openai/models?api-version=2024-10-21'),
    );
    for (const call of fetchMock.mock.calls) {
      expect((call[1] as RequestInit).headers).toMatchObject({ 'api-key': 'azure-test-key' });
    }
    expect(JSON.stringify(await p.health())).not.toContain('azure-test-key');
    expect(JSON.stringify(await p.health())).not.toContain('api-version');
  });
  it('rejects unsafe paths and headers that could override authentication', () => {
    expect(() => loadLlmConfig({ ...input, LLM_CHAT_PATH: 'https://attacker.test/path' })).toThrow(
      /LLM_CHAT_PATH/,
    );
    expect(() => loadLlmConfig({ ...input, LLM_HEADERS: '{"authorization":"secret"}' })).toThrow(
      /cannot contain/,
    );
  });
  it('bounds stalled requests with a timeout', async () => {
    const stalled = vi.fn(
      (_url: string | URL | Request, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () =>
            reject(new DOMException('aborted', 'AbortError')),
          );
        }),
    );
    const p = createLlmProvider(loadLlmConfig({ ...input, LLM_MAX_RETRIES: '0' }), stalled);
    await expect(p.answer(answerInput)).rejects.toThrow(/after 1 attempt/);
  });
  it('reports reachable authentication failure without exposing credentials', async () => {
    const p = createLlmProvider(
      loadLlmConfig(input),
      vi.fn().mockResolvedValue(new Response('', { status: 401 })),
    );
    const health = await p.health();
    expect(health).toMatchObject({
      provider: 'nvidia',
      configured: true,
      reachable: true,
      connected: false,
      authentication: 'rejected',
      status: 'authentication-failed',
      model: 'vendor/model',
      baseUrl: 'https://integrate.example/v1',
    });
    expect(JSON.stringify(health)).not.toContain('test-only-key');
  });
});
