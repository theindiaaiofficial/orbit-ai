import { describe, expect, it, vi } from 'vitest';
import { createTtsProvider, cleanTtsText } from '../src/providers/tts.js';
import { loadTtsConfig } from '../src/config/tts.js';

describe('provider-agnostic TTS', () => {
  const env = { TTS_PROVIDER_TYPE: 'openai-compatible', TTS_PROVIDER_NAME: 'groq', TTS_BASE_URL: 'https://api.groq.com/openai/v1', TTS_MODEL: 'playai-tts', TTS_VOICE: 'Fritz-PlayAI', TTS_API_KEY: 'test-secret' };
  it('selects Groq defaults from the single production key and rejects a missing key', () => {
    const config = loadTtsConfig({ TTS_API_KEY: env.TTS_API_KEY });
    expect(config.providerName).toBe('groq');
    expect(config.baseUrl).toBe('https://api.groq.com/openai/v1');
    expect(config.model).toBe('canopylabs/orpheus-v1-english');
    expect(config.voice).toBe('hannah');
    expect(config.apiKeyEnv).toBe('TTS_API_KEY');
    expect(() => loadTtsConfig({ TTS_PROVIDER_TYPE: 'openai-compatible' })).toThrow(/TTS_API_KEY/);
  });
  it('cleans markdown/citations and bounds text', () => { expect(cleanTtsText('**Hello** [1] https://example.com')).toBe('Hello'); expect(cleanTtsText('x'.repeat(20), 10)).toHaveLength(10); });
  it('sends a successful speech request without exposing the key', async () => { const fetcher = vi.fn().mockResolvedValue(new Response(new Uint8Array([1,2,3]), { status: 200 })); const p = createTtsProvider(loadTtsConfig(env), fetcher); const out = await p.synthesize('**Hello**'); expect(out.body).toEqual(new Uint8Array([1,2,3])); expect(new Headers(fetcher.mock.calls[0]?.[1]?.headers).get('authorization')).toBe('Bearer test-secret'); });
  it('contains provider errors and timeouts', async () => { const failed = createTtsProvider(loadTtsConfig(env), vi.fn().mockResolvedValue(new Response('bad', { status: 500 }))); await expect(failed.synthesize('hello')).rejects.toThrow('TTS request failed'); const slowFetcher: typeof fetch = async (_u, init) => await new Promise<Response>((_r, rej) => init?.signal?.addEventListener('abort', () => rej(new Error('aborted')))); const slow = createTtsProvider(loadTtsConfig({ ...env, TTS_TIMEOUT_MS: '100' }), slowFetcher); await expect(slow.synthesize('hello')).rejects.toThrow('TTS request failed'); });
});
