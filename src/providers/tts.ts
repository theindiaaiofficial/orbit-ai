import type { ProviderHealth } from '../domain/types.js';
import type { TtsConfig } from '../config/tts.js';
export type TtsAudio = { body: Uint8Array; contentType: string };
export interface TtsProvider { readonly name: string; synthesize(text: string): Promise<TtsAudio>; health(): Promise<ProviderHealth>; }
export function cleanTtsText(text: string, max = 4000) {
  return text.replace(/```[\s\S]*?```/g, '').replace(/\[\d+\]/g, '').replace(/https?:\/\/\S+/g, '')
    .replace(/[*_~#>`]/g, '').replace(/\s+/g, ' ').trim().slice(0, max);
}
const mime: Record<string, string> = { wav: 'audio/wav', mp3: 'audio/mpeg', opus: 'audio/ogg; codecs=opus', aac: 'audio/aac', flac: 'audio/flac' };
export class LocalTts implements TtsProvider {
  readonly name = 'local';
  async synthesize(_text: string): Promise<TtsAudio> { throw new Error('TTS provider is not configured'); }
  async health(): Promise<ProviderHealth> { return { provider: this.name, type: 'local', connected: true, configured: true, reachable: true, authentication: 'not-required', status: 'ready' }; }
}
export class OpenAICompatibleTts implements TtsProvider {
  readonly name: string;
  constructor(private config: TtsConfig, private fetchImpl: typeof fetch = fetch) { this.name = config.providerName; }
  private url(path: string) { return new URL(path.slice(1), `${this.config.baseUrl}/`); }
  private headers() { return { [this.config.authHeaderName]: this.config.authScheme ? `${this.config.authScheme} ${this.config.apiKey!}` : this.config.apiKey!, 'content-type': 'application/json' }; }
  async synthesize(text: string) {
    const clean = cleanTtsText(text, this.config.maxInputChars); if (!clean) throw new Error('TTS input is empty');
    const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), this.config.timeoutMs);
    try {
      const r = await this.fetchImpl(this.url(this.config.speechPath), { method: 'POST', headers: this.headers(), body: JSON.stringify({ model: this.config.model, voice: this.config.voice, input: clean, response_format: this.config.responseFormat }), signal: controller.signal });
      if (!r.ok) throw new Error(`TTS provider failed (${r.status})`);
      return { body: new Uint8Array(await r.arrayBuffer()), contentType: mime[this.config.responseFormat] ?? 'application/octet-stream' };
    } catch (e) { throw new Error('TTS request failed', { cause: e }); } finally { clearTimeout(timer); }
  }
  async health() {
    const common = { provider: this.name, type: this.config.providerType, model: this.config.model, baseUrl: this.config.baseUrl, configured: true };
    try { const r = await this.fetchImpl(this.url(this.config.healthPath), { headers: this.headers(), signal: AbortSignal.timeout(this.config.timeoutMs) }); const auth = [401,403].includes(r.status) ? 'rejected' : r.ok ? 'accepted' : 'unknown'; return { ...common, connected: r.ok, reachable: true, authentication: auth, status: r.ok ? 'ready' : 'upstream-error', detail: `diagnostic HTTP ${r.status}` } as ProviderHealth; }
    catch { return { ...common, connected: false, reachable: false, authentication: 'unknown', status: 'unreachable' } as ProviderHealth; }
  }
}
export function createTtsProvider(config: TtsConfig, fetchImpl: typeof fetch = fetch): TtsProvider { return config.providerType === 'local' ? new LocalTts() : new OpenAICompatibleTts(config, fetchImpl); }
