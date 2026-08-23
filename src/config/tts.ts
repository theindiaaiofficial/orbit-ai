import { z } from 'zod';

export type TtsConfig = {
  providerName: string;
  providerType: 'local' | 'openai-compatible';
  baseUrl?: string;
  apiKey?: string;
  apiKeyEnv: 'TTS_API_KEY';
  model?: string;
  voice?: string;
  responseFormat: 'wav' | 'mp3' | 'opus' | 'aac' | 'flac';
  timeoutMs: number;
  maxInputChars: number;
  speechPath: string;
  healthPath: string;
  authHeaderName: string;
  authScheme: string;
};

const GROQ_BASE_URL = 'https://api.groq.com/openai/v1';
const GROQ_MODEL = 'playai-tts';
const GROQ_VOICE = 'Fritz-PlayAI';
const token = /^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/;
function path(value: string | undefined, fallback: string, name: string) {
  const p = value ?? fallback;
  if (!p.startsWith('/') || p.startsWith('//') || /[?#\r\n]/.test(p) || new URL(p, 'https://invalid').pathname !== p)
    throw new Error(`${name} must be a normalized absolute path`);
  return p;
}

/**
 * Production convention: TTS_API_KEY is the only required deployment setting.
 * When present, the built-in Groq OpenAI-compatible defaults are selected.
 * The provider interface remains generic so a future provider can be added in code.
 */
export function loadTtsConfig(input: NodeJS.ProcessEnv): TtsConfig {
  const hasKey = Boolean(input.TTS_API_KEY);
  const type = z.enum(['local', 'openai-compatible']).parse(input.TTS_PROVIDER_TYPE ?? (hasKey ? 'openai-compatible' : 'local'));
  const name = input.TTS_PROVIDER_NAME ?? (type === 'local' ? 'local' : 'groq');
  const apiKey = input.TTS_API_KEY;
  const authHeaderName = input.TTS_AUTH_HEADER_NAME ?? 'authorization';
  if (!token.test(authHeaderName)) throw new Error('TTS_AUTH_HEADER_NAME must be a valid header name');
  const authScheme = input.TTS_AUTH_SCHEME ?? 'Bearer';
  if (authScheme && !token.test(authScheme)) throw new Error('TTS_AUTH_SCHEME must be empty or a valid HTTP authentication scheme');
  const config: TtsConfig = {
    providerName: name, providerType: type,
    baseUrl: input.TTS_BASE_URL ?? (type === 'openai-compatible' ? GROQ_BASE_URL : undefined),
    apiKey, apiKeyEnv: 'TTS_API_KEY',
    model: input.TTS_MODEL ?? (type === 'openai-compatible' ? GROQ_MODEL : undefined),
    voice: input.TTS_VOICE ?? (type === 'openai-compatible' ? GROQ_VOICE : undefined),
    responseFormat: z.enum(['wav', 'mp3', 'opus', 'aac', 'flac']).parse(input.TTS_RESPONSE_FORMAT ?? 'wav'),
    timeoutMs: z.coerce.number().int().min(100).parse(input.TTS_TIMEOUT_MS ?? 15000),
    maxInputChars: z.coerce.number().int().min(100).max(10000).parse(input.TTS_MAX_INPUT_CHARS ?? 4000),
    speechPath: path(input.TTS_SPEECH_PATH, '/audio/speech', 'TTS_SPEECH_PATH'),
    healthPath: path(input.TTS_HEALTH_PATH, '/models', 'TTS_HEALTH_PATH'), authHeaderName, authScheme,
  };
  if (type !== 'local') {
    if (!config.baseUrl || !config.model || !config.apiKey) throw new Error('TTS_API_KEY is required for production TTS');
    const u = new URL(config.baseUrl); if (u.username || u.password || u.search || u.hash) throw new Error('TTS_BASE_URL must not contain credentials, query parameters, or fragments');
    config.baseUrl = u.toString().replace(/\/$/, '');
  }
  return config;
}
