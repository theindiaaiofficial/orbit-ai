import crypto from 'node:crypto';
import type { ProviderHealth } from '../domain/types.js';
export interface EmbeddingProvider {
  readonly name: string;
  embed(texts: string[]): Promise<number[][]>;
  health(): Promise<ProviderHealth>;
}
export class LocalEmbedding implements EmbeddingProvider {
  name = 'local-deterministic';
  constructor(private dimensions = 128) {
    if (!Number.isInteger(dimensions) || dimensions < 8 || dimensions > 8192) throw new Error('Embedding dimensions must be an integer between 8 and 8192');
  }
  async embed(texts: string[]) {
    return texts.map((t) => {
      const v = Array(this.dimensions).fill(0) as number[];
      for (const token of t.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []) {
        const h = crypto.createHash('sha256').update(token).digest();
        const i = h.readUInt32BE(0) % this.dimensions;
        v[i] = (v[i] ?? 0) + (h[4]! % 2 ? 1 : -1);
      }
      const n = Math.hypot(...v) || 1;
      return v.map((x) => x / n);
    });
  }
  async health() { return { provider: this.name, connected: true, detail: `${this.dimensions} dimensions` }; }
}
export class OpenAIEmbedding implements EmbeddingProvider {
  name = 'openai-compatible';
  constructor(
    private base: string,
    private key: string | undefined,
    private model: string,
    private dimensions = 128,
    private timeoutMs = 15000,
  ) {}
  async embed(texts: string[]) {
    if (!this.key) throw new Error('OPENAI_API_KEY is required');
    if (!texts.length) return [];
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const payload: Record<string, unknown> = { model: this.model, input: texts };
      // text-embedding-3 models support reduced dimensions. Other models may
      // ignore or reject this field, so validate their actual output below.
      if (/^text-embedding-3(?:-|$)/i.test(this.model)) payload.dimensions = this.dimensions;
      const r = await fetch(`${this.base}/embeddings`, {
        method: 'POST',
        headers: { authorization: `Bearer ${this.key}`, 'content-type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!r.ok) throw new Error(`Embedding provider failed (${r.status})`);
      const j = await r.json() as { data?: { index?: number; embedding?: number[] }[] };
      if (!Array.isArray(j.data) || j.data.length !== texts.length) throw new Error('Embedding provider returned an unexpected number of vectors');
      const ordered = j.data.slice().sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
      const vectors = ordered.map((row) => row.embedding);
      if (vectors.some((v) => !Array.isArray(v) || v.length !== this.dimensions || v.some((n) => !Number.isFinite(n)))) {
        const actual = vectors.find((v) => Array.isArray(v))?.length ?? 0;
        throw new Error(`Embedding dimension mismatch: expected ${this.dimensions}, received ${actual}. Use a compatible model/configuration or rebuild the vector schema and index.`);
      }
      return vectors as number[][];
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') throw new Error(`Embedding request timed out after ${this.timeoutMs}ms`, { cause: error });
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }
  async health() {
    return { provider: this.name, connected: Boolean(this.key), detail: this.key ? `configured; expected ${this.dimensions} dimensions` : 'missing credentials' };
  }
}
