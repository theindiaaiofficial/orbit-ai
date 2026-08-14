import fs from 'node:fs';
import path from 'node:path';
import type { ProviderHealth, RetrievedChunk, VectorRecord } from '../domain/types.js';
export interface VectorProvider {
  readonly name: string;
  upsert(clientId: string, rows: VectorRecord[]): Promise<void>;
  replaceSource(clientId: string, source: string, rows: VectorRecord[]): Promise<void>;
  search(clientId: string, v: number[], topK: number, min: number): Promise<RetrievedChunk[]>;
  deleteTenant(clientId: string): Promise<void>;
  health(): Promise<ProviderHealth>;
}
const cosine = (a: number[], b: number[]) =>
  a.reduce((s, x, i) => s + x * (b[i] ?? 0), 0) /
  ((Math.hypot(...a) || 1) * (Math.hypot(...b) || 1));
export class LocalVector implements VectorProvider {
  name = 'local-json-cosine';
  private rows: VectorRecord[] = [];
  constructor(private file: string) {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    if (fs.existsSync(file))
      this.rows = JSON.parse(fs.readFileSync(file, 'utf8')) as VectorRecord[];
  }
  private save() {
    const t = `${this.file}.tmp`;
    fs.writeFileSync(t, JSON.stringify(this.rows));
    fs.renameSync(t, this.file);
  }
  async upsert(c: string, r: VectorRecord[]) {
    const incoming = new Set(r.map((x) => x.id));
    this.rows = this.rows.filter((x) => x.clientId !== c || !incoming.has(x.id)).concat(r);
    this.save();
  }
  async replaceSource(c: string, source: string, r: VectorRecord[]) {
    this.rows = this.rows.filter((x) => x.clientId !== c || x.source !== source).concat(r);
    this.save();
  }
  async search(c: string, v: number[], k: number, min: number) {
    return this.rows
      .filter((x) => x.clientId === c)
      .map((x) => ({ id: x.id, text: x.text, source: x.source, score: cosine(x.embedding, v) }))
      .filter((x) => x.score >= min)
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }
  async deleteTenant(c: string) {
    this.rows = this.rows.filter((x) => x.clientId !== c);
    this.save();
  }
  async health() {
    return { provider: this.name, connected: true, detail: `${this.rows.length} vectors` };
  }
}
export class QdrantVector implements VectorProvider {
  name = 'qdrant';
  constructor(
    private url: string,
    private key?: string,
  ) {}
  private headers() {
    return { 'content-type': 'application/json', ...(this.key ? { 'api-key': this.key } : {}) };
  }
  async ensure(dim: number) {
    await fetch(`${this.url}/collections/tenant_knowledge`, {
      method: 'PUT',
      headers: this.headers(),
      body: JSON.stringify({ vectors: { size: dim, distance: 'Cosine' } }),
    });
  }
  async upsert(c: string, r: VectorRecord[]) {
    if (r[0]) await this.ensure(r[0].embedding.length);
    const res = await fetch(`${this.url}/collections/tenant_knowledge/points?wait=true`, {
      method: 'PUT',
      headers: this.headers(),
      body: JSON.stringify({
        points: r.map((x) => ({
          id: x.id,
          vector: x.embedding,
          payload: { clientId: c, text: x.text, source: x.source },
        })),
      }),
    });
    if (!res.ok) throw new Error(`Qdrant upsert failed (${res.status})`);
  }
  async replaceSource(c: string, source: string, r: VectorRecord[]) {
    const deleted = await fetch(
      `${this.url}/collections/tenant_knowledge/points/delete?wait=true`,
      {
        method: 'POST',
        headers: this.headers(),
        body: JSON.stringify({
          filter: {
            must: [
              { key: 'clientId', match: { value: c } },
              { key: 'source', match: { value: source } },
            ],
          },
        }),
      },
    );
    if (!deleted.ok && deleted.status !== 404)
      throw new Error(`Qdrant source replacement failed (${deleted.status})`);
    await this.upsert(c, r);
  }
  async search(c: string, v: number[], k: number, min: number) {
    const r = await fetch(`${this.url}/collections/tenant_knowledge/points/search`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify({
        vector: v,
        limit: k,
        score_threshold: min,
        filter: { must: [{ key: 'clientId', match: { value: c } }] },
        with_payload: true,
      }),
    });
    if (!r.ok) throw new Error(`Qdrant search failed (${r.status})`);
    const j = (await r.json()) as {
      result: { id: string | number; score: number; payload: { text: string; source: string } }[];
    };
    return j.result.map((x) => ({ id: String(x.id), ...x.payload, score: x.score }));
  }
  async deleteTenant(c: string) {
    await fetch(`${this.url}/collections/tenant_knowledge/points/delete?wait=true`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify({ filter: { must: [{ key: 'clientId', match: { value: c } }] } }),
    });
  }
  async health() {
    try {
      const r = await fetch(`${this.url}/healthz`);
      return { provider: this.name, connected: r.ok };
    } catch {
      return { provider: this.name, connected: false };
    }
  }
}
