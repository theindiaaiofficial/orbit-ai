/* eslint-disable @typescript-eslint/no-explicit-any */
import fs from 'node:fs';
import path from 'node:path';
import type { ProviderHealth, RetrievedChunk, VectorRecord } from '../domain/types.js';
export interface VectorProvider {
  readonly name: string;
  upsert(clientId: string, rows: VectorRecord[]): Promise<void>;
  replaceSource(clientId: string, source: string, rows: VectorRecord[]): Promise<void>;
  /** Upsert a complete tenant generation, then remove stale point IDs only after all upserts succeed. */
  replaceTenant(clientId: string, rows: VectorRecord[]): Promise<void>;
  search(clientId: string, v: number[], topK: number, min: number): Promise<RetrievedChunk[]>;
  /** Bounded tenant-only text corpus used by exact, keyword, and BM25 recall. */
  listTenant(clientId: string, limit?: number): Promise<RetrievedChunk[]>;
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
    if (!r.length) return;
    const dimensions = r[0]!.embedding.length;
    if (!dimensions || r.some((row) => row.embedding.length !== dimensions)) throw new Error('Local vector upsert contains inconsistent embedding dimensions');
    const current = this.rows.find((row) => row.clientId === c);
    if (current && current.embedding.length !== dimensions) throw new Error(`Local vector dimension mismatch: index=${current.embedding.length}, embedding=${dimensions}; rebuild the tenant index`);
    const previous = this.rows;
    const incoming = new Set(r.map((x) => x.id));
    this.rows = this.rows.filter((x) => x.clientId !== c || !incoming.has(x.id)).concat(r);
    try { this.save(); } catch (error) { this.rows = previous; throw error; }
  }
  async replaceSource(c: string, source: string, r: VectorRecord[]) {
    const incomingDimension = r[0]?.embedding.length;
    const conflicting = this.rows.find((row) => row.clientId === c && row.source !== source && incomingDimension !== undefined && row.embedding.length !== incomingDimension);
    if (conflicting) throw new Error(`Local vector dimension mismatch: tenant has ${conflicting.embedding.length} dimensions but replacement has ${incomingDimension}; perform a full tenant rebuild`);
    const previous = this.rows;
    const newIds = new Set(r.map((x) => x.id));
    this.rows = previous.filter((x) => x.clientId !== c || x.source !== source || newIds.has(x.id));
    this.rows = this.rows.filter((x) => x.clientId !== c || !newIds.has(x.id)).concat(r);
    try { this.save(); } catch (error) { this.rows = previous; throw error; }
  }
  async replaceTenant(c: string, r: VectorRecord[]) {
    if (r.length && r.some((row) => row.embedding.length !== r[0]!.embedding.length)) throw new Error('Tenant rebuild contains inconsistent embedding dimensions');
    const previous = this.rows;
    this.rows = previous.filter((x) => x.clientId !== c).concat(r);
    try { this.save(); } catch (error) { this.rows = previous; throw error; }
  }
  async search(c: string, v: number[], k: number, min: number) {
    const tenantRows = this.rows.filter((x) => x.clientId === c);
    if (tenantRows.some((x) => x.embedding.length !== v.length)) throw new Error(`Local vector dimension mismatch: query=${v.length}, index=${tenantRows.find((x) => x.embedding.length !== v.length)!.embedding.length}; rebuild the tenant index`);
    return tenantRows
      .map((x) => ({ id: x.id, text: x.text, source: x.source, score: cosine(x.embedding, v) }))
      .filter((x) => x.score >= min)
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }
  async listTenant(c: string, limit = 2000) {
    const rows = this.rows.filter((x) => x.clientId === c);
    if (rows.length > limit) throw new Error(`Local vector corpus contains ${rows.length} chunks, above the configured scan limit ${limit}; increase the retrieval budget`);
    return rows.map((x) => ({ id: x.id, text: x.text, source: x.source, score: 0 }));
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
    const existing = await fetch(`${this.url}/collections/tenant_knowledge`, { headers: this.headers() });
    if (existing.ok) {
      const body = await existing.json() as { result?: { config?: { params?: { vectors?: { size?: number } | Record<string, { size?: number }> } } } };
      const vectors = body.result?.config?.params?.vectors;
      if (!vectors || typeof vectors !== 'object') throw new Error('Qdrant collection configuration is missing vector dimensions');
      if (!('size' in vectors)) throw new Error('Qdrant collection uses named vectors, but this adapter requires a single unnamed vector; create a compatible collection or update the adapter');
      const current = vectors.size;
      if (typeof current !== 'number' || current !== dim) {
        throw new Error(`Qdrant collection dimension mismatch: collection=${String(current)}, embedding=${dim}; recreate/reindex the collection with matching dimensions`);
      }
      return;
    }
    const created = await fetch(`${this.url}/collections/tenant_knowledge`, {
      method: 'PUT', headers: this.headers(),
      body: JSON.stringify({ vectors: { size: dim, distance: 'Cosine' } }),
    });
    if (!created.ok && created.status !== 409) throw new Error(`Qdrant collection setup failed (${created.status})`);
  }
  private async scrollIds(filter: Record<string, unknown>, limit = 100000): Promise<Array<string | number>> {
    const ids: Array<string | number> = [];
    let offset: string | number | undefined;
    while (ids.length < limit) {
      const r = await fetch(`${this.url}/collections/tenant_knowledge/points/scroll`, {
        method: 'POST', headers: this.headers(),
        body: JSON.stringify({ limit: Math.min(256, limit - ids.length), with_payload: false, filter, ...(offset === undefined ? {} : { offset }) }),
      });
      if (!r.ok) throw new Error(`Qdrant scroll failed (${r.status})`);
      const j = await r.json() as { result?: { points?: { id: string | number }[]; next_page_offset?: string | number | null } };
      const page = j.result?.points ?? [];
      ids.push(...page.map((point) => point.id));
      offset = j.result?.next_page_offset ?? undefined;
      if (page.length === 0 || offset === undefined) break;
    }
    if (ids.length >= limit && offset !== undefined) throw new Error(`Qdrant scan reached safety limit (${limit}); refusing destructive replacement`);
    return ids;
  }
  private async deletePointIds(ids: Array<string | number>) {
    for (let i = 0; i < ids.length; i += 256) {
      const response = await fetch(`${this.url}/collections/tenant_knowledge/points/delete?wait=true`, {
        method: 'POST', headers: this.headers(), body: JSON.stringify({ points: ids.slice(i, i + 256) }),
      });
      if (!response.ok) throw new Error(`Qdrant point cleanup failed (${response.status})`);
    }
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
    if (r[0]) await this.ensure(r[0].embedding.length);
    else {
      const check = await fetch(`${this.url}/collections/tenant_knowledge`, { headers: this.headers() });
      if (check.status === 404) return;
      if (!check.ok) throw new Error(`Qdrant collection check failed (${check.status})`);
    }
    const filter = { must: [{ key: 'clientId', match: { value: c } }, { key: 'source', match: { value: source } }] };
    const existingIds = await this.scrollIds(filter);
    // Upsert first. If this fails, keep the old source points searchable.
    if (r.length) await this.upsert(c, r);
    const incoming = new Set(r.map((row) => row.id));
    const stale = existingIds.filter((id) => !incoming.has(String(id)));
    if (stale.length) await this.deletePointIds(stale);
  }
  async replaceTenant(c: string, r: VectorRecord[]) {
    if (r[0]) await this.ensure(r[0].embedding.length);
    else {
      const check = await fetch(`${this.url}/collections/tenant_knowledge`, { headers: this.headers() });
      if (check.status === 404) return;
      if (!check.ok) throw new Error(`Qdrant collection check failed (${check.status})`);
    }
    const filter = { must: [{ key: 'clientId', match: { value: c } }] };
    const existingIds = await this.scrollIds(filter);
    // All chunks are upserted before stale chunks are removed. Partial failures
    // may leave extra old rows, but do not destroy the previous searchable index.
    for (let i = 0; i < r.length; i += 128) await this.upsert(c, r.slice(i, i + 128));
    const incoming = new Set(r.map((row) => row.id));
    const stale = existingIds.filter((id) => !incoming.has(String(id)));
    if (stale.length) await this.deletePointIds(stale);
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
  async listTenant(c: string, limit = 2000) {
    const points: { id: string | number; payload?: { text?: string; source?: string } }[] = [];
    let offset: string | number | undefined;
    const safeLimit = Math.max(0, Math.min(limit, 100000));
    while (points.length < safeLimit) {
      const r = await fetch(`${this.url}/collections/tenant_knowledge/points/scroll`, {
        method: 'POST', headers: this.headers(),
        body: JSON.stringify({ limit: Math.min(256, safeLimit - points.length), with_payload: true, filter: { must: [{ key: 'clientId', match: { value: c } }] }, ...(offset === undefined ? {} : { offset }) }),
      });
      if (!r.ok) throw new Error(`Qdrant tenant scan failed (${r.status})`);
      const j = await r.json() as { result?: { points?: typeof points; next_page_offset?: string | number | null } };
      const page = j.result?.points ?? [];
      points.push(...page);
      offset = j.result?.next_page_offset ?? undefined;
      if (!page.length || offset === undefined) break;
    }
    if (points.length >= safeLimit && safeLimit > 0 && offset !== undefined) throw new Error(`Qdrant tenant scan reached configured limit (${safeLimit}); use a larger explicit limit to avoid partial recall`);
    return points.map((x) => ({ id: String(x.id), text: String(x.payload?.text ?? ''), source: String(x.payload?.source ?? ''), score: 0 }));
  }
  async deleteTenant(c: string) {
    const collection = await fetch(`${this.url}/collections/tenant_knowledge`, { headers: this.headers() });
    if (collection.status === 404) return; // idempotent cleanup for an uninitialized index
    if (!collection.ok) throw new Error(`Qdrant collection check failed (${collection.status})`);
    const r = await fetch(`${this.url}/collections/tenant_knowledge/points/delete?wait=true`, {
      method: 'POST', headers: this.headers(),
      body: JSON.stringify({ filter: { must: [{ key: 'clientId', match: { value: c } }] } }),
    });
    if (!r.ok) throw new Error(`Qdrant tenant deletion failed (${r.status})`);
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

import { createClient, type SupabaseClient } from '@supabase/supabase-js';
/** pgvector adapter. Every operation supplies client_id; SQL RPC also enforces it. */
export class SupabaseVector implements VectorProvider {
  readonly name = 'supabase-pgvector';
  private readonly client: SupabaseClient<any>;
  constructor(urlOrClient: string | SupabaseClient<any>, key?: string, private readonly dimensions = 128) {
    this.client =
      typeof urlOrClient === 'string'
        ? createClient(urlOrClient, key ?? '', {
            auth: { persistSession: false, autoRefreshToken: false },
          })
        : urlOrClient;
  }
  // Keep configured dimensions aligned with the SQL vector(N) declaration.
  private validate(v: number[]) {
    if (v.length !== this.dimensions)
      throw new Error(`pgvector embedding must have exactly ${this.dimensions} dimensions; received ${v.length}. Configure the embedding provider or reindex with a matching schema.`);
    return v;
  }
  async upsert(clientId: string, rows: VectorRecord[]) {
    if (!rows.length) return;
    const { error } = await this.client.from('knowledge_vectors').upsert(
      rows.map((x) => ({
        id: x.id,
        client_id: clientId,
        source: x.source,
        content: x.text,
        embedding: this.validate(x.embedding),
      })),
      { onConflict: 'id' },
    );
    if (error) throw new Error(`pgvector upsert failed: ${error.message}`);
  }
  async replaceSource(clientId: string, source: string, rows: VectorRecord[]) {
    const { data: prior, error: readError } = await this.client.from('knowledge_vectors')
      .select('id').eq('client_id', clientId).eq('source', source);
    if (readError) throw new Error(`pgvector source listing failed: ${readError.message}`);
    // Upsert replacement rows first so a provider failure cannot erase the only copy.
    await this.upsert(clientId, rows);
    const incoming = new Set(rows.map((row) => row.id));
    const stale = (prior ?? []).map((row: any) => String(row.id)).filter((id: string) => !incoming.has(id));
    for (let i = 0; i < stale.length; i += 200) {
      const { error } = await this.client.from('knowledge_vectors').delete().eq('client_id', clientId).in('id', stale.slice(i, i + 200));
      if (error) throw new Error(`pgvector stale source cleanup failed: ${error.message}`);
    }
  }
  async replaceTenant(clientId: string, rows: VectorRecord[]) {
    const priorIds: string[] = [];
    let from = 0;
    const pageSize = 500;
    while (true) {
      const { data, error } = await this.client.from('knowledge_vectors').select('id')
        .eq('client_id', clientId).order('created_at', { ascending: true }).range(from, from + pageSize - 1);
      if (error) throw new Error(`pgvector tenant listing failed: ${error.message}`);
      const page = data ?? [];
      priorIds.push(...page.map((row: any) => String(row.id)));
      if (page.length < pageSize) break;
      from += pageSize;
      if (from >= 100000) throw new Error('pgvector tenant index exceeds safe rebuild limit; refusing destructive replacement');
    }
    // New index rows are committed first; stale IDs are removed only afterward.
    for (let i = 0; i < rows.length; i += 200) await this.upsert(clientId, rows.slice(i, i + 200));
    const incoming = new Set(rows.map((row) => row.id));
    const stale = priorIds.filter((id) => !incoming.has(id));
    for (let i = 0; i < stale.length; i += 200) {
      const { error } = await this.client.from('knowledge_vectors').delete().eq('client_id', clientId).in('id', stale.slice(i, i + 200));
      if (error) throw new Error(`pgvector stale tenant cleanup failed: ${error.message}`);
    }
  }
  async search(clientId: string, v: number[], topK: number, min: number) {
    const { data, error } = await this.client.rpc('match_knowledge_vectors', {
      query_embedding: this.validate(v),
      match_client_id: clientId,
      match_threshold: min,
      match_count: topK,
    });
    if (error) throw new Error(`pgvector search failed: ${error.message}`);
    return (data ?? []).map((x: any) => ({
      id: String(x.id),
      text: String(x.text ?? x.content),
      source: String(x.source),
      score: Number(x.similarity ?? x.score),
    }));
  }
  async listTenant(clientId: string, limit = 2000) {
    const safeLimit = Math.max(0, Math.min(limit, 100000));
    if (safeLimit === 0) return [];
    const rows: any[] = [];
    const pageSize = 500;
    for (let from = 0; from < safeLimit; from += pageSize) {
      const to = Math.min(from + pageSize - 1, safeLimit - 1);
      const { data, error } = await this.client.from('knowledge_vectors').select('id,content,source')
        .eq('client_id', clientId).order('created_at', { ascending: true }).order('id', { ascending: true }).range(from, to);
      if (error) throw new Error(`pgvector tenant scan failed: ${error.message}`);
      const page = data ?? [];
      rows.push(...page);
      if (page.length < to - from + 1) return rows.map((x) => ({ id: String(x.id), text: String(x.content), source: String(x.source), score: 0 }));
    }
    const { data: extra, error: extraError } = await this.client.from('knowledge_vectors').select('id')
      .eq('client_id', clientId).order('created_at', { ascending: true }).order('id', { ascending: true }).range(safeLimit, safeLimit);
    if (extraError) throw new Error(`pgvector tenant scan verification failed: ${extraError.message}`);
    if (extra?.length) throw new Error(`pgvector tenant scan reached configured limit (${safeLimit}); increase it to avoid partial recall`);
    return rows.map((x) => ({ id: String(x.id), text: String(x.content), source: String(x.source), score: 0 }));
  }
  async deleteTenant(clientId: string) {
    const { error } = await this.client
      .from('knowledge_vectors')
      .delete()
      .eq('client_id', clientId);
    if (error) throw new Error(`pgvector tenant delete failed: ${error.message}`);
  }
  async health(): Promise<ProviderHealth> {
    const { error } = await this.client.from('knowledge_vectors').select('id').limit(1);
    return { provider: this.name, connected: !error, detail: error?.message };
  }
}
