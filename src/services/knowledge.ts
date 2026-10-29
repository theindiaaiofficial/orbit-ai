import crypto from 'node:crypto';
import path from 'node:path';
import type { EmbeddingProvider } from '../providers/embedding.js';
import type { VectorProvider } from '../providers/vector.js';
import type { ObjectStorage } from '../providers/storage.js';
import { chunks, parseDocument } from './parser.js';
import type { Repository } from '../repositories/repository.js';

/** Deterministic RFC-4122-shaped UUID acceptable to Postgres UUID and Qdrant. */
function stableChunkUuid(identity: string): string {
  const bytes = crypto.createHash('sha256').update(identity).digest().subarray(0, 16);
  bytes[6] = (bytes[6]! & 0x0f) | 0x50;
  bytes[8] = (bytes[8]! & 0x3f) | 0x80;
  const h = bytes.toString('hex');
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20, 32)}`;
}

/** Match both storage adapters so object names and vector source metadata agree. */
function canonicalFilename(name: string): string {
  const clean = path.basename(name).replace(/[^a-zA-Z0-9._-]/g, '_');
  if (!clean || clean === '.' || clean === '..' || clean.length > 180) throw new Error('Unsafe filename');
  return clean;
}

export class KnowledgeService {
  constructor(
    private storage: ObjectStorage,
    private embeddings: EmbeddingProvider,
    private vectors: VectorProvider,
    private repo?: Repository,
  ) {}

  async upload(clientId: string, name: string, data: Buffer) {
    const filename = canonicalFilename(name);
    const text = await parseDocument(filename, data);
    if (!text.trim()) throw new Error('Document contains no text');
    const parts = chunks(text);
    if (!parts.length) throw new Error('Document produced no indexable chunks');
    const vectors = await this.embeddings.embed(parts);
    if (vectors.length !== parts.length || vectors.some((v) => !Array.isArray(v) || !v.length || v.some((n) => !Number.isFinite(n)))) {
      throw new Error('Embedding provider returned an invalid vector set');
    }

    // Storage returns the canonical persisted object name. Write/validate the
    // vector replacement before recording ready metadata. On failure, remove a
    // newly stored object only when there was no existing object to overwrite.
    // Existing files are kept on failures where possible; replaceSource itself
    // upserts before stale-record cleanup for remote providers.
    const existingNames = await this.storage.list(clientId);
    const existed = existingNames.includes(filename);
    const previousData = existed ? await this.storage.read(clientId, filename) : undefined;
    const previousParts = previousData
      ? chunks(await parseDocument(filename, previousData))
      : [];
    const previousEmbeddings = previousParts.length ? await this.embeddings.embed(previousParts) : [];
    try {
      await this.storage.put(clientId, filename, data);
      await this.vectors.replaceSource(
        clientId,
        filename,
        parts.map((x, i) => ({
          id: stableChunkUuid(`${clientId}\0${filename}\0${i}\0${x}`),
          clientId, source: filename, text: x, embedding: vectors[i]!,
        })),
      );
      await this.repo?.saveKnowledge(clientId, filename, data.byteLength, parts.length, 'ready');
    } catch (error) {
      // Best-effort compensation keeps file contents and vectors aligned. The
      // original provider error remains the surfaced cause if compensation fails.
      try {
        if (previousData !== undefined) {
          await this.storage.put(clientId, filename, previousData);
          await this.vectors.replaceSource(clientId, filename, previousParts.map((text, i) => ({
            id: stableChunkUuid(`${clientId}\0${filename}\0${i}\0${text}`), clientId, source: filename, text,
            embedding: previousEmbeddings[i]!,
          })));
        } else {
          await this.vectors.replaceSource(clientId, filename, []);
          await this.storage.remove(clientId, filename);
        }
      } catch { /* a durable cleanup/reindex is required if provider rollback fails */ }
      // If this was a brand-new file, keep an explicit failed status for retry.
      // If it was an overwrite and rollback restored the prior file, preserve its
      // existing metadata instead of incorrectly labelling valid old data failed.
      if (previousData === undefined) {
        try { await this.repo?.saveKnowledge(clientId, filename, data.byteLength, parts.length, 'failed'); } catch { /* best effort status */ }
      }
      throw error;
    }
    return { filename, size: data.byteLength, chunks: parts.length, embeddingStatus: 'ready' };
  }

  async list(clientId: string) {
    const metadata = (await this.repo?.knowledge(clientId)) as
      | Array<{ filename: string; size: number; chunks: number; status: string; updatedAt: string }>
      | undefined;
    if (metadata?.length) return metadata.map((x) => ({ ...x, bytes: x.size, embeddingStatus: x.status }));
    const names = await this.storage.list(clientId);
    return Promise.all(names.map(async (filename) => {
      const size = (await this.storage.read(clientId, filename)).byteLength;
      return { filename, size, bytes: size, chunks: 0, embeddingStatus: 'unknown' };
    }));
  }

  async remove(clientId: string, inputName: string) {
    const filename = canonicalFilename(inputName);
    const existed = (await this.storage.list(clientId)).includes(filename);
    const previousData = existed ? await this.storage.read(clientId, filename) : undefined;
    const previousParts = previousData === undefined ? [] : chunks(await parseDocument(filename, previousData));
    const previousEmbeddings = previousParts.length ? await this.embeddings.embed(previousParts) : [];

    // Remove source vectors first. If storage or metadata deletion then fails,
    // best-effort compensation restores the prior object and vector rows.
    await this.vectors.replaceSource(clientId, filename, []);
    try {
      await this.storage.remove(clientId, filename);
      await this.repo?.removeKnowledge(clientId, filename);
    } catch (error) {
      if (previousData !== undefined) {
        try {
          await this.storage.put(clientId, filename, previousData);
          await this.vectors.replaceSource(clientId, filename, previousParts.map((text, i) => ({
            id: stableChunkUuid(`${clientId}\0${filename}\0${i}\0${text}`), clientId, source: filename, text,
            embedding: previousEmbeddings[i]!,
          })));
        } catch { /* durable cleanup/reindex is required if remote rollback fails */ }
      }
      throw error;
    }
    return this.rebuild(clientId);
  }

  async rebuild(clientId: string) {
    const names = (await this.storage.list(clientId)).map(canonicalFilename);
    const all: { source: string; text: string; sourceChunkIndex: number }[] = [];
    for (const name of names) {
      const text = await parseDocument(name, await this.storage.read(clientId, name));
      chunks(text).forEach((chunk, sourceChunkIndex) => all.push({ source: name, text: chunk, sourceChunkIndex }));
    }
    const embeddings = all.length ? await this.embeddings.embed(all.map((x) => x.text)) : [];
    if (embeddings.length !== all.length || embeddings.some((v) => !Array.isArray(v) || !v.length || v.some((n) => !Number.isFinite(n)))) {
      throw new Error('Embedding provider returned an invalid vector set during rebuild');
    }
    const rows = all.map((x, i) => ({
      id: stableChunkUuid(`${clientId}\0${x.source}\0${x.sourceChunkIndex}\0${x.text}`),
      clientId, source: x.source, text: x.text, embedding: embeddings[i]!,
    }));
    await this.vectors.replaceTenant(clientId, rows);
    return { files: names.length, chunks: all.length };
  }
}
