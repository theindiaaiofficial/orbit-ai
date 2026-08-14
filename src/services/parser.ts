import pdf from 'pdf-parse';
import mammoth from 'mammoth';
import path from 'node:path';
import { AppError } from '../lib/errors.js';
const allowed = new Set(['.txt', '.md', '.pdf', '.docx']);
export async function parseDocument(name: string, data: Buffer) {
  const ext = path.extname(name).toLowerCase();
  if (!allowed.has(ext)) throw new AppError(415, 'UNSUPPORTED_MEDIA_TYPE', 'Unsupported file type');
  if (ext === '.pdf') return (await pdf(data)).text;
  if (ext === '.docx') return (await mammoth.extractRawText({ buffer: data })).value;
  return data.toString('utf8');
}
export function chunks(text: string, size = 900, overlap = 120) {
  const clean = text.replace(/\0/g, '').replace(/\s+/g, ' ').trim();
  if (!clean) return [];
  const out: string[] = [];
  for (let i = 0; i < clean.length; i += size - overlap) {
    out.push(clean.slice(i, i + size));
    if (i + size >= clean.length) break;
  }
  return out;
}
