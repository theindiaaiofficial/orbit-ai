import fs from 'node:fs/promises';
import path from 'node:path';
import { safeSegment } from '../lib/security.js';
export interface ObjectStorage {
  put(clientId: string, name: string, data: Buffer): Promise<string>;
  list(clientId: string): Promise<string[]>;
  read(clientId: string, name: string): Promise<Buffer>;
  remove(clientId: string, name: string): Promise<void>;
  removeTenant(clientId: string): Promise<void>;
}
export class LocalStorage implements ObjectStorage {
  constructor(private root: string) {}
  private dir(id: string) {
    return path.join(this.root, safeSegment(id));
  }
  private file(id: string, name: string) {
    const clean = path.basename(name).replace(/[^a-zA-Z0-9._-]/g, '_');
    if (!clean || clean === '.' || clean === '..') throw new Error('Unsafe filename');
    return path.join(this.dir(id), clean);
  }
  async put(id: string, n: string, d: Buffer) {
    await fs.mkdir(this.dir(id), { recursive: true });
    const f = this.file(id, n);
    await fs.writeFile(f, d, { flag: 'wx' }).catch(async (e) => {
      if ((e as NodeJS.ErrnoException).code === 'EEXIST') await fs.writeFile(f, d);
      else throw e;
    });
    return path.basename(f);
  }
  async list(id: string) {
    return fs.readdir(this.dir(id)).catch(() => []);
  }
  async read(id: string, n: string) {
    return fs.readFile(this.file(id, n));
  }
  async remove(id: string, n: string) {
    await fs.rm(this.file(id, n), { force: true });
  }
  async removeTenant(id: string) {
    await fs.rm(this.dir(id), { recursive: true, force: true });
  }
}
