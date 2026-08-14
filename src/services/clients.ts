import crypto from 'node:crypto';
import type { Repository } from '../repositories/repository.js';
import type { ObjectStorage } from '../providers/storage.js';
import type { VectorProvider } from '../providers/vector.js';
import type { ClientConfig } from '../domain/types.js';
import { generateApiKey, hashKey, normalizeDomain } from '../lib/security.js';
export class ClientService {
  constructor(
    private repo: Repository,
    private storage: ObjectStorage,
    private vectors: VectorProvider,
    private baseUrl: string,
  ) {}
  private credentials(apiKey: string) {
    console.log('[DEBUG][ClientService.credentials][START]', {
      apiKeyExists: Boolean(apiKey),
      apiKeyPreview: `${apiKey.slice(0, 8)}...`,
    });
    const widgetUrl = new URL('/widget.js', this.baseUrl).toString();

    console.log('[DEBUG][ClientService.credentials]', {
      apiKeyExists: Boolean(apiKey),
      apiKeyPreview: `${apiKey.slice(0, 8)}...`,
      widgetUrl,
      embedCodeHasPlaceholder: `<script src="${widgetUrl}" data-api-key="${apiKey}" async></script>`.includes('YOUR_API_KEY'),
    });

    const embedCode = `<script src="${widgetUrl}" data-api-key="${apiKey}" async></script>`;
    
    console.log('[DEBUG][ClientService.credentials][RESULT]', {
      apiKeyExists: Boolean(apiKey),
      apiKeyPreview: `${apiKey.slice(0, 8)}...`,
      widgetUrl,
      embedCodeHasPlaceholder: embedCode.includes('YOUR_API_KEY'),
      embedCodePreview: embedCode.replace(apiKey, `${apiKey.slice(0, 8)}...`),
    });

    return {
      apiKey,
      embedCode: `<script src="${widgetUrl}" data-api-key="${apiKey}" async></script>`,
      copyOnce: true as const,
    };
  }
  create(v: {
    name: string;
    slug: string;
    domains: string[];
    config: ClientConfig;
    prompt: string;
  }) {
    const id = crypto.randomUUID(),
      apiKey = generateApiKey();
    const client = this.repo.createClient({
      ...v,
      id,
      domains: v.domains.map(normalizeDomain),
      keyHash: hashKey(apiKey),
    });
    this.repo.audit(id, 'client.created', { slug: v.slug });
    return { client, ...this.credentials(apiKey) };
  }
  rotate(id: string) {
    if (!this.repo.getClient(id)) return;
    const apiKey = generateApiKey();
    this.repo.rotateKey(id, hashKey(apiKey));
    this.repo.audit(id, 'key.rotated', {});
    return this.credentials(apiKey);
  }
  async remove(id: string) {
    const ok = this.repo.deleteClient(id);
    if (ok) {
      await this.storage.removeTenant(id);
      await this.vectors.deleteTenant(id);
    }
    return ok;
  }
}
