import { z } from 'zod';
import path from 'node:path';
import { loadLlmConfig, type LlmConfig } from './llm.js';
const schema = z.object({
  NODE_ENV: z.enum(['development', 'test', 'production']).default('development'),
  HOST: z.string().default('0.0.0.0'),
  PORT: z.coerce.number().int().positive().default(3000),
  ADMIN_API_KEY: z.string().min(16).default('local-development-admin-key'),
  DATA_DIR: z.string().default('./data'),
  PUBLIC_BASE_URL: z.string().url().default('http://localhost:3000'),
  EMBEDDING_PROVIDER: z.enum(['local', 'openai']).default('local'),
  VECTOR_PROVIDER: z.enum(['local', 'qdrant']).default('local'),
  OPENAI_BASE_URL: z.string().url().default('https://api.openai.com/v1'),
  OPENAI_API_KEY: z.string().optional(),
  OPENAI_EMBEDDING_MODEL: z.string().default('text-embedding-3-small'),
  QDRANT_URL: z.string().url().default('http://localhost:6333'),
  QDRANT_API_KEY: z.string().optional(),
  NOTIFICATION_PROVIDER: z.enum(['outbox', 'smtp']).default('outbox'),
  SMTP_HOST: z.string().optional(),
  SMTP_PORT: z.coerce.number().default(587),
  SMTP_USER: z.string().optional(),
  SMTP_PASS: z.string().optional(),
  SMTP_FROM: z.string().optional(),
  CORS_ORIGINS: z.string().default(''),
});
export type Env = Omit<z.infer<typeof schema>, 'CORS_ORIGINS'> & {
  llm: LlmConfig;
  CORS_ORIGINS: string[];
};
export function loadEnv(input: NodeJS.ProcessEnv = process.env): Env {
  const compatibleInput = {
    ...input,
    OPENAI_API_KEY: input.OPENAI_API_KEY || input.EMBEDDING_API_KEY,
    OPENAI_EMBEDDING_MODEL: input.OPENAI_EMBEDDING_MODEL || input.EMBEDDING_MODEL,
  };
  const e = schema.parse(compatibleInput);
  return {
    ...e,
    CORS_ORIGINS: e.CORS_ORIGINS.split(',')
      .map((x) => x.trim())
      .filter(Boolean),
    DATA_DIR: path.resolve(e.DATA_DIR),
    llm: loadLlmConfig(input),
  };
}
