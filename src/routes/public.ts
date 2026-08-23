import type { FastifyInstance, FastifyRequest } from 'fastify';
import type { Context } from './context.js';
import { TtsProviderError } from '../providers/tts.js';
import { hashKey, originMatches } from '../lib/security.js';
import { AppError } from '../lib/errors.js';
import { chatSchema, leadSchema, ttsSchema } from '../domain/schemas.js';

type TenantRequest = FastifyRequest & { tenantId: string };
export async function publicRoutes(app: FastifyInstance, c: Context) {
  const tenant = async (req: FastifyRequest) => {
    const key = req.headers['x-api-key'], origin = req.headers.origin;
    if (typeof key !== 'string') throw new AppError(401, 'INVALID_API_KEY', 'Valid public client credential required');
    const client = await c.repo.getClientByKeyHash(hashKey(key));
    if (!client?.enabled) throw new AppError(401, 'INVALID_API_KEY', 'Valid public client credential required');
    if (typeof origin !== 'string' || !originMatches(origin, await c.repo.domains(client.id))) throw new AppError(403, 'ORIGIN_FORBIDDEN', 'Origin is not allowed');
    (req as TenantRequest).tenantId = client.id;
  };
  app.addHook('preHandler', tenant);
  app.get('/config', async (req) => {
    const x = await c.repo.getClient((req as TenantRequest).tenantId);
    if (!x) throw new AppError(404, 'NOT_FOUND', 'Client not found');
    return { assistantName: x.config.assistantName, welcomeMessage: x.config.welcomeMessage, widget: x.config.widget ?? {}, theme: x.config.theme ?? {} };
  });
  app.post('/chat', async (req) => {
    const b = chatSchema.parse(req.body);
    return c.chat.chat((req as TenantRequest).tenantId, b.message, b.sessionId);
  });
  app.post('/chat/stream', async (req, reply) => {
    const b = chatSchema.parse(req.body);
    // Hijacking bypasses Fastify's normal response hooks, so preserve the
    // already-authorized exact origin for browser-readable tenant streaming.
    const origin = req.headers.origin;
    if (typeof origin === 'string') {
      reply.raw.setHeader('access-control-allow-origin', origin);
      reply.raw.setHeader('vary', 'Origin');
    }
    reply.hijack();
    reply.raw.statusCode = 200;
    reply.raw.setHeader('content-type', 'text/event-stream; charset=utf-8');
    reply.raw.setHeader('cache-control', 'no-cache, no-transform');
    reply.raw.setHeader('connection', 'keep-alive');
    reply.raw.flushHeaders?.();
    try {
      const result = await c.chat.stream((req as TenantRequest).tenantId, b.message, b.sessionId, (token) => {
        reply.raw.write(`event: token\ndata: ${JSON.stringify({ token })}\n\n`);
      });
      reply.raw.write(`event: done\ndata: ${JSON.stringify(result)}\n\n`);
    } catch {
      reply.raw.write(`event: error\ndata: ${JSON.stringify({ message: 'Unable to complete the response' })}\n\n`);
    } finally { reply.raw.end(); }
  });
  app.post('/tts', async (req, reply) => {
    const b = ttsSchema.parse(req.body);
    try {
      const audio = await c.tts.synthesize(b.text);
      return reply.type(audio.contentType).header('cache-control', 'no-store').send(Buffer.from(audio.body));
    } catch (error) {
      // Keep diagnostics server-side and non-secret; the public contract remains generic.
      const cause = error instanceof Error ? error.cause : undefined;
      if (error instanceof TtsProviderError) app.log.warn({ event: 'tts.provider_error', status: error.status }, 'TTS provider rejected synthesis');
      else if (cause instanceof TtsProviderError) app.log.warn({ event: 'tts.provider_error', status: cause.status, detail: cause.detail.replace(/bearer\s+\S+/ig, 'Bearer [redacted]').slice(0, 240) }, 'TTS provider rejected synthesis');
      else app.log.warn({ event: 'tts.request_error' }, 'TTS synthesis failed');
      throw new AppError(503, 'TTS_UNAVAILABLE', 'Audio is temporarily unavailable');
    }
  });
  app.post('/leads', async (req) => {
    const b = leadSchema.parse(req.body); const tid = (req as TenantRequest).tenantId; const { conversationId, ...data } = b;
    if (conversationId && !(await c.repo.conversationForClient(tid, conversationId))) throw new AppError(404, 'CONVERSATION_NOT_FOUND', 'Conversation not found');
    const id = await c.repo.saveLead(tid, conversationId, data as Record<string, string>); const client = (await c.repo.getClient(tid))!;
    await c.notification.notify(client.config.notificationEmail ?? client.config.teamEmail, `New lead for ${client.name}`, { id, ...data }, tid);
    await c.repo.audit(tid, 'lead.created', { leadId: id }); return { id, status: 'accepted' };
  });
}
