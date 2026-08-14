import type { FastifyInstance, FastifyRequest } from 'fastify';
import type { Context } from './context.js';
import { hashKey, originMatches } from '../lib/security.js';
import { AppError } from '../lib/errors.js';
import { chatSchema, leadSchema } from '../domain/schemas.js';
export async function publicRoutes(app: FastifyInstance, c: Context) {
  const tenant = async (req: FastifyRequest) => {
    const key = req.headers['x-api-key'],
      origin = req.headers.origin;
    if (typeof key !== 'string')
      throw new AppError(401, 'INVALID_API_KEY', 'Valid public client credential required');
    const client = c.repo.getClientByKeyHash(hashKey(key));
    if (!client?.enabled)
      throw new AppError(401, 'INVALID_API_KEY', 'Valid public client credential required');
    if (typeof origin !== 'string' || !originMatches(origin, c.repo.domains(client.id)))
      throw new AppError(403, 'ORIGIN_FORBIDDEN', 'Origin is not allowed');
    (req as FastifyRequest & { tenantId: string }).tenantId = client.id;
  };
  app.addHook('preHandler', tenant);
  app.get('/config', async (req) => {
    const x = c.repo.getClient((req as FastifyRequest & { tenantId: string }).tenantId)!;
    return {
      assistantName: x.config.assistantName,
      welcomeMessage: x.config.welcomeMessage,
      widget: x.config.widget ?? {},
      theme: x.config.theme ?? {},
    };
  });
  app.post('/chat', async (req) => {
    const b = chatSchema.parse(req.body);
    return c.chat.chat(
      (req as FastifyRequest & { tenantId: string }).tenantId,
      b.message,
      b.sessionId,
    );
  });
  app.post('/leads', async (req) => {
    const b = leadSchema.parse(req.body);
    const tid = (req as FastifyRequest & { tenantId: string }).tenantId;
    const { conversationId, ...data } = b;
    if (conversationId && !c.repo.conversationForClient(tid, conversationId))
      throw new AppError(404, 'CONVERSATION_NOT_FOUND', 'Conversation not found');
    const id = c.repo.saveLead(tid, conversationId, data as Record<string, string>);
    const client = c.repo.getClient(tid)!;
    await c.notification.notify(
      client.config.notificationEmail ?? client.config.teamEmail,
      `New lead for ${client.name}`,
      { id, ...data },
    );
    c.repo.audit(tid, 'lead.created', { leadId: id });
    return { id, status: 'accepted' };
  });
}
