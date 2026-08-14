import fs from 'node:fs/promises';
import path from 'node:path';
import nodemailer from 'nodemailer';
import crypto from 'node:crypto';
import type { ProviderHealth } from '../domain/types.js';
export interface NotificationProvider {
  readonly name: string;
  notify(to: string | undefined, subject: string, data: unknown): Promise<void>;
  health(): Promise<ProviderHealth>;
}
export class OutboxNotification implements NotificationProvider {
  name = 'file-outbox';
  constructor(private dir: string) {}
  async notify(to: string | undefined, subject: string, data: unknown) {
    await fs.mkdir(this.dir, { recursive: true });
    await fs.writeFile(
      path.join(this.dir, `${Date.now()}-${crypto.randomUUID()}.json`),
      JSON.stringify({ to: to ?? null, subject, data, createdAt: new Date().toISOString() }),
    );
  }
  async health() {
    return { provider: this.name, connected: true };
  }
}
export class SmtpNotification implements NotificationProvider {
  name = 'smtp';
  private tx;
  constructor(options: {
    host?: string;
    port: number;
    user?: string;
    pass?: string;
    from?: string;
  }) {
    this.tx = nodemailer.createTransport({
      host: options.host,
      port: options.port,
      secure: options.port === 465,
      auth: options.user ? { user: options.user, pass: options.pass } : undefined,
    });
    this.from = options.from;
  }
  private from?: string;
  async notify(to: string | undefined, subject: string, data: unknown) {
    if (!to) throw new Error('Notification email is not configured');
    await this.tx.sendMail({ from: this.from, to, subject, text: JSON.stringify(data, null, 2) });
  }
  async health() {
    try {
      await this.tx.verify();
      return { provider: this.name, connected: true };
    } catch {
      return { provider: this.name, connected: false };
    }
  }
}
