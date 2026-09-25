import type { Env } from "../lib";
import { whatsapp } from "./whatsapp";

export interface SendResult { ok: boolean; id?: string; error?: string }

export interface Channel {
  validateRecipient(id: string): boolean;
  send(env: Env, a: { to: string; audio: ArrayBuffer; jobId: number }): Promise<SendResult>;
}

export const channels: Record<string, Channel> = { whatsapp };