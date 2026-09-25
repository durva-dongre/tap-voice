import { n, json, type Env } from "../lib";
import { apiKeyOk } from "../auth";
import { channels } from "../channels";

const LANGS = new Set(["en", "hi", "mr", "pa", "kn"]);
const VOICE = /^[a-z0-9_]{1,40}$/;
type In = { text?: string; language?: string; contact_id?: string; channel?: string; source?: string; voice_id?: string; dedup_key?: string };

function validate(j: In, env: Env): string | null {
  const t = (j.text ?? "").trim();
  if (!t || t.length > n(env.MAX_TEXT_CHARS)) return "text empty or too long";
  if (!j.language || !LANGS.has(j.language)) return "unsupported language";
  const ch = channels[j.channel ?? "whatsapp"];
  if (!ch) return "unknown channel";
  if (!j.contact_id || !ch.validateRecipient(j.contact_id)) return "bad contact_id";
  if (j.voice_id && !VOICE.test(j.voice_id)) return "bad voice_id";
  if (j.dedup_key && j.dedup_key.length > 128) return "dedup_key too long";
  return null;
}

export async function createJobs(req: Request, env: Env) {
  if (!(await apiKeyOk(req, env.INTAKE_API_KEY))) return json({ error: "unauthorized" }, 401);
  let body: In | In[];
  try { body = await req.json(); } catch { return json({ error: "invalid json" }, 400); }
  const list = Array.isArray(body) ? body : [body];
  if (list.length < 1 || list.length > 100) return json({ error: "1-100 jobs per request" }, 400);
  for (const [i, j] of list.entries()) { const e = validate(j, env); if (e) return json({ error: e, index: i }, 400); }
  try {
    const res = await env.DB.batch(list.map((j) => env.DB.prepare(
      `INSERT INTO jobs (dedup_key,source,channel,contact_id,text,language,voice_id) VALUES (?1,?2,?3,?4,?5,?6,?7)
       ON CONFLICT(dedup_key) DO UPDATE SET dedup_key=dedup_key RETURNING id, status`)
      .bind(j.dedup_key ?? null, j.source ?? "api", j.channel ?? "whatsapp", j.contact_id, j.text!.trim(), j.language, j.voice_id ?? null)));
    const out = res.map((r) => { const x = r.results[0] as { id: number; status: string }; return { job_id: x.id, status: x.status }; });
    return json(Array.isArray(body) ? { jobs: out } : out[0], 201);
  } catch (e) {
    console.error("intake D1 failure", e);
    return json({ error: "queue write failed, retry" }, 503);
  }
}

export async function getJob(req: Request, env: Env, id: string) {
  if (!(await apiKeyOk(req, env.INTAKE_API_KEY))) return json({ error: "unauthorized" }, 401);
  const r = await env.DB.prepare(
    "SELECT id,status,attempts,error,delivery_status,created_at,sent_at,channel_msg_id FROM jobs WHERE id=?").bind(id).first();
  return r ? json(r) : json({ error: "not found" }, 404);
}