import { n, json, type Env } from "../lib";
import { channels } from "../channels";
import { requeueStale, finalizeCounts, terminateRun } from "../scheduler";

const active = (s: string) => s === "starting" || s === "running";
type R = { ok: boolean; id?: string; error?: string };

export async function podRoute(req: Request, env: Env, runId: string, path: string, ctx: ExecutionContext) {
  const [, , action, id] = path.split("/");
  if (req.method !== "POST") return json({ error: "method" }, 405);
  switch (action) {
    case "claim": return claim(req, env, runId);
    case "beat": return beat(req, env, runId);
    case "deliver": return deliver(req, env, runId, id);
    case "fail": return fail(req, env, runId, id);
    case "complete": return complete(req, env, runId, ctx);
  }
  return json({ error: "not found" }, 404);
}

async function claim(req: Request, env: Env, runId: string) {
  const run = await env.DB.prepare("SELECT status, claimed FROM runs WHERE id=?").bind(runId).first<{ status: string; claimed: number }>();
  if (!run || !active(run.status)) return json({ error: "run not active" }, 409);
  const body = await req.json<{ limit?: number }>().catch(() => ({} as { limit?: number }));
  const room = n(env.MAX_JOBS_PER_CLAIM) - run.claimed;
  const lim = Math.max(0, Math.min(body.limit ?? room, room));
  if (lim === 0) return json({ jobs: [] });
  const [, res] = await env.DB.batch([
    env.DB.prepare("UPDATE runs SET status='running', first_beat_at=COALESCE(first_beat_at,datetime('now')), last_beat_at=datetime('now') WHERE id=? AND status IN ('starting','running')").bind(runId),
    env.DB.prepare(
      `UPDATE jobs SET status='claimed', run_id=?1, claimed_at=datetime('now'), attempts=attempts+1
       WHERE id IN (SELECT id FROM jobs WHERE status='pending' ORDER BY id LIMIT ?2)
         AND EXISTS (SELECT 1 FROM runs WHERE id=?1 AND status IN ('starting','running'))
       RETURNING id, text, language, voice_id`).bind(runId, lim),
    env.DB.prepare(
      `UPDATE runs SET claimed=(SELECT COUNT(*) FROM jobs WHERE run_id=?1) WHERE id=?1`).bind(runId),
  ]);
  return json({ jobs: res.results });
}

async function beat(req: Request, env: Env, runId: string) {
  const b = await req.json<{ phase?: string; remaining?: number; models?: string }>().catch(() => ({} as any));
  const r = await env.DB.prepare(
    `UPDATE runs SET status='running', last_beat_at=datetime('now'), first_beat_at=COALESCE(first_beat_at,datetime('now')),
       phase=?2, jobs_remaining=?3, models=?4 WHERE id=?1 AND status IN ('starting','running')`)
    .bind(runId, b.phase ?? null, b.remaining ?? null, b.models ?? null).run();
  return json({ ok: true, stop: r.meta.changes === 0 });
}

async function settle(env: Env, jobId: string | number, r: R) {
  if (r.ok) {
    return env.DB.prepare("UPDATE jobs SET status='sent', channel_msg_id=?2, sent_at=datetime('now'), error=NULL WHERE id=?1")
      .bind(jobId, r.id ?? null).run();
  }
  const maxAttempts = n(env.MAX_ATTEMPTS);
  return env.DB.prepare(
    `UPDATE jobs SET
       status=CASE WHEN attempts < ?3 THEN 'pending' ELSE 'failed' END,
       run_id=CASE WHEN attempts < ?3 THEN NULL ELSE run_id END,
       error=?2
     WHERE id=?1`)
    .bind(jobId, (r.error ?? "unknown").slice(0, 300), maxAttempts).run();
}

async function deliver(req: Request, env: Env, runId: string, jobId: string) {
  const row = await env.DB.prepare(
    `SELECT j.status, j.channel, j.contact_id, r.status AS rs FROM jobs j JOIN runs r ON r.id=j.run_id
     WHERE j.id=?1 AND j.run_id=?2`).bind(jobId, runId).first<{ status: string; channel: string; contact_id: string; rs: string }>();
  if (!row) return json({ error: "not your job" }, 404);
  if (row.status === "sent") return json({ ok: true, dup: true });
  if (!active(row.rs)) return json({ error: "run not active" }, 409);
  if (row.status !== "claimed") return json({ error: "not claimed" }, 410);
  const audio = await req.arrayBuffer();
  if (audio.byteLength < 200 || audio.byteLength > 5_000_000) return json({ error: "bad audio size" }, 400);
  const r = await channels[row.channel].send(env, { to: row.contact_id, audio, jobId: Number(jobId) });
  await settle(env, jobId, r);
  return json({ ok: r.ok, error: r.ok ? undefined : r.error });
}

async function fail(req: Request, env: Env, runId: string, jobId: string) {
  const b = await req.json<{ error?: string }>().catch(() => ({} as any));
  const row = await env.DB.prepare("SELECT status FROM jobs WHERE id=?1 AND run_id=?2").bind(jobId, runId).first<{ status: string }>();
  if (row?.status === "claimed") await settle(env, jobId, { ok: false, error: b.error });
  return json({ ok: true });
}

async function complete(req: Request, env: Env, runId: string, ctx: ExecutionContext) {
  const b = await req.json<{ reason?: string; gpu_seconds?: number }>().catch(() => ({} as any));
  const run = await env.DB.prepare("SELECT pod_id FROM runs WHERE id=?").bind(runId).first<{ pod_id: string | null }>();
  if (!run) return json({ error: "no such run" }, 404);
  await env.DB.prepare(
    `UPDATE runs SET status='done', ended_at=datetime('now'), reason=?2, gpu_seconds=?3, phase='ended'
     WHERE id=?1 AND status IN ('starting','running')`).bind(runId, String(b.reason ?? "").slice(0, 80), b.gpu_seconds ?? null).run();
  await requeueStale(env);
  await finalizeCounts(env, runId);
  ctx.waitUntil(terminateRun(env, runId, run.pod_id).catch((e) => console.error("terminate after complete", e)));
  return json({ ok: true });
}