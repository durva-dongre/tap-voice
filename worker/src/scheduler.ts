import { n, notify, type Env } from "./lib";
import { createPod, terminatePod, findPodId, listTtsPods } from "./runpod";

const ACTIVE = "('starting','running')";
const toMin = (s: string) => { const [h, m] = s.split(":").map(Number); return h * 60 + m; };

export function local(env: Env, ms = Date.now()) {
  const iso = new Date(ms + n(env.TZ_OFFSET_MIN) * 60000).toISOString();
  return { date: iso.slice(0, 10), hhmm: iso.slice(11, 16), min: toMin(iso.slice(11, 16)) };
}

export async function tick(env: Env, ms: number) {
  const safe = (name: string, p: Promise<unknown>) => p.catch((e) => console.error(name, e));
  await safe("requeue", requeueStale(env));
  await safe("watchdog", watchdog(env));
  await safe("start", maybeStart(env, ms));
  if (new Date(ms).getUTCMinutes() % 15 === 0) await safe("sweep", sweepOrphans(env));
  if (local(env, ms).hhmm === env.MAINT_LOCAL_TIME) await safe("maint", maintenance(env));
}

async function maybeStart(env: Env, ms: number) {
  const { date, min } = local(env, ms);
  const slot = env.SLOTS.split(",").map((s) => s.trim())
    .find((s) => min >= toMin(s) && min < toMin(s) + n(env.SLOT_WINDOW_MIN));
  if (!slot) return;
  const h = await env.DB.prepare(
    `SELECT SUM(status!='start_failed') ok, SUM(status='start_failed') bad FROM runs WHERE slot_date=?1 AND slot=?2`)
    .bind(date, slot).first<{ ok: number | null; bad: number | null }>();
  if ((h?.ok ?? 0) > 0 || (h?.bad ?? 0) >= 3) return;

  if (env.ENFORCE_WA_WINDOW === "1") await expireDoomed(env);
  const p = await env.DB.prepare("SELECT COUNT(*) n FROM jobs WHERE status='pending'").first<{ n: number }>();
  if ((p?.n ?? 0) < n(env.MIN_PENDING_TO_START)) return skip(env, date, slot, "queue_below_min");
  const cap = await env.DB.prepare(
    `SELECT COALESCE(SUM((julianday(COALESCE(ended_at,datetime('now')))-julianday(started_at))*1440),0) m
     FROM runs WHERE slot_date=? AND status IN ('done','killed','running','starting')`).bind(date).first<{ m: number }>();
  if ((cap?.m ?? 0) >= n(env.DAILY_RUN_MINUTES_CAP)) { await notify(env, "TTS daily minutes cap hit"); return skip(env, date, slot, "daily_cap"); }

  const id = await startRun(env, date, slot);
  if (!id) {
    const b = await env.DB.prepare("SELECT COUNT(*) n FROM runs WHERE slot_date=?1 AND slot=?2 AND status='start_failed'").bind(date, slot).first<{ n: number }>();
    if ((b?.n ?? 0) >= 3) await notify(env, `TTS slot ${slot}: pod start failed 3x, giving up until next slot`);
  }
}

const skip = (env: Env, date: string, slot: string, why: string) =>
  env.DB.prepare(`INSERT INTO runs (id,slot_date,slot,status,started_at,ended_at,reason)
                  VALUES (?1,?2,?3,'skipped',datetime('now'),datetime('now'),?4)`)
    .bind(crypto.randomUUID(), date, slot, why).run();

export async function startRun(env: Env, date: string, slot: string, noPod = false): Promise<string | null> {
  const id = crypto.randomUUID();
  const ins = await env.DB.prepare(
    `INSERT INTO runs (id,slot_date,slot,status,started_at)
     SELECT ?1,?2,?3,'starting',datetime('now') WHERE NOT EXISTS (SELECT 1 FROM runs WHERE status IN ${ACTIVE})`)
    .bind(id, date, slot).run();
  if (!ins.meta.changes) return null;
  if (noPod) return id;
  try {
    const podId = await createPod(env, id);
    await env.DB.prepare("UPDATE runs SET pod_id=?1 WHERE id=?2").bind(podId, id).run();
    return id;
  } catch (e) {
    await env.DB.prepare("UPDATE runs SET status='start_failed', ended_at=datetime('now'), reason=?2 WHERE id=?1")
      .bind(id, String(e).slice(0, 300)).run();
    return null;
  }
}

export async function watchdog(env: Env) {
  const rows = await env.DB.prepare(
    `SELECT id, pod_id, status FROM runs WHERE status IN ${ACTIVE} AND (
       started_at < datetime('now', ?1)
       OR (status='starting' AND started_at < datetime('now', ?2))
       OR (status='running' AND COALESCE(last_beat_at, first_beat_at, started_at) < datetime('now', ?3)))`)
    .bind(`-${n(env.MAX_RUN_SECONDS)} seconds`, `-${n(env.BOOT_GRACE_SECONDS)} seconds`, `-${n(env.BEAT_TIMEOUT_SECONDS)} seconds`)
    .all<{ id: string; pod_id: string | null; status: string }>();
  for (const r of rows.results) await killRun(env, r.id, r.pod_id, `watchdog(${r.status})`);
}

export async function killRun(env: Env, id: string, podId: string | null, reason: string) {
  await env.DB.prepare(`UPDATE runs SET status='killed', ended_at=datetime('now'), reason=?2 WHERE id=?1 AND status IN ${ACTIVE}`)
    .bind(id, reason).run();
  await terminateRun(env, id, podId).catch((e) => console.error("terminate failed", id, e));
  await requeueStale(env);
  await finalizeCounts(env, id);
  await notify(env, `TTS run ${id.slice(0, 8)} killed: ${reason}`);
}

export async function terminateRun(env: Env, id: string, podId: string | null) {
  const pid = podId ?? (await findPodId(env, id));
  if (pid) await terminatePod(env, pid);
}

export const requeueStale = (env: Env) =>
  env.DB.batch([
    env.DB.prepare(
      `UPDATE jobs SET status='pending', run_id=NULL
       WHERE status='claimed' AND (run_id IS NULL OR run_id NOT IN (SELECT id FROM runs WHERE status IN ${ACTIVE}))
         AND attempts < ?1`).bind(n(env.MAX_ATTEMPTS)),
    env.DB.prepare(
      `UPDATE jobs SET status='failed', error='run_ended_before_send'
       WHERE status='claimed' AND (run_id IS NULL OR run_id NOT IN (SELECT id FROM runs WHERE status IN ${ACTIVE}))
         AND attempts >= ?1`).bind(n(env.MAX_ATTEMPTS)),
  ]);

export const finalizeCounts = (env: Env, id: string) =>
  env.DB.prepare(`UPDATE runs SET
      sent=(SELECT COUNT(*) FROM jobs WHERE run_id=?1 AND status='sent'),
      failed=(SELECT COUNT(*) FROM jobs WHERE run_id=?1 AND status='failed') WHERE id=?1`).bind(id).run();

export async function sweepOrphans(env: Env) {
  const pods = await listTtsPods(env);
  if (!pods.length) return;
  const act = new Set((await env.DB.prepare(`SELECT id FROM runs WHERE status IN ${ACTIVE}`).all<{ id: string }>()).results.map((r) => r.id));
  for (const p of pods) if (!act.has(p.name.slice(4))) {
    await notify(env, `TTS orphan pod ${p.id} terminated`);
    await terminatePod(env, p.id).catch((e) => console.error("orphan terminate", e));
  }
}

async function expireDoomed(env: Env) {
  await env.DB.prepare(
    `UPDATE jobs SET status='failed', error='wa_window_closed' WHERE status='pending' AND channel='whatsapp'
     AND NOT EXISTS (SELECT 1 FROM contacts c WHERE c.contact_id=jobs.contact_id AND c.last_inbound_at > datetime('now','-23 hours'))`).run();
}

async function maintenance(env: Env) {
  await env.DB.batch([
    env.DB.prepare("DELETE FROM jobs WHERE status IN ('sent','failed') AND created_at < datetime('now', ?1)").bind(`-${n(env.RETENTION_DAYS)} days`),
    env.DB.prepare("UPDATE jobs SET text=NULL WHERE text IS NOT NULL AND status IN ('sent','failed') AND created_at < datetime('now', ?1)").bind(`-${n(env.TEXT_RETENTION_DAYS)} days`),
    env.DB.prepare("DELETE FROM runs WHERE started_at < datetime('now','-180 days')"),
    env.DB.prepare("DELETE FROM contacts WHERE last_inbound_at < datetime('now','-3 days')"),
  ]);
}