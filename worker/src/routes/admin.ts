import { json, type Env } from "../lib";
import { apiKeyOk, runToken } from "../auth";
import { startRun, killRun, local } from "../scheduler";

export async function admin(req: Request, env: Env, path: string) {
  if (!(await apiKeyOk(req, env.ADMIN_API_KEY))) return json({ error: "unauthorized" }, 401);
  if (path === "/admin/status" && req.method === "GET") {
    const [active, jobs, langs, runs] = await env.DB.batch([
      env.DB.prepare("SELECT * FROM runs WHERE status IN ('starting','running')"),
      env.DB.prepare("SELECT status, COUNT(*) n FROM jobs GROUP BY status"),
      env.DB.prepare("SELECT language, COUNT(*) n FROM jobs WHERE status='pending' GROUP BY language"),
      env.DB.prepare("SELECT * FROM runs ORDER BY started_at DESC LIMIT 10"),
    ]);
    return json({ active: active.results, jobs: jobs.results, pending_by_language: langs.results, runs: runs.results });
  }
  if (path === "/admin/run" && req.method === "POST") {
    const b = await req.json<{ no_pod?: boolean }>().catch(() => ({} as { no_pod?: boolean }));
    const id = await startRun(env, local(env).date, "manual", !!b.no_pod);
    return id ? json({ run_id: id, ...(b.no_pod ? { token: await runToken(env.POD_SECRET, id) } : {}) })
              : json({ error: "a run is active, or pod creation failed (see runs.reason)" }, 409);
  }
  if (path === "/admin/kill" && req.method === "POST") {
    const { run_id } = await req.json<{ run_id: string }>();
    const r = await env.DB.prepare("SELECT pod_id FROM runs WHERE id=?").bind(run_id).first<{ pod_id: string | null }>();
    if (!r) return json({ error: "no such run" }, 404);
    await killRun(env, run_id, r.pod_id, "admin");
    return json({ ok: true });
  }
  return json({ error: "not found" }, 404);
}