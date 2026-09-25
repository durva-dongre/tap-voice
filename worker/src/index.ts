import { json, type Env } from "./lib";
import { verifyRun } from "./auth";
import { tick } from "./scheduler";
import { createJobs, getJob } from "./routes/jobs";
import { podRoute } from "./routes/pod";
import { admin } from "./routes/admin";

export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const p = new URL(req.url).pathname;
    try {
      if (p === "/healthz") return new Response("ok");
      if (p === "/jobs" && req.method === "POST") return await createJobs(req, env);
      if (p.startsWith("/jobs/") && req.method === "GET") return await getJob(req, env, p.slice(6));
      if (p.startsWith("/admin/")) return await admin(req, env, p);
      if (p.startsWith("/pod/")) {
        const run = await verifyRun(req, env);
        return run ? await podRoute(req, env, run, p, ctx) : json({ error: "unauthorized" }, 401);
      }
    } catch (e) { console.error("unhandled", p, e); return json({ error: "internal" }, 500); }
    return json({ error: "not found" }, 404);
  },
  async scheduled(ev: ScheduledController, env: Env, ctx: ExecutionContext) {
    ctx.waitUntil(tick(env, ev.scheduledTime));
  },
};