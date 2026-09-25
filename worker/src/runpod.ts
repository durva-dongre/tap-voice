import { n, type Env } from "./lib";
import { runToken } from "./auth";

const call = (env: Env, path: string, init: RequestInit = {}) =>
  fetch(env.RUNPOD_API_BASE + path, {
    ...init, headers: { Authorization: `Bearer ${env.RUNPOD_API_KEY}`, "Content-Type": "application/json" },
  });

export async function createPod(env: Env, runId: string): Promise<string> {
  const podEnv = {
    RUN_ID: runId, RUN_TOKEN: await runToken(env.POD_SECRET, runId), WORKER_URL: env.WORKER_PUBLIC_URL,
    RUNPOD_API_BASE: env.RUNPOD_API_BASE, MAX_JOBS: env.MAX_JOBS_PER_CLAIM,
    POD_LIMIT_SECONDS: String(n(env.MAX_RUN_SECONDS) - 300),
  };
  let last = "no POD_GPU_IDS configured";
  for (const gpu of env.POD_GPU_IDS.split(",").map((s) => s.trim()).filter(Boolean)) {
    const res = await call(env, "/pods", { method: "POST", body: JSON.stringify({
      name: `tts-${runId}`, image: env.POD_IMAGE, gpu: { id: gpu, count: 1 }, cloud: env.POD_CLOUD,
      disk: n(env.POD_DISK_GB), env: podEnv,
      ...(env.RUNPOD_REGISTRY_AUTH_ID ? { registry: { id: env.RUNPOD_REGISTRY_AUTH_ID } } : {}),
    }) });
    const body = await res.text();
    if (res.ok) { const id = JSON.parse(body).id; if (id) return id as string; last = `no id in 2xx: ${body.slice(0, 150)}`; }
    else last = `${gpu}: ${res.status} ${body.slice(0, 150)}`;
  }
  throw new Error(last);
}

export async function terminatePod(env: Env, podId: string) {
  const r = await call(env, `/pods/${podId}`, { method: "DELETE" });
  if (!r.ok && r.status !== 404) throw new Error(`terminate ${r.status} ${(await r.text()).slice(0, 120)}`);
}

export async function listTtsPods(env: Env): Promise<{ id: string; name: string }[]> {
  const r = await call(env, "/pods");
  if (!r.ok) return [];
  const j = await r.json<{ pods?: { id: string; name?: string }[] }>();
  return (j.pods ?? []).filter((p) => p.name?.startsWith("tts-")).map((p) => ({ id: p.id, name: p.name! }));
}

export const findPodId = async (env: Env, runId: string) =>
  (await listTtsPods(env)).find((p) => p.name === `tts-${runId}`)?.id ?? null;