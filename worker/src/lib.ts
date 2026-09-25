export type Env = { DB: D1Database } & Record<
  | "TZ_OFFSET_MIN" | "SLOTS" | "SLOT_WINDOW_MIN" | "MAINT_LOCAL_TIME" | "MAX_JOBS_PER_CLAIM" | "MAX_RUN_SECONDS"
  | "BEAT_TIMEOUT_SECONDS" | "BOOT_GRACE_SECONDS" | "MAX_ATTEMPTS" | "MAX_TEXT_CHARS" | "MIN_PENDING_TO_START"
  | "DAILY_RUN_MINUTES_CAP" | "ENFORCE_WA_WINDOW" | "WA_DRY_RUN" | "WA_API_VERSION" | "RETENTION_DAYS"
  | "TEXT_RETENTION_DAYS" | "POD_IMAGE" | "POD_GPU_IDS" | "POD_CLOUD" | "POD_DISK_GB" | "RUNPOD_API_BASE"
  | "RUNPOD_REGISTRY_AUTH_ID" | "WORKER_PUBLIC_URL" | "ALERT_WEBHOOK_URL"
  | "INTAKE_API_KEY" | "ADMIN_API_KEY" | "POD_SECRET" | "RUNPOD_API_KEY"
  | "WA_TOKEN" | "WA_PHONE_ID" | "WA_APP_SECRET" | "WA_VERIFY_TOKEN", string>;

export const n = (v: string) => parseInt(v, 10);

export const json = (o: unknown, status = 200) =>
  new Response(JSON.stringify(o), { status, headers: { "content-type": "application/json" } });

export const notify = (env: Env, text: string) =>
  env.ALERT_WEBHOOK_URL
    ? fetch(env.ALERT_WEBHOOK_URL, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text }) }).then(() => {}, () => {})
    : Promise.resolve();