import type { Env } from "./lib";

const enc = new TextEncoder();
const key = (s: string, use: KeyUsage[]) => crypto.subtle.importKey("raw", enc.encode(s), { name: "HMAC", hash: "SHA-256" }, false, use);
const hex = (b: ArrayBuffer) => [...new Uint8Array(b)].map((x) => x.toString(16).padStart(2, "0")).join("");
const unhex = (h: string) => new Uint8Array(h.match(/../g)?.map((x) => parseInt(x, 16)) ?? []);

async function safeEq(a: string, b: string) {
  const [x, y] = await Promise.all([a, b].map((s) => crypto.subtle.digest("SHA-256", enc.encode(s))));
  const xb = new Uint8Array(x), yb = new Uint8Array(y);
  let diff = 0;
  for (let i = 0; i < xb.length; i++) diff |= xb[i] ^ yb[i];
  return diff === 0;
}

export async function apiKeyOk(req: Request, expected: string) {
  const got = req.headers.get("authorization")?.replace(/^Bearer /i, "") ?? req.headers.get("x-api-key") ?? "";
  return !!expected && !!got && (await safeEq(got, expected));
}

export async function runToken(secret: string, runId: string) {
  return hex(await crypto.subtle.sign("HMAC", await key(secret, ["sign"]), enc.encode("run:" + runId)));
}

export async function verifyRun(req: Request, env: Env): Promise<string | null> {
  const id = req.headers.get("x-run-id") ?? "", tok = req.headers.get("x-run-token") ?? "";
  if (!/^[0-9a-f-]{36}$/.test(id) || !tok) return null;
  const ok = await crypto.subtle.verify("HMAC", await key(env.POD_SECRET, ["verify"]), unhex(tok), enc.encode("run:" + id));
  return ok ? id : null;
}

export async function hmacOk(secret: string, body: string, sigHex: string) {
  return crypto.subtle.verify("HMAC", await key(secret, ["verify"]), unhex(sigHex), enc.encode(body));
}