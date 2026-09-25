import type { Channel, SendResult } from "./index";

async function fail(res: Response): Promise<SendResult> {
  const t = await res.text();
  return { ok: false, error: `wa ${res.status}: ${t.slice(0, 200)}` };
}

export const whatsapp: Channel = {
  validateRecipient: (id) => /^\d{8,15}$/.test(id),
  async send(env, { to, audio, jobId }) {
    if (env.WA_DRY_RUN === "1") return { ok: true, id: `dry-${jobId}` };
    try {
      const base = `https://graph.facebook.com/${env.WA_API_VERSION}/${env.WA_PHONE_ID}`;
      const auth = { Authorization: `Bearer ${env.WA_TOKEN}` };
      const fd = new FormData();
      fd.append("messaging_product", "whatsapp"); fd.append("type", "audio/ogg");
      fd.append("file", new Blob([audio], { type: "audio/ogg; codecs=opus" }), "voice.ogg");
      const up = await fetch(`${base}/media`, { method: "POST", headers: auth, body: fd });
      if (!up.ok) return fail(up);
      const { id } = await up.json<{ id: string }>();
      const m = await fetch(`${base}/messages`, { method: "POST", headers: { ...auth, "Content-Type": "application/json" },
        body: JSON.stringify({ messaging_product: "whatsapp", to, type: "audio", audio: { id } }) });
      if (!m.ok) return fail(m);
      const msgId = (await m.json<{ messages?: { id: string }[] }>()).messages?.[0]?.id;
      return { ok: true, id: msgId };
    } catch (e) { return { ok: false, error: `net: ${String(e).slice(0, 150)}` }; }
  },
};