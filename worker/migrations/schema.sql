CREATE TABLE jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dedup_key TEXT UNIQUE,
  source TEXT NOT NULL DEFAULT 'api',
  channel TEXT NOT NULL DEFAULT 'whatsapp',
  contact_id TEXT NOT NULL,
  text TEXT,
  language TEXT NOT NULL,
  voice_id TEXT,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','claimed','sent','failed')),
  run_id TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  channel_msg_id TEXT,
  delivery_status TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  claimed_at TEXT,
  sent_at TEXT
);
CREATE INDEX jobs_status_id ON jobs(status, id);
CREATE INDEX jobs_run ON jobs(run_id) WHERE run_id IS NOT NULL;
CREATE INDEX jobs_msg ON jobs(channel_msg_id) WHERE channel_msg_id IS NOT NULL;
CREATE INDEX jobs_created ON jobs(created_at);

CREATE TABLE runs (
  id TEXT PRIMARY KEY,
  slot_date TEXT NOT NULL,
  slot TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('starting','running','done','killed','start_failed','skipped')),
  pod_id TEXT,
  started_at TEXT NOT NULL,
  first_beat_at TEXT, last_beat_at TEXT, ended_at TEXT,
  phase TEXT, jobs_remaining INTEGER, models TEXT,
  claimed INTEGER NOT NULL DEFAULT 0, sent INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0,
  gpu_seconds REAL, reason TEXT
);
CREATE INDEX runs_status ON runs(status);
CREATE INDEX runs_slot ON runs(slot_date, slot);

CREATE TABLE contacts (
  contact_id TEXT PRIMARY KEY,
  last_inbound_at TEXT NOT NULL
);