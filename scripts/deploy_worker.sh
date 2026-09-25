#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
WORKER_DIR="$ROOT_DIR/worker"

ENV_FILE="$ROOT_DIR/.env"
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE"; exit 1; }
set -a
. "$ENV_FILE"
set +a

[ -d "$WORKER_DIR" ] || { echo "missing $WORKER_DIR"; exit 1; }
cd "$WORKER_DIR"

need() { for v in "$@"; do [ -n "${!v:-}" ] || { echo "missing $v in .env"; exit 1; }; done; }
need CLOUDFLARE_API_TOKEN CLOUDFLARE_ACCOUNT_ID INTAKE_API_KEY ADMIN_API_KEY POD_SECRET RUNPOD_API_KEY WA_TOKEN WA_PHONE_ID WA_APP_SECRET WA_VERIFY_TOKEN

for t in npx jq curl; do command -v "$t" >/dev/null || { echo "need $t"; exit 1; }; done

export CLOUDFLARE_API_TOKEN CLOUDFLARE_ACCOUNT_ID

if [ -n "${POD_IMAGE:-}" ]; then
  sed -i.bak "s|^POD_IMAGE = .*|POD_IMAGE = \"$POD_IMAGE\"|" wrangler.toml
fi
if [ -n "${RUNPOD_REGISTRY_AUTH_ID:-}" ]; then
  sed -i.bak "s|^RUNPOD_REGISTRY_AUTH_ID = .*|RUNPOD_REGISTRY_AUTH_ID = \"$RUNPOD_REGISTRY_AUTH_ID\"|" wrangler.toml
fi
if [ -n "${WORKER_PUBLIC_URL:-}" ]; then
  sed -i.bak "s|^WORKER_PUBLIC_URL = .*|WORKER_PUBLIC_URL = \"$WORKER_PUBLIC_URL\"|" wrangler.toml
fi

if [ -f package-lock.json ]; then
  npm ci
else
  npm install
fi
npx tsc --noEmit

OUT=$(npx wrangler deploy 2>&1)
echo "$OUT"

if [ -z "${WORKER_PUBLIC_URL:-}" ]; then
  URL=$(echo "$OUT" | grep -o 'https://[^ ]*workers\.dev' | head -1)
  if [ -z "$URL" ]; then
    echo "could not detect WORKER_PUBLIC_URL, set it in .env and rerun"
    exit 1
  fi
  sed -i.bak "s|^WORKER_PUBLIC_URL = .*|WORKER_PUBLIC_URL = \"$URL\"|" wrangler.toml
  echo "WORKER_PUBLIC_URL=$URL" >> "$ENV_FILE"
  npx wrangler deploy
  WORKER_PUBLIC_URL="$URL"
fi

T=$(mktemp)
trap 'rm -f "$T"' EXIT
jq -n \
  --arg a "$INTAKE_API_KEY" \
  --arg b "$ADMIN_API_KEY" \
  --arg c "$POD_SECRET" \
  --arg d "$RUNPOD_API_KEY" \
  --arg e "$WA_TOKEN" \
  --arg f "$WA_PHONE_ID" \
  --arg g "$WA_APP_SECRET" \
  --arg h "$WA_VERIFY_TOKEN" \
  '{INTAKE_API_KEY:$a,ADMIN_API_KEY:$b,POD_SECRET:$c,RUNPOD_API_KEY:$d,WA_TOKEN:$e,WA_PHONE_ID:$f,WA_APP_SECRET:$g,WA_VERIFY_TOKEN:$h}' > "$T"

npx wrangler secret bulk "$T"

curl -fsS "$WORKER_PUBLIC_URL/healthz" && echo " worker healthy"