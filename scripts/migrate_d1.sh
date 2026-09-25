#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
WORKER_DIR="$ROOT_DIR/worker"
SCHEMA_FILE="$ROOT_DIR/migrations/schema.sql"

ENV_FILE="$ROOT_DIR/.env"
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE"; exit 1; }
set -a
. "$ENV_FILE"
set +a

[ -d "$WORKER_DIR" ] || { echo "missing $WORKER_DIR"; exit 1; }
[ -f "$SCHEMA_FILE" ] || { echo "missing $SCHEMA_FILE"; exit 1; }
cd "$WORKER_DIR"

: "${CLOUDFLARE_API_TOKEN:?missing CLOUDFLARE_API_TOKEN in .env}"
: "${CLOUDFLARE_ACCOUNT_ID:?missing CLOUDFLARE_ACCOUNT_ID in .env}"
: "${D1_DATABASE_NAME:=tts-jobs}"

export CLOUDFLARE_API_TOKEN CLOUDFLARE_ACCOUNT_ID

command -v npx >/dev/null || { echo "need npx"; exit 1; }

ID=$(npx wrangler d1 list --json | jq -r --arg n "$D1_DATABASE_NAME" '.[] | select(.name==$n) | .uuid' || true)
if [ -z "$ID" ]; then
  npx wrangler d1 create "$D1_DATABASE_NAME"
  ID=$(npx wrangler d1 list --json | jq -r --arg n "$D1_DATABASE_NAME" '.[] | select(.name==$n) | .uuid')
fi

echo "database_id=$ID"

if [ -f wrangler.toml ]; then
  sed -i.bak "s|^database_id = .*|database_id = \"$ID\"|" wrangler.toml
fi

set +e
OUT=$(npx wrangler d1 execute "$D1_DATABASE_NAME" --remote --file="$SCHEMA_FILE" 2>&1)
CODE=$?
set -e
echo "$OUT"

if [ $CODE -ne 0 ] && ! echo "$OUT" | grep -q "already exists"; then
  exit $CODE
fi

echo "migration applied to $D1_DATABASE_NAME ($ID)"