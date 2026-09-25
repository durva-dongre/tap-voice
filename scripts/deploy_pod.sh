#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "cp .env.example .env and fill it"; exit 1; }
set -a; . ./.env; set +a

for v in RUNPOD_API_KEY POD_GPU_IDS POD_CLOUD POD_DISK_GB; do
  [ -n "${!v:-}" ] || { echo "missing $v in .env"; exit 1; }
done
for t in jq curl; do command -v $t >/dev/null || { echo "need $t"; exit 1; }; done

IMAGE="${POD_IMAGE:-${IMAGE_TAG:-}}"
[ -n "$IMAGE" ] || { echo "missing POD_IMAGE in .env; set it to the tag printed by the build-pod-image GitHub Actions workflow"; exit 1; }

B="${RUNPOD_API_BASE:-https://api.runpod.io/v2}"
H="Authorization: Bearer $RUNPOD_API_KEY"
curl -fsS -H "$H" "$B/pods" | jq -e '.pods|type=="array"' >/dev/null
GPU=$(echo "$POD_GPU_IDS" | cut -d, -f1 | xargs)

BODY=$(jq -n --arg name "tts-selftest-$(date +%s)" --arg image "$IMAGE" --arg gpu "$GPU" --arg cloud "$POD_CLOUD" \
  --argjson disk "$POD_DISK_GB" --arg reg "${RUNPOD_REGISTRY_AUTH_ID:-}" \
  '{name:$name, image:$image, gpu:{id:$gpu,count:1}, cloud:$cloud, disk:$disk,
    dockerArgs:"python -m app.selftest",
    env:{MOCK_ENGINES:"0"}} + (if $reg != "" then {registry:{id:$reg}} else {} end)')

POD_ID=$(curl -fsS -H "$H" -H "Content-Type: application/json" -X POST "$B/pods" -d "$BODY" | jq -r .id)
[ -n "$POD_ID" ] && [ "$POD_ID" != "null" ] || { echo "pod create failed"; exit 1; }
echo "selftest pod: $POD_ID"
echo "check output via RunPod console (Pods > $POD_ID > Logs)"

for i in $(seq 1 40); do
  sleep 15
  STATUS=$(curl -fsS -H "$H" "$B/pods/$POD_ID" | jq -r '.desiredStatus // .status // "unknown"')
  echo "[$i] $STATUS"
  [ "$STATUS" = "EXITED" ] || [ "$STATUS" = "TERMINATED" ] && break
done

curl -fsS -H "$H" -X DELETE "$B/pods/$POD_ID" >/dev/null
echo "terminated $POD_ID"
echo "image verified: $IMAGE"