#!/bin/bash
# Probe an OpenAI-compatible endpoint to discover which path it answers on.
# Usage:
#   export PULSAR_KEY=sk-...
#   ./test1.sh
set -euo pipefail

: "${PULSAR_KEY:?set PULSAR_KEY env var to a bearer token before running}"
: "${PULSAR_BASE:=https://pulsar.corti.com}"
: "${PULSAR_MODEL:=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"

for p in "" "/" "/chat/completions" "/completions" "/generate" "/models"; do
  echo "=== POST ${PULSAR_BASE}/v1${p} ==="
  curl -s -o /tmp/body -w "HTTP %{http_code}\n" -X POST "${PULSAR_BASE}/v1${p}" \
    -H "Authorization: Bearer ${PULSAR_KEY}" -H "Content-Type: application/json" \
    -d "{\"model\":\"${PULSAR_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" \
    -m 15
  head -c 300 /tmp/body
  echo
done
