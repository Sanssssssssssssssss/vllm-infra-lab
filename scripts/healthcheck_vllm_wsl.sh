#!/usr/bin/env bash
set -euo pipefail

HOST_NAME="${1:-127.0.0.1}"
PORT="${2:-8000}"
API_KEY="${3:-}"

echo "GET http://${HOST_NAME}:${PORT}/health"
curl -fsS "http://${HOST_NAME}:${PORT}/health"
echo

echo "GET http://${HOST_NAME}:${PORT}/v1/models"
if [ -n "${API_KEY}" ]; then
  curl -fsS -H "Authorization: Bearer ${API_KEY}" "http://${HOST_NAME}:${PORT}/v1/models"
else
  curl -fsS "http://${HOST_NAME}:${PORT}/v1/models"
fi
echo

