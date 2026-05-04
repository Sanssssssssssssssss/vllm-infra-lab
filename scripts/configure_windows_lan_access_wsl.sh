#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${1:?workspace path is required}"
PORT="${2:-8000}"
LISTEN_ADDRESS="${3:-0.0.0.0}"

WIN_SCRIPT="$(wslpath -w "${WORKSPACE_DIR}/scripts/configure_windows_lan_access.ps1")"

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Start-Process PowerShell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File ""${WIN_SCRIPT}"" -ListenAddress ${LISTEN_ADDRESS} -Port ${PORT}'"

echo "An elevated PowerShell window should open for Windows portproxy/firewall configuration."

