param(
  [Parameter(Mandatory = $true)][string]$HostName,
  [Parameter(Mandatory = $true)][string]$Message,
  [int]$Port = 8001,
  [string]$ApiKey = "change-this-before-lan-use",
  [string]$Model = "Qwen3-8B-GGUF-q4_k_m-local",
  [int]$MaxTokens = 64,
  [int]$Timeout = 120,
  [switch]$Thinking,
  [string]$WorkspaceDir = ""
)

$ErrorActionPreference = "Stop"

if (-not $WorkspaceDir) {
  $WorkspaceDir = Split-Path -Parent $PSScriptRoot
}

$args = @(
  (Join-Path $WorkspaceDir "scripts\test_openai_api.py"),
  "--host", $HostName,
  "--port", "$Port",
  "--api-key", $ApiKey,
  "--model", $Model,
  "--message", $Message,
  "--max-tokens", "$MaxTokens",
  "--timeout", "$Timeout"
)

if ($Thinking) {
  # Intentionally omit backend-specific disable-thinking payload.
} 

py -3 @args
