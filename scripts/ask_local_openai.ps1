param(
  [Parameter(Mandatory = $true)][string]$Message,
  [int]$MaxTokens = 64,
  [int]$Timeout = 120,
  [switch]$Thinking,
  [string]$WorkspaceDir = "",
  [string]$Profile = ""
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "common.ps1")

$cfg = Get-RuntimeConfig -WorkspaceDir $WorkspaceDir -ProfileName $Profile
$root = $cfg["WORKSPACE_DIR"]
$apiHost = Get-ConfigValue -Config $cfg -Key "API_HOST" -DefaultValue "127.0.0.1"
if ($apiHost -eq "0.0.0.0") {
  $apiHost = "127.0.0.1"
}
$port = Get-ConfigValue -Config $cfg -Key "API_PORT" -DefaultValue "8000"
$apiKey = Get-ConfigValue -Config $cfg -Key "API_KEY"
$model = Get-ConfigValue -Config $cfg -Key "SERVED_MODEL_NAME" -DefaultValue (Get-ConfigValue -Config $cfg -Key "MODEL_ID")
$backend = Get-ConfigValue -Config $cfg -Key "BACKEND"

$args = @(
  (Join-Path $root "scripts\test_openai_api.py"),
  "--host", $apiHost,
  "--port", $port,
  "--api-key", $apiKey,
  "--model", $model,
  "--message", $Message,
  "--max-tokens", "$MaxTokens",
  "--timeout", "$Timeout"
)

if ((-not $Thinking) -and $backend -eq "vllm") {
  $args += "--disable-thinking"
}

py -3 @args
