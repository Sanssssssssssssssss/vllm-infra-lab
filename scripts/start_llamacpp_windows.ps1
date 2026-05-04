param(
  [string]$WorkspaceDir = "",
  [string]$Profile = ""
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "common.ps1")

$cfg = Get-RuntimeConfig -WorkspaceDir $WorkspaceDir -ProfileName $Profile
if ((Get-ConfigValue -Config $cfg -Key "BACKEND") -ne "llamacpp") {
  throw "Active profile is not a llama.cpp profile."
}

$serverPath = Join-Path (Get-ConfigValue -Config $cfg -Key "LLAMACPP_ROOT") "llama-server.exe"
$modelPath = Get-ModelPathFromConfig -Config $cfg
$apiHost = Get-ConfigValue -Config $cfg -Key "API_HOST" -DefaultValue "0.0.0.0"
$port = Get-ConfigValue -Config $cfg -Key "API_PORT" -DefaultValue "8000"
$apiKey = Get-ConfigValue -Config $cfg -Key "API_KEY"
$alias = Get-ConfigValue -Config $cfg -Key "SERVED_MODEL_NAME"
$ctxSize = Get-ConfigValue -Config $cfg -Key "LLAMACPP_CTX_SIZE" -DefaultValue "4096"
$threads = Get-ConfigValue -Config $cfg -Key "LLAMACPP_THREADS" -DefaultValue "8"
$nGpuLayers = Get-ConfigValue -Config $cfg -Key "LLAMACPP_N_GPU_LAYERS" -DefaultValue "999"
$flashAttn = Get-ConfigValue -Config $cfg -Key "LLAMACPP_FLASH_ATTN" -DefaultValue "0"
$extraArgs = Get-ConfigValue -Config $cfg -Key "LLAMACPP_EXTRA_ARGS"

if (-not (Test-Path -LiteralPath $serverPath)) {
  throw "Missing llama-server.exe at $serverPath. Run scripts/bootstrap_llamacpp_windows.ps1 first."
}
if (-not (Test-Path -LiteralPath $modelPath)) {
  throw "Missing model file at $modelPath. Run scripts/download_active_model_windows.ps1 first."
}

$args = @(
  "--host", $apiHost,
  "--port", $port,
  "--api-key", $apiKey,
  "--alias", $alias,
  "-m", $modelPath,
  "-c", $ctxSize,
  "-t", $threads,
  "-ngl", $nGpuLayers
)

if ($flashAttn -eq "1") {
  $args += @("--flash-attn", "on")
}

if ($extraArgs) {
  $args += @($extraArgs -split "\s+")
}

Write-Host "Starting $serverPath"
Write-Host ("Arguments: " + ($args -join " "))
& $serverPath @args
