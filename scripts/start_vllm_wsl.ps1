$ErrorActionPreference = "Stop"

$workspace = (Get-Location).Path
$workspaceForWsl = "/mnt/" + ($workspace.Substring(0,1).ToLower()) + ($workspace.Substring(2) -replace "\\","/")
$envFile = Join-Path $workspace "config\\runtime.env"

if (-not (Test-Path $envFile)) {
  throw "Missing config/runtime.env. Copy config/runtime.env.example first."
}

$scriptForWsl = "$workspaceForWsl/scripts/start_vllm_wsl.sh"
wsl -e bash $scriptForWsl $workspaceForWsl
