$ErrorActionPreference = "Stop"

$workspace = (Get-Location).Path
$workspaceForWsl = "/mnt/" + ($workspace.Substring(0,1).ToLower()) + ($workspace.Substring(2) -replace "\\","/")
$scriptForWsl = "$workspaceForWsl/scripts/bootstrap_vllm_wsl.sh"
wsl -e bash $scriptForWsl $workspaceForWsl
