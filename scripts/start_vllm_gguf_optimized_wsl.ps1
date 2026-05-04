$ErrorActionPreference = "Stop"

$workspace = (Get-Location).Path
$workspaceForWsl = "/mnt/" + ($workspace.Substring(0,1).ToLower()) + ($workspace.Substring(2) -replace "\\","/")
$scriptForWsl = "$workspaceForWsl/scripts/start_vllm_gguf_optimized_wsl.sh"

wsl -e bash $scriptForWsl $workspaceForWsl
