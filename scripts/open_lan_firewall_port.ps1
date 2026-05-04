param(
  [int]$Port = 8000,
  [string]$RuleName = "vLLM LAN API"
)

$ErrorActionPreference = "Stop"

$existing = Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
if ($existing) {
  Write-Host "Firewall rule '$RuleName' already exists."
  exit 0
}

New-NetFirewallRule `
  -DisplayName $RuleName `
  -Direction Inbound `
  -Action Allow `
  -Protocol TCP `
  -LocalPort $Port `
  -Profile Private

Write-Host "Created inbound firewall rule '$RuleName' for TCP port $Port on the Private profile."
