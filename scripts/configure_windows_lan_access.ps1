param(
  [string]$ListenAddress = "0.0.0.0",
  [int]$Port = 8000,
  [string]$WslAddress
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

& (Join-Path $scriptRoot "add_wsl_portproxy.ps1") -ListenAddress $ListenAddress -Port $Port -WslAddress $WslAddress
& (Join-Path $scriptRoot "open_lan_firewall_port.ps1") -Port $Port

Write-Host "Windows LAN access configuration finished."

