param(
  [string]$ListenAddress = "0.0.0.0",
  [int]$Port = 8000,
  [string]$WslAddress
)

$ErrorActionPreference = "Stop"

if (-not $WslAddress) {
  $ips = wsl -e bash -lc "hostname -I"
  if ($ips) {
    $parts = @(($ips -split "\s+") | Where-Object { $_ })
    $WslAddress = $parts[0]
  }
}

if (-not $WslAddress) {
  throw "Could not determine the WSL IP address."
}

Write-Host "Adding portproxy: ${ListenAddress}:${Port} -> ${WslAddress}:${Port}"
netsh interface portproxy delete v4tov4 listenaddress=$ListenAddress listenport=$Port | Out-Null
if ($LASTEXITCODE -ne 0) {
  Write-Warning "Existing portproxy delete returned exit code $LASTEXITCODE. Continuing."
}

netsh interface portproxy add v4tov4 listenaddress=$ListenAddress listenport=$Port connectaddress=$WslAddress connectport=$Port
if ($LASTEXITCODE -ne 0) {
  throw "netsh portproxy add failed with exit code $LASTEXITCODE. Run this script from an elevated PowerShell window."
}

Write-Host "Portproxy configured."
