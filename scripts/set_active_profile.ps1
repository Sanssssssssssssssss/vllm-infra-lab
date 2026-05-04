param(
  [Parameter(Mandatory = $true)][string]$Profile,
  [string]$WorkspaceDir = ""
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-WorkspaceDir -WorkspaceDir $WorkspaceDir
$runtimePath = Join-Path $root "config\runtime.env"
$profilePath = Get-ProfilePath -WorkspaceDir $root -ProfileName $Profile

if (-not (Test-Path -LiteralPath $profilePath)) {
  throw "Unknown profile '$Profile'. Run scripts/list_profiles.ps1 first."
}

$content = Get-Content -LiteralPath $runtimePath
$updated = $false
$newContent = foreach ($line in $content) {
  if ($line -match '^ACTIVE_PROFILE=') {
    $updated = $true
    "ACTIVE_PROFILE=$Profile"
  } else {
    $line
  }
}

if (-not $updated) {
  $newContent = @("ACTIVE_PROFILE=$Profile") + $newContent
}

Set-Content -LiteralPath $runtimePath -Value $newContent -Encoding utf8
Write-Host "Active profile set to $Profile"
