param(
  [string]$WorkspaceDir = ""
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-WorkspaceDir -WorkspaceDir $WorkspaceDir
$profileDir = Join-Path $root "config\profiles"

Get-ChildItem -LiteralPath $profileDir -Filter *.env | ForEach-Object {
  $cfg = Import-EnvFile -Path $_.FullName
  [PSCustomObject]@{
    profile = Get-ConfigValue -Config $cfg -Key "PROFILE_ID" -DefaultValue $_.BaseName
    backend = Get-ConfigValue -Config $cfg -Key "BACKEND"
    model = Get-ConfigValue -Config $cfg -Key "MODEL_ID"
    description = Get-ConfigValue -Config $cfg -Key "PROFILE_DESCRIPTION"
  }
} | Format-Table -AutoSize
