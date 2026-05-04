param(
  [string]$WorkspaceDir = "",
  [string]$Profile = ""
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "common.ps1")

$cfg = Get-RuntimeConfig -WorkspaceDir $WorkspaceDir -ProfileName $Profile
$apiHost = Get-ConfigValue -Config $cfg -Key "API_HOST" -DefaultValue "127.0.0.1"
if ($apiHost -eq "0.0.0.0") {
  $apiHost = "127.0.0.1"
}
$port = [int](Get-ConfigValue -Config $cfg -Key "API_PORT" -DefaultValue "8000")
$apiKey = Get-ConfigValue -Config $cfg -Key "API_KEY"
$expectedModel = Get-ConfigValue -Config $cfg -Key "SERVED_MODEL_NAME" -DefaultValue (Get-ConfigValue -Config $cfg -Key "MODEL_ID")

$healthUrl = "http://${apiHost}:${port}/health"
$modelsUrl = "http://${apiHost}:${port}/v1/models"

Write-Host "GET $healthUrl"
try {
  Invoke-RestMethod -Method Get -Uri $healthUrl | Out-Host
} catch {
  throw "Health endpoint failed: $($_.Exception.Message)"
}

Write-Host "GET $modelsUrl"
$headers = @{}
if ($apiKey) {
  $headers["Authorization"] = "Bearer $apiKey"
}
$models = Invoke-RestMethod -Method Get -Uri $modelsUrl -Headers $headers
$models | ConvertTo-Json -Depth 6

$modelIds = @($models.data | ForEach-Object { $_.id })
if ($expectedModel -and ($modelIds -notcontains $expectedModel)) {
  Write-Warning "Active profile expects model '$expectedModel', but the running API reported: $($modelIds -join ', ')"
}
