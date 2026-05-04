param(
  [string]$HostName = "127.0.0.1",
  [int]$Port = 8000,
  [string]$ApiKey = ""
)

$ErrorActionPreference = "Stop"

$healthUrl = "http://${HostName}:${Port}/health"
$modelsUrl = "http://${HostName}:${Port}/v1/models"

Write-Host "GET $healthUrl"
try {
  Invoke-RestMethod -Method Get -Uri $healthUrl | Out-Host
} catch {
  Write-Warning "Health endpoint failed: $($_.Exception.Message)"
}

Write-Host "GET $modelsUrl"
try {
  $headers = @{}
  if ($ApiKey) {
    $headers["Authorization"] = "Bearer $ApiKey"
  }

  Invoke-RestMethod -Method Get -Uri $modelsUrl -Headers $headers | ConvertTo-Json -Depth 6
} catch {
  Write-Warning "Models endpoint failed: $($_.Exception.Message)"
}

