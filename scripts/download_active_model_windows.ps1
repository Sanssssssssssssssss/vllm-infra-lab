param(
  [string]$WorkspaceDir = "",
  [string]$Profile = "",
  [switch]$Force
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "common.ps1")

$cfg = Get-RuntimeConfig -WorkspaceDir $WorkspaceDir -ProfileName $Profile
$repo = Get-ConfigValue -Config $cfg -Key "HF_REPO"
$fileGlob = Get-ConfigValue -Config $cfg -Key "HF_FILE_GLOB"
$modelDir = Get-ConfigValue -Config $cfg -Key "MODEL_DIR_WINDOWS"

if (-not $repo -or -not $fileGlob -or -not $modelDir) {
  throw "HF_REPO, HF_FILE_GLOB, and MODEL_DIR_WINDOWS are required for this profile."
}

New-Item -ItemType Directory -Path $modelDir -Force | Out-Null

$apiUrl = "https://huggingface.co/api/models/$repo"
$modelMeta = Invoke-RestMethod -Uri $apiUrl
$files = @($modelMeta.siblings | Where-Object { $_.rfilename -like $fileGlob } | Select-Object -ExpandProperty rfilename)

if ($files.Count -eq 0) {
  throw "No files matched '$fileGlob' in repo '$repo'."
}

foreach ($file in $files) {
  $target = Join-Path $modelDir $file
  $remoteUrl = "https://huggingface.co/$repo/resolve/main/${file}?download=true"
  $remoteLength = $null
  try {
    $headResponse = Invoke-WebRequest -Uri $remoteUrl -Method Head
    $remoteLength = [int64]$headResponse.Headers["Content-Length"]
  } catch {
    Write-Warning "Could not determine remote size for $file. The file will be refreshed from cache."
  }

  if ((-not $Force) -and (Test-Path -LiteralPath $target)) {
    $localLength = (Get-Item -LiteralPath $target).Length
    if ($remoteLength -and ($localLength -eq $remoteLength)) {
      Write-Host "Skipping existing $file"
      continue
    }
  }

  Write-Host "Downloading $file with Hugging Face CLI"
  $env:HF_HUB_DISABLE_XET = "1"
  $cacheArgs = @(
    "-m", "huggingface_hub.cli.hf", "download",
    $repo,
    $file,
    "--quiet"
  )

  if ($Force) {
    $cacheArgs += "--force-download"
  }

  $cachedPath = (py -3 @cacheArgs | Select-Object -Last 1).Trim()
  if (-not $cachedPath) {
    throw "Failed to resolve cached path for $file"
  }

  Copy-Item -LiteralPath $cachedPath -Destination $target -Force
}

Write-Host "Model files are ready under $modelDir"
