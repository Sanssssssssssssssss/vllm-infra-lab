param(
  [string]$WorkspaceDir = "",
  [string]$Profile = "",
  [switch]$Force
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "common.ps1")

$cfg = Get-RuntimeConfig -WorkspaceDir $WorkspaceDir -ProfileName $Profile
if ((Get-ConfigValue -Config $cfg -Key "BACKEND") -ne "llamacpp") {
  throw "Active profile is not a llama.cpp profile."
}

$version = Get-ConfigValue -Config $cfg -Key "LLAMACPP_VERSION"
$variant = Get-ConfigValue -Config $cfg -Key "LLAMACPP_VARIANT"
$installDir = Get-ConfigValue -Config $cfg -Key "LLAMACPP_ROOT"

if (-not $version -or -not $variant -or -not $installDir) {
  throw "LLAMACPP_VERSION, LLAMACPP_VARIANT, and LLAMACPP_ROOT are required."
}

$releaseBase = "https://github.com/ggml-org/llama.cpp/releases/download/$version"
$mainAsset = "llama-$version-bin-$variant.zip"
$cudaRuntimeAsset = "cudart-llama-bin-$variant.zip"

$downloadDir = Join-Path $env:TEMP "llamacpp-$version-$variant"
New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
New-Item -ItemType Directory -Path $installDir -Force | Out-Null

$assets = @(
  @{ Name = $mainAsset; Url = "$releaseBase/$mainAsset" },
  @{ Name = $cudaRuntimeAsset; Url = "$releaseBase/$cudaRuntimeAsset" }
)

foreach ($asset in $assets) {
  $zipPath = Join-Path $downloadDir $asset.Name
  if ($Force -or -not (Test-Path -LiteralPath $zipPath)) {
    Write-Host "Downloading $($asset.Name)"
    Invoke-WebRequest -Uri $asset.Url -OutFile $zipPath
  }
  Write-Host "Extracting $($asset.Name) to $installDir"
  Expand-Archive -LiteralPath $zipPath -DestinationPath $installDir -Force
}

$serverPath = Join-Path $installDir "llama-server.exe"
if (-not (Test-Path -LiteralPath $serverPath)) {
  throw "llama-server.exe was not found under $installDir after extraction."
}

Write-Host "llama.cpp is ready at $installDir"
