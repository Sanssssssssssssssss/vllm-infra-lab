Set-StrictMode -Version Latest

function Get-WorkspaceDir {
  param([string]$WorkspaceDir)

  if ($WorkspaceDir) {
    return $WorkspaceDir
  }

  return (Split-Path -Parent $PSScriptRoot)
}

function Import-EnvFile {
  param([Parameter(Mandatory = $true)][string]$Path)

  if (-not (Test-Path -LiteralPath $Path)) {
    throw "Missing env file: $Path"
  }

  $data = @{}
  foreach ($rawLine in Get-Content -LiteralPath $Path) {
    $line = $rawLine.Trim()
    if (-not $line) {
      continue
    }
    if ($line.StartsWith("#")) {
      continue
    }

    $splitIndex = $line.IndexOf("=")
    if ($splitIndex -lt 1) {
      continue
    }

    $key = $line.Substring(0, $splitIndex).Trim()
    $value = $line.Substring($splitIndex + 1).Trim()

    if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
      $value = $value.Substring(1, $value.Length - 2)
    }

    $data[$key] = $value
  }

  return $data
}

function Merge-Config {
  param(
    [Parameter(Mandatory = $true)][hashtable]$Base,
    [Parameter(Mandatory = $true)][hashtable]$Overlay
  )

  $merged = @{}
  foreach ($entry in $Base.GetEnumerator()) {
    $merged[$entry.Key] = $entry.Value
  }
  foreach ($entry in $Overlay.GetEnumerator()) {
    $merged[$entry.Key] = $entry.Value
  }
  return $merged
}

function Get-ProfilePath {
  param(
    [Parameter(Mandatory = $true)][string]$WorkspaceDir,
    [Parameter(Mandatory = $true)][string]$ProfileName
  )

  return (Join-Path $WorkspaceDir "config\profiles\$ProfileName.env")
}

function Get-RuntimeConfig {
  param(
    [string]$WorkspaceDir,
    [string]$ProfileName
  )

  $root = Get-WorkspaceDir -WorkspaceDir $WorkspaceDir
  $runtimePath = Join-Path $root "config\runtime.env"
  $runtime = Import-EnvFile -Path $runtimePath

  $activeProfile = $ProfileName
  if (-not $activeProfile) {
    $activeProfile = $runtime["ACTIVE_PROFILE"]
  }
  if (-not $activeProfile) {
    throw "ACTIVE_PROFILE is not set in $runtimePath"
  }

  $profilePath = Get-ProfilePath -WorkspaceDir $root -ProfileName $activeProfile
  $profile = Import-EnvFile -Path $profilePath

  $config = Merge-Config -Base $runtime -Overlay $profile
  $config["WORKSPACE_DIR"] = $root
  $config["PROFILE_PATH"] = $profilePath
  $config["ACTIVE_PROFILE"] = $activeProfile
  return $config
}

function Get-ConfigValue {
  param(
    [Parameter(Mandatory = $true)][hashtable]$Config,
    [Parameter(Mandatory = $true)][string]$Key,
    [string]$DefaultValue = ""
  )

  if ($Config.ContainsKey($Key) -and $null -ne $Config[$Key] -and "$($Config[$Key])" -ne "") {
    return "$($Config[$Key])"
  }

  return $DefaultValue
}

function Get-ModelPathFromConfig {
  param([Parameter(Mandatory = $true)][hashtable]$Config)

  $modelDir = Get-ConfigValue -Config $Config -Key "MODEL_DIR_WINDOWS"
  $entryFile = Get-ConfigValue -Config $Config -Key "MODEL_ENTRY_FILE"
  if (-not $modelDir -or -not $entryFile) {
    throw "MODEL_DIR_WINDOWS and MODEL_ENTRY_FILE must both be set for this profile."
  }

  return (Join-Path $modelDir $entryFile)
}
