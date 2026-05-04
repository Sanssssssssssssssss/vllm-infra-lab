$ErrorActionPreference = "Stop"

Write-Host "== Host OS =="
Get-CimInstance Win32_OperatingSystem |
  Select-Object Caption, Version, OSArchitecture |
  Format-List

Write-Host "== CPU =="
Get-CimInstance Win32_Processor |
  Select-Object Name |
  Format-List

Write-Host "== GPU =="
Get-CimInstance Win32_VideoController |
  Select-Object Name, DriverVersion |
  Format-Table -Auto

Write-Host "== RAM =="
Get-CimInstance Win32_ComputerSystem |
  Select-Object @{Name="TotalPhysicalMemoryGB"; Expression={[Math]::Round($_.TotalPhysicalMemory / 1GB, 2)}} |
  Format-List

Write-Host "== Python =="
try {
  py -0p
} catch {
  Write-Warning "Python launcher was not found."
}

Write-Host "== NVIDIA-SMI =="
try {
  nvidia-smi
} catch {
  Write-Warning "nvidia-smi is not available."
}

Write-Host "== WSL Status =="
$distros = @()
try {
  $distros = wsl -l -q 2>$null | Where-Object { $_.Trim() }
} catch {
  $distros = @()
}

if ($distros.Count -eq 0) {
  Write-Warning "WSL is present, but no Linux distribution is installed yet."
} else {
  $distros | ForEach-Object { Write-Host $_ }
}

Write-Host "== IPv4 Addresses =="
Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown" } |
  Select-Object InterfaceAlias, IPAddress, PrefixLength |
  Format-Table -Auto
