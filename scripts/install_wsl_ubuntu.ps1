$ErrorActionPreference = "Stop"

try {
  $current = wsl -l -q 2>$null
} catch {
  $current = @()
}

if ($current -match "Ubuntu") {
  Write-Host "Ubuntu is already installed in WSL."
  exit 0
}

Write-Host "Installing Ubuntu for WSL2..."
Write-Host "If Windows asks for a reboot, reboot first and then finish Ubuntu setup."
wsl --install -d Ubuntu

