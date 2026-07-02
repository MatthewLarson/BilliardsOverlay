# Install the Pool Guide "brain" to auto-start at logon on Windows.
#
# Uses a Startup-folder shortcut (no admin needed). Also opens the firewall for
# the brain's ports if run elevated, and starts the brain immediately.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_brain_service.ps1
#
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pyw  = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pyw)) { throw "pythonw not found at $pyw -- create the venv first (pip install -e .)" }

# 1) Startup shortcut (runs at logon, no console window, no admin).
$startup = [Environment]::GetFolderPath("Startup")
$lnkPath = Join-Path $startup "PoolGuideBrain.lnk"
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut($lnkPath)
$lnk.TargetPath = $pyw
$lnk.Arguments = "-m pool_guide.apps.webui"
$lnk.WorkingDirectory = $root
$lnk.WindowStyle = 7
$lnk.Description = "Pool Guide brain control panel"
$lnk.Save()
Write-Host "Startup shortcut created: $lnkPath"

# 2) Firewall (needs admin; best-effort).
try {
    Remove-NetFirewallRule -DisplayName "Pool Guide brain" -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName "Pool Guide brain" -Direction Inbound -Action Allow `
        -Protocol TCP -LocalPort 5555,5556,8080 -Program $pyw -ErrorAction Stop | Out-Null
    Write-Host "Firewall rule added (ports 5555/5556/8080)."
} catch {
    Write-Warning "Firewall rule not added (needs Administrator). If the Pi can't reach the brain, run this script once in an elevated PowerShell, or allow python through Windows Firewall for a Private network."
}

# 3) Start now if not already listening on 8080.
$up = $false
try { $up = (Test-NetConnection -ComputerName 127.0.0.1 -Port 8080 -WarningAction SilentlyContinue).TcpTestSucceeded } catch {}
if (-not $up) {
    Start-Process -FilePath $pyw -ArgumentList "-m pool_guide.apps.webui" -WorkingDirectory $root -WindowStyle Hidden
    Write-Host "Brain started."
} else {
    Write-Host "Brain already running on :8080."
}
$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -like "192.168.*" } | Select-Object -First 1).IPAddress
Write-Host "Brain control panel: http://$ip:8080"
