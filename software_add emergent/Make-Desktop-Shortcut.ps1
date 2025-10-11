
# Make-Desktop-Shortcut.ps1
param(
  [string]$TargetBat = "$(Join-Path $PSScriptRoot 'Run_LEAP_GUI.bat')",
  [string]$ShortcutName = "LEAP GUI.lnk"
)

$WshShell = New-Object -ComObject WScript.Shell
$Desktop = [Environment]::GetFolderPath("Desktop")
$Shortcut = $WshShell.CreateShortcut((Join-Path $Desktop $ShortcutName))
$Shortcut.TargetPath = $TargetBat
$Shortcut.WorkingDirectory = Split-Path -Parent $TargetBat
$Shortcut.WindowStyle = 1
$Shortcut.Description = "Run python main.py (LEAP GUI)"
$Shortcut.Save()
Write-Host "Shortcut created on Desktop: $ShortcutName"
