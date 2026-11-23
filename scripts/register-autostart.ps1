<#
.SYNOPSIS
  Start the shared alliance-slurm MCP server automatically when you log in to Windows.

.DESCRIPTION
  Registers a per-user scheduled task (no administrator rights needed) that
  runs scripts\shared-server.ps1 at logon. Remove it with -Unregister.

.EXAMPLE
  .\scripts\register-autostart.ps1
  .\scripts\register-autostart.ps1 -Unregister
#>
param(
    [int]$Port = 8765,
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"
$taskName = "alliance-slurm-mcp shared server"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Output "Removed scheduled task '$taskName'."
    return
}

$script = (Resolve-Path (Join-Path $PSScriptRoot "shared-server.ps1")).Path
$shell = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $shell) { $shell = (Get-Command powershell).Source }
$action = New-ScheduledTaskAction -Execute $shell `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`" -Port $Port"
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Shared alliance-slurm MCP server on http://127.0.0.1:$Port/mcp" -Force | Out-Null
Write-Output "Registered '$taskName': the shared server starts at each logon. Start it now with scripts\shared-server.ps1."
