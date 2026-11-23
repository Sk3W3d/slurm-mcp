<#
.SYNOPSIS
  Start, stop, or check the shared alliance-slurm MCP server (HTTP on localhost).

.DESCRIPTION
  One shared server holds the SSH connections for every client (Claude Code
  sessions, Codex), so each cluster needs one Duo approval while it runs.
  Clients connect to http://127.0.0.1:<port>/mcp with the bearer token from
  ~/.slurm-mcp/server-token. Logs go to ~/.slurm-mcp/server.log.

.EXAMPLE
  .\scripts\shared-server.ps1            # start (does nothing if already running)
  .\scripts\shared-server.ps1 -Status
  .\scripts\shared-server.ps1 -Stop
#>
param(
    [int]$Port = 8765,
    [string]$Config = (Join-Path $PSScriptRoot "..\config.yaml"),
    [switch]$Stop,
    [switch]$Status
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "..\.venv\Scripts\alliance-slurm-mcp.exe"
$stateDir = Join-Path $HOME ".slurm-mcp"
$log = Join-Path $stateDir "server.log"
$errLog = Join-Path $stateDir "server.err.log"

function Get-ServerProcesses {
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -like "*alliance-slurm-mcp*--serve*" -and $_.CommandLine -like "*--port $Port*" }
}

function Test-Listening {
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $client.Connect("127.0.0.1", $Port)
        $client.Dispose()
        return $true
    } catch {
        return $false
    }
}

if ($Status) {
    $procs = @(Get-ServerProcesses)
    if (Test-Listening) {
        Write-Output "Running on http://127.0.0.1:$Port/mcp (process $($procs.ProcessId -join ', ')). Log: $errLog"
    } else {
        Write-Output "Not running on port $Port."
    }
    return
}

if ($Stop) {
    $procs = @(Get-ServerProcesses)
    if (-not $procs) { Write-Output "Not running on port $Port."; return }
    foreach ($proc in $procs) { Stop-Process -Id $proc.ProcessId -Force -Confirm:$false }
    Write-Output "Stopped (process $($procs.ProcessId -join ', ')). Open SSH connections are closed; the next login needs Duo."
    return
}

if (Test-Listening) {
    Write-Output "Already running on http://127.0.0.1:$Port/mcp."
    return
}
if (-not (Test-Path $exe)) { throw "Server executable not found: $exe (create the venv and pip install -e . first)." }
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
$configPath = (Resolve-Path $Config).Path
Start-Process -FilePath $exe `
    -ArgumentList @("--config", "`"$configPath`"", "--serve", "--port", "$Port") `
    -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $errLog | Out-Null

for ($i = 0; $i -lt 40; $i++) {
    if (Test-Listening) {
        Write-Output "Started on http://127.0.0.1:$Port/mcp. Token: $(Join-Path $stateDir 'server-token'). Log: $errLog"
        return
    }
    Start-Sleep -Milliseconds 250
}
throw "The server did not start listening on port $Port within 10 s. See $errLog."
