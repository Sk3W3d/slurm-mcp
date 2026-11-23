#!/bin/sh
# Start, stop, or check the shared alliance-slurm MCP server (HTTP on localhost).
#
# One shared server holds the SSH connections for every client (Claude Code
# sessions, Codex), so each cluster needs one Duo approval while it runs.
# Clients connect to http://127.0.0.1:<port>/mcp with the bearer token in
# ~/.slurm-mcp/server-token. Logs go to ~/.slurm-mcp/server.log.
#
# Usage:
#   scripts/shared-server.sh [start|stop|status|restart] [--port N] [--config PATH]
#
# For systemd autostart see scripts/install-systemd-user.sh.
set -eu

action=start
port=8765
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(dirname -- "$script_dir")
config="$repo/config.yaml"

while [ $# -gt 0 ]; do
    case "$1" in
        start|stop|status|restart) action=$1 ;;
        --port) port=$2; shift ;;
        --port=*) port=${1#--port=} ;;
        --config) config=$2; shift ;;
        --config=*) config=${1#--config=} ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# The venv layout differs: .venv/bin on Linux/macOS, .venv/Scripts on Windows (Git Bash).
if [ -x "$repo/.venv/bin/alliance-slurm-mcp" ]; then
    exe="$repo/.venv/bin/alliance-slurm-mcp"; python="$repo/.venv/bin/python"
else
    exe="$repo/.venv/Scripts/alliance-slurm-mcp.exe"; python="$repo/.venv/Scripts/python.exe"
fi
exe=${SLURM_MCP_EXE:-$exe}
state_dir="$HOME/.slurm-mcp"
log="$state_dir/server.log"
pidfile="$state_dir/server-$port.pid"

listening() {
    "$python" -c 'import socket, sys
s = socket.socket(); s.settimeout(1)
sys.exit(s.connect_ex(("127.0.0.1", int(sys.argv[1]))))' "$port" 2>/dev/null
}

running_pid() {
    [ -f "$pidfile" ] || return 1
    pid=$(cat "$pidfile")
    kill -0 "$pid" 2>/dev/null && { echo "$pid"; return 0; }
    rm -f "$pidfile"
    return 1
}

start() {
    if listening; then
        echo "Already running on http://127.0.0.1:$port/mcp."
        return 0
    fi
    [ -x "$exe" ] || { echo "Server executable not found: $exe (create the venv and pip install -e . first)." >&2; exit 1; }
    [ -f "$config" ] || { echo "Config not found: $config" >&2; exit 1; }
    mkdir -p "$state_dir"
    chmod 700 "$state_dir" 2>/dev/null || true
    umask 077  # the token file the server creates stays private
    nohup "$exe" --config "$config" --serve --port "$port" >>"$log" 2>&1 &
    echo $! >"$pidfile"
    i=0
    while [ $i -lt 40 ]; do
        if listening; then
            echo "Started on http://127.0.0.1:$port/mcp (pid $(cat "$pidfile")). Token: $state_dir/server-token. Log: $log"
            return 0
        fi
        sleep 0.25
        i=$((i + 1))
    done
    echo "The server did not start listening on port $port within 10 s. Last log lines:" >&2
    tail -n 20 "$log" >&2
    exit 1
}

stop() {
    if pid=$(running_pid); then
        kill "$pid"
        i=0
        while kill -0 "$pid" 2>/dev/null && [ $i -lt 20 ]; do sleep 0.25; i=$((i + 1)); done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        rm -f "$pidfile"
        echo "Stopped (pid $pid). Paramiko connections are closed (the next login needs Duo); OpenSSH ControlMaster connections stay open."
    elif listening; then
        echo "Something is listening on port $port but was not started by this script (systemd?). Try: systemctl --user stop alliance-slurm-mcp" >&2
        exit 1
    else
        echo "Not running on port $port."
    fi
}

case "$action" in
    start) start ;;
    stop) stop ;;
    restart) stop; start ;;
    status)
        if listening; then
            pid=$(running_pid || echo "unknown, maybe systemd")
            echo "Running on http://127.0.0.1:$port/mcp (pid $pid). Log: $log"
        else
            echo "Not running on port $port."
        fi
        ;;
esac
