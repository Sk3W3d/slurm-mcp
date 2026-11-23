#!/bin/sh
# Run the shared alliance-slurm MCP server as a systemd user service (Linux).
#
# Usage:
#   scripts/install-systemd-user.sh [--port N] [--config PATH]   # install, enable, start
#   scripts/install-systemd-user.sh --uninstall
#
# The service runs while you are logged in. To keep it running after you log
# out (e.g. on a workstation you ssh into), enable lingering once:
#   loginctl enable-linger "$USER"     (may need an administrator on some systems)
set -eu

port=8765
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(dirname -- "$script_dir")
config="$repo/config.yaml"
uninstall=0
unit=alliance-slurm-mcp.service
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

while [ $# -gt 0 ]; do
    case "$1" in
        --port) port=$2; shift ;;
        --config) config=$2; shift ;;
        --uninstall) uninstall=1 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ "$uninstall" -eq 1 ]; then
    systemctl --user disable --now "$unit" 2>/dev/null || true
    rm -f "$unit_dir/$unit"
    systemctl --user daemon-reload
    echo "Removed $unit."
    exit 0
fi

exe="$repo/.venv/bin/alliance-slurm-mcp"
[ -x "$exe" ] || { echo "Server executable not found: $exe (python -m venv .venv && .venv/bin/pip install -e .)" >&2; exit 1; }
config=$(CDPATH= cd -- "$(dirname -- "$config")" && pwd)/$(basename -- "$config")
[ -f "$config" ] || { echo "Config not found: $config" >&2; exit 1; }

mkdir -p "$unit_dir" "$HOME/.slurm-mcp"
chmod 700 "$HOME/.slurm-mcp"
cat >"$unit_dir/$unit" <<EOF
[Unit]
Description=Shared alliance-slurm MCP server (http://127.0.0.1:$port/mcp)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=$exe --config $config --serve --port $port
# The token file the server creates stays private.
UMask=0077
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "$unit"
echo "Installed and started $unit on http://127.0.0.1:$port/mcp."
echo "Status: systemctl --user status $unit   Logs: journalctl --user -u $unit"
