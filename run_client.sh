#!/usr/bin/env bash
# ==============================================================================
# MOSAIC: Client
# Run this on any machine with a display.
#
# Usage (interactive, script will ask for server details):
#   ./run_client.sh
#
# Usage (non-interactive, pass values as environment variables):
#   MOSAIC_DAEMON_TARGET=192.168.0.4:50055 ./run_client.sh
#
# The server must be running before launching the client:
#   Local server:   ./run_server.sh  (on this machine)
#   Remote server:  ssh user@<ip-address> 'cd ~/mosaic && bash run_server.sh'
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

mkdir -p var/logs

# Local .venv takes priority over any PYTHON_BIN set in .env (which may be from another machine)
if [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
fi

# ------------------------------------------------------------------------------
# Interactive configuration
# Only runs when MOSAIC_DAEMON_TARGET is not already set via the environment.
# To skip all prompts:
#   MOSAIC_DAEMON_TARGET=192.168.0.4:50055 ./run_client.sh
# ------------------------------------------------------------------------------
if [[ -z "${MOSAIC_DAEMON_TARGET:-}" ]]; then
    echo ""
    echo "=== MOSAIC Client Setup ==="
    echo "Press Enter to accept the value shown in [brackets]."
    echo ""

    read -p "Server machine name (for display, e.g. my-gpu-server) [server]: " _machine
    SERVER_MACHINE="${_machine:-server}"

    read -p "Server IP address [127.0.0.1]: " _host
    SERVER_HOST="${_host:-127.0.0.1}"

    read -p "Server port [50055]: " _port
    MOSAIC_DAEMON_TARGET_PORT="${_port:-50055}"

    MOSAIC_DAEMON_TARGET="${SERVER_HOST}:${MOSAIC_DAEMON_TARGET_PORT}"
    echo ""
    echo "Connecting to: ${SERVER_MACHINE} at ${MOSAIC_DAEMON_TARGET}"
    echo ""
else
    SERVER_MACHINE="server"
    MOSAIC_DAEMON_TARGET_PORT="${MOSAIC_DAEMON_TARGET##*:}"
fi

DAEMON_TARGET="${MOSAIC_DAEMON_TARGET}"
DAEMON_HOST="${DAEMON_TARGET%%:*}"

echo "Connecting to MOSAIC server at ${DAEMON_TARGET} ..."

# ------------------------------------------------------------------------------
# Verify the server is reachable before opening the GUI
# ------------------------------------------------------------------------------
if ! "$PYTHON_BIN" -c "
import asyncio, grpc, sys
async def check():
    ch = grpc.aio.insecure_channel('${DAEMON_TARGET}')
    try:
        await asyncio.wait_for(ch.channel_ready(), timeout=5.0)
        print('Server reachable at ${DAEMON_TARGET}')
    except Exception as e:
        print(f'ERROR: Cannot reach server at ${DAEMON_TARGET}: {e}', file=sys.stderr)
        sys.exit(1)
    finally:
        await ch.close()
asyncio.run(check())
"; then
    if [[ "$DAEMON_HOST" == "127.0.0.1" || "$DAEMON_HOST" == "localhost" || "$DAEMON_HOST" == "::1" ]]; then
        echo "Hint: start the local server first:" >&2
        echo "  bash run_server.sh" >&2
        echo "Common fix if proto errors: bash tools/generate_protos.sh && bash run_server.sh" >&2
    else
        echo "Hint: start the remote server on ${DAEMON_HOST}:" >&2
        echo "  ssh user@${DAEMON_HOST} 'cd ~/mosaic && bash run_server.sh'" >&2
    fi
    exit 1
fi

# ------------------------------------------------------------------------------
# Launch MOSAIC GUI
# ------------------------------------------------------------------------------
echo "Launching MOSAIC..."
QT_API=PyQt6 QT_DEBUG_PLUGINS=0 \
    MOSAIC_DAEMON_TARGET="${DAEMON_TARGET}" \
    "$PYTHON_BIN" -m gym_gui.app
