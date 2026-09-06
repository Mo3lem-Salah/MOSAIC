#!/usr/bin/env bash
# ==============================================================================
# MOSAIC: Server
# Run this on the compute / GPU node (Hamid@a1-R8428-G11).
# The GUI client connects from another machine via:
#   MOSAIC_DAEMON_TARGET=<server-ip>:50055 ./run_client.sh
#
# Usage:
#   ./run_server.sh                                                # start server
#   ./run_server.sh status                                         # check if running
#   ./run_server.sh stop                                           # stop server
#   ./run_server.sh attach                                         # reattach to live terminal
#   ./run_server.sh logs                                           # tail log file
#   DAEMON_PORT=50056 ./run_server.sh                              # custom port
#   DAEMON_HOST=127.0.0.1 ./run_server.sh                          # loopback only (local test)
#   MOSAIC_GPUS=0,1 ./run_server.sh                                # restrict to specific GPUs
#   MOSAIC_START_VLLM=1 MOSAIC_VLLM_MODEL=<path> ./run_server.sh  # with vLLM
#
# Persistence:
#   The daemon runs inside a tmux session named "mosaic-server".
#   It survives SSH disconnects. Reattach at any time with:
#     tmux attach -t mosaic-server
#   or:
#     ./run_server.sh attach
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SESSION="mosaic-server"
LOG_FILE="var/logs/trainer_daemon.log"

# ==============================================================================
# Helpers (defined before the case block that uses them)
# ==============================================================================

is_running() {
    tmux has-session -t "$SESSION" 2>/dev/null
}

# ==============================================================================
# Sub-commands
# ==============================================================================

case "${1:-start}" in
    status)
        if is_running; then
            echo "MOSAIC server is running (tmux session: $SESSION)"
            echo "  Attach : tmux attach -t $SESSION"
            echo "  Logs   : $LOG_FILE"
            exit 0
        else
            echo "MOSAIC server is not running"
            exit 1
        fi
        ;;
    stop)
        if is_running; then
            tmux kill-session -t "$SESSION"
            echo "MOSAIC server stopped"
        else
            echo "No server running"
        fi
        exit 0
        ;;
    attach)
        if is_running; then
            tmux attach -t "$SESSION"
        else
            echo "No server running. Start with: ./run_server.sh"
            exit 1
        fi
        exit 0
        ;;
    logs)
        tail -f "$LOG_FILE"
        exit 0
        ;;
    start|"")
        # Fall through to startup section
        ;;
    *)
        echo "Usage: $0 [start|status|stop|attach|logs]"
        exit 1
        ;;
esac

# ==============================================================================
# Startup
# ==============================================================================

if is_running; then
    echo "Server already running in tmux session '$SESSION'."
    echo "  Attach : ./run_server.sh attach"
    echo "  Status : ./run_server.sh status"
    exit 0
fi

if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

mkdir -p var/logs var/trainer

# Local .venv takes priority over any PYTHON_BIN set in .env (which may be from another machine)
if [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
fi

# ------------------------------------------------------------------------------
# Interactive configuration
# Only runs when DAEMON_PORT and MOSAIC_GPUS are not already set via the environment.
# To skip all prompts:
#   DAEMON_HOST=0.0.0.0 DAEMON_PORT=50055 MOSAIC_GPUS=0 ./run_server.sh
# ------------------------------------------------------------------------------
if [[ -z "${DAEMON_PORT:-}" ]]; then
    echo ""
    echo "=== MOSAIC Server Setup ==="
    echo "Press Enter to accept the value shown in [brackets]."
    echo ""

    read -p "Machine name (for display, e.g. my-gpu-server) [$(hostname)]: " _machine
    SERVER_MACHINE="${_machine:-$(hostname)}"

    read -p "Bind address (0.0.0.0 = all interfaces, 127.0.0.1 = local only) [0.0.0.0]: " _host
    DAEMON_HOST="${_host:-0.0.0.0}"

    read -p "Listen port [50055]: " _port
    DAEMON_PORT="${_port:-50055}"

    if command -v nvidia-smi &>/dev/null; then
        echo ""
        echo "Available GPUs:"
        nvidia-smi --list-gpus
        echo ""
        read -p "GPU device indices to expose (comma-separated, e.g. 0 or 0,1) [0]: " _gpus
        MOSAIC_GPUS="${_gpus:-0}"
    else
        echo "No NVIDIA GPU detected; running on CPU."
        MOSAIC_GPUS=""
    fi

    echo ""
    echo "Server will listen on: ${DAEMON_HOST}:${DAEMON_PORT}"
    [[ -n "$MOSAIC_GPUS" ]] && echo "GPUs exposed: ${MOSAIC_GPUS}"
    echo ""
else
    SERVER_MACHINE="$(hostname)"
    DAEMON_HOST="${DAEMON_HOST:-0.0.0.0}"
    DAEMON_PORT="${DAEMON_PORT:-50055}"
    MOSAIC_GPUS="${MOSAIC_GPUS:-0}"
fi

LISTEN="${DAEMON_HOST}:${DAEMON_PORT}"

# ------------------------------------------------------------------------------
# Optional: launch vLLM alongside the daemon
# Set MOSAIC_START_VLLM=1 and MOSAIC_VLLM_MODEL=<model_path> to enable.
# vLLM runs detached via nohup since it is a separate long-running process.
# ------------------------------------------------------------------------------
if [[ "${MOSAIC_START_VLLM:-0}" == "1" ]]; then
    VLLM_MODEL="${MOSAIC_VLLM_MODEL:-}"
    VLLM_PORT="${MOSAIC_VLLM_PORT:-8000}"
    if [ -z "$VLLM_MODEL" ]; then
        echo "WARNING: MOSAIC_START_VLLM=1 but MOSAIC_VLLM_MODEL is not set; skipping vLLM."
    else
        echo "Starting vLLM on 0.0.0.0:${VLLM_PORT} with model: ${VLLM_MODEL}"
        nohup vllm serve "$VLLM_MODEL" \
            --host 0.0.0.0 \
            --port "$VLLM_PORT" \
            --gpu-memory-utilization "${MOSAIC_VLLM_GPU_UTIL:-0.85}" \
            --enforce-eager \
            > "var/logs/vllm_server_${VLLM_PORT}.log" 2>&1 &
        echo "vLLM started (PID $!), logs: var/logs/vllm_server_${VLLM_PORT}.log"
    fi
fi

# ------------------------------------------------------------------------------
# Launch trainer daemon inside a detached tmux session.
#
# Why tmux over nohup+setsid:
#   - Survives SSH disconnects (session is independent of any terminal)
#   - Reattachable: "tmux attach -t mosaic-server" shows live output
#   - No PID file management, no EXIT trap contradictions
#
# The venv is sourced inside the tmux session independently because the
# session gets a fresh shell that does not inherit the outer environment.
# "|| true" ensures the daemon still starts if .venv is absent.
# ------------------------------------------------------------------------------
echo "Starting MOSAIC server on ${LISTEN} (tmux session: $SESSION, GPUs: ${MOSAIC_GPUS}) ..."

tmux new-session -d -s "$SESSION" \
    "cd '${ROOT_DIR}' && \
     [ -f .venv/bin/activate ] && source .venv/bin/activate || true; \
     export CUDA_VISIBLE_DEVICES='${MOSAIC_GPUS}'; \
     '${PYTHON_BIN}' -m gym_gui.services.trainer_daemon \
         --listen '${LISTEN}' \
         2>&1 | tee '${LOG_FILE}'"

# Brief pause to catch immediate startup failures (bad import, port in use, etc.)
sleep 1

if is_running; then
    echo ""
    echo "MOSAIC server started"
    echo "  Session  : $SESSION"
    echo "  Listening: $LISTEN"
    echo "  GPUs     : $MOSAIC_GPUS  (CUDA_VISIBLE_DEVICES)"
    echo "  Log file : $LOG_FILE"
    echo ""
    echo "Server will continue running after this SSH session ends."
    echo "  Reattach : ./run_server.sh attach   (or: tmux attach -t $SESSION)"
    echo "  Detach   : Ctrl+B  D"
    echo "  Status   : ./run_server.sh status"
    echo "  Stop     : ./run_server.sh stop"
else
    echo "ERROR: tmux session died immediately. Last log output:"
    tail -20 "$LOG_FILE" 2>/dev/null || echo "(no log file found)"
    exit 1
fi
