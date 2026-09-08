"""Deployment mode detection: local vs. remote daemon.

Reads MOSAIC_DAEMON_TARGET at import time to determine whether GPU-heavy
components (vLLM, operators) should be treated as local or remote.

All modules that need host-aware URLs import from here so the source of
truth is a single env var, not scattered hardcoded strings.
"""
from __future__ import annotations

import os

_target = os.environ.get("MOSAIC_DAEMON_TARGET", "127.0.0.1:50055")
_host = _target.split(":")[0]
_LOCAL_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

IS_REMOTE_MODE: bool = _host not in _LOCAL_HOSTS
"""True when the trainer daemon (and GPU resources) are on a remote machine."""

DAEMON_HOST: str = _host
"""Hostname or IP of the machine running the trainer daemon."""

DAEMON_TARGET: str = _target
"""Full host:port of the trainer daemon (e.g. '192.168.0.6:50055')."""

# ── Server SSH config (used when IS_REMOTE_MODE=True) ────────────────────────
# Override via env vars if your server layout differs from the defaults.
SERVER_SSH_USER: str = os.environ.get("MOSAIC_SERVER_SSH_USER", "hamid")
"""SSH username on the server machine."""

SERVER_PROJECT_ROOT: str = os.environ.get(
    "MOSAIC_SERVER_PROJECT_ROOT",
    f"/home/{SERVER_SSH_USER}/projects/2x6000/mosaic",
)
"""Absolute path to the MOSAIC project root on the server."""

SERVER_PYTHON: str = os.environ.get(
    "MOSAIC_SERVER_PYTHON",
    f"{SERVER_PROJECT_ROOT}/.venv/bin/python",
)
"""Python executable on the server (inside the server venv)."""

SERVER_SSH_HOST: str = f"{SERVER_SSH_USER}@{DAEMON_HOST}" if IS_REMOTE_MODE else ""
"""SSH target string (user@host) for connecting to the server."""


def vllm_host() -> str:
    """vLLM bind address: 0.0.0.0 on the server, 127.0.0.1 locally."""
    return "0.0.0.0" if IS_REMOTE_MODE else "127.0.0.1"


def vllm_base_url(port: int = 8000) -> str:
    """OpenAI-compatible base URL for connecting to a vLLM server."""
    host = DAEMON_HOST if IS_REMOTE_MODE else "127.0.0.1"
    return f"http://{host}:{port}/v1"


def vllm_health_url(port: int) -> str:
    """Health endpoint URL for a vLLM server at the given port."""
    host = DAEMON_HOST if IS_REMOTE_MODE else "127.0.0.1"
    return f"http://{host}:{port}/health"


__all__ = [
    "IS_REMOTE_MODE",
    "DAEMON_HOST",
    "DAEMON_TARGET",
    "SERVER_SSH_USER",
    "SERVER_PROJECT_ROOT",
    "SERVER_PYTHON",
    "SERVER_SSH_HOST",
    "vllm_host",
    "vllm_base_url",
    "vllm_health_url",
]
