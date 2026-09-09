from __future__ import annotations

"""InferenceServicer — vLLM server lifecycle management on the daemon machine."""

import asyncio
import logging
import urllib.request
from typing import Any, Dict, Set, cast

from gym_gui.services.trainer.proto import inference_pb2 as _pb2_module
from gym_gui.services.trainer.proto import inference_pb2_grpc

pb2 = cast(Any, _pb2_module)

_LOGGER = logging.getLogger("gym_gui.trainer.inference_service")
_BASE_PORT = 8000
_HEALTH_POLL_INTERVAL_S = 2.0
_HEALTH_MAX_ATTEMPTS = 300  # 10 minutes


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """Manages vLLM server subprocesses on the machine running the daemon.

    server_id to port mapping: server_id 0 to port 8000, server_id 1 to port 8001, etc.
    Start calls are idempotent: a second StartVLLMServer for a running server
    returns immediately with its current status.
    """

    def __init__(self) -> None:
        # server_id -> {process, port, model_id, status}
        self._registry: Dict[int, dict] = {}
        self._event_queues: Set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # RPCs
    # ------------------------------------------------------------------

    async def StartVLLMServer(self, request, context):
        server_id = request.server_id
        port = _BASE_PORT + server_id

        async with self._lock:
            entry = self._registry.get(server_id)
            if entry and entry["status"] in ("starting", "running"):
                return pb2.StartVLLMResponse(port=entry["port"], status=entry["status"])

        await self._broadcast(server_id, "starting", port, request.model_path, "")

        gpu_util = request.gpu_memory_utilization if request.gpu_memory_utilization > 0 else 0.85
        # Build argument list explicitly to avoid shell injection.
        args = [
            "vllm", "serve", request.model_path,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_util),
        ]
        if request.enforce_eager:
            args.append("--enforce-eager")

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except Exception as exc:
            _LOGGER.error("Failed to start vLLM server %d: %s", server_id, exc)
            await self._broadcast(server_id, "error", port, request.model_path, str(exc))
            return pb2.StartVLLMResponse(port=port, status="error", message=str(exc))

        async with self._lock:
            self._registry[server_id] = {
                "process": proc,
                "port": port,
                "model_id": request.model_path,
                "status": "starting",
            }

        asyncio.create_task(
            self._poll_health(server_id, port, request.model_path),
            name=f"vllm-health-{server_id}",
        )
        return pb2.StartVLLMResponse(port=port, status="starting")

    async def StopVLLMServer(self, request, context):
        server_id = request.server_id
        async with self._lock:
            entry = self._registry.pop(server_id, None)

        if entry:
            proc = entry["process"]
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            await self._broadcast(server_id, "stopped", entry["port"], entry["model_id"], "")

        return pb2.StopVLLMResponse(status="stopped")

    async def WatchVLLMServers(self, request, context):
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._event_queues.add(q)
        # Send a snapshot of current state to the new subscriber.
        async with self._lock:
            for sid, entry in self._registry.items():
                await q.put(pb2.VLLMServerEvent(
                    server_id=sid,
                    port=entry["port"],
                    status=entry["status"],
                    model_id=entry["model_id"],
                ))
        try:
            while True:
                yield await q.get()
        finally:
            self._event_queues.discard(q)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _poll_health(self, server_id: int, port: int, model_id: str) -> None:
        """Poll the vLLM /health endpoint until running or the process exits."""
        url = f"http://127.0.0.1:{port}/health"
        for _ in range(_HEALTH_MAX_ATTEMPTS):
            await asyncio.sleep(_HEALTH_POLL_INTERVAL_S)

            async with self._lock:
                entry = self._registry.get(server_id)
                if not entry or entry["status"] == "stopped":
                    return
                proc = entry["process"]

            if proc.returncode is not None:
                _LOGGER.warning("vLLM server %d exited (rc=%s)", server_id, proc.returncode)
                async with self._lock:
                    if server_id in self._registry:
                        self._registry[server_id]["status"] = "error"
                await self._broadcast(server_id, "error", port, model_id,
                                      f"Process exited with code {proc.returncode}")
                return

            try:
                with urllib.request.urlopen(url, timeout=2):
                    pass
                _LOGGER.info("vLLM server %d healthy on port %d", server_id, port)
                async with self._lock:
                    if server_id in self._registry:
                        self._registry[server_id]["status"] = "running"
                await self._broadcast(server_id, "running", port, model_id, "")
                return
            except Exception:
                continue

        _LOGGER.error("vLLM server %d never became healthy after %d attempts",
                      server_id, _HEALTH_MAX_ATTEMPTS)
        await self._broadcast(server_id, "error", port, model_id, "Health check timed out")

    async def _broadcast(self, server_id: int, status: str, port: int,
                         model_id: str, error: str) -> None:
        event = pb2.VLLMServerEvent(
            server_id=server_id, port=port, status=status,
            model_id=model_id, error_message=error,
        )
        stale: Set[asyncio.Queue] = set()
        for q in self._event_queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                stale.add(q)
        self._event_queues -= stale


__all__ = ["InferenceServicer"]
