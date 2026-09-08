from __future__ import annotations

"""SystemServicer — GPU and system resource monitoring on the daemon machine."""

import asyncio
import logging
from typing import Any, cast

from gym_gui.services.trainer.proto import system_pb2 as _pb2_module
from gym_gui.services.trainer.proto import system_pb2_grpc

pb2 = cast(Any, _pb2_module)

_LOGGER = logging.getLogger("gym_gui.trainer.system_service")
_DEFAULT_INTERVAL_S = 1.0


class SystemServicer(system_pb2_grpc.SystemServiceServicer):
    """Streams GPU and system resource info from the daemon machine.

    pynvml is initialised once at construction time (NVML best practice).
    If pynvml is unavailable the service still runs, returning empty GPU lists.
    """

    def __init__(self) -> None:
        self._nvml_ok = False
        self._gpu_count = 0
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._gpu_count = pynvml.nvmlDeviceGetCount()
            self._nvml_ok = True
            _LOGGER.info("SystemServicer: pynvml ready, %d GPU(s) detected", self._gpu_count)
        except Exception as exc:
            _LOGGER.warning("SystemServicer: pynvml unavailable (%s) — GPU info will be empty", exc)

    # ------------------------------------------------------------------
    # RPCs
    # ------------------------------------------------------------------

    async def GetSystemInfo(self, request, context):
        return self._snapshot()

    async def WatchSystemInfo(self, request, context):
        interval = (request.interval_ms / 1000.0) if request.interval_ms > 0 else _DEFAULT_INTERVAL_S
        while True:
            yield self._snapshot()
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot(self):
        """Build a SystemInfoEvent from current NVML + psutil readings."""
        gpus = []
        if self._nvml_ok:
            pynvml = self._pynvml
            for i in range(self._gpu_count):
                try:
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    try:
                        temp = float(pynvml.nvmlDeviceGetTemperature(
                            h, pynvml.NVML_TEMPERATURE_GPU))
                    except Exception:
                        temp = 0.0
                    name = pynvml.nvmlDeviceGetName(h)
                    gpus.append(pb2.GPUInfo(
                        index=i,
                        name=str(name),
                        memory_used_gb=int(mem.used) / 1e9,
                        memory_total_gb=int(mem.total) / 1e9,
                        utilization_percent=float(util.gpu),
                        temperature_celsius=temp,
                    ))
                except Exception as exc:
                    _LOGGER.debug("Could not read GPU %d info: %s", i, exc)

        try:
            import psutil
            vm = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)
        except Exception:
            return pb2.SystemInfoEvent(gpus=gpus)

        return pb2.SystemInfoEvent(
            gpus=gpus,
            cpu_percent=cpu,
            ram_used_gb=vm.used / 1e9,
            ram_total_gb=vm.total / 1e9,
        )


__all__ = ["SystemServicer"]
