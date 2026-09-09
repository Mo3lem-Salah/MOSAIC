"""Foreground launcher for a LinkGroup's shared xuance_worker subprocess.

A LinkGroup binds N agents to one shared MAPPO/IPPO checkpoint. Instead
of launching N subprocess workers (one per agent), this module launches
ONE shared xuance_worker subprocess that serves all N agents via per-call
``player_id`` routing. This saves memory (1x model vs Nx) and latency
(1x policy load vs N sequential loads).

The launch runs on the main Qt thread but pumps the Qt event loop
between I/O polls, so the GUI stays responsive during the 30-120 second
policy-load phase. This avoids the thread-safety pitfalls of calling
``subprocess.PIPE`` reads from a non-main thread while the main thread
may also touch the pipe.

See docs/Development_Progress/1.0_DAY_73/TASK_4/PLAN.md Diagram 1 for
the architectural rationale.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable, Optional

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

if TYPE_CHECKING:
    from gym_gui.services.operator import LinkGroup
    from gym_gui.services.operator_launcher import (
        OperatorLauncher,
        OperatorProcessHandle,
    )


_LOGGER = logging.getLogger(__name__)


class LinkGroupLauncher(QObject):
    """Foreground launcher for a LinkGroup's shared xuance_worker.

    Pumps the Qt event loop between I/O polls so the GUI and progress
    dialog stay responsive during the 30-120 second policy load.

    Signals (emitted on the main thread):
        progress(group_id, status, pct): 0-100 percent progress.
        ready(group_id, handle): launch succeeded.
        error(group_id, message): launch failed.
    """

    progress = pyqtSignal(str, str, int)
    ready = pyqtSignal(str, object)
    error = pyqtSignal(str, str)

    def __init__(
        self,
        group_id: str,
        link_group: "LinkGroup",
        env_name: str,
        task: str,
        view_size: Optional[int],
        launcher: "OperatorLauncher",
        operator_id: str,
        display_name: str,
        parent=None,
    ):
        super().__init__(parent)
        self.group_id = group_id
        self.link_group = link_group
        self.env_name = env_name
        self.task = task
        self.view_size = view_size
        self.launcher = launcher
        self.operator_id = operator_id
        self.display_name = display_name
        self._handle: Optional["OperatorProcessHandle"] = None
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        if self._handle is not None:
            try:
                self._handle.stop()
            except Exception:
                pass

    def run_sync(self) -> None:
        """Drive the launch on the main thread, pumping Qt events between I/O polls."""
        from gym_gui.services.operator import OperatorConfig

        try:
            # --- Phase 1: Spawn subprocess ---
            self._emit_progress("Spawning subprocess...", 10)
            if self._cancelled:
                return

            primary_agent = self.link_group.primary_agent
            settings = {
                "algorithm": self.link_group.algorithm,
                "policy_path": self.link_group.policy_path,
            }
            player_config = OperatorConfig.single_agent(
                operator_id=f"{self.operator_id}_linkgroup_{self.group_id}",
                display_name=f"{self.display_name} - shared RL ({self.group_id})",
                worker_id="xuance_worker",
                worker_type="rl",
                env_name=self.env_name,
                task=self.task,
                settings=settings,
                view_size=self.view_size,
            )

            try:
                self._handle = self.launcher.launch_operator(
                    player_config, interactive=True,
                )
            except Exception as exc:
                _LOGGER.exception(
                    "Failed to spawn shared RL subprocess for %s",
                    self.group_id,
                )
                self.error.emit(self.group_id, f"Spawn failed: {exc}")
                return

            if self._cancelled:
                self._handle.stop()
                return

            # --- Phase 2: Wait for startup ({"type": "init"}) ---
            self._emit_progress("Waiting for worker startup...", 30)
            startup_msg = self._read_with_event_loop(
                timeout_total=60.0,
                progress_start=30,
                progress_end=40,
                status="Waiting for worker startup...",
            )
            if self._cancelled:
                if self._handle:
                    self._handle.stop()
                return
            if startup_msg is None:
                self.error.emit(
                    self.group_id,
                    "Worker did not emit startup init message within 60s",
                )
                return

            # --- Phase 3: init_agent + wait for agent_initialized ---
            self._emit_progress("Loading policy checkpoint...", 50)
            self._handle.send_init_agent(
                game_name=self.task, player_id=primary_agent,
            )

            init_resp = self._read_with_event_loop(
                timeout_total=180.0,
                progress_start=50,
                progress_end=90,
                status="Loading policy checkpoint...",
            )
            if self._cancelled:
                if self._handle:
                    self._handle.stop()
                return
            if init_resp is None:
                self.error.emit(
                    self.group_id,
                    "Worker did not respond to init_agent within 180s",
                )
                return
            if init_resp.get("type") not in ("agent_ready", "agent_initialized"):
                self.error.emit(
                    self.group_id,
                    f"Unexpected init_agent response: {init_resp}",
                )
                return

            # --- Phase 4: Ready ---
            self._emit_progress("Ready", 100)
            self.ready.emit(self.group_id, self._handle)

        except Exception as exc:
            _LOGGER.exception("LinkGroupLauncher %s crashed", self.group_id)
            if self._handle is not None:
                try:
                    self._handle.stop()
                except Exception:
                    pass
            self.error.emit(self.group_id, f"Internal error: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit_progress(self, status: str, pct: int) -> None:
        self.progress.emit(self.group_id, status, pct)
        # Process Qt events to make the progress bar actually update
        QApplication.processEvents()

    def _read_with_event_loop(
        self,
        timeout_total: float,
        progress_start: int,
        progress_end: int,
        status: str,
        poll_chunk: float = 1.0,
    ):
        """Read one JSON response while pumping the Qt event loop.

        This is THE critical design decision in this module: instead of
        blocking on ``select()`` in a thread (which has Python GIL and
        subprocess-pipe issues), we poll in short 100ms chunks on the
        main thread and call ``QApplication.processEvents()`` between
        polls. The GUI repaints, the progress bar updates, and the
        user can click Cancel.

        We also don't call the handle's built-in ``read_response`` with a
        long timeout because that would block the event loop. Short
        polls + processEvents is the right pattern for this use case.
        """
        if self._handle is None:
            return None
        deadline = time.monotonic() + timeout_total
        start_time = time.monotonic()
        last_pct_emit = progress_start

        while time.monotonic() < deadline:
            if self._cancelled:
                return None

            # Non-blocking read (timeout=0.1s)
            resp = self._handle.try_read_response(timeout=poll_chunk)
            if resp is not None:
                return resp

            # Pump Qt events so the GUI stays alive
            QApplication.processEvents()

            # Emit progress tick based on elapsed fraction
            elapsed = time.monotonic() - start_time
            pct = progress_start + int(
                (progress_end - progress_start) * min(elapsed / timeout_total, 1.0)
            )
            if pct > last_pct_emit:
                self.progress.emit(self.group_id, status, pct)
                last_pct_emit = pct
                QApplication.processEvents()

        return None
