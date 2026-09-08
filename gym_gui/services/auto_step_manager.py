"""AutoStepManager -- pre-run N steps, cache frames, replay at user interval.

Architecture:
  1. Collection phase: emits reset_requested / step_requested signals; main_window
     routes responses back via on_ready_received / on_step_collected.  No threads --
     fully event-driven via the existing QTimer poll path.
  2. Replay phase: internal QTimer emits display_frame each tick; main_window calls
     display_operator_payload with the cached payload.

The manager is idle by default and becomes active only after start_collection().
main_window checks is_active_for(operator_id) in _handle_operator_response to
route step/ready responses here instead of the normal display path.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

_LOGGER = logging.getLogger(__name__)


class AutoStepManager(QObject):
    """Event-driven pre-run + replay controller for RL operators."""

    # Phase 1 requests -- main_window handles these
    reset_requested = pyqtSignal(str, int)   # operator_id, seed
    step_requested = pyqtSignal(str)          # operator_id

    # Progress -- drives the QProgressDialog in main_window
    frame_collected = pyqtSignal(str, int, int)  # operator_id, step, total

    # Phase 1 done -- all operators collected their frames
    collection_done = pyqtSignal()

    # Phase 2 replay -- main_window calls display_operator_payload with this
    display_frame = pyqtSignal(str, dict)   # operator_id, payload dict

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._collecting: bool = False
        self._replaying: bool = False
        self._operator_ids: List[str] = []
        self._n_steps: int = 1000
        self._seed: int = 0
        self._frames: Dict[str, List[dict]] = {}
        self._step_counts: Dict[str, int] = {}
        self._done_operators: set[str] = set()
        self._replay_pos: Dict[str, int] = {}
        self._replay_timer = QTimer(self)
        self._replay_timer.timeout.connect(self._replay_tick)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_active_for(self, operator_id: str) -> bool:
        """True while collecting frames for this operator."""
        return self._collecting and operator_id in self._operator_ids

    @property
    def is_collecting(self) -> bool:
        return self._collecting

    @property
    def is_replaying(self) -> bool:
        return self._replaying

    def frame_count(self, operator_id: str) -> int:
        return len(self._frames.get(operator_id, []))

    # ------------------------------------------------------------------
    # Phase 1 -- collection
    # ------------------------------------------------------------------

    def start_collection(
        self,
        operator_ids: List[str],
        seed: int,
        n_steps: int = 1000,
    ) -> None:
        self.stop()
        self._collecting = True
        self._operator_ids = list(operator_ids)
        self._n_steps = n_steps
        self._seed = seed
        self._frames = {op: [] for op in operator_ids}
        self._step_counts = {op: 0 for op in operator_ids}
        self._done_operators = set()
        _LOGGER.info(
            "AutoStepManager: starting collection ops=%s seed=%d n_steps=%d",
            operator_ids, seed, n_steps,
        )
        for op_id in operator_ids:
            self.reset_requested.emit(op_id, seed)

    def on_ready_received(self, operator_id: str) -> None:
        """Called by main_window when operator emits 'ready' during collection."""
        if not self._collecting or operator_id not in self._operator_ids:
            return
        self.step_requested.emit(operator_id)

    def on_step_collected(
        self,
        operator_id: str,
        payload: dict,
        episode_done: bool,
    ) -> None:
        """Called by main_window for each step/episode_done response during collection."""
        if not self._collecting or operator_id not in self._operator_ids:
            return

        if payload.get("render_payload"):
            self._frames[operator_id].append(payload)

        self._step_counts[operator_id] = self._step_counts.get(operator_id, 0) + 1
        count = self._step_counts[operator_id]
        self.frame_collected.emit(operator_id, count, self._n_steps)

        if episode_done or count >= self._n_steps:
            _LOGGER.info(
                "AutoStepManager: %s done collecting (%d frames, episode_done=%s)",
                operator_id, len(self._frames[operator_id]), episode_done,
            )
            self._done_operators.add(operator_id)
            if len(self._done_operators) >= len(self._operator_ids):
                self._collecting = False
                self.collection_done.emit()
        else:
            self.step_requested.emit(operator_id)

    # ------------------------------------------------------------------
    # Phase 2 -- replay
    # ------------------------------------------------------------------

    def start_replay(self, interval_ms: int = 500) -> None:
        if not any(self._frames.values()):
            _LOGGER.warning("AutoStepManager: no frames to replay")
            return
        self._replaying = True
        self._replay_pos = {op: 0 for op in self._operator_ids}
        self._replay_timer.start(interval_ms)
        _LOGGER.info("AutoStepManager: replay started interval_ms=%d", interval_ms)

    def stop_replay(self) -> None:
        self._replay_timer.stop()
        self._replaying = False

    # ------------------------------------------------------------------
    # Full stop
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._replay_timer.stop()
        self._collecting = False
        self._replaying = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _replay_tick(self) -> None:
        for op_id in self._operator_ids:
            frames = self._frames.get(op_id)
            if not frames:
                continue
            pos = self._replay_pos.get(op_id, 0)
            self.display_frame.emit(op_id, frames[pos])
            self._replay_pos[op_id] = (pos + 1) % len(frames)


__all__ = ["AutoStepManager"]
