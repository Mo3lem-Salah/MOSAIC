"""Episode loading from the telemetry service for GUI replay.

================================================================================
WHEN TO USE THIS MODULE
================================================================================

    Use EpisodeLoader when you need to list or load recorded episodes
    from the telemetry service (SQLite). This is the read-path for the
    GUI replay tab.

    It provides:
    - Listing recent episodes with metadata (reward, steps, status)
    - Loading full episode data (step records) for playback
    - In-memory representation of episodes for the GUI

    NOTE: This module reads from TelemetryService (SQLite), not directly
    from HDF5. To get actual frame images, you need to resolve frame_ref
    URIs using FrameResolver (gym_gui.replays.FrameResolver).

================================================================================
ARCHITECTURE
================================================================================

    TelemetryService (gym_gui/services/telemetry.py)
        |
        +---> recent_episodes() -> list[EpisodeRollup]
        +---> episode_steps(id) -> list[StepRecord]
        |
        v
    EpisodeLoader (this module)
        |
        +---> recent_episode_ids(limit=20) -> list[str]
        +---> load_episode(id) -> EpisodeReplay
        |
        v
    _ReplayTab (gym_gui/ui/widgets/render_tabs.py)
        |
        +---> displays episode list in table
        +---> shows step-by-step playback

================================================================================
RELATIONSHIP TO OTHER MODULES
================================================================================

    EpisodeLoader  --->  reads episode metadata from SQLite via TelemetryService
                         (this module)
    FrameResolver  --->  resolves frame_ref URIs to numpy arrays from HDF5
                         (gym_gui.replays.FrameResolver)
    ReplayWriter   --->  writes frames/observations to HDF5 during training
                         (gym_gui.replays.ReplayWriter)
    ReplayReader   --->  reads HDF5 files directly (no SQLite needed)
                         (gym_gui.replays.ReplayReader)

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

from gym_gui.core.data_model import EpisodeRollup, StepRecord
from gym_gui.services.telemetry import TelemetryService


@dataclass(slots=True)
class EpisodeReplay:
    """In-memory representation of a telemetry-backed episode.

    This is a container for episode data loaded from SQLite via
    TelemetryService. It holds the episode metadata (rollup) and
    all step records.

    To get actual frame images for display, you need to resolve
    frame_ref URIs using FrameResolver::

        resolver = FrameResolver(replay_dir)
        for step in replay.steps:
            if step.frame_ref:
                frame = resolver.resolve(step.frame_ref)
                # frame is now a numpy array (H, W, C)
    """

    rollup: EpisodeRollup
    steps: List[StepRecord]

    def __iter__(self) -> Iterator[StepRecord]:
        return iter(self.steps)

    @property
    def total_reward(self) -> float:
        return self.rollup.total_reward

    @property
    def episode_id(self) -> str:
        return self.rollup.episode_id


__all__ = ["EpisodeReplay", "EpisodeLoader"]


class EpisodeLoader:
    """Load full episodes from telemetry services for playback.

    This is the READ-ONLY loader used by the GUI replay tab to list
    and load episodes from SQLite via TelemetryService.

    Usage::

        loader = EpisodeLoader(telemetry_service)

        # List recent episodes
        for ep_id in loader.recent_episode_ids(limit=20):
            print(ep_id)

        # Load full episode
        replay = loader.load_episode("run_abc123-ep0042")
        if replay:
            for step in replay.steps:
                print(step.step_index, step.reward)

    To get actual frame images, pair with FrameResolver::

        resolver = FrameResolver(replay_dir)
        replay = loader.load_episode(episode_id)
        for step in replay.steps:
            if step.frame_ref:
                frame = resolver.resolve(step.frame_ref)
    """

    def __init__(self, telemetry: TelemetryService) -> None:
        self._telemetry = telemetry

    def recent_episode_ids(self, *, limit: int = 20) -> Iterable[str]:
        return [episode.episode_id for episode in self._telemetry.recent_episodes()][:limit]

    def load_episode(self, episode_id: str) -> Optional[EpisodeReplay]:
        steps = list(self._telemetry.episode_steps(episode_id))
        if not steps:
            return None
        rollup = next(
            (episode for episode in self._telemetry.recent_episodes() if episode.episode_id == episode_id),
            None,
        )
        if rollup is None:
            rollup = EpisodeRollup(
                episode_id=episode_id,
                total_reward=sum(step.reward for step in steps),
                steps=len(steps),
                terminated=steps[-1].terminated,
                truncated=steps[-1].truncated,
                metadata={},
                agent_id=steps[-1].agent_id,
                game_id=steps[-1].render_payload.get("game_id") if isinstance(steps[-1].render_payload, dict) else None,
            )
        return EpisodeReplay(rollup=rollup, steps=steps)
