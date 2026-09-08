"""HDF5-based replay reader for RL training experiences.

================================================================================
WHEN TO USE THIS MODULE
================================================================================

    Use ReplayReader to read back episode data from HDF5 files that were
    written by ReplayWriter. This is useful for:

    - Offline replay of recorded episodes (playback in GUI)
    - Training data loading (batch iteration for imitation learning)
    - Post-hoc analysis of training runs
    - Generating videos from recorded frames

    NOTE: The GUI replay tab currently uses EpisodeLoader + FrameResolver
    instead of this class. ReplayReader provides a standalone alternative
    that reads directly from HDF5 without going through SQLite/TelemetryService.
    It may be useful for offline analysis scripts or future training pipelines.

================================================================================
ARCHITECTURE
================================================================================

    HDF5 file (var/replay/{run_id}.h5)
        /frames        (N, H, W, C) uint8
        /observations  (N, ...) float32 or uint8
        /actions       (N,) int32 or (N, action_dim) float32
        /rewards       (N,) float32
        /dones         (N,) bool
        /episodes/starts   [0, 1523, 3102, ...]
        /episodes/lengths  [1523, 1579, ...]
        |
        v
    ReplayReader (this module)
        |
        +---> get_episode(idx)     -> full episode with frames
        +---> get_step(idx)        -> single step
        +---> iter_batches(32)     -> training batches

================================================================================
RELATIONSHIP TO OTHER MODULES
================================================================================

    ReplayWriter  --->  writes HDF5 files (gym_gui.replays.ReplayWriter)
    ReplayReader  --->  reads HDF5 files back (this module)
    FrameResolver --->  resolves frame_ref URIs from SQLite to numpy arrays
                        (gym_gui.replays.FrameResolver)

    The key difference between ReplayReader and FrameResolver:
    - ReplayReader reads entire episodes directly from HDF5 (no SQLite needed)
    - FrameResolver resolves individual frame_ref URIs stored in SQLite

================================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)

# Lazy import h5py to avoid hard dependency at module load time
_h5py = None


def _get_h5py():
    """Lazy import of h5py."""
    global _h5py
    if _h5py is None:
        try:
            import h5py
            _h5py = h5py
        except ImportError as e:
            raise ImportError(
                "h5py is required for HDF5 replay storage. "
                "Install with: pip install h5py"
            ) from e
    return _h5py


class ReplayReader:
    """Reads replay data from HDF5 for playback or training.

    This is a READ-ONLY class for loading episodes from HDF5 files that
    were written by ReplayWriter. It provides random access to individual
    steps, full episode extraction, and batch iteration for training.

    Usage::

        with ReplayReader(path) as reader:
            # Get metadata
            print(reader.num_steps, reader.num_episodes)

            # Get single step
            step = reader.get_step(100)

            # Get entire episode
            episode = reader.get_episode(0)

            # Iterate in batches for training
            for batch in reader.iter_batches(batch_size=32):
                train(batch)

    To write data, use ReplayWriter (gym_gui.replays.ReplayWriter).
    To resolve frame_ref URIs from SQLite, use FrameResolver
    (gym_gui.replays.FrameResolver).
    """

    def __init__(self, path: Path) -> None:
        """Initialize the replay reader.

        Args:
            path: Path to the HDF5 file
        """
        self._path = Path(path)
        self._file: Any = None  # h5py.File

    def open(self) -> None:
        """Open the HDF5 file for reading."""
        h5py = _get_h5py()
        self._file = h5py.File(self._path, "r")

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self) -> "ReplayReader":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def num_steps(self) -> int:
        """Return total number of steps."""
        if self._file is None:
            return 0
        if "actions" in self._file:
            return self._file["actions"].shape[0]
        return 0

    @property
    def num_episodes(self) -> int:
        """Return total number of episodes."""
        if self._file is None:
            return 0
        if "episodes/starts" in self._file:
            return len(self._file["episodes/starts"])
        return 0

    @property
    def metadata(self) -> dict:
        """Return file metadata as dict."""
        if self._file is None:
            return {}
        return dict(self._file.attrs)

    @property
    def has_frames(self) -> bool:
        """Return True if file contains frames dataset."""
        return self._file is not None and "frames" in self._file

    @property
    def has_observations(self) -> bool:
        """Return True if file contains observations dataset."""
        return self._file is not None and "observations" in self._file

    def get_step(self, index: int) -> dict:
        """Get a single step by index.

        Args:
            index: Step index (0-based)

        Returns:
            Dict with action, reward, done, and optionally frame/observation
        """
        if self._file is None:
            raise RuntimeError("Reader not open")

        result = {
            "action": int(self._file["actions"][index]),
            "reward": float(self._file["rewards"][index]),
            "done": bool(self._file["dones"][index]),
        }

        if "frames" in self._file:
            result["frame"] = self._file["frames"][index]
        if "observations" in self._file:
            result["observation"] = self._file["observations"][index]

        return result

    def get_episode(self, episode_idx: int) -> dict:
        """Get all data for an episode.

        Args:
            episode_idx: Episode index (0-based)

        Returns:
            Dict with arrays for the entire episode
        """
        if self._file is None:
            raise RuntimeError("Reader not open")

        starts = self._file["episodes/starts"][:]
        lengths = self._file["episodes/lengths"][:]

        if episode_idx >= len(starts):
            raise IndexError(f"Episode {episode_idx} not found (max: {len(starts)-1})")

        start = int(starts[episode_idx])
        length = int(lengths[episode_idx])
        end = start + length

        result = {
            "actions": self._file["actions"][start:end],
            "rewards": self._file["rewards"][start:end],
            "dones": self._file["dones"][start:end],
            "episode_index": episode_idx,
            "start_step": start,
            "length": length,
        }

        if "frames" in self._file:
            result["frames"] = self._file["frames"][start:end]
        if "observations" in self._file:
            result["observations"] = self._file["observations"][start:end]

        return result

    def get_frame(self, step_index: int) -> Optional[np.ndarray]:
        """Get a single frame by step index.

        Args:
            step_index: Step index (0-based)

        Returns:
            Frame array or None if not found
        """
        if self._file is None:
            return None
        if "frames" not in self._file:
            return None
        if step_index >= self._file["frames"].shape[0]:
            return None
        return self._file["frames"][step_index]

    def get_observation(self, step_index: int) -> Optional[np.ndarray]:
        """Get a single observation by step index.

        Args:
            step_index: Step index (0-based)

        Returns:
            Observation array or None if not found
        """
        if self._file is None:
            return None
        if "observations" not in self._file:
            return None
        if step_index >= self._file["observations"].shape[0]:
            return None
        return self._file["observations"][step_index]

    def iter_batches(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        include_frames: bool = False,
    ) -> Iterator[dict]:
        """Iterate through data in batches for training.

        Args:
            batch_size: Number of steps per batch
            shuffle: Whether to shuffle indices
            include_frames: Whether to include frame data

        Yields:
            Dict with batch arrays
        """
        if self._file is None:
            raise RuntimeError("Reader not open")

        indices = np.arange(self.num_steps)
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            # Sort for efficient HDF5 read (contiguous access)
            sorted_indices = np.sort(batch_indices)

            batch = {
                "actions": self._file["actions"][sorted_indices],
                "rewards": self._file["rewards"][sorted_indices],
                "dones": self._file["dones"][sorted_indices],
            }

            if "observations" in self._file:
                batch["observations"] = self._file["observations"][sorted_indices]

            if include_frames and "frames" in self._file:
                batch["frames"] = self._file["frames"][sorted_indices]

            yield batch


__all__ = ["ReplayReader"]
