"""Replay system for recording, storing, and playing back RL episodes.

================================================================================
OVERVIEW
================================================================================

    This package provides the complete replay pipeline for MOSAIC:

    1. WRITE (training time):
       ReplayWriter records frames, observations, actions, and rewards
       into HDF5 files with background threading. It produces frame_ref
       URIs that get stored in SQLite alongside scalar telemetry.

    2. STORE (on disk):
       Data is split between SQLite (scalars + references) and HDF5
       (large arrays). Frame_ref URIs like "h5://run_abc123/frames/1523"
       bridge the two stores.

    3. READ (replay time):
       - EpisodeLoader lists and loads episodes from SQLite via
         TelemetryService (for the GUI replay tab)
       - FrameResolver resolves frame_ref URIs to numpy arrays
         from HDF5 (for displaying frames in the GUI)
       - ReplayReader reads entire episodes directly from HDF5
         (for offline analysis or training data loading)

================================================================================
QUICK REFERENCE
================================================================================

    Module              | Purpose                        | When to use
    --------------------|--------------------------------|--------------------------
    ReplayWriter        | Write episodes to HDF5         | During training (db_sink)
    ReplayReader        | Read episodes from HDF5        | Offline analysis, training
    FrameResolver       | Resolve h5:// URIs to arrays   | GUI display, telemetry
    EpisodeLoader       | List/load episodes from SQLite | GUI replay tab

================================================================================
FILE LAYOUT
================================================================================

    gym_gui/replays/
        __init__.py         This file (public API exports)
        ReplayWriter.py     HDF5 write path (training time)
        ReplayReader.py     HDF5 read path (offline analysis)
        FrameResolver.py    Resolves frame_ref URIs to numpy arrays
        EpisodeLoader.py    Episode listing from TelemetryService

================================================================================
DATA FLOW
================================================================================

    TRAINING TIME:
        Worker -> TelemetryDBSink -> ReplayWriter -> HDF5 (frames/obs)
                                -> SQLite (scalars + frame_ref URIs)

    REPLAY TIME (GUI):
        _ReplayTab -> EpisodeLoader -> TelemetryService -> SQLite
                                   -> FrameResolver -> HDF5 (resolved frames)

    REPLAY TIME (offline):
        Script -> ReplayReader -> HDF5 (direct read, no SQLite needed)

================================================================================
"""

from gym_gui.replays.ReplayWriter import ReplayWriter
from gym_gui.replays.ReplayReader import ReplayReader
from gym_gui.replays.FrameResolver import FrameRef, FrameResolver, make_frame_ref
from gym_gui.replays.EpisodeLoader import EpisodeReplay, EpisodeLoader

__all__ = [
    # Write path (training time)
    "ReplayWriter",
    # Read path (HDF5 direct)
    "ReplayReader",
    # Read path (SQLite -> HDF5 resolution)
    "FrameRef",
    "FrameResolver",
    "make_frame_ref",
    # Read path (SQLite episode listing)
    "EpisodeReplay",
    "EpisodeLoader",
]
