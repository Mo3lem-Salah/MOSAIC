"""MeltingPot environment previewer.

Extracted from `MainWindow._on_initialize_operator` (backup lines 5076-5123).
Uses Shimmy's MeltingPotCompatibilityV0 wrapper. Prefers `WORLD.RGB` (full
40x72 world view) over per-agent `RGB` (40x40) when available.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from gym_gui.ui.handlers.env_previewers.base import EnvPreviewError, EnvPreviewImportError

if TYPE_CHECKING:
    from gym_gui.services.operator import OperatorConfig


_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class MeltingpotEnvPreview:
    """Preview MeltingPot social multi-agent scenarios."""

    def preview(
        self,
        config: "OperatorConfig",
        seed: int,
        operator_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[Tuple[str, int]]]:
        try:
            from shimmy import MeltingPotCompatibilityV0
        except ImportError as e:
            raise EnvPreviewImportError("MeltingPot not installed - cannot preview") from e

        task = config.task
        try:
            # Extract substrate name from task (format: meltingpot/substrate_name)
            substrate_name = task.split("/", 1)[-1] if "/" in task else task

            # Create environment via Shimmy wrapper (handles roles automatically)
            env = MeltingPotCompatibilityV0(substrate_name=substrate_name)

            observations, _ = env.reset()

            # Get RGB from first agent
            # Prefer WORLD.RGB (40x72 full world view) over individual RGB (40x40)
            first_agent = env.agents[0]
            rgb_frame: Optional[np.ndarray] = None
            if first_agent in observations:
                if "WORLD.RGB" in observations[first_agent]:
                    rgb_frame = observations[first_agent]["WORLD.RGB"]
                elif "RGB" in observations[first_agent]:
                    rgb_frame = observations[first_agent]["RGB"]

            env.close()

            if rgb_frame is None:
                raise EnvPreviewError(f"No RGB observation available for {substrate_name}")

            return rgb_frame, None, None
        except EnvPreviewError:
            raise
        except Exception as e:
            raise EnvPreviewError(f"Cannot preview MeltingPot {task}: {e}") from e


__all__ = ["MeltingpotEnvPreview"]
