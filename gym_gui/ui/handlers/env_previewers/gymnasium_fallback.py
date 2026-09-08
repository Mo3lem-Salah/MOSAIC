"""Generic Gymnasium fallback previewer.

Extracted from `MainWindow._on_initialize_operator` (backup lines 5224-5237,
the final `else:` branch of the if/elif chain).

Used for any env_name that does not match a dedicated previewer. The
dispatcher in main_window falls through to this previewer via a `.get(...)`
call whose default is the fallback instance.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple, cast

import numpy as np

from gym_gui.ui.handlers.env_previewers.base import EnvPreviewError

if TYPE_CHECKING:
    from gym_gui.services.operator import OperatorConfig


_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class GymnasiumFallbackEnvPreview:
    """Generic gymnasium.make(task, render_mode='rgb_array') fallback.

    Catches any exception (including ImportError) and rethrows as
    EnvPreviewError with the env_name and task in the message. This matches
    the original inline fallback which used a bare `except Exception:` clause
    with the message "Cannot preview {env_name}/{task}: {e}".
    """

    def preview(
        self,
        config: "OperatorConfig",
        seed: int,
        operator_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[Tuple[str, int]]]:
        try:
            import gymnasium as gym
            env = gym.make(config.task, render_mode="rgb_array")
            env.reset(seed=seed)
            rgb_frame = cast(Optional[np.ndarray], env.render())
            env.close()

            return rgb_frame, None, None
        except Exception as e:
            raise EnvPreviewError(
                f"Cannot preview {config.env_name}/{config.task}: {e}"
            ) from e


__all__ = ["GymnasiumFallbackEnvPreview"]
