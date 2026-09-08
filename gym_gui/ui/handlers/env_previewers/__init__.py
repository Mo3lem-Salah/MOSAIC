"""Environment previewers for `MainWindow._on_initialize_operator`.

Stateless one-shot snapshot classes; distinct from the stateful lifecycle
managers in `gym_gui/ui/handlers/env_loaders/` (which own tabs, controllers,
and long-lived resources for Human-vs-Agent mode).

See `base.py` for the `EnvPreview` protocol and the expected exceptions
(`EnvPreviewError`, `EnvPreviewImportError`).

Each previewer implements a single method:

    def preview(
        self,
        config: OperatorConfig,
        seed: int,
        operator_id: Optional[str] = None,
    ) -> tuple[np.ndarray | None, dict | None, tuple[str, int] | None]:
        ...

Extraction status (refactor plan steps 1a + 1b, 2026-08-10):
- All 11 previewers extracted: minigrid (also handles babyai), crafter, nle,
  minihack, textworld, pettingzoo (chess/connect_four/go/tictactoe),
  mosaic_multigrid, ini_multigrid, meltingpot, overcooked, socialjax, plus
  the generic gymnasium fallback (`GymnasiumFallbackEnvPreview`) used for
  any env_name not in the dispatcher dict.
"""

from gym_gui.ui.handlers.env_previewers.base import (
    EnvPreview,
    EnvPreviewError,
    EnvPreviewImportError,
)
from gym_gui.ui.handlers.env_previewers.crafter import CrafterEnvPreview
from gym_gui.ui.handlers.env_previewers.gymnasium_fallback import GymnasiumFallbackEnvPreview
from gym_gui.ui.handlers.env_previewers.ini_multigrid import IniMultigridEnvPreview
from gym_gui.ui.handlers.env_previewers.meltingpot import MeltingpotEnvPreview
from gym_gui.ui.handlers.env_previewers.minigrid import MinigridEnvPreview
from gym_gui.ui.handlers.env_previewers.minihack import MinihackEnvPreview
from gym_gui.ui.handlers.env_previewers.mosaic_multigrid import MosaicMultigridEnvPreview
from gym_gui.ui.handlers.env_previewers.nle import NLEEnvPreview
from gym_gui.ui.handlers.env_previewers.overcooked import OvercookedEnvPreview
from gym_gui.ui.handlers.env_previewers.pettingzoo import PettingzooEnvPreview
from gym_gui.ui.handlers.env_previewers.socialjax import SocialjaxEnvPreview
from gym_gui.ui.handlers.env_previewers.textworld import TextworldEnvPreview

__all__ = [
    # Protocol + exceptions
    "EnvPreview",
    "EnvPreviewError",
    "EnvPreviewImportError",
    # Concrete previewers (alphabetical)
    "CrafterEnvPreview",
    "GymnasiumFallbackEnvPreview",
    "IniMultigridEnvPreview",
    "MeltingpotEnvPreview",
    "MinigridEnvPreview",
    "MinihackEnvPreview",
    "MosaicMultigridEnvPreview",
    "NLEEnvPreview",
    "OvercookedEnvPreview",
    "PettingzooEnvPreview",
    "SocialjaxEnvPreview",
    "TextworldEnvPreview",
]
