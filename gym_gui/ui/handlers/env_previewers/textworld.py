"""TextWorld environment previewer.

Extracted from `MainWindow._on_initialize_operator` (backup lines 4766-4826).
TextWorld games are text-based; the preview finds a game file under
`var/data/tw_games/<task>/`, opens it, resets, and renders the initial
observation text as a PIL image so the operator preview UI has something
to show.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from gym_gui.ui.handlers.env_previewers.base import EnvPreviewError, EnvPreviewImportError

if TYPE_CHECKING:
    from gym_gui.services.operator import OperatorConfig


_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class TextworldEnvPreview:
    """Preview TextWorld interactive-fiction environments as text-in-image."""

    def preview(
        self,
        config: "OperatorConfig",
        seed: int,
        operator_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[Tuple[str, int]]]:
        try:
            import textworld
            import textworld.gym
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as e:
            raise EnvPreviewImportError("TextWorld not installed - cannot preview") from e

        try:
            # Preserve the original path calculation exactly. In main_window.py
            # this was `Path(__file__).parent.parent.parent.parent / "var" / ...`
            # (4 parents from a file at repo/gym_gui/ui/main_window.py), which
            # actually resolves to ABOVE the repo root (a pre-existing bug from
            # the inline branch). To preserve behaviour byte-for-byte from this
            # deeper location (repo/gym_gui/ui/handlers/env_previewers/textworld.py),
            # we use 6 parents so the resulting path matches the original.
            # If the path bug is to be fixed, do so in a separate PR.
            games_path = Path(__file__).parent.parent.parent.parent.parent.parent / "var" / "data" / "tw_games" / config.task
            game_files = list(games_path.glob("*.ulx")) + list(games_path.glob("*.z8"))

            if not game_files:
                raise EnvPreviewError(
                    f"No TextWorld games found for '{config.task}' in var/data/tw_games/"
                )

            # Register and create environment from first game file
            game_file = str(game_files[seed % len(game_files)])
            request_infos = textworld.EnvInfos(
                objective=True, description=True, score=True, max_score=True, won=True
            )
            env_id = textworld.gym.register_game(game_file, request_infos, max_episode_steps=100)
            env = textworld.gym.make(env_id)
            obs, _info = env.reset()
            env.close()

            # Render text observation as image
            text = obs if isinstance(obs, str) else str(obs)
            # Wrap long lines
            lines = []
            for line in text.split("\n"):
                while len(line) > 80:
                    lines.append(line[:80])
                    line = line[80:]
                lines.append(line)
            text = "\n".join(lines[:40])  # Limit to 40 lines

            # Create image from text
            img_width, img_height = 640, 480
            img = Image.new("RGB", (img_width, img_height), color=(20, 20, 30))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
            except OSError:
                font = ImageFont.load_default()
            draw.text((10, 10), text, fill=(200, 200, 200), font=font)
            rgb_frame = np.array(img)

            return rgb_frame, None, None
        except EnvPreviewError:
            raise
        except Exception as e:
            raise EnvPreviewError(f"Cannot preview TextWorld: {e}") from e


__all__ = ["TextworldEnvPreview"]
