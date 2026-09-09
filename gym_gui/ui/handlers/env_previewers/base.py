"""Base protocol and exceptions for environment previewers.

Environment previewers are stateless one-shot snapshot functions used by
`MainWindow._on_initialize_operator` to render a single RGB frame of an
environment for the operator preview UI. They are distinct from the stateful
lifecycle managers in `gym_gui/ui/handlers/env_loaders/` (which own tabs,
controllers, and long-lived resources for Human-vs-Agent mode).

Each previewer implements the `EnvPreview` protocol:

    class MyEnvPreview:
        def preview(
            self,
            config: OperatorConfig,
            seed: int,
        ) -> tuple[np.ndarray | None, dict | None, tuple[str, int] | None]:
            # Create env, reset with seed, render one RGB frame, close.
            # Return (frame, board_game_payload, (status_text, timeout_ms)).
            ...

Return contract:
- `frame`: RGB numpy array (H, W, 3) or None if no render is available.
- `board_game_payload`: game-specific structured payload (e.g., chess FEN +
  legal moves) for BoardGameRendererStrategy, or None for generic RGB rendering.
- `status_message`: optional (text, timeout_ms) tuple the dispatcher shows in
  the status bar (e.g., ("Custom MiniGrid configuration applied!", 3000)).
  None if no message. The timeout preserves the exact per-branch UX from the
  original inline code (success toasts typically 3000ms, warning toasts 5000ms).

Expected error signalling:
- Missing optional dependency: raise `EnvPreviewImportError`. Dispatcher shows
  a "not installed" message and aborts the preview.
- Any other expected failure (env-specific setup problem, malformed config,
  etc.): raise `EnvPreviewError`. Dispatcher shows the message and aborts.
- Unexpected errors: let them propagate as regular exceptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, Tuple

if TYPE_CHECKING:
    import numpy as np

    from gym_gui.services.operator import OperatorConfig


class EnvPreviewError(Exception):
    """A previewer failed for an expected reason.

    The message is shown to the user via the status bar. Do not use for
    programmer errors or unexpected exceptions.
    """


class EnvPreviewImportError(EnvPreviewError):
    """A previewer failed because an optional dependency is not installed.

    Raised by previewers that need packages outside the core install
    (e.g., minigrid, crafter, nle, textworld). The dispatcher shows a
    dedicated "not installed" message and aborts.
    """


class EnvPreview(Protocol):
    """Protocol implemented by every environment previewer."""

    def preview(
        self,
        config: "OperatorConfig",
        seed: int,
        operator_id: Optional[str] = None,
    ) -> Tuple[Optional["np.ndarray"], Optional[dict], Optional[Tuple[str, int]]]:
        """Render a single RGB snapshot of the environment.

        Args:
            config: The operator configuration. Used to read env-specific
                settings (e.g., `config.settings["square_size"]`) and per-worker
                settings such as `initial_state`.
            seed: Random seed for `env.reset(seed=seed)`.
            operator_id: Optional operator identifier, used only by previewers
                that emit `log_constant` calls with `extra={"operator_id": ...}`
                (currently pettingzoo, for LOG_UI_BOARD_CONFIG_ENV_INIT_CUSTOM
                and related informational events). Other previewers ignore it.

        Returns:
            A 3-tuple `(frame, board_game_payload, status_message)`:
            - `frame`: RGB array `(H, W, 3)` or None.
            - `board_game_payload`: Structured payload for board games, or None.
            - `status_message`: Optional `(text, timeout_ms)` tuple. None means
              the dispatcher emits no status-bar message for this preview.

        Raises:
            EnvPreviewImportError: An optional dependency is missing.
            EnvPreviewError: Any other expected preview failure.
        """
        ...


__all__ = ["EnvPreview", "EnvPreviewError", "EnvPreviewImportError"]
