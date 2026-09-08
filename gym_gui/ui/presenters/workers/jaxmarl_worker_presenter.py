"""Presenter for JaxMARL worker analytics lane."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

_LOGGER = logging.getLogger(__name__)


class JaxMARLWorkerPresenter:
    """Presenter for the JaxMARL GPU-accelerated multi-agent RL worker.

    JaxMARL runs IPPO and MAPPO on JAX-native environments (MOSAIC sports,
    SocialJax social dilemmas) and saves checkpoints as .npz files.
    This presenter enables policy loading and interactive evaluation via the
    Operators tab.

    Supported algorithms:
    - IPPO  (Independent PPO, per-agent networks)
    - MAPPO (shared network + global critic)

    Supported environments:
    - mosaic_multigrid: Soccer, Basketball, American Football
    - socialjax: CoopMining, CoinGame, CleanUp, HarvestCommonOpen, etc.
    """

    @property
    def id(self) -> str:
        return "jaxmarl_worker"

    def build_train_request(self, policy_path: Any, current_game: Optional[Any]) -> dict:
        """Build a training request for the JaxMARL worker.

        JaxMARL training is launched via the Script Experiments tab.
        This method is a placeholder for future integration.
        """
        raise NotImplementedError(
            "JaxMARL training is launched via Script Experiments, not the Operators tab."
        )

    def create_tabs(
        self, run_id: str, agent_id: str, first_payload: dict, parent: Any
    ) -> List[Any]:
        """Create worker-specific UI tabs for a running JaxMARL session.

        JaxMARL interactive mode does not emit FastLane telemetry during
        policy evaluation (action-selection only). Tabs are created when
        training mode emits telemetry payloads.

        Args:
            run_id: Unique run identifier.
            agent_id: Agent identifier.
            first_payload: First telemetry payload containing metadata.
            parent: Parent Qt widget.

        Returns:
            List of QWidget tab instances (empty for interactive eval mode).
        """
        _LOGGER.info(
            "JaxMARL create_tabs: run_id=%s, agent_id=%s", run_id, agent_id
        )
        return []
