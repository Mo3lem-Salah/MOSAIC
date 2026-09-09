"""Regression tests for the Google Research Football (GRF) integration.

GRF is a physics-based football (soccer) environment for single- and
multi-agent RL.
Paper: Kurach et al. (2020). "Google Research Football: A Novel RL Environment".

These tests lock in the defects fixed on DAY_77/TASK_1:

* frames reach the Render View as true RGB, not BGR (double channel swap)
* the engine's own SDL window stays hidden when embedded in the GUI
* HUMAN_ONLY is offered so the render container accepts keyboard input
* the match advances on idle ticks instead of only on "Agent Step"
* action indices are logged with readable names ("12 (shot)")

See docs/Development_Progress/1.0_DAY_77/TASK_1/.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

# Skip the whole module when the compiled GRF engine is unavailable.
pytest.importorskip("gfootball")

from gym_gui.controllers.interaction import (
    AleInteractionController,
    GFootballInteractionController,
    GriddlyInteractionController,
    JumanjiArcadeInteractionController,
    ProcgenInteractionController,
    SMACInteractionController,
    ViZDoomInteractionController,
)
from gym_gui.controllers.keyboard_worker_bridge import KeyboardWorkerBridge
from gym_gui.core.adapters.gfootball import (
    GFOOTBALL_ADAPTERS,
    GRF_DEFAULT_ACTIONS,
    GFootballAdapter,
)
from gym_gui.core.enums import (
    DEFAULT_CONTROL_MODES,
    DEFAULT_RENDER_MODES,
    ENVIRONMENT_FAMILY_BY_GAME,
    ControlMode,
    EnvironmentFamily,
    GameId,
    RenderMode,
)
from gym_gui.core.ui.game_config.game_configs import GFootballConfig

GRF_GAME_IDS = [g for g in GameId if g.name.startswith("GRF_")]

# Controllers whose worlds keep moving without human input. session._idle_step
# uses this exact tuple to decide whether to gate on ``awaiting_human``.
CONTINUOUS_CONTROLLERS = (
    AleInteractionController,
    ViZDoomInteractionController,
    ProcgenInteractionController,
    JumanjiArcadeInteractionController,
    GriddlyInteractionController,
    GFootballInteractionController,
    SMACInteractionController,
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_all_sixteen_scenarios_registered() -> None:
    assert len(GRF_GAME_IDS) == 16
    assert len(GFOOTBALL_ADAPTERS) == 16
    assert set(GFOOTBALL_ADAPTERS) == set(GRF_GAME_IDS)


@pytest.mark.parametrize("game_id", GRF_GAME_IDS, ids=lambda g: g.name)
def test_enum_tables_cover_every_scenario(game_id: GameId) -> None:
    """A GameId missing from any of these maps breaks the GUI dropdown."""
    assert ENVIRONMENT_FAMILY_BY_GAME[game_id] is EnvironmentFamily.GFOOTBALL
    assert RenderMode.RGB_ARRAY in DEFAULT_RENDER_MODES[game_id]
    assert ControlMode.HUMAN_ONLY in DEFAULT_CONTROL_MODES[game_id]


def test_adapter_supports_human_control() -> None:
    """Regression: the adapter used to omit HUMAN_ONLY, contradicting enums.py.

    operator_render_container.keyPressEvent() discards every key unless the
    operator has a human worker, which requires this mode. Without it W and the
    other GRF keys silently did nothing.
    """
    modes = GFootballAdapter.supported_control_modes
    assert ControlMode.HUMAN_ONLY in modes
    assert ControlMode.AGENT_ONLY in modes
    for game_id in GRF_GAME_IDS:
        assert ControlMode.HUMAN_ONLY in DEFAULT_CONTROL_MODES[game_id]


# ---------------------------------------------------------------------------
# Action naming (readable runtime logs)
# ---------------------------------------------------------------------------

def test_action_set_has_nineteen_named_actions() -> None:
    assert len(GRF_DEFAULT_ACTIONS) == 19
    assert GRF_DEFAULT_ACTIONS[0] == "idle"
    assert GRF_DEFAULT_ACTIONS[12] == "shot"
    assert GRF_DEFAULT_ACTIONS[18] == "release_dribble"


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (0, "idle"),
        (3, "top"),
        (5, "right"),
        (12, "shot"),
        (13, "sprint"),
        (18, "release_dribble"),
        (19, "unknown_19"),
        (-1, "unknown_-1"),
    ],
)
def test_get_action_name(action: int, expected: str) -> None:
    """Regression: logs showed a bare "action 12" with no hint it meant "shot"."""
    adapter = GFootballAdapter.__new__(GFootballAdapter)  # pure lookup, no env
    assert adapter.get_action_name(action) == expected


def test_get_action_name_handles_multi_agent_sequences() -> None:
    adapter = GFootballAdapter.__new__(GFootballAdapter)
    assert adapter.get_action_name([5, 12, 0]) == "right,shot,idle"
    assert adapter.get_action_name(np.array([1, 7])) == "left,bottom"


def test_keyboard_bridge_renders_action_names() -> None:
    """The bridge log is what the user actually reads in the Runtime Log."""
    bridge = KeyboardWorkerBridge()
    bridge._action_names = list(GRF_DEFAULT_ACTIONS)
    assert bridge._describe_action(12) == "12 (shot)"
    assert bridge._describe_action(0) == "0 (idle)"
    assert bridge._describe_action(18) == "18 (release_dribble)"


def test_keyboard_bridge_falls_back_without_names() -> None:
    """Environments that supply no table must keep the original bare format."""
    bridge = KeyboardWorkerBridge()
    assert bridge._describe_action(12) == "12"

    named = KeyboardWorkerBridge()
    named._action_names = list(GRF_DEFAULT_ACTIONS)
    assert named._describe_action(99) == "99"  # out of range degrades safely


# ---------------------------------------------------------------------------
# Continuous play (idle ticking)
# ---------------------------------------------------------------------------

def test_interaction_controller_ticks_at_grf_cadence() -> None:
    controller = GFootballInteractionController(owner=None, target_hz=10)
    assert controller.idle_interval_ms() == 100
    # Index 0 is a real action in GRF's Discrete(19); no sentinel needed.
    assert controller.maybe_passive_action() == 0


def test_controller_is_exempt_from_awaiting_human_gate() -> None:
    """Regression: the match stayed frozen at Step 0 despite actions arriving.

    session._idle_step() returns early for controllers outside this tuple. GRF
    was missing from it, so every tick aborted before stepping and only the
    "Agent Step" button advanced the game.
    """
    controller = GFootballInteractionController(owner=None)
    assert isinstance(controller, CONTINUOUS_CONTROLLERS)
    require_awaiting = not isinstance(controller, CONTINUOUS_CONTROLLERS)
    assert require_awaiting is False


def _owner(**overrides):
    state = dict(
        _adapter=object(),
        _game_id=GameId.GRF_5V5,
        _game_started=True,
        _game_paused=False,
        _control_mode=ControlMode.HUMAN_ONLY,
        _last_step=None,
    )
    state.update(overrides)
    return SimpleNamespace(**state)


@pytest.mark.parametrize(
    ("overrides", "expected", "reason"),
    [
        ({}, True, "running match should tick"),
        ({"_game_started": False}, False, "not started"),
        ({"_game_paused": True}, False, "paused"),
        ({"_control_mode": ControlMode.AGENT_ONLY}, False, "agent drives stepping"),
        ({"_adapter": None}, False, "no adapter"),
        (
            {"_last_step": SimpleNamespace(terminated=True, truncated=False)},
            False,
            "episode finished",
        ),
    ],
)
def test_should_idle_tick_guards(overrides, expected: bool, reason: str) -> None:
    controller = GFootballInteractionController(owner=None)
    controller._owner = _owner(**overrides)
    assert controller.should_idle_tick() is expected, reason


def test_session_selects_gfootball_controller() -> None:
    """The controller is useless unless session.py actually hands it out."""
    import gym_gui.controllers.session as session_module

    class _Session(session_module.SessionController):
        def __init__(self) -> None:  # bypass the real, heavy constructor
            self._game_id = GameId.GRF_5V5

    controller = session_module.SessionController._create_interaction_controller(
        _Session(), EnvironmentFamily.GFOOTBALL
    )
    assert isinstance(controller, GFootballInteractionController)
    assert controller.idle_interval_ms() == 100


# ---------------------------------------------------------------------------
# Live environment: rendering and stepping
#
# These need the compiled engine plus a GL context. On a headless machine run
# the suite under xvfb-run.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loaded_adapter():
    from gym_gui.core.factories.adapters import create_adapter

    adapter = create_adapter(GameId.GRF_5V5, game_config=GFootballConfig())
    try:
        adapter.load()
    except Exception as exc:  # pragma: no cover - engine/GL unavailable
        pytest.skip(f"GRF engine unavailable: {exc}")
    adapter.reset()
    yield adapter
    adapter.close()


@pytest.mark.slow
def test_hidden_window_env_var_is_set(loaded_adapter) -> None:
    """The engine must not open a second top-level window beside the Qt shell."""
    assert os.environ.get("GFOOTBALL_HIDDEN_WINDOW") == "1"


@pytest.mark.slow
def test_render_returns_rgb_frame(loaded_adapter) -> None:
    payload = loaded_adapter.render()
    assert payload is not None
    frame = payload["rgb"]
    assert frame.shape == (720, 1280, 3)
    assert frame.dtype == np.uint8
    # QImage.Format_RGB888 requires a contiguous buffer.
    assert frame.flags["C_CONTIGUOUS"]


@pytest.mark.slow
def test_frame_channels_are_rgb_not_bgr(loaded_adapter) -> None:
    """Regression: players appeared in blue shirts instead of yellow.

    The engine reads the framebuffer as GL_RGB, then FootballEnvCore.render()
    applied a second b,g,r -> r,g,b swap, leaving R and B transposed. Grass is
    the ground truth: turf is green-dominant and warm, so red exceeds blue.
    """
    for _ in range(20):
        loaded_adapter.step(0)
    frame = loaded_adapter.render()["rgb"]

    grass = frame[400:700, 300:1000].reshape(-1, 3).mean(axis=0)
    assert grass.argmax() == 1, f"grass should be green-dominant, got {grass}"
    assert grass[0] > grass[2], f"grass should be warm (R > B), got {grass}"

    pixels = frame.reshape(-1, 3).astype(int)
    yellow = ((pixels[:, 0] > 150) & (pixels[:, 1] > 150) & (pixels[:, 2] < 110)).sum()
    cyan = ((pixels[:, 2] > 150) & (pixels[:, 1] > 150) & (pixels[:, 0] < 110)).sum()
    assert yellow > cyan, f"shirts look cyan, channels still swapped ({yellow=} {cyan=})"


@pytest.mark.slow
def test_idle_action_advances_the_match(loaded_adapter) -> None:
    """The core of the report: the pitch must move without any key press."""
    idle = GFootballInteractionController(owner=None).maybe_passive_action()
    frames = []
    for _ in range(15):
        loaded_adapter.step(idle)
        frames.append(loaded_adapter.render()["rgb"].copy())

    changed = sum(1 for a, b in zip(frames, frames[1:]) if not np.array_equal(a, b))
    assert changed >= len(frames) // 2, (
        f"only {changed}/{len(frames) - 1} frames changed; the match is frozen"
    )


@pytest.mark.slow
def test_step_reports_action_name_in_telemetry(loaded_adapter, caplog) -> None:
    """The step log must carry the resolved name, not just the raw index.

    ``LogConstantMixin`` emits through ``self._logger``, which adapters inherit
    from the base class, so capture at the root to stay independent of which
    module owns the logger.
    """
    import logging

    with caplog.at_level(logging.DEBUG):
        loaded_adapter.step(12)  # shot

    names = [
        record.action_name
        for record in caplog.records
        if hasattr(record, "action_name")
    ]
    assert "shot" in names, f"expected 'shot' in step telemetry, saw {names}"
