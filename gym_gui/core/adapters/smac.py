"""SMAC v1 (StarCraft Multi-Agent Challenge) adapter for the MOSAIC GUI.

Bridges SMAC's custom ``MultiAgentEnv`` API to MOSAIC's ``EnvironmentAdapter``
interface.  SMAC does NOT use Gymnasium -- it has its own API with methods like
``get_obs()``, ``get_state()``, and ``get_avail_agent_actions()``.

Supports three renderer modes:
- ``"3d"``: Full SC2 engine 3D rendering via EGL (GPU-accelerated).
- ``"heatmap"``: Custom 2x2 feature-layer panel (pure numpy).
- ``"classic"``: SMAC's built-in PyGame 2D renderer (colored circles).

Repository: https://github.com/oxwhirl/smac
Paper:      https://arxiv.org/abs/1902.04043
"""

from __future__ import annotations

import logging
import os
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    AgentSnapshot,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import (
    ControlMode,
    GameId,
    RenderMode,
    SteppingParadigm,
)
from gym_gui.logging_config.log_constants import (
    LOG_SMAC_ACTION_MASK_WARN,
    LOG_SMAC_BATTLE_RESULT,
    LOG_SMAC_ENV_CLOSED,
    LOG_SMAC_ENV_CREATED,
    LOG_SMAC_ENV_RESET,
    LOG_SMAC_RENDER_ERROR,
    LOG_SMAC_SC2_PATH_MISSING,
    LOG_SMAC_STEP_SUMMARY,
)

_LOGGER = logging.getLogger(__name__)


def _patch_pysc2_duplicate_map_tolerance() -> None:
    """Make ``pysc2.maps.lib.get_maps()`` tolerate duplicate map names.

    SMAC v1 and SMACv2 each define an independent ``Map`` subclass registry
    (``smac.env.starcraft2.maps.smac_maps`` and
    ``smacv2.env.starcraft2.maps.smac_maps``). SMACv2's registry
    re-declares every v1 map name (``3m``, ``8m``, ``2s3z``, ...) for
    backward compatibility, in addition to its own ``10gen_*`` maps, and
    both classes resolve to an identical ``.path`` (e.g.
    ``SMAC_Maps/3m.SC2Map``) for shared names.

    ``pysc2.maps.lib.get_maps()`` walks *every* ``Map`` subclass that has
    ever been imported into the process, with no per-package namespacing.
    Because MOSAIC registers both the SMAC and SMACv2 adapters
    unconditionally at startup (``gym_gui/core/adapters/__init__.py``),
    both packages' top-level modules import their map registries as a
    side effect, and any subsequent call to ``pysc2.maps.lib.get()`` or
    ``get_maps()`` -- for either package, regardless of which one is
    actually being launched -- raises ``DuplicateMapError``.

    This patches ``get_maps()`` to keep the first-registered class for a
    given name instead of raising, which is safe here because the
    colliding classes are functionally interchangeable (same map file).
    Applied once, idempotently, at import time of this module (which both
    ``SMACAdapter`` and ``SMACv2Adapter`` depend on for 3D-render patching
    and shared map metadata).
    """
    from pysc2.maps import lib as _pysc2_maps_lib

    if getattr(_pysc2_maps_lib, "_mosaic_duplicate_tolerant", False):
        return

    def _get_maps_tolerant() -> Dict[str, Any]:
        maps: Dict[str, Any] = {}
        for mp in _pysc2_maps_lib.Map.all_subclasses():
            if mp.filename or mp.battle_net:
                map_name = mp.__name__
                if map_name not in maps:  # keep first registration, skip dupes
                    maps[map_name] = mp
        return maps

    _pysc2_maps_lib.get_maps = _get_maps_tolerant
    _pysc2_maps_lib._mosaic_duplicate_tolerant = True


_patch_pysc2_duplicate_map_tolerance()


def _patch_pysc2_shuffled_hue_py311() -> None:
    """Fix ``pysc2.lib.colors.shuffled_hue()`` for Python 3.9+.

    ``shuffled_hue()`` calls ``random.shuffle(palette, lambda: 0.5)`` --
    the old (pre-3.9) ``random.shuffle(x, random)`` signature that accepted
    a custom RNG function to produce a deterministic "fixed shuffle" (per
    the function's own comment). Python 3.9 removed that second parameter
    entirely, so this raises ``TypeError: Random.shuffle() takes 2
    positional arguments but 3 were given`` on any Python >= 3.9.

    This is triggered the moment SMAC's ``renderer='classic'`` (PyGame 2D
    renderer) is used: ``smac.env.starcraft2.render`` imports
    ``pysc2.lib.renderer_human``, which imports ``pysc2.lib.features``,
    which evaluates ``pysc2.lib.colors.shuffled_hue()`` at *module import
    time* to build its palette table -- so it fails before any of MOSAIC's
    own code runs, regardless of camera settings.

    This patches ``shuffled_hue()`` with a reimplementation of the exact
    same deterministic Fisher-Yates shuffle algorithm Python's old
    ``random.shuffle(x, random_func)`` used internally, using
    ``random_func = lambda: 0.5`` (matching the original call), so the
    resulting palette is bit-for-bit identical to what upstream pysc2
    intended on old Python versions. Applied once, idempotently, at import
    time of this module.
    """
    from pysc2.lib import colors as _pysc2_colors

    if getattr(_pysc2_colors, "_mosaic_shuffle_patched", False):
        return

    def _fixed_shuffle(x: list, random_func) -> None:
        # Reimplementation of CPython's pre-3.9 random.shuffle(x, random)
        # Fisher-Yates algorithm, since Python 3.9+ removed the ability to
        # pass a custom random function as the second positional argument.
        for i in reversed(range(1, len(x))):
            j = int(random_func() * (i + 1))
            x[i], x[j] = x[j], x[i]

    def _shuffled_hue_patched(scale):
        import numpy
        palette = list(_pysc2_colors.smooth_hue_palette(scale))
        _fixed_shuffle(palette, lambda: 0.5)  # Return a fixed shuffle
        return numpy.array(palette)

    _pysc2_colors.shuffled_hue = _shuffled_hue_patched
    _pysc2_colors._mosaic_shuffle_patched = True


_patch_pysc2_shuffled_hue_py311()

# Action names for telemetry and display
SMAC_BASE_ACTIONS: List[str] = [
    "NO-OP",       # 0 -- only valid action for dead agents
    "STOP",        # 1 -- stop current movement/attack
    "MOVE_NORTH",  # 2
    "MOVE_SOUTH",  # 3
    "MOVE_EAST",   # 4
    "MOVE_WEST",   # 5
    # indices 6+ are ATTACK_ENEMY_0, ATTACK_ENEMY_1, ...
]

# Map metadata: map_name -> (ally_count, enemy_description, difficulty_tier, episode_limit)
# episode_limit values are the real per-map SMAC defaults, taken from
# smac.env.starcraft2.maps.smac_maps (also cross-referenced against
# epymarl's src/config/envs/sc2.yaml / smacv2_configs/*.yaml, which are
# authoritative reference configs for both SMAC v1 map defaults and the
# SMACv2 procedural scenario table).
SMAC_MAP_INFO: Dict[str, tuple[int, str, str, int]] = {
    "3m": (3, "3 Marines", "Easy", 60),
    "8m": (8, "8 Marines", "Easy", 120),
    "25m": (25, "25 Marines", "Easy", 150),
    "2s3z": (5, "2 Stalkers + 3 Zealots", "Easy", 120),
    "3s5z": (8, "3 Stalkers + 5 Zealots", "Easy", 150),
    "5m_vs_6m": (5, "6 Marines", "Hard", 70),
    "8m_vs_9m": (8, "9 Marines", "Hard", 120),
    "10m_vs_11m": (10, "11 Marines", "Hard", 150),
    "27m_vs_30m": (27, "30 Marines", "Super Hard", 180),
    "MMM": (10, "1 Medivac + 2 Marauders + 7 Marines", "Hard", 150),
    "MMM2": (10, "1 Medivac + 3 Marauders + 8 Marines", "Super Hard", 180),
    "3s5z_vs_3s6z": (8, "3 Stalkers + 6 Zealots", "Super Hard", 170),
    "3s_vs_3z": (3, "3 Zealots", "Easy", 150),
    "3s_vs_4z": (3, "4 Zealots", "Hard", 200),
    "3s_vs_5z": (3, "5 Zealots", "Super Hard", 250),
    "1c3s5z": (9, "1 Colossus + 3 Stalkers + 5 Zealots", "Hard", 180),
    "2m_vs_1z": (2, "1 Zealot", "Easy", 150),
    "corridor": (6, "24 Zerglings", "Hard", 400),
    "6h_vs_8z": (6, "8 Zealots", "Super Hard", 150),
    "2s_vs_1sc": (2, "1 Spine Crawler", "Easy", 300),
    "so_many_baneling": (7, "32 Banelings", "Hard", 100),
    "bane_vs_bane": (24, "24 Zerglings + Banelings", "Easy", 200),
    "2c_vs_64zg": (2, "64 Zerglings", "Super Hard", 400),
}


# Default resolution for 3D GPU rendering
_3D_RENDER_SIZE = 512
_3D_MINIMAP_SIZE = 256


def _patch_launch_for_3d(
    smac_env: Any,
    render_size: int = _3D_RENDER_SIZE,
    minimap_size: int = _3D_MINIMAP_SIZE,
) -> None:
    """Monkey-patch ``_launch()`` on a SMAC env to enable 3D GPU rendering.

    SMAC hardcodes ``want_rgb=False`` (which adds ``-headlessNoRender`` to the
    SC2 command line) and omits the ``render`` field from ``InterfaceOptions``.
    This patch overrides ``_launch()`` to pass ``want_rgb=True`` and add a
    ``SpatialCameraSetup`` so the SC2 engine returns RGB frames via EGL.

    Args:
        smac_env: The SMAC/SMACv2 environment instance to patch.
        render_size: Main 3D camera output resolution in pixels (square).
            Comes from ``SMACConfig.render_resolution`` (user-adjustable).
        minimap_size: Minimap inset render resolution in pixels (square).
            Kept independent of ``render_size`` since the minimap is
            composited at a fixed small inset size regardless of the main
            frame's resolution (see ``_composite_minimap_inset``).

    Note on camera FOV: ``InterfaceOptions.render.width`` (formerly used
    here) is documented in Blizzard's own ``sc2api.proto`` as relevant
    *only* to feature-layer cameras, not the 3D render camera --
    ``SpatialCameraSetup.width``: "Below are only relevant for feature
    layers." Measured directly: setting it to any value from 4.0 to 200.0
    produces byte-for-byte identical 3D render output. The 3D camera's
    field of view is fixed by the SC2 engine per-map (measured ~7-17 world
    units depending on the map) and is not adjustable via this interface
    option. Only camera *position* is controllable during live play, via
    ``ActionRaw.camera_move`` (see ``_move_camera_to``).
    """

    def _launch_with_render(self_env: Any = smac_env) -> None:

        # Ensure absl flags are parsed (pysc2 requirement)
        try:
            import sys

            from absl import flags
            if not flags.FLAGS.is_parsed():
                flags.FLAGS(sys.argv)
        except Exception:
            pass

        # Resolve the map directly from the matching package's own map
        # module, bypassing pysc2.maps.lib.get()'s global subclass scan.
        #
        # IMPORTANT: SMAC v1 and SMACv2 each define their own map registry
        # (both subclass pysc2.maps.lib.Map), and SMACv2's registry already
        # re-declares every v1 map name (3m, 8m, 2s3z, ...) for backward
        # compatibility, in addition to its own 10gen_* maps. Because
        # ``pysc2.maps.lib.Map.all_subclasses()`` walks *every* Map
        # subclass that has ever been imported into the process with no
        # namespacing, simply *importing* both packages' map modules in
        # the same process (which MOSAIC does at startup, since both the
        # SMAC and SMACv2 adapters are registered unconditionally) causes
        # ``pysc2.maps.lib.get_maps()`` to raise ``DuplicateMapError`` on
        # any subsequent call -- for either package, regardless of which
        # one is actually being launched. This function is shared by both
        # v1 (SMACAdapter) and v2 (via _patch_launch_for_3d called from
        # SMACv2Adapter), so it resolves the map class from the specific
        # module matching the *actual* environment instance being
        # launched, rather than going through the global registry.
        from pysc2 import run_configs
        from s2clientprotocol import sc2api_pb2 as sc_pb

        env_module = type(self_env).__module__
        if env_module.startswith("smacv2."):
            import smacv2.env.starcraft2.maps.smac_maps as _maps_mod
        else:
            import smac.env.starcraft2.maps.smac_maps as _maps_mod
        _map_cls = getattr(_maps_mod, self_env.map_name, None)
        if _map_cls is None:
            raise ValueError(
                f"Map '{self_env.map_name}' not found in "
                f"{_maps_mod.__name__}.get_smac_map_registry()"
            )

        # python-dotenv does not expand shell variables ($PWD, ~, etc.) in
        # values it loads into os.environ.  The XuanCe training subprocess
        # inherits that literal unexpanded string, so expand it here — before
        # pysc2 reads os.environ["SC2PATH"] inside run_configs.get().
        _sc2_env_val = os.environ.get("SC2PATH", "")
        if _sc2_env_val:
            _sc2_expanded = os.path.expandvars(os.path.expanduser(_sc2_env_val))
            if _sc2_expanded != _sc2_env_val:
                os.environ["SC2PATH"] = _sc2_expanded

        self_env._run_config = run_configs.get(version=self_env.game_version)
        # SMACv2 stores version (v1 doesn't, but setting it is harmless)
        if hasattr(self_env._run_config, "version"):
            self_env.version = self_env._run_config.version
        _map = _map_cls()

        # Enable 3D rendering: want_rgb=True prevents -headlessNoRender
        interface_options = sc_pb.InterfaceOptions(raw=True, score=False)
        interface_options.render.resolution.x = render_size
        interface_options.render.resolution.y = render_size
        interface_options.render.minimap_resolution.x = minimap_size
        interface_options.render.minimap_resolution.y = minimap_size

        self_env._sc2_proc = self_env._run_config.start(
            window_size=self_env.window_size, want_rgb=True,
        )
        self_env._controller = self_env._sc2_proc.controller

        # Create game (copied from SMAC's _launch)
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=self_env._run_config.map_data(_map.path),
            ),
            realtime=False,
            random_seed=self_env._seed,
        )
        create.player_setup.add(type=sc_pb.Participant)

        from smac.env.starcraft2.starcraft2 import difficulties, races
        create.player_setup.add(
            type=sc_pb.Computer,
            race=races[self_env._bot_race],
            difficulty=difficulties[self_env.difficulty],
        )
        self_env._controller.create_game(create)

        join = sc_pb.RequestJoinGame(
            race=races[self_env._agent_race], options=interface_options,
        )
        # SMACv2 stores the join request as self.game
        self_env.game = join
        self_env._controller.join_game(join)

        # Post-join setup (terrain, pathing, map dimensions) -- same as original
        game_info = self_env._controller.game_info()
        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        # SMACv2 stores these as instance attributes
        self_env.map_play_area_min = map_play_area_min
        self_env.map_play_area_max = map_play_area_max
        self_env.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self_env.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self_env.map_x = map_info.map_size.x
        self_env.map_y = map_info.map_size.y

        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self_env.map_x, int(self_env.map_y / 8)
            )
            self_env.pathing_grid = np.transpose(
                np.array(
                    [
                        [(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                        for row in vals
                    ],
                    dtype=bool,
                )
            )
        else:
            self_env.pathing_grid = np.invert(
                np.flip(
                    np.transpose(
                        np.array(
                            list(map_info.pathing_grid.data), dtype=bool
                        ).reshape(self_env.map_x, self_env.map_y)
                    ),
                    axis=1,
                )
            )

        self_env.terrain_height = (
            np.flip(
                np.transpose(
                    np.array(list(map_info.terrain_height.data)).reshape(
                        self_env.map_x, self_env.map_y
                    )
                ),
                1,
            )
            / 255
        )

    smac_env._launch = types.MethodType(lambda self: _launch_with_render(self), smac_env)


def _center_camera_on_units(smac_env: Any) -> tuple[float, float] | None:
    """Move the 3D render camera to center on the unit centroid.

    After each ``reset()``, SC2's camera reverts to its default position which
    is typically NOT over the SMAC battle area.  This function computes the
    centroid of all units from the latest observation and sends an
    ``ActionRaw.camera_move`` to reposition the camera, followed by a minimal
    ``step(1)`` so the next ``observe()`` returns the correctly-framed view.

    The 3D camera's field of view is fixed per-map by the SC2 engine
    (typically 7-17 world units, not user-adjustable -- see
    ``_patch_launch_for_3d``), so on larger maps the camera cannot show the
    whole battlefield at once regardless of centering strategy. Use this
    function (rather than :func:`_center_camera_on_map`) when you want the
    camera to track the action instead of staying fixed on the map's
    geometric center.

    Returns the (cx, cy) world coordinates of the new camera center, or None.
    """
    try:
        from s2clientprotocol import sc2api_pb2 as sc_pb

        obs = smac_env._obs
        if obs is None:
            return None
        units = obs.observation.raw_data.units
        if not units:
            return None

        # Compute centroid of all units (allies + enemies)
        cx = sum(u.pos.x for u in units) / len(units)
        cy = sum(u.pos.y for u in units) / len(units)

        _move_camera_to(smac_env, cx, cy)
        return (cx, cy)
    except Exception:
        return None


def _center_camera_on_map(smac_env: Any) -> tuple[float, float] | None:
    """Move the 3D render camera to the map's fixed geometric center.

    Unlike :func:`_center_camera_on_units`, this centers on the *map's*
    playable-area midpoint, which never moves during an episode. Since the
    3D camera's FOV is fixed by the engine and not adjustable (see
    ``_patch_launch_for_3d``), this only shows the whole map on maps small
    enough to fit within that fixed FOV; on larger maps it still shows a
    partial, but stable and centered, view.

    Returns the (cx, cy) world coordinates of the map center, or None.
    """
    try:
        gi = smac_env._controller.game_info()
        pa = gi.start_raw.playable_area
        cx = (pa.p0.x + pa.p1.x) / 2.0
        cy = (pa.p0.y + pa.p1.y) / 2.0
        _move_camera_to(smac_env, cx, cy)
        return (cx, cy)
    except Exception:
        return None


def _move_camera_to(smac_env: Any, cx: float, cy: float) -> None:
    """Send a real ``ActionRaw.camera_move`` to the given world coordinates.

    This is the SC2 engine's actual in-game camera control API. Verified
    empirically: moving the camera and re-observing produces a genuinely
    different rendered frame (663K/786K pixels changed in a 4-world-unit
    pan test), unlike the previous ``render.width`` interface option which
    had zero effect on 3D render output.

    Does NOT call ``controller.step()``. An earlier version of this
    function called ``step(1)`` between the camera-move action and the
    re-observe, under the mistaken assumption that a simulation tick was
    needed for the camera move to be reflected. That was a real bug: SC2's
    ``RequestStep`` genuinely advances the game simulation by one game
    loop -- it is not a rendering-only operation. Every mouse-drag pan
    gesture (which can fire many camera moves per second) was silently
    advancing the actual episode simulation outside the normal
    action/step loop, with no corresponding entry in the episode's
    step count or reward accounting. Verified via direct measurement
    (``observation.game_loop`` and the render frame both) that
    ``act(camera_move)`` followed by a plain ``observe()`` -- no
    ``step()`` in between -- already reflects the new camera position in
    the very next observation, with ``game_loop`` staying unchanged. The
    ``step()`` call was unnecessary and has been removed.
    """
    try:
        from s2clientprotocol import sc2api_pb2 as sc_pb

        action = sc_pb.Action()
        action.action_raw.camera_move.center_world_space.x = cx
        action.action_raw.camera_move.center_world_space.y = cy

        smac_env._controller.act(action)
        smac_env._obs = smac_env._controller.observe()
    except Exception:
        pass


def _extract_player_status(smac_env: Any) -> Dict[str, Any]:
    """Extract resource/army status for the SMAC Dashboard status panel.

    Reads ``observation.player_common`` (``PlayerCommon`` -- minerals,
    vespene, food_used/food_cap, army_count, idle_worker_count) and the
    allied units' resolved type **names** from
    ``observation.raw_data.units``, both already available since SMAC
    requests both ``raw=True`` (which populates ``player_common`` in
    addition to ``raw_data``) at launch -- no extra data needs to be
    requested from the engine for this beyond one cached ``data_raw()``
    call for the name lookup table.

    ``player_common`` is a top-level field on ``Observation``, a sibling
    of ``raw_data`` (not nested inside it) -- confirmed against the actual
    protobuf schema after an earlier version of this function incorrectly
    read ``raw_data.player`` (``PlayerRaw``, which only has ``camera``,
    ``power_sources``, ``upgrade_ids`` -- no resource fields at all) and
    silently returned nothing every time due to an ``AttributeError``
    inside the broad ``except Exception`` guard below.

    Returns unit **names**, not raw ``unit_type`` IDs, because SMAC's
    custom ``.SC2Map`` files define modified "_RL" unit variants (e.g.
    "Marine_RL") with dynamically assigned runtime IDs that differ per
    map and per process -- confirmed by direct measurement (the same
    allied Marine unit reported ``unit_type`` 1970 on one map and 1971 on
    another, never pysc2's static catalog ID). Names are stable and
    resolvable via ``data_raw()``, so icon lookup keys off the name
    instead (see ``smac_unit_icons.get_unit_icon_path``).

    Returns a dict with keys ``player_common`` (dict of the PlayerCommon
    fields) and ``ally_unit_names`` (list of resolved unit type name
    strings, e.g. ``["Marine_RL", "Marine_RL", "Marine_RL"]``, for
    currently-alive allied units), suitable for
    ``SmacDashboardWidget.update_from_step_info()``. Returns an empty dict
    if raw unit data isn't available (defensive; should not normally
    happen once the env is running).
    """
    try:
        obs = smac_env._obs
        if obs is None or not obs.observation.HasField("raw_data"):
            return {}
        pc = obs.observation.player_common
        player_common = {
            "minerals": pc.minerals,
            "vespene": pc.vespene,
            "food_used": pc.food_used,
            "food_cap": pc.food_cap,
            "army_count": pc.army_count,
            "idle_worker_count": pc.idle_worker_count,
        }
        # Alliance.Self == 1 (raw.proto)
        ally_units = [u for u in obs.observation.raw_data.units if u.alliance == 1]

        from gym_gui.core.adapters.smac_unit_icons import resolve_unit_type_name
        ally_unit_names = [resolve_unit_type_name(smac_env, u.unit_type) for u in ally_units]

        return {"player_common": player_common, "ally_unit_names": ally_unit_names}
    except Exception:
        return {}


_MINIMAP_INSET_FRACTION = 0.20  # Inset width/height as a fraction of the main frame
_MINIMAP_INSET_MARGIN = 10  # Pixels from the main frame's corner
_MINIMAP_BORDER_COLOR = (255, 255, 0)  # Yellow border around the inset (fallback, no frame texture)
_MINIMAP_VIEWPORT_COLOR = (255, 60, 60)  # Red box marking the current camera FOV

# Native SC2 minimap console-panel HUD skin, extracted by the user from the
# game's own asset archive (see gym_gui/assets/SMAC/ATTRIBUTION.md and
# gym_gui/assets/SMACv2/ATTRIBUTION.md). Used to give the minimap inset the
# same frame/border look as the native client.
#
# SMAC and SMACv2 each have their own extracted asset tree
# (gym_gui/assets/SMAC/ and gym_gui/assets/SMACv2/) -- they currently
# contain byte-identical texture files (same extraction, same game assets),
# but they are NOT the same directory and must be resolved independently
# per game family rather than one hardcoded to the other. If the user (or a
# future extraction) ever swaps in family-specific skins, this must already
# be wired to pick up the right one.
_MINIMAP_ASSET_FAMILIES = ("SMAC", "SMACv2")


def _minimap_frame_texture_path(asset_family: str) -> Path:
    """Resolve the console-panel minimap texture path for a given game family.

    ``asset_family`` must be ``"SMAC"`` or ``"SMACv2"`` -- each has its own
    extracted asset tree under ``gym_gui/assets/<family>/``.
    """
    if asset_family not in _MINIMAP_ASSET_FAMILIES:
        raise ValueError(
            f"Unknown SMAC asset family {asset_family!r}; expected one of "
            f"{_MINIMAP_ASSET_FAMILIES}"
        )
    return (
        Path(__file__).resolve().parent.parent.parent
        / "assets"
        / asset_family
        / "mods"
        / "liberty.sc2mod"
        / "base.sc2assets"
        / "assets"
        / "textures"
        / "ui_console_panel_minimap_simulant.png"
    )


# Crop box (left, top, right, bottom) isolating the minimap panel from the
# full HUD strip texture, measured directly from the source PNG.
_MINIMAP_FRAME_CROP = (0, 100, 345, 500)
# The black cutout "hole" in the frame where minimap pixel data belongs,
# measured directly from the source PNG (rows 198-482, cols 31-320) and
# expressed relative to _MINIMAP_FRAME_CROP's origin.
_MINIMAP_FRAME_HOLE = (31, 98, 320, 382)  # (x0, y0, x1, y1) within the cropped frame

# Cache key is (asset_family, inset_size) so SMAC and SMACv2 never share a
# cached texture, even though today's source files happen to be identical.
_minimap_frame_cache: dict[
    tuple[str, int], tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None
] = {}


def _load_minimap_frame_texture(
    inset_size: int, asset_family: str = "SMACv2"
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
    """Load and scale the native minimap HUD frame texture for a given inset size.

    Returns ``(rgb, alpha, hole_box)`` where ``rgb``/``alpha`` are the full
    scaled frame texture (larger than ``inset_size`` -- it includes the
    decorative border beyond the hole) and ``hole_box`` is
    ``(x0, y0, x1, y1)`` locating the black cutout hole within that texture,
    scaled so the hole is exactly ``inset_size`` x ``inset_size``. Minimap
    pixel data should be pasted into ``hole_box`` before compositing this
    texture on top (alpha-blended) so the frame's border decorates around it.
    Returns None if the texture asset is unavailable (falls back to the
    plain yellow border).
    """
    cache_key = (asset_family, inset_size)
    if cache_key in _minimap_frame_cache:
        return _minimap_frame_cache[cache_key]
    try:
        from PIL import Image

        texture_path = _minimap_frame_texture_path(asset_family)
        if not texture_path.is_file():
            _minimap_frame_cache[cache_key] = None
            return None

        img = Image.open(texture_path).convert("RGBA")
        cropped = img.crop(_MINIMAP_FRAME_CROP)

        hx0, hy0, hx1, hy1 = _MINIMAP_FRAME_HOLE
        hole_w = hx1 - hx0
        hole_h = hy1 - hy0
        scale = inset_size / max(hole_w, hole_h, 1)

        scaled_w = max(1, int(cropped.size[0] * scale))
        scaled_h = max(1, int(cropped.size[1] * scale))
        scaled = cropped.resize((scaled_w, scaled_h), Image.LANCZOS)

        arr = np.array(scaled)
        rgb = arr[:, :, :3].astype(np.float64)
        alpha = arr[:, :, 3].astype(np.float64) / 255.0

        hole_box = (
            int(hx0 * scale),
            int(hy0 * scale),
            int(hx0 * scale) + inset_size,
            int(hy0 * scale) + inset_size,
        )
        # The source texture's cutout "hole" is drawn as opaque black pixels
        # (the original artist's placeholder for "the live minimap shows
        # through here"), not as transparent pixels -- confirmed by direct
        # pixel inspection (alpha > 240 inside the hole). Force alpha to 0
        # there so compositing this frame on top of the real minimap pixels
        # doesn't paint solid black over them.
        hbx0, hby0, hbx1, hby1 = hole_box
        hbx0c, hby0c = max(0, hbx0), max(0, hby0)
        hbx1c, hby1c = min(alpha.shape[1], hbx1), min(alpha.shape[0], hby1)
        if hbx1c > hbx0c and hby1c > hby0c:
            alpha[hby0c:hby1c, hbx0c:hbx1c] = 0.0

        result = (rgb, alpha, hole_box)
        _minimap_frame_cache[cache_key] = result
        return result
    except Exception:
        _minimap_frame_cache[cache_key] = None
        return None


def _composite_minimap_inset(
    main_frame: np.ndarray,
    obs: Any,
    playable_area: tuple[float, float, float, float] | None,
    camera_center: tuple[float, float] | None,
    asset_family: str = "SMACv2",
) -> np.ndarray:
    """Draw SC2's always-full-map minimap as a picture-in-picture inset.

    The main 3D camera has a fixed, engine-controlled field of view (~7-17
    world units, smaller than most SMAC maps -- see the module docstring
    investigation), so it can never show the whole battlefield at once.
    SC2 separately renders a minimap (``render_data.minimap``) that always
    covers the entire map regardless of the main camera's zoom/position.
    Compositing it into a corner of the main frame gives full map awareness
    (all four edges always visible) without needing any camera FOV control
    that the engine doesn't expose.

    Args:
        main_frame: The full-resolution 3D render, shape (H, W, 3).
        obs: The latest SC2 ``ResponseObservation`` (for ``render_data.minimap``).
        playable_area: (x0, y0, x1, y1) world-unit bounds of the playable area,
            used to draw the current camera viewport as a rectangle on the inset.
        camera_center: (cx, cy) world coordinates of the main camera's center,
            used for the viewport rectangle. If None, no rectangle is drawn.
        asset_family: ``"SMAC"`` or ``"SMACv2"`` -- selects which family's
            extracted asset tree the HUD frame texture is loaded from
            (``gym_gui/assets/SMAC/`` vs ``gym_gui/assets/SMACv2/``). These
            are separate directories, not the same source, even though
            their current texture content happens to be identical.

    Returns:
        A copy of ``main_frame`` with the minimap inset composited into the
        bottom-left corner (matching the native SC2 client's minimap
        position), or the original frame unchanged if the minimap data is
        unavailable.
    """
    try:
        observation = obs.observation
        if not observation.HasField("render_data"):
            return main_frame
        mm = observation.render_data.minimap
        if len(mm.data) == 0:
            return main_frame

        ch = mm.bits_per_pixel // 8
        mini = np.frombuffer(mm.data, dtype=np.uint8).reshape(mm.size.y, mm.size.x, ch)
        if ch == 4:
            mini = mini[:, :, :3]
        mini = np.ascontiguousarray(mini)

        h, w = main_frame.shape[:2]
        inset_size = max(32, int(min(h, w) * _MINIMAP_INSET_FRACTION))

        # Nearest-neighbor resize the minimap to the inset size
        row_idx = np.linspace(0, mini.shape[0] - 1, inset_size).astype(int)
        col_idx = np.linspace(0, mini.shape[1] - 1, inset_size).astype(int)
        mini_resized = mini[np.ix_(row_idx, col_idx)]

        frame = main_frame.copy()
        margin = _MINIMAP_INSET_MARGIN
        y0 = h - inset_size - margin
        x0 = margin
        y1 = y0 + inset_size
        x1 = x0 + inset_size
        if y0 < 0 or x1 > w:
            return frame  # Frame too small for an inset at this size

        # Paste the raw minimap pixels first. The native HUD frame texture's
        # cutout hole has near-zero alpha there, so compositing it on top
        # afterwards decorates the border without covering the minimap image.
        frame[y0:y1, x0:x1] = mini_resized

        frame_texture = _load_minimap_frame_texture(inset_size, asset_family=asset_family)
        if frame_texture is not None:
            tex_rgb, tex_alpha, hole_box = frame_texture
            hx0, hy0, hx1, hy1 = hole_box
            tex_h, tex_w = tex_alpha.shape

            # Position the full frame texture so its hole aligns with the
            # minimap we just pasted at (x0, y0)-(x1, y1).
            place_x0 = x0 - hx0
            place_y0 = y0 - hy0
            place_x1 = place_x0 + tex_w
            place_y1 = place_y0 + tex_h

            # Clip the placement (and the corresponding texture region) to
            # the frame's bounds so the border doesn't index out of range.
            src_x0 = max(0, -place_x0)
            src_y0 = max(0, -place_y0)
            src_x1 = tex_w - max(0, place_x1 - w)
            src_y1 = tex_h - max(0, place_y1 - h)
            dst_x0 = max(0, place_x0)
            dst_y0 = max(0, place_y0)
            dst_x1 = min(w, place_x1)
            dst_y1 = min(h, place_y1)

            if dst_x1 > dst_x0 and dst_y1 > dst_y0 and src_x1 > src_x0 and src_y1 > src_y0:
                region = frame[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float64)
                a = tex_alpha[src_y0:src_y1, src_x0:src_x1, None]
                blended = tex_rgb[src_y0:src_y1, src_x0:src_x1] * a + region * (1.0 - a)
                frame[dst_y0:dst_y1, dst_x0:dst_x1] = blended.astype(np.uint8)
        else:
            # Fallback: plain colored border if the frame texture asset
            # isn't available.
            border = 2
            frame[y0:y0 + border, x0:x1] = _MINIMAP_BORDER_COLOR
            frame[y1 - border:y1, x0:x1] = _MINIMAP_BORDER_COLOR
            frame[y0:y1, x0:x0 + border] = _MINIMAP_BORDER_COLOR
            frame[y0:y1, x1 - border:x1] = _MINIMAP_BORDER_COLOR

        # Draw the main camera's approximate viewport as a rectangle on the
        # inset, so the user can see where the zoomed-in view sits relative
        # to the whole map. The main camera's FOV isn't queryable at
        # runtime, so this uses a conservative fixed estimate (the smallest
        # measured FOV, ~7 world units) rather than claiming false precision.
        if camera_center is not None and playable_area is not None:
            px0, py0, px1, py1 = playable_area
            map_w = max(px1 - px0, 1e-6)
            map_h = max(py1 - py0, 1e-6)
            fov_estimate = 10.0  # world units; conservative mid-estimate

            cx, cy = camera_center
            # World -> inset pixel space. SC2 world Y increases upward;
            # image rows increase downward, so flip Y.
            def world_to_inset(wx: float, wy: float) -> tuple[int, int]:
                fx = (wx - px0) / map_w
                fy = 1.0 - (wy - py0) / map_h
                return (
                    x0 + int(fx * inset_size),
                    y0 + int(fy * inset_size),
                )

            half = fov_estimate / 2.0
            rx0, ry0 = world_to_inset(cx - half, cy + half)
            rx1, ry1 = world_to_inset(cx + half, cy - half)
            rx0, rx1 = sorted((max(x0, min(x1 - 1, rx0)), max(x0, min(x1 - 1, rx1))))
            ry0, ry1 = sorted((max(y0, min(y1 - 1, ry0)), max(y0, min(y1 - 1, ry1))))
            if rx1 > rx0 and ry1 > ry0:
                frame[ry0, rx0:rx1 + 1] = _MINIMAP_VIEWPORT_COLOR
                frame[ry1, rx0:rx1 + 1] = _MINIMAP_VIEWPORT_COLOR
                frame[ry0:ry1 + 1, rx0] = _MINIMAP_VIEWPORT_COLOR
                frame[ry0:ry1 + 1, rx1] = _MINIMAP_VIEWPORT_COLOR

        return frame
    except Exception:
        return main_frame


# Native SC2 top-bar HUD skin (resource bar background) + small resource
# icons, both extracted by the user from the game's own asset archive.
# Used to draw the SMAC/SMACv2 dashboard (minerals/vespene/supply/army/
# idle-workers + army composition icons) directly into the 3D render's
# top-right corner, matching the native client's resource-bar position,
# instead of a separate Qt widget in the side panel.
_HUD_BAR_TEXTURE_NAME = "ui_console_menubar_background_simulant.png"
_HUD_ICON_NAMES = {
    "minerals": ("core", "icon-mineral.png"),
    "vespene": ("liberty", "btn-doodad-vespenegeyser.png"),
    "supply": ("core", "icon-supply.png"),
}
_HUD_MARGIN = 10  # Pixels from the main frame's top-right corner
_HUD_BAR_HEIGHT_FRACTION = 0.045  # Bar height as a fraction of the main frame height
_HUD_TEXT_COLOR = (235, 235, 235)
_HUD_MINERAL_COLOR = (93, 174, 255)
_HUD_VESPENE_COLOR = (93, 255, 143)

_hud_bar_texture_cache: dict[tuple[str, int, int], np.ndarray | None] = {}
_hud_icon_cache: dict[tuple[str, str, int], Any] = {}
_hud_font_cache: dict[int, Any] = {}


def _hud_asset_root(asset_family: str) -> tuple[Path, Path]:
    """Return (liberty_textures_dir, core_textures_dir) for a game family.

    ``liberty.sc2mod`` holds the console panel/HUD chrome textures;
    ``core.sc2mod`` holds the small resource icons (minerals/supply).
    Both live under the same per-family asset tree, resolved independently
    for SMAC vs SMACv2 (see ``_minimap_frame_texture_path`` for why).
    """
    base = Path(__file__).resolve().parent.parent.parent / "assets" / asset_family / "mods"
    return (
        base / "liberty.sc2mod" / "base.sc2assets" / "assets" / "textures",
        base / "core.sc2mod" / "base.sc2assets" / "assets" / "textures",
    )


def _load_hud_bar_texture(
    asset_family: str, width: int, height: int
) -> np.ndarray | None:
    """Load and resize the HUD resource-bar background texture (RGBA float array)."""
    cache_key = (asset_family, width, height)
    if cache_key in _hud_bar_texture_cache:
        return _hud_bar_texture_cache[cache_key]
    try:
        from PIL import Image

        liberty_dir, _ = _hud_asset_root(asset_family)
        path = liberty_dir / _HUD_BAR_TEXTURE_NAME
        if not path.is_file():
            _hud_bar_texture_cache[cache_key] = None
            return None
        img = Image.open(path).convert("RGBA").resize((width, height), Image.LANCZOS)
        arr = np.array(img).astype(np.float64)
        _hud_bar_texture_cache[cache_key] = arr
        return arr
    except Exception:
        _hud_bar_texture_cache[cache_key] = None
        return None



def _load_hud_icon(asset_family: str, key: str, size: int) -> Any:
    """Load and resize a small HUD resource icon. Returns a PIL RGBA Image.

    Each entry in ``_HUD_ICON_NAMES`` specifies which sub-mod directory
    the file lives in (``"core"`` for ``core.sc2mod`` -- resource icons
    like minerals/supply -- or ``"liberty"`` for ``liberty.sc2mod`` --
    unit/command icons). An earlier version of this function always
    looked in ``core.sc2mod`` regardless, which silently failed (returned
    None, drawing nothing) for any icon actually living under
    ``liberty.sc2mod``.
    """
    cache_key = (asset_family, key, size)
    if cache_key in _hud_icon_cache:
        return _hud_icon_cache[cache_key]
    try:
        from PIL import Image

        entry = _HUD_ICON_NAMES.get(key)
        if entry is None:
            _hud_icon_cache[cache_key] = None
            return None
        submod, filename = entry
        liberty_dir, core_dir = _hud_asset_root(asset_family)
        base_dir = liberty_dir if submod == "liberty" else core_dir
        path = base_dir / filename
        if not path.is_file():
            _hud_icon_cache[cache_key] = None
            return None
        img = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
        _hud_icon_cache[cache_key] = img
        return img
    except Exception:
        _hud_icon_cache[cache_key] = None
        return None


def _hud_font(size: int) -> Any:
    if size in _hud_font_cache:
        return _hud_font_cache[size]
    from PIL import ImageFont

    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            font = ImageFont.truetype(font_path, size)
            _hud_font_cache[size] = font
            return font
        except (OSError, IOError):
            continue
    font = ImageFont.load_default()
    _hud_font_cache[size] = font
    return font


def _alpha_composite_region(
    frame: np.ndarray, x0: int, y0: int, patch_rgba: np.ndarray
) -> None:
    """Alpha-blend ``patch_rgba`` (H, W, 4 float array) onto ``frame`` in place at (x0, y0).

    Clips to ``frame``'s bounds so partially off-frame patches don't index
    out of range.
    """
    h, w = frame.shape[:2]
    ph, pw = patch_rgba.shape[:2]

    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    src_x1, src_y1 = pw - max(0, x0 + pw - w), ph - max(0, y0 + ph - h)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(w, x0 + pw), min(h, y0 + ph)

    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0 or src_x1 <= src_x0 or src_y1 <= src_y0:
        return

    region = frame[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float64)
    patch = patch_rgba[src_y0:src_y1, src_x0:src_x1]
    a = (patch[:, :, 3:4] / 255.0)
    blended = patch[:, :, :3] * a + region * (1.0 - a)
    frame[dst_y0:dst_y1, dst_x0:dst_x1] = blended.astype(np.uint8)


def _composite_resource_hud(
    main_frame: np.ndarray,
    player_common: Dict[str, Any] | None,
    ally_unit_names: List[str] | None,
    asset_family: str = "SMACv2",
) -> np.ndarray:
    """Draw the SMAC/SMACv2 resource + army-composition HUD in the top-right corner.

    Replaces the separate ``SmacDashboardWidget`` Qt panel in the side
    control panel -- this renders the same information (minerals, vespene,
    supply, army count, idle workers, per-unit-type army composition icons
    with live counts) directly into the 3D render frame using the native
    SC2 HUD console-panel skin as a background, matching where the native
    client actually shows its resource bar (top-right).

    Args:
        main_frame: The full-resolution 3D render (already has the
            minimap inset composited in, if enabled), shape (H, W, 3).
        player_common: dict of PlayerCommon fields (minerals, vespene,
            food_used, food_cap, army_count, idle_worker_count), or None
            if not yet available (draws nothing).
        ally_unit_names: resolved unit type name strings for currently
            alive allied units (e.g. ``["Marine_RL", "Marine_RL"]``), used
            to build the army composition icon row. None draws no army row.
        asset_family: ``"SMAC"`` or ``"SMACv2"`` -- selects which family's
            extracted asset tree to load HUD textures/icons from. These
            are separate directories, not the same source, even though
            their current texture content happens to be identical.

    Returns:
        A copy of ``main_frame`` with the HUD composited in, or the
        original frame unchanged if there's nothing to draw or the PIL
        drawing pipeline is unavailable.
    """
    if player_common is None:
        return main_frame
    try:
        from PIL import Image, ImageDraw

        h, w = main_frame.shape[:2]
        bar_h = max(20, int(h * _HUD_BAR_HEIGHT_FRACTION))
        bar_w = max(120, int(w * 0.30))
        bar_x1 = w - _HUD_MARGIN
        bar_x0 = bar_x1 - bar_w
        bar_y0 = _HUD_MARGIN
        bar_y1 = bar_y0 + bar_h
        if bar_x0 < 0 or bar_y1 > h:
            return main_frame  # Frame too small for the HUD at this size

        frame = main_frame.copy()

        # A plain dark translucent panel behind the resource icons/text.
        # An earlier version stretched the native
        # "ui_console_menubar_background_simulant.png" texture to fit --
        # but that texture bakes in 4 fixed-width bordered slot dividers
        # at its native 241x56 size, so stretching it to an arbitrary
        # bar_w/bar_h (to match the render resolution) visibly misaligned
        # the slot borders against the icon/text cursor positions, which
        # are laid out independently. A plain panel avoids that mismatch.
        frame[bar_y0:bar_y1, bar_x0:bar_x1] = (
            frame[bar_y0:bar_y1, bar_x0:bar_x1].astype(np.float64) * 0.35
        ).astype(np.uint8)

        # Army composition row: one small unit icon + count per distinct
        # allied unit type, drawn *below* the resource bar (top-right
        # corner), so the resource fields read first -- the resource
        # bar's supply field ("used/cap") already reflects total army
        # size, so a separate "Army N" text label is redundant with
        # these icons.
        if ally_unit_names:
            from collections import Counter

            from gym_gui.core.adapters.smac_unit_icons import get_unit_icon_path

            counts = Counter(ally_unit_names)
            comp_icon_size = max(14, int(bar_h * 0.9))
            comp_overlay_w = bar_w
            comp_overlay_h = comp_icon_size + 6
            comp_y0 = bar_y1 + 4
            if comp_y0 + comp_overlay_h > h:
                comp_y0 = None  # Not enough room below; skip the row

            if comp_y0 is not None:
                comp_overlay = Image.new("RGBA", (comp_overlay_w, comp_overlay_h), (0, 0, 0, 0))
                comp_draw = ImageDraw.Draw(comp_overlay)
                comp_font = _hud_font(max(8, int(comp_icon_size * 0.42)))

                # Lay out right-to-left so the row hugs the top-right corner
                # like the resource bar above it.
                x = comp_overlay_w
                for unit_name, count in sorted(counts.items()):
                    icon_path = get_unit_icon_path(unit_name, asset_family=asset_family)
                    label = f"x{count}"
                    label_w = comp_draw.textlength(label, font=comp_font)
                    entry_w = comp_icon_size + 3 + label_w + 6
                    x -= entry_w
                    if x < 0:
                        break
                    if icon_path is not None:
                        icon_img = Image.open(icon_path).convert("RGBA").resize(
                            (comp_icon_size, comp_icon_size), Image.LANCZOS
                        )
                        comp_overlay.alpha_composite(icon_img, (int(x), 0))
                    comp_draw.text(
                        (x + comp_icon_size + 3, comp_icon_size // 2),
                        label,
                        font=comp_font,
                        fill=_HUD_TEXT_COLOR,
                        anchor="lm",
                    )

                _alpha_composite_region(
                    frame, bar_x1 - comp_overlay_w, comp_y0, np.array(comp_overlay)
                )

        # Draw resource text onto a transparent overlay, then composite
        # that (so anti-aliased glyph edges blend correctly onto the bar).
        icon_size = max(12, int(bar_h * 0.6))
        overlay = Image.new("RGBA", (bar_w, bar_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = _hud_font(max(9, int(bar_h * 0.36)))

        minerals = int(player_common.get("minerals", 0))
        vespene = int(player_common.get("vespene", 0))
        food_used = int(player_common.get("food_used", 0))
        food_cap = int(player_common.get("food_cap", 0))

        pad = max(3, int(bar_h * 0.12))
        text_y = bar_h // 2
        cursor_x = pad

        mineral_icon = _load_hud_icon(asset_family, "minerals", icon_size)
        if mineral_icon is not None:
            overlay.alpha_composite(mineral_icon, (cursor_x, (bar_h - icon_size) // 2))
            cursor_x += icon_size + 3
        text = f"{minerals}"
        draw.text((cursor_x, text_y), text, font=font, fill=_HUD_MINERAL_COLOR, anchor="lm")
        cursor_x += draw.textlength(text, font=font) + pad * 2

        vespene_icon = _load_hud_icon(asset_family, "vespene", icon_size)
        if vespene_icon is not None:
            overlay.alpha_composite(vespene_icon, (int(cursor_x), (bar_h - icon_size) // 2))
            cursor_x += icon_size + 3
        text = f"{vespene}"
        draw.text((cursor_x, text_y), text, font=font, fill=_HUD_VESPENE_COLOR, anchor="lm")
        cursor_x += draw.textlength(text, font=font) + pad * 2

        # Supply icon doubles as the "unit counter" field (used/cap, e.g.
        # "15/20"), matching the native client's rightmost resource-bar
        # slot. Only these three fields (minerals, vespene, supply/unit
        # count) belong in the top bar -- idle-worker count and per-unit
        # army composition are shown in the row below instead.
        supply_icon = _load_hud_icon(asset_family, "supply", icon_size)
        if supply_icon is not None:
            overlay.alpha_composite(supply_icon, (int(cursor_x), (bar_h - icon_size) // 2))
            cursor_x += icon_size + 3
        text = f"{food_used}/{food_cap}"
        draw.text((cursor_x, text_y), text, font=font, fill=_HUD_TEXT_COLOR, anchor="lm")

        _alpha_composite_region(frame, bar_x0, bar_y0, np.array(overlay))

        return frame
    except Exception:
        return main_frame


_HEALTH_BAR_WORLD_WIDTH = 0.8  # World units wide (roughly one unit's diameter)
_HEALTH_BAR_Z_OFFSET = 1.6  # World units above the unit's ground position
_HEALTH_BAR_ALLY_COLOR = (60, 220, 60)
_HEALTH_BAR_ENEMY_COLOR = (220, 60, 60)
_HEALTH_BAR_SHIELD_COLOR = (80, 160, 255)
_HEALTH_BAR_BG_COLOR = (90, 90, 90)


def _draw_unit_health_bars(smac_env: Any) -> None:
    """Draw a health/shield bar directly above each unit, in world space.

    Uses SC2's ``RequestDebug`` / ``DebugDraw`` API to draw world-space
    lines exactly at each unit's position. The SC2 engine itself projects
    these onto the screen using its real camera matrix, so the bars track
    the unit and camera perfectly with no drift -- unlike an earlier
    version of this function, which approximated the world-to-screen
    projection in Python using a fixed guessed field-of-view constant.
    That approximation was measurably wrong (predicted screen shifts did
    not match the unit's actual measured on-screen movement after a
    camera pan) and got visibly worse the further a unit was from the
    frame center, because the real 3D camera has genuine perspective/tilt
    (confirmed via direct pixel-shift measurement: panning shifts pixels
    non-uniformly across the frame). Debug-drawn world-space lines have no
    such error since the engine does the projection, not us.

    Must be called *before* the next ``observe()`` -- like camera moves,
    debug draws take effect on the following render frame. Does not
    affect ``game_loop`` (verified: ``RequestDebug`` is not a simulation
    step, unlike ``RequestStep``).

    Silently does nothing if raw unit data isn't available.
    """
    try:
        from s2clientprotocol import sc2api_pb2 as sc_pb

        obs = smac_env._obs
        if obs is None or not obs.observation.HasField("raw_data"):
            return
        units = obs.observation.raw_data.units
        if not units:
            return

        debug_cmd = sc_pb.RequestDebug()
        draw = debug_cmd.debug.add().draw

        for unit in units:
            health_max = unit.health_max
            if health_max <= 0:
                continue  # Avoid div-by-zero for units with no defined max health

            half_w = _HEALTH_BAR_WORLD_WIDTH / 2.0
            bar_z = unit.pos.z + _HEALTH_BAR_Z_OFFSET
            left_x = unit.pos.x - half_w
            right_x = unit.pos.x + half_w

            # Background (full-width bar showing max health extent)
            bg_line = draw.lines.add()
            bg_line.color.r, bg_line.color.g, bg_line.color.b = _HEALTH_BAR_BG_COLOR
            bg_line.line.p0.x, bg_line.line.p0.y, bg_line.line.p0.z = left_x, unit.pos.y, bar_z
            bg_line.line.p1.x, bg_line.line.p1.y, bg_line.line.p1.z = right_x, unit.pos.y, bar_z

            # Health fill, from the left edge, proportional to current/max health
            health_frac = max(0.0, min(1.0, unit.health / health_max))
            if health_frac > 0:
                fill_x = left_x + _HEALTH_BAR_WORLD_WIDTH * health_frac
                is_ally = unit.alliance == 1  # raw.proto Alliance.Self == 1
                color = _HEALTH_BAR_ALLY_COLOR if is_ally else _HEALTH_BAR_ENEMY_COLOR
                hp_line = draw.lines.add()
                hp_line.color.r, hp_line.color.g, hp_line.color.b = color
                hp_line.line.p0.x, hp_line.line.p0.y, hp_line.line.p0.z = left_x, unit.pos.y, bar_z
                hp_line.line.p1.x, hp_line.line.p1.y, hp_line.line.p1.z = fill_x, unit.pos.y, bar_z

            # Shield fill, drawn as a second line slightly above the health bar
            shield_max = unit.shield_max
            if shield_max > 0 and unit.shield > 0:
                shield_frac = max(0.0, min(1.0, unit.shield / shield_max))
                shield_fill_x = left_x + _HEALTH_BAR_WORLD_WIDTH * shield_frac
                shield_bar_z = bar_z + 0.15
                sh_line = draw.lines.add()
                sh_line.color.r, sh_line.color.g, sh_line.color.b = _HEALTH_BAR_SHIELD_COLOR
                sh_line.line.p0.x, sh_line.line.p0.y, sh_line.line.p0.z = left_x, unit.pos.y, shield_bar_z
                sh_line.line.p1.x, sh_line.line.p1.y, sh_line.line.p1.z = shield_fill_x, unit.pos.y, shield_bar_z

        smac_env._controller._client.send(debug=debug_cmd)
    except Exception:
        pass


class SMACAdapter(EnvironmentAdapter[List[np.ndarray], List[int]]):
    """Adapter bridging SMAC v1's ``MultiAgentEnv`` to MOSAIC's adapter interface.

    SMAC v1 uses hand-designed maps with fixed team compositions.
    All agents act simultaneously each timestep (parallel stepping).
    """

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP)

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: Any | None = None,
    ) -> None:
        super().__init__(context)

        # Import config type locally to avoid circular imports
        from gym_gui.core.ui.game_config.game_configs import SMACConfig

        if config is None:
            config = SMACConfig()
        if not isinstance(config, SMACConfig):
            config = SMACConfig()

        self._config: Any = config
        self._map_name: str = config.map_name
        self._smac_env: Any = None
        self._n_agents: int = 0
        self._n_actions: int = 0
        self._obs_shape: int = 0
        self._state_shape: int = 0
        self._episode_limit: int = 0
        self._step_counter: int = 0
        self._camera_center: tuple[float, float] | None = None

    @property
    def stepping_paradigm(self) -> SteppingParadigm:  # type: ignore[override]
        return SteppingParadigm.SIMULTANEOUS

    @property
    def action_space(self) -> gym.Space[Any]:
        """Per-agent action space: Discrete(n_actions).

        SMAC is not a Gymnasium environment, so we construct a synthetic
        space from the env_info metadata returned after load().
        """
        if self._n_actions == 0:
            return gym.spaces.Discrete(1)
        return gym.spaces.Discrete(self._n_actions)

    @property
    def observation_space(self) -> gym.Space[Any]:
        """Per-agent observation space: Box of shape (obs_shape,).

        Each agent receives a float vector containing distances, health,
        shield, unit type, and relative positions of allies/enemies.
        """
        if self._obs_shape == 0:
            return gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        return gym.spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_shape,), dtype=np.float32,
        )

    def load(self) -> None:
        """Instantiate the SMAC v1 environment.

        Validates SC2PATH and imports from the ``smac`` package.

        SC2 path resolution order:
            1. ``config.sc2_path`` (UI text field)
            2. ``SC2PATH`` environment variable (`.env` or shell)
            3. ``var/data`` project-local directory (paths.VAR_SC2_DIR)
            4. Error with download instructions
        """
        from gym_gui.config.paths import VAR_SC2_DIR

        # Resolve StarCraft II installation path
        sc2_path = (
            self._config.sc2_path
            or os.environ.get("SC2PATH")
            or (str(VAR_SC2_DIR) if VAR_SC2_DIR.is_dir() else None)
        )
        # python-dotenv does not expand shell variables ($PWD, ~, etc.) — expand here
        if sc2_path:
            sc2_path = os.path.expandvars(os.path.expanduser(sc2_path))
        if sc2_path and os.path.isdir(sc2_path):
            os.environ["SC2PATH"] = sc2_path
        elif not os.environ.get("SC2PATH"):
            self.log_constant(
                LOG_SMAC_SC2_PATH_MISSING,
                extra={"map_name": self._map_name},
            )
            raise RuntimeError(
                "StarCraft II installation not found. "
                "Set SC2PATH environment variable, provide sc2_path in config, "
                "or install into var/data/. "
                "Download from: https://github.com/Blizzard/s2client-proto#linux-packages"
            )

        # Ensure protobuf compatibility (s2clientprotocol requires pure-python impl)
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

        from smac.env import StarCraft2Env

        env_kwargs: Dict[str, Any] = {
            "map_name": self._map_name,
            "difficulty": self._config.difficulty,
            "reward_sparse": self._config.reward_sparse,
            "reward_only_positive": self._config.reward_only_positive,
            "reward_scale": self._config.reward_scale,
            "reward_scale_rate": self._config.reward_scale_rate,
            "obs_own_health": self._config.obs_own_health,
            "obs_pathing_grid": self._config.obs_pathing_grid,
            "obs_terrain_height": self._config.obs_terrain_height,
        }
        if self._config.seed is not None:
            env_kwargs["seed"] = self._config.seed
        if self._config.episode_limit is not None:
            env_kwargs["episode_limit"] = self._config.episode_limit

        self._smac_env = StarCraft2Env(**env_kwargs)

        # Patch _launch() for 3D GPU rendering before any reset() call
        if getattr(self._config, "renderer", "3d") == "3d":
            _patch_launch_for_3d(
                self._smac_env,
                render_size=getattr(self._config, "render_resolution", _3D_RENDER_SIZE),
            )

        env_info = self._smac_env.get_env_info()
        self._n_agents = env_info["n_agents"]
        self._n_actions = env_info["n_actions"]
        self._obs_shape = env_info["obs_shape"]
        self._state_shape = env_info["state_shape"]
        self._episode_limit = env_info["episode_limit"]

        self.log_constant(
            LOG_SMAC_ENV_CREATED,
            extra={
                "map_name": self._map_name,
                "n_agents": self._n_agents,
                "n_actions": self._n_actions,
                "obs_shape": self._obs_shape,
                "state_shape": self._state_shape,
                "episode_limit": self._episode_limit,
            },
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[List[np.ndarray]]:
        """Reset the SMAC environment and return initial observations."""
        if self._smac_env is None:
            self.load()

        self._smac_env.reset()
        self._step_counter = 0

        # Cache playable area after first reset (SC2 is now running)
        if not hasattr(self, "_playable_area"):
            try:
                gi = self._smac_env._controller.game_info()
                pa = gi.start_raw.playable_area
                self._playable_area = (pa.p0.x, pa.p0.y, pa.p1.x, pa.p1.y)
            except Exception:
                self._playable_area = (
                    0.0, 0.0,
                    float(self._smac_env.map_x),
                    float(self._smac_env.map_y),
                )

        # Center 3D camera on the map's fixed midpoint (default camera
        # sits at world origin (0,0), which typically misses the SMAC
        # battle area entirely). Skippable via
        # SMACConfig.camera_auto_center=False if the user prefers SC2's
        # default camera position.
        #
        # The 3D camera's field of view is fixed by the SC2 engine per-map
        # (measured ~7-17 world units, smaller than the 28x28 playable
        # area on most maps) and is NOT adjustable -- see
        # ``_patch_launch_for_3d`` for the measured evidence that
        # ``render.width`` has no effect on 3D output. So this always
        # starts the camera centered on the map, giving a stable reference
        # point; the user can then pan with the mouse (``move_camera``,
        # which sends a real ``ActionRaw.camera_move``) to look around the
        # rest of the map.
        if (
            getattr(self._config, "renderer", "3d") == "3d"
            and getattr(self._config, "camera_auto_center", True)
        ):
            center = _center_camera_on_map(self._smac_env)
            if center is not None:
                self._camera_center = center
            else:
                _LOGGER.debug(
                    "SMAC camera auto-center failed (no units observed yet or "
                    "camera_move action rejected); leaving default camera position."
                )

        obs = self._smac_env.get_obs()
        state = self._smac_env.get_state()
        avail_actions = [
            self._smac_env.get_avail_agent_actions(i)
            for i in range(self._n_agents)
        ]

        self.log_constant(
            LOG_SMAC_ENV_RESET,
            extra={
                "map_name": self._map_name,
                "n_agents": self._n_agents,
                "seed": seed,
            },
        )

        return self._package_step(
            observation=obs,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={
                "num_agents": self._n_agents,
                "agent_observations": obs,
                "global_state": state,
                "avail_actions": avail_actions,
                "action_masks": avail_actions,
                "step": 0,
                **_extract_player_status(self._smac_env),
            },
        )

    def step(self, action: List[int]) -> AdapterStep[List[np.ndarray]]:
        """Execute simultaneous actions for all agents.

        Args:
            action: List of integer actions, one per agent.

        Returns:
            AdapterStep with per-agent observations, shared reward, and info
            containing global_state and action_masks.
        """
        if self._smac_env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        reward, terminated, info = self._smac_env.step(action)
        self._step_counter += 1

        obs = self._smac_env.get_obs()
        state = self._smac_env.get_state()
        avail_actions = [
            self._smac_env.get_avail_agent_actions(i)
            for i in range(self._n_agents)
        ]

        battle_won = info.get("battle_won", False)

        # SMAC returns a single shared reward (fully cooperative)
        agent_rewards = [float(reward)] * self._n_agents

        step_info: Dict[str, Any] = {
            "num_agents": self._n_agents,
            "agent_observations": obs,
            "global_state": state,
            "avail_actions": avail_actions,
            "action_masks": avail_actions,
            "agent_rewards": agent_rewards,
            "battle_won": battle_won,
            "step": self._step_counter,
        }
        step_info.update(_extract_player_status(self._smac_env))
        step_info.update(info)

        self.log_constant(
            LOG_SMAC_STEP_SUMMARY,
            extra={
                "step": self._step_counter,
                "reward": float(reward),
                "terminated": terminated,
                "battle_won": battle_won,
            },
        )

        if terminated:
            self.log_constant(
                LOG_SMAC_BATTLE_RESULT,
                extra={
                    "map_name": self._map_name,
                    "battle_won": battle_won,
                    "steps": self._step_counter,
                    "total_reward": float(reward),
                },
            )

        return self._package_step(
            observation=obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=False,
            info=step_info,
        )

    def render(self) -> Optional[Dict[str, Any]]:
        """Render using 3D engine, heatmap overlays, or classic PyGame.

        The renderer is chosen by ``self._config.renderer``:

        - ``"3d"``: Full SC2 engine 3D rendering via EGL (GPU-accelerated).
        - ``"heatmap"``: 2x2 panel view with terrain, health, unit type,
          and shield/energy overlays (pure numpy).
        - ``"classic"``: SMAC's built-in PyGame renderer (coloured circles
          with health arcs).
        """
        if self._smac_env is None:
            return None

        renderer = getattr(self._config, "renderer", "3d")

        if renderer == "classic":
            return self._render_pygame()

        if renderer == "3d":
            result = self._render_3d()
            if result is not None:
                return result
            # Fall through to heatmap if 3D fails
            renderer = "heatmap"

        if renderer == "heatmap":
            return self._render_heatmap()

        return self._render_pygame()

    def _render_3d(self) -> Optional[Dict[str, Any]]:
        """Extract 3D GPU-rendered RGB frame from SC2 engine via render_data."""
        try:
            if getattr(self._config, "unit_health_bars", True):
                # Debug draws (like camera moves) take effect on the NEXT
                # observation, so draw first, then re-observe, so the
                # frame we extract pixels from already has the bars in it.
                _draw_unit_health_bars(self._smac_env)
                self._smac_env._obs = self._smac_env._controller.observe()

            obs = self._smac_env._obs
            if obs is None:
                return None
            observation = obs.observation
            if not observation.HasField("render_data"):
                return None
            map_img = observation.render_data.map
            if len(map_img.data) == 0:
                return None
            channels = map_img.bits_per_pixel // 8
            frame = np.frombuffer(map_img.data, dtype=np.uint8).reshape(
                map_img.size.y, map_img.size.x, channels,
            )
            if channels == 4:
                frame = frame[:, :, :3]  # Drop alpha channel
            if getattr(self._config, "minimap_inset", True):
                frame = _composite_minimap_inset(
                    frame,
                    obs,
                    getattr(self, "_playable_area", None),
                    self._camera_center,
                    asset_family="SMAC",
                )
            if getattr(self._config, "smac_hud", True):
                status = _extract_player_status(self._smac_env)
                frame = _composite_resource_hud(
                    frame,
                    status.get("player_common"),
                    status.get("ally_unit_names"),
                    asset_family="SMAC",
                )
            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": frame,
                "game_id": self._map_name,
                "num_agents": self._n_agents,
                "step": self._step_counter,
            }
        except Exception as exc:
            self.log_constant(
                LOG_SMAC_RENDER_ERROR,
                exc_info=exc,
                extra={"map_name": self._map_name, "renderer": "3d"},
            )
            return None

    def _render_heatmap(self) -> Optional[Dict[str, Any]]:
        """Render using custom 2x2 heatmap feature-layer panels."""
        try:
            from gym_gui.rendering.smac_heatmap import (
                SMACHeatmapRenderer,
                extract_frame_data,
            )

            if not hasattr(self, "_heatmap_renderer"):
                self._heatmap_renderer = SMACHeatmapRenderer()

            frame_data = extract_frame_data(
                self._smac_env,
                self._step_counter,
                self._map_name,
                getattr(self, "_playable_area", (0.0, 0.0, 32.0, 32.0)),
            )
            if frame_data is None:
                return self._render_pygame()

            frame = self._heatmap_renderer.render(frame_data)
            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": frame,
                "game_id": self._map_name,
                "num_agents": self._n_agents,
                "step": self._step_counter,
            }
        except Exception as exc:
            self.log_constant(
                LOG_SMAC_RENDER_ERROR,
                exc_info=exc,
                extra={"map_name": self._map_name, "renderer": "heatmap"},
            )
            return self._render_pygame()

    def _render_pygame(self) -> Optional[Dict[str, Any]]:
        """Fallback: SMAC's built-in PyGame 2D renderer."""
        if self._smac_env is None:
            return None
        try:
            frame = self._smac_env.render(mode="rgb_array")
            if isinstance(frame, np.ndarray):
                return {
                    "mode": RenderMode.RGB_ARRAY.value,
                    "rgb": frame,
                    "game_id": self._map_name,
                    "num_agents": self._n_agents,
                    "step": self._step_counter,
                }
        except Exception as exc:
            self.log_constant(
                LOG_SMAC_RENDER_ERROR,
                exc_info=exc,
                extra={"map_name": self._map_name},
            )
        return None

    def close(self) -> None:
        """Close the SMAC environment and terminate the SC2 process."""
        if self._smac_env is not None:
            self.log_constant(
                LOG_SMAC_ENV_CLOSED,
                extra={"map_name": self._map_name},
            )
            self._smac_env.close()
            self._smac_env = None

    def save_replay(self) -> None:
        """Save a StarCraft II replay file for the current episode."""
        if self._smac_env is not None and hasattr(self._smac_env, "save_replay"):
            self._smac_env.save_replay()

    # ─────────────────────────────────────────────────────────────────
    # 3D Camera control (mouse panning in the render widget)
    # ─────────────────────────────────────────────────────────────────
    #
    # NOTE: there is no engine-level zoom/FOV control available during live
    # gameplay. ``InterfaceOptions.render.width`` (formerly used here) is
    # documented as feature-layer-only and measured to have zero effect on
    # 3D render output. ``ObserverAction.camera_move`` (which does have a
    # ``distance`` zoom field) is replay-only -- pysc2 makes it a no-op
    # during live play (``@valid_status(Status.in_replay)``). The only real
    # camera control available live is *position*, via
    # ``ActionRaw.camera_move``, which is what ``move_camera`` below sends.

    # World units of camera pan per pixel of mouse drag. This is a fixed UI
    # sensitivity constant, not a measurement of the engine's actual FOV
    # (which is fixed per-map by SC2 itself, typically ~7-17 world units,
    # and cannot be queried or changed via any interface option).
    _PAN_SENSITIVITY = 0.12

    def move_camera(self, dx_world: float, dy_world: float) -> None:
        """Pan the real in-engine 3D camera via ``ActionRaw.camera_move``.

        This sends an actual SC2 API action and re-observes, so it moves
        the true render camera (not a numpy crop of a static frame).
        Clamped to stay within the map's playable area so panning can't
        drive the camera off into empty space with no landmarks.

        Args:
            dx_world: Pan right (positive) or left (negative) in world units.
            dy_world: Pan up (positive) or down (negative) in world units.
        """
        if self._smac_env is None:
            return
        cx, cy = self._camera_center or (0.0, 0.0)
        new_cx = cx + dx_world
        new_cy = cy + dy_world

        pa = getattr(self, "_playable_area", None)
        if pa is not None:
            x0, y0, x1, y1 = pa
            new_cx = max(x0, min(x1, new_cx))
            new_cy = max(y0, min(y1, new_cy))

        _move_camera_to(self._smac_env, new_cx, new_cy)
        self._camera_center = (new_cx, new_cy)

    def build_step_state(
        self,
        observation: Any,
        info: Any,
    ) -> StepState:
        """Construct the canonical StepState for multi-agent display."""
        agent_snapshots: List[AgentSnapshot] = []
        info_dict = dict(info) if isinstance(info, dict) else {}

        for i in range(self._n_agents):
            snapshot = AgentSnapshot(
                name=f"agent_{i}",
                role="active",
                info={
                    "reward": info_dict.get("agent_rewards", [0.0] * self._n_agents)[i]
                    if i < len(info_dict.get("agent_rewards", []))
                    else 0.0,
                },
            )
            agent_snapshots.append(snapshot)

        return StepState(
            active_agent=None,  # simultaneous -- no single active agent
            agents=tuple(agent_snapshots),
            metrics={
                "step_count": self._step_counter,
                "num_agents": self._n_agents,
                "battle_won": info_dict.get("battle_won", False),
            },
            environment={
                "map_name": self._map_name,
                "family": "smac",
                "paradigm": "simultaneous",
            },
            raw=info_dict,
        )

    # ─────────────────────────────────────────────────────────────────
    # Multi-agent helper methods
    # ─────────────────────────────────────────────────────────────────

    def get_avail_actions(self) -> List[List[int]]:
        """Get available action masks for all agents."""
        if self._smac_env is None:
            return []
        return [
            self._smac_env.get_avail_agent_actions(i)
            for i in range(self._n_agents)
        ]

    def get_global_state(self) -> Optional[np.ndarray]:
        """Get the global state vector (for centralized training)."""
        if self._smac_env is None:
            return None
        return self._smac_env.get_state()

    @property
    def num_agents(self) -> int:
        return self._n_agents

    @property
    def num_actions(self) -> int:
        return self._n_actions


# ═══════════════════════════════════════════════════════════════════════════
# Concrete adapter subclasses for each SMAC v1 map
# ═══════════════════════════════════════════════════════════════════════════


class SMAC3MAdapter(SMACAdapter):
    """3 Marines vs 3 Marines (Easy)."""

    id = GameId.SMAC_3M.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="3m")
        super().__init__(context, config=config)


class SMAC8MAdapter(SMACAdapter):
    """8 Marines vs 8 Marines (Easy)."""

    id = GameId.SMAC_8M.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="8m")
        super().__init__(context, config=config)


class SMAC2S3ZAdapter(SMACAdapter):
    """2 Stalkers + 3 Zealots vs same (Easy)."""

    id = GameId.SMAC_2S3Z.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="2s3z")
        super().__init__(context, config=config)


class SMAC3S5ZAdapter(SMACAdapter):
    """3 Stalkers + 5 Zealots vs same (Easy)."""

    id = GameId.SMAC_3S5Z.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="3s5z")
        super().__init__(context, config=config)


class SMAC5Mvs6MAdapter(SMACAdapter):
    """5 Marines vs 6 Marines (Hard, asymmetric)."""

    id = GameId.SMAC_5M_VS_6M.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="5m_vs_6m")
        super().__init__(context, config=config)


class SMACMMM2Adapter(SMACAdapter):
    """1 Medivac + 2 Marauders + 7 Marines vs 1M+3Ma+8Mar (Super Hard)."""

    id = GameId.SMAC_MMM2.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="MMM2")
        super().__init__(context, config=config)


class SMAC25MAdapter(SMACAdapter):
    """25 Marines vs 25 Marines (Easy, large-scale symmetric)."""

    id = GameId.SMAC_25M.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="25m")
        super().__init__(context, config=config)


class SMAC8Mvs9MAdapter(SMACAdapter):
    """8 Marines vs 9 Marines (Hard, asymmetric)."""

    id = GameId.SMAC_8M_VS_9M.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="8m_vs_9m")
        super().__init__(context, config=config)


class SMAC10Mvs11MAdapter(SMACAdapter):
    """10 Marines vs 11 Marines (Hard, asymmetric)."""

    id = GameId.SMAC_10M_VS_11M.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="10m_vs_11m")
        super().__init__(context, config=config)


class SMAC27Mvs30MAdapter(SMACAdapter):
    """27 Marines vs 30 Marines (Super Hard, large-scale asymmetric)."""

    id = GameId.SMAC_27M_VS_30M.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="27m_vs_30m")
        super().__init__(context, config=config)


class SMACMMMAdapter(SMACAdapter):
    """1 Medivac + 2 Marauders + 7 Marines vs same (Hard, mixed, symmetric)."""

    id = GameId.SMAC_MMM.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="MMM")
        super().__init__(context, config=config)


class SMAC3S5Zvs3S6ZAdapter(SMACAdapter):
    """3 Stalkers + 5 Zealots vs 3 Stalkers + 6 Zealots (Super Hard, asymmetric)."""

    id = GameId.SMAC_3S5Z_VS_3S6Z.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="3s5z_vs_3s6z")
        super().__init__(context, config=config)


class SMAC3Svs3ZAdapter(SMACAdapter):
    """3 Stalkers vs 3 Zealots (Easy, kiting)."""

    id = GameId.SMAC_3S_VS_3Z.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="3s_vs_3z")
        super().__init__(context, config=config)


class SMAC3Svs4ZAdapter(SMACAdapter):
    """3 Stalkers vs 4 Zealots (Hard, kiting)."""

    id = GameId.SMAC_3S_VS_4Z.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="3s_vs_4z")
        super().__init__(context, config=config)


class SMAC3Svs5ZAdapter(SMACAdapter):
    """3 Stalkers vs 5 Zealots (Super Hard, kiting)."""

    id = GameId.SMAC_3S_VS_5Z.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="3s_vs_5z")
        super().__init__(context, config=config)


class SMAC1C3S5ZAdapter(SMACAdapter):
    """1 Colossus + 3 Stalkers + 5 Zealots vs same (Hard, mixed, symmetric)."""

    id = GameId.SMAC_1C3S5Z.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="1c3s5z")
        super().__init__(context, config=config)


class SMAC2Mvs1ZAdapter(SMACAdapter):
    """2 Marines vs 1 Zealot (Easy, asymmetric races)."""

    id = GameId.SMAC_2M_VS_1Z.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="2m_vs_1z")
        super().__init__(context, config=config)


class SMACCorridorAdapter(SMACAdapter):
    """6 Zealots vs 24 Zerglings (Hard, chokepoint kiting)."""

    id = GameId.SMAC_CORRIDOR.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="corridor")
        super().__init__(context, config=config)


class SMAC6Hvs8ZAdapter(SMACAdapter):
    """6 Hydralisks vs 8 Zealots (Super Hard, asymmetric races)."""

    id = GameId.SMAC_6H_VS_8Z.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="6h_vs_8z")
        super().__init__(context, config=config)


class SMAC2Svs1SCAdapter(SMACAdapter):
    """2 Stalkers vs 1 Spine Crawler (Easy, asymmetric races)."""

    id = GameId.SMAC_2S_VS_1SC.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="2s_vs_1sc")
        super().__init__(context, config=config)


class SMACSoManyBanelingAdapter(SMACAdapter):
    """7 Zealots vs 32 Banelings (Hard, swarm)."""

    id = GameId.SMAC_SO_MANY_BANELING.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="so_many_baneling")
        super().__init__(context, config=config)


class SMACBaneVsBaneAdapter(SMACAdapter):
    """24 Zerglings + Banelings vs same (Easy, mixed, symmetric)."""

    id = GameId.SMAC_BANE_VS_BANE.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="bane_vs_bane")
        super().__init__(context, config=config)


class SMAC2Cvs64ZGAdapter(SMACAdapter):
    """2 Colossi vs 64 Zerglings (Super Hard, swarm)."""

    id = GameId.SMAC_2C_VS_64ZG.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.core.ui.game_config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="2c_vs_64zg")
        super().__init__(context, config=config)


# ═══════════════════════════════════════════════════════════════════════════
# Adapter registry for factory pattern
# ═══════════════════════════════════════════════════════════════════════════

SMAC_ADAPTERS: Dict[GameId, type[SMACAdapter]] = {
    GameId.SMAC_3M: SMAC3MAdapter,
    GameId.SMAC_8M: SMAC8MAdapter,
    GameId.SMAC_25M: SMAC25MAdapter,
    GameId.SMAC_2S3Z: SMAC2S3ZAdapter,
    GameId.SMAC_3S5Z: SMAC3S5ZAdapter,
    GameId.SMAC_5M_VS_6M: SMAC5Mvs6MAdapter,
    GameId.SMAC_8M_VS_9M: SMAC8Mvs9MAdapter,
    GameId.SMAC_10M_VS_11M: SMAC10Mvs11MAdapter,
    GameId.SMAC_27M_VS_30M: SMAC27Mvs30MAdapter,
    GameId.SMAC_MMM: SMACMMMAdapter,
    GameId.SMAC_MMM2: SMACMMM2Adapter,
    GameId.SMAC_3S5Z_VS_3S6Z: SMAC3S5Zvs3S6ZAdapter,
    GameId.SMAC_3S_VS_3Z: SMAC3Svs3ZAdapter,
    GameId.SMAC_3S_VS_4Z: SMAC3Svs4ZAdapter,
    GameId.SMAC_3S_VS_5Z: SMAC3Svs5ZAdapter,
    GameId.SMAC_1C3S5Z: SMAC1C3S5ZAdapter,
    GameId.SMAC_2M_VS_1Z: SMAC2Mvs1ZAdapter,
    GameId.SMAC_CORRIDOR: SMACCorridorAdapter,
    GameId.SMAC_6H_VS_8Z: SMAC6Hvs8ZAdapter,
    GameId.SMAC_2S_VS_1SC: SMAC2Svs1SCAdapter,
    GameId.SMAC_SO_MANY_BANELING: SMACSoManyBanelingAdapter,
    GameId.SMAC_BANE_VS_BANE: SMACBaneVsBaneAdapter,
    GameId.SMAC_2C_VS_64ZG: SMAC2Cvs64ZGAdapter,
}

__all__ = [
    "SMACAdapter",
    "SMAC_ADAPTERS",
    "SMAC_BASE_ACTIONS",
    "SMAC_MAP_INFO",
    "SMAC3MAdapter",
    "SMAC8MAdapter",
    "SMAC25MAdapter",
    "SMAC2S3ZAdapter",
    "SMAC3S5ZAdapter",
    "SMAC5Mvs6MAdapter",
    "SMAC8Mvs9MAdapter",
    "SMAC10Mvs11MAdapter",
    "SMAC27Mvs30MAdapter",
    "SMACMMMAdapter",
    "SMACMMM2Adapter",
    "SMAC3S5Zvs3S6ZAdapter",
    "SMAC3Svs3ZAdapter",
    "SMAC3Svs4ZAdapter",
    "SMAC3Svs5ZAdapter",
    "SMAC1C3S5ZAdapter",
    "SMAC2Mvs1ZAdapter",
    "SMACCorridorAdapter",
    "SMAC6Hvs8ZAdapter",
    "SMAC2Svs1SCAdapter",
    "SMACSoManyBanelingAdapter",
    "SMACBaneVsBaneAdapter",
    "SMAC2Cvs64ZGAdapter",
]
