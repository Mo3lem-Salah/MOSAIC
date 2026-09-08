"""Configuration widgets for SMAC / SMACv2 multi-agent environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from PyQt6 import QtWidgets

from gym_gui.core.ui.game_config.game_configs import SMACConfig
from gym_gui.core.enums import GameId

# SMAC v1 hand-designed maps
SMAC_GAME_IDS: tuple[GameId, ...] = (
    GameId.SMAC_3M,
    GameId.SMAC_8M,
    GameId.SMAC_25M,
    GameId.SMAC_2S3Z,
    GameId.SMAC_3S5Z,
    GameId.SMAC_5M_VS_6M,
    GameId.SMAC_8M_VS_9M,
    GameId.SMAC_10M_VS_11M,
    GameId.SMAC_27M_VS_30M,
    GameId.SMAC_MMM,
    GameId.SMAC_MMM2,
    GameId.SMAC_3S5Z_VS_3S6Z,
    GameId.SMAC_3S_VS_3Z,
    GameId.SMAC_3S_VS_4Z,
    GameId.SMAC_3S_VS_5Z,
    GameId.SMAC_1C3S5Z,
    GameId.SMAC_2M_VS_1Z,
    GameId.SMAC_CORRIDOR,
    GameId.SMAC_6H_VS_8Z,
    GameId.SMAC_2S_VS_1SC,
    GameId.SMAC_SO_MANY_BANELING,
    GameId.SMAC_BANE_VS_BANE,
    GameId.SMAC_2C_VS_64ZG,
)

# SMACv2 procedural maps
SMACV2_GAME_IDS: tuple[GameId, ...] = (
    GameId.SMACV2_TERRAN,
    GameId.SMACV2_PROTOSS,
    GameId.SMACV2_ZERG,
    # EPyMARL scenario presets (n_units vs n_enemies overrides)
    GameId.SMACV2_TERRAN_5V5,
    GameId.SMACV2_TERRAN_10V10,
    GameId.SMACV2_TERRAN_20V20,
    GameId.SMACV2_TERRAN_10V11,
    GameId.SMACV2_TERRAN_20V23,
    GameId.SMACV2_PROTOSS_5V5,
    GameId.SMACV2_PROTOSS_10V10,
    GameId.SMACV2_PROTOSS_20V20,
    GameId.SMACV2_PROTOSS_10V11,
    GameId.SMACV2_PROTOSS_20V23,
    GameId.SMACV2_ZERG_5V5,
    GameId.SMACV2_ZERG_10V10,
    GameId.SMACV2_ZERG_20V20,
    GameId.SMACV2_ZERG_10V11,
    GameId.SMACV2_ZERG_20V23,
)

# Combined for config panel dispatch
ALL_SMAC_GAME_IDS: tuple[GameId, ...] = SMAC_GAME_IDS + SMACV2_GAME_IDS

# Per-map metadata for info display: GameId -> (n_allies, ally_desc, difficulty)
_MAP_INFO: dict[GameId, tuple[int, str, str]] = {
    GameId.SMAC_3M: (3, "3 Marines", "Easy"),
    GameId.SMAC_8M: (8, "8 Marines", "Easy"),
    GameId.SMAC_25M: (25, "25 Marines", "Easy"),
    GameId.SMAC_2S3Z: (5, "2 Stalkers + 3 Zealots", "Easy"),
    GameId.SMAC_3S5Z: (8, "3 Stalkers + 5 Zealots", "Easy"),
    GameId.SMAC_5M_VS_6M: (5, "5 Marines (vs 6)", "Hard"),
    GameId.SMAC_8M_VS_9M: (8, "8 Marines (vs 9)", "Hard"),
    GameId.SMAC_10M_VS_11M: (10, "10 Marines (vs 11)", "Hard"),
    GameId.SMAC_27M_VS_30M: (27, "27 Marines (vs 30)", "Super Hard"),
    GameId.SMAC_MMM: (10, "1 Medivac + 2 Marauders + 7 Marines", "Hard"),
    GameId.SMAC_MMM2: (10, "1 Medivac + 2 Marauders + 7 Marines", "Super Hard"),
    GameId.SMAC_3S5Z_VS_3S6Z: (8, "3 Stalkers + 5 Zealots (vs 3S+6Z)", "Super Hard"),
    GameId.SMAC_3S_VS_3Z: (3, "3 Stalkers (vs 3 Zealots)", "Easy"),
    GameId.SMAC_3S_VS_4Z: (3, "3 Stalkers (vs 4 Zealots)", "Hard"),
    GameId.SMAC_3S_VS_5Z: (3, "3 Stalkers (vs 5 Zealots)", "Super Hard"),
    GameId.SMAC_1C3S5Z: (9, "1 Colossus + 3 Stalkers + 5 Zealots", "Hard"),
    GameId.SMAC_2M_VS_1Z: (2, "2 Marines (vs 1 Zealot)", "Easy"),
    GameId.SMAC_CORRIDOR: (6, "6 Zealots (vs 24 Zerglings)", "Hard"),
    GameId.SMAC_6H_VS_8Z: (6, "6 Hydralisks (vs 8 Zealots)", "Super Hard"),
    GameId.SMAC_2S_VS_1SC: (2, "2 Stalkers (vs 1 Spine Crawler)", "Easy"),
    GameId.SMAC_SO_MANY_BANELING: (7, "7 Zealots (vs 32 Banelings)", "Hard"),
    GameId.SMAC_BANE_VS_BANE: (24, "24 Zerglings + Banelings", "Easy"),
    GameId.SMAC_2C_VS_64ZG: (2, "2 Colossi (vs 64 Zerglings)", "Super Hard"),
    GameId.SMACV2_TERRAN: (10, "Random Terran (procedural)", "Varies"),
    GameId.SMACV2_PROTOSS: (10, "Random Protoss (procedural)", "Varies"),
    GameId.SMACV2_ZERG: (10, "Random Zerg (procedural)", "Varies"),
    GameId.SMACV2_TERRAN_5V5: (5, "5 Random Terran (vs 5)", "Varies"),
    GameId.SMACV2_TERRAN_10V10: (10, "10 Random Terran (vs 10)", "Varies"),
    GameId.SMACV2_TERRAN_20V20: (20, "20 Random Terran (vs 20)", "Varies"),
    GameId.SMACV2_TERRAN_10V11: (10, "10 Random Terran (vs 11)", "Varies"),
    GameId.SMACV2_TERRAN_20V23: (20, "20 Random Terran (vs 23)", "Varies"),
    GameId.SMACV2_PROTOSS_5V5: (5, "5 Random Protoss (vs 5)", "Varies"),
    GameId.SMACV2_PROTOSS_10V10: (10, "10 Random Protoss (vs 10)", "Varies"),
    GameId.SMACV2_PROTOSS_20V20: (20, "20 Random Protoss (vs 20)", "Varies"),
    GameId.SMACV2_PROTOSS_10V11: (10, "10 Random Protoss (vs 11)", "Varies"),
    GameId.SMACV2_PROTOSS_20V23: (20, "20 Random Protoss (vs 23)", "Varies"),
    GameId.SMACV2_ZERG_5V5: (5, "5 Random Zerg (vs 5)", "Varies"),
    GameId.SMACV2_ZERG_10V10: (10, "10 Random Zerg (vs 10)", "Varies"),
    GameId.SMACV2_ZERG_20V20: (20, "20 Random Zerg (vs 20)", "Varies"),
    GameId.SMACV2_ZERG_10V11: (10, "10 Random Zerg (vs 11)", "Varies"),
    GameId.SMACV2_ZERG_20V23: (20, "20 Random Zerg (vs 23)", "Varies"),
}

# AI difficulty levels: display name -> SMAC difficulty string
DIFFICULTY_OPTIONS: dict[str, str] = {
    "1 - Very Easy": "1",
    "2 - Easy": "2",
    "3 - Medium": "3",
    "4 - Medium-Hard": "4",
    "5 - Hard": "5",
    "6 - Harder": "6",
    "7 - Very Hard (Standard)": "7",
    "8 - Cheat Vision": "8",
    "9 - Cheat Money": "9",
    "10 - Cheat Insane": "10",
}


@dataclass(slots=True)
class ControlCallbacks:
    """Bridge callbacks for propagating UI changes to session state."""

    on_change: Callable[[str, Any], None]


def build_smac_controls(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    game_id: GameId,
    overrides: Dict[str, Any],
    defaults: SMACConfig | None = None,
    callbacks: ControlCallbacks | None = None,
) -> None:
    """Populate SMAC/SMACv2-specific configuration widgets.

    Args:
        parent: Parent widget for controls.
        layout: Form layout to add controls to.
        game_id: The selected SMAC game ID.
        overrides: Dictionary to store user-modified values.
        defaults: Default configuration values.
        callbacks: Optional callbacks for value changes.
    """
    def emit(key: str, value: Any) -> None:
        overrides[key] = value
        if callbacks:
            callbacks.on_change(key, value)

    cfg = defaults if isinstance(defaults, SMACConfig) else SMACConfig()

    # -------- Map Info (read-only) --------
    info = _MAP_INFO.get(game_id, (0, "Unknown", "Unknown"))
    n_agents, composition, difficulty_tier = info

    map_info_label = QtWidgets.QLabel(
        f"<b>{composition}</b><br>"
        f"Agents: {n_agents} | Difficulty Tier: {difficulty_tier}",
        parent,
    )
    map_info_label.setWordWrap(True)
    layout.addRow("Map Info", map_info_label)

    # -------- AI Difficulty --------
    difficulty_combo = QtWidgets.QComboBox(parent)
    difficulty_combo.addItems(list(DIFFICULTY_OPTIONS.keys()))

    current_diff = overrides.get("difficulty", cfg.difficulty)
    current_diff_text = "7 - Very Hard (Standard)"
    for name, val in DIFFICULTY_OPTIONS.items():
        if val == str(current_diff):
            current_diff_text = name
            break
    difficulty_combo.setCurrentText(current_diff_text)

    def on_difficulty_changed(text: str) -> None:
        diff_val = DIFFICULTY_OPTIONS.get(text, "7")
        emit("difficulty", diff_val)

    difficulty_combo.currentTextChanged.connect(on_difficulty_changed)
    difficulty_combo.setToolTip(
        "SC2 built-in AI difficulty level.\n"
        "Level 7 ('Very Hard') is the standard benchmark setting.\n"
        "Levels 8-10 give the AI unfair advantages (vision/resources)."
    )
    layout.addRow("AI Difficulty", difficulty_combo)

    # -------- Reward Type --------
    reward_combo = QtWidgets.QComboBox(parent)
    reward_combo.addItems(["Shaped (default)", "Sparse (+1 win only)"])

    current_sparse = overrides.get("reward_sparse", cfg.reward_sparse)
    reward_combo.setCurrentIndex(1 if current_sparse else 0)

    def on_reward_changed(index: int) -> None:
        emit("reward_sparse", index == 1)

    reward_combo.currentIndexChanged.connect(on_reward_changed)
    reward_combo.setToolTip(
        "Shaped: per-step reward based on damage dealt/received + win bonus.\n"
        "Sparse: +1 for winning the battle, 0 otherwise."
    )
    layout.addRow("Reward Type", reward_combo)

    # -------- Renderer --------
    renderer_combo = QtWidgets.QComboBox(parent)
    _RENDERER_OPTIONS = [
        ("3D (GPU Rendered)", "3d"),
        ("Heatmap (Feature Layers)", "heatmap"),
        ("Classic (PyGame)", "classic"),
    ]
    renderer_combo.addItems([label for label, _ in _RENDERER_OPTIONS])

    current_renderer = overrides.get("renderer", cfg.renderer)
    _renderer_idx = next(
        (i for i, (_, val) in enumerate(_RENDERER_OPTIONS) if val == current_renderer),
        0,
    )
    renderer_combo.setCurrentIndex(_renderer_idx)

    def on_renderer_changed(index: int) -> None:
        emit("renderer", _RENDERER_OPTIONS[index][1] if index < len(_RENDERER_OPTIONS) else "3d")

    renderer_combo.currentIndexChanged.connect(on_renderer_changed)
    renderer_combo.setToolTip(
        "3D (GPU Rendered): Full SC2 engine 3D rendering via EGL (requires GPU).\n"
        "Heatmap: 2x2 panel view with terrain, health, unit type, and shield overlays.\n"
        "Classic: SMAC's built-in PyGame renderer (colored circles with health arcs)."
    )
    layout.addRow("Renderer", renderer_combo)

    # -------- Render Resolution --------
    render_resolution_spin = QtWidgets.QSpinBox(parent)
    render_resolution_spin.setRange(256, 2048)
    render_resolution_spin.setSingleStep(256)
    render_resolution_spin.setSuffix(" px")
    render_resolution_spin.setValue(int(overrides.get("render_resolution", cfg.render_resolution)))

    def on_render_resolution_changed(value: int) -> None:
        emit("render_resolution", value)

    render_resolution_spin.valueChanged.connect(on_render_resolution_changed)
    render_resolution_spin.setToolTip(
        "3D camera output resolution, in pixels (square frame).\n"
        "Higher = sharper image, more GPU/CPU cost per frame.\n"
        "Applied at environment launch -- reload the environment for changes\n"
        "to take effect. Only applies when Renderer is '3D (GPU Rendered)'.\n"
        "The minimap inset always renders at a fixed 256x256 regardless of this."
    )
    layout.addRow("Render Resolution", render_resolution_spin)

    # -------- Camera Auto-Center --------
    # Note: there is no adjustable camera zoom/FOV setting. SC2's 3D camera
    # field of view is fixed per-map by the engine itself (measured ~7-17
    # world units) and cannot be changed via any interface option --
    # InterfaceOptions.render.width was tried and measured to have zero
    # effect on 3D render output. Once running, pan the camera by clicking
    # and dragging on the video, which sends a real ActionRaw.camera_move.
    camera_center_cb = QtWidgets.QCheckBox("Auto-center camera on map each episode", parent)
    camera_center_cb.setChecked(bool(overrides.get("camera_auto_center", cfg.camera_auto_center)))

    def on_camera_center_changed(state: int) -> None:
        emit("camera_auto_center", state == 2)

    camera_center_cb.stateChanged.connect(on_camera_center_changed)
    camera_center_cb.setToolTip(
        "If checked (default): after each reset, the camera automatically moves\n"
        "to the map's fixed geometric center, since SC2's default camera position\n"
        "(world origin) usually misses the battle area entirely.\n"
        "If unchecked: use StarCraft II's default camera position.\n"
        "Once the game is running, click-and-drag on the video to pan the camera\n"
        "(a real ActionRaw.camera_move -- the field of view itself is fixed by\n"
        "the SC2 engine per-map and cannot be zoomed)."
    )
    layout.addRow("", camera_center_cb)

    # -------- Minimap Inset (full-map awareness) --------
    minimap_cb = QtWidgets.QCheckBox("Show full-map minimap inset", parent)
    minimap_cb.setChecked(bool(overrides.get("minimap_inset", cfg.minimap_inset)))

    def on_minimap_changed(state: int) -> None:
        emit("minimap_inset", state == 2)

    minimap_cb.stateChanged.connect(on_minimap_changed)
    minimap_cb.setToolTip(
        "If checked (default): overlay StarCraft II's always-full-map minimap\n"
        "as a picture-in-picture inset in the corner of the 3D view, so all four\n"
        "map edges stay visible even though the main camera's field of view is\n"
        "fixed and zoomed in. A red box on the inset marks the main camera's\n"
        "approximate current viewport.\n"
        "If unchecked: show only the unmodified full-frame 3D render."
    )
    layout.addRow("", minimap_cb)

    # -------- Unit Health Bars --------
    health_bars_cb = QtWidgets.QCheckBox("Show health/shield bars over units", parent)
    health_bars_cb.setChecked(bool(overrides.get("unit_health_bars", cfg.unit_health_bars)))

    def on_health_bars_changed(state: int) -> None:
        emit("unit_health_bars", state == 2)

    health_bars_cb.stateChanged.connect(on_health_bars_changed)
    health_bars_cb.setToolTip(
        "If checked (default): overlay a green (ally) or red (enemy) health bar,\n"
        "plus a blue shield bar when applicable, above each unit in the 3D view.\n"
        "Built from the same per-unit health/shield data already available via\n"
        "the raw interface, drawn using SC2's own debug-draw API so the engine\n"
        "places bars using its real camera projection -- they track each unit\n"
        "and the camera exactly, with no drift.\n"
        "If unchecked: show only the unmodified render (with minimap inset, if enabled)."
    )
    layout.addRow("", health_bars_cb)

    # -------- Observation: Own Health --------
    obs_health_cb = QtWidgets.QCheckBox("Include own health in observation", parent)
    obs_health_cb.setChecked(bool(overrides.get("obs_own_health", cfg.obs_own_health)))

    def on_obs_health_changed(state: int) -> None:
        emit("obs_own_health", state == 2)

    obs_health_cb.stateChanged.connect(on_obs_health_changed)
    obs_health_cb.setToolTip("Include the agent's own health value in its observation vector.")
    layout.addRow("", obs_health_cb)

    # -------- Observation: Pathing Grid --------
    obs_pathing_cb = QtWidgets.QCheckBox("Include pathing grid", parent)
    obs_pathing_cb.setChecked(bool(overrides.get("obs_pathing_grid", cfg.obs_pathing_grid)))

    def on_obs_pathing_changed(state: int) -> None:
        emit("obs_pathing_grid", state == 2)

    obs_pathing_cb.stateChanged.connect(on_obs_pathing_changed)
    obs_pathing_cb.setToolTip("Include terrain walkability grid in observations (increases obs size).")
    layout.addRow("", obs_pathing_cb)

    # -------- Observation: Terrain Height --------
    obs_terrain_cb = QtWidgets.QCheckBox("Include terrain height", parent)
    obs_terrain_cb.setChecked(bool(overrides.get("obs_terrain_height", cfg.obs_terrain_height)))

    def on_obs_terrain_changed(state: int) -> None:
        emit("obs_terrain_height", state == 2)

    obs_terrain_cb.stateChanged.connect(on_obs_terrain_changed)
    obs_terrain_cb.setToolTip("Include terrain height map in observations (increases obs size).")
    layout.addRow("", obs_terrain_cb)

    # -------- Episode Limit --------
    ep_limit_spin = QtWidgets.QSpinBox(parent)
    ep_limit_spin.setRange(0, 10000)
    ep_limit_spin.setSpecialValueText("Map Default")
    ep_limit_spin.setValue(int(overrides.get("episode_limit", 0) or 0))

    def on_ep_limit_changed(value: int) -> None:
        emit("episode_limit", value if value > 0 else None)

    ep_limit_spin.valueChanged.connect(on_ep_limit_changed)
    ep_limit_spin.setToolTip(
        "Maximum number of steps per episode.\n"
        "0 = use the map's default limit."
    )
    layout.addRow("Episode Limit", ep_limit_spin)

    # -------- Seed --------
    seed_spin = QtWidgets.QSpinBox(parent)
    seed_spin.setRange(-1, 999999)
    seed_spin.setSpecialValueText("Random")
    seed_spin.setValue(int(overrides.get("seed", -1) if overrides.get("seed") is not None else -1))

    def on_seed_changed(value: int) -> None:
        emit("seed", value if value >= 0 else None)

    seed_spin.valueChanged.connect(on_seed_changed)
    seed_spin.setToolTip("Random seed for environment. -1 = random each episode.")
    layout.addRow("Seed", seed_spin)

    # -------- SC2 Path --------
    sc2_path_edit = QtWidgets.QLineEdit(parent)
    sc2_path_edit.setPlaceholderText("auto: SC2PATH env var -> var/data/")
    current_sc2_path = overrides.get("sc2_path", cfg.sc2_path)
    if current_sc2_path:
        sc2_path_edit.setText(str(current_sc2_path))

    def on_sc2_path_changed(text: str) -> None:
        emit("sc2_path", text.strip() if text.strip() else None)

    sc2_path_edit.textChanged.connect(on_sc2_path_changed)
    sc2_path_edit.setToolTip(
        "Path to StarCraft II installation directory.\n"
        "Resolution order: this field -> SC2PATH env var -> var/data/\n"
        "Download from: https://github.com/Blizzard/s2client-proto#linux-packages"
    )
    layout.addRow("SC2 Path", sc2_path_edit)

    # -------- Info Label --------
    is_v2 = game_id in SMACV2_GAME_IDS
    version_info = (
        "SMACv2: Procedural unit generation -- team compositions change every episode."
        if is_v2
        else "SMAC v1: Fixed map with hand-designed unit compositions."
    )
    info_label = QtWidgets.QLabel(
        f"<i>{version_info}<br>"
        "All agents act simultaneously (parallel stepping).<br>"
        "Cooperative: shared team reward (CTDE paradigm).<br><br>"
        "<b>Requires:</b> StarCraft II binary installed (set SC2PATH env var).<br>"
        "<b>Renderer:</b> Built-in PyGame 2D top-down view.</i>",
        parent,
    )
    info_label.setWordWrap(True)
    layout.addRow("", info_label)
