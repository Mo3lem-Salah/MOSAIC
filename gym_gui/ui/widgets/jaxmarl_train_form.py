"""JaxMARL worker training form.

Provides a configuration dialog for JaxMARL RL training runs with:
- Algorithm selection: MAPPO (centralized critic) or IPPO (independent)
- Sport and variant selection
- Core hyperparameters
- TensorBoard tracking
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_TRAIN_FORM_ERROR,
    LOG_UI_TRAIN_FORM_INFO,
    LOG_UI_TRAIN_FORM_TRACE,
    LOG_UI_TRAIN_FORM_WARNING,
)
from gym_gui.validations.validation_jaxmarl_worker_form import run_jaxmarl_dry_run

_LOGGER = logging.getLogger("gym_gui.ui.jaxmarl_train_form")
REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Schema loading (metadata/jaxmarl/<version>/schemas.json)
# ---------------------------------------------------------------------------

def _load_jaxmarl_schemas() -> Tuple[Dict[str, Any], Optional[str]]:
    """Load JaxMARL schema from metadata/jaxmarl/*/schemas.json.

    Sorts version subdirectories descending (e.g. 0.2.0 > 0.1.0) and picks
    the first parseable schemas.json. Falls back to metadata/jaxmarl/schemas.json
    if no version subdirectory contains one.

    Returns:
        (schema_dict, jaxmarl_version) — both may be empty/None if no schema
        is found. Callers must handle absent keys via fallbacks below.
    """
    schema_root = REPO_ROOT / "metadata" / "jaxmarl"
    if not schema_root.exists():
        return {}, None

    candidates: List[Tuple[str, Path]] = []
    for entry in schema_root.iterdir():
        if not entry.is_dir():
            continue
        schema_file = entry / "schemas.json"
        if schema_file.exists():
            candidates.append((entry.name, schema_file))

    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        fallback = schema_root / "schemas.json"
        if fallback.exists():
            candidates.append(("latest", fallback))

    for _, schema_file in candidates:
        try:
            data = json.loads(schema_file.read_text())
        except Exception as exc:
            _LOGGER.warning("Failed to parse %s: %s", schema_file, exc)
            continue
        return data, data.get("jaxmarl_version")

    return {}, None


_JAXMARL_SCHEMAS, _JAXMARL_SCHEMA_VERSION = _load_jaxmarl_schemas()


# ---------------------------------------------------------------------------
# Fallback tables (used when schemas.json is missing or malformed)
# ---------------------------------------------------------------------------

_FALLBACK_FAMILIES: Dict[str, Dict[str, Any]] = {
    "mosaic_multigrid": {
        "label": "MOSAIC MultiGrid (Sports)",
        "launcher": "native",
        "environments": [
            ("soccer", "Soccer"),
            ("af",     "American Football"),
            ("bb",     "Basketball"),
        ],
        "variants": [
            ("1v1",   "1 vs 1"),
            ("2v2",   "2 vs 2"),
            ("3v3",   "3 vs 3"),
            ("G-1v0", "Green solo (G-1v0)"),
            ("G-2v0", "Green 2-agent (G-2v0)"),
            ("G-3v0", "Green 3-agent (G-3v0)"),
            ("B-0v1", "Blue solo (B-0v1)"),
            ("B-0v2", "Blue 2-agent (B-0v2)"),
            ("B-0v3", "Blue 3-agent (B-0v3)"),
        ],
        "supported_algorithms": ["mappo", "ippo"],
    },
    "socialjax": {
        "label": "SocialJax (Social Dilemmas)",
        "launcher": "native",
        "environments": [
            ("cleanup",     "Cleanup"),
            ("coins",       "Coin Game"),
            ("coop_mining", "Cooperative Mining"),
        ],
        "variants": [],
        "supported_algorithms": ["mappo", "ippo", "mat", "commnet", "coma"],
    },
    "mpe": {
        "label": "MPE (Multi-Agent Particle)",
        "launcher": "upstream",
        "environments": [
            ("MPE_simple_v3",                  "Simple"),
            ("MPE_simple_tag_v3",              "Simple Tag"),
            ("MPE_simple_spread_v3",           "Simple Spread"),
            ("MPE_simple_speaker_listener_v4", "Simple Speaker-Listener"),
        ],
        "variants": [],
        "supported_algorithms": ["mappo", "ippo"],
    },
    "mabrax": {
        "label": "MABrax (Multi-Agent MuJoCo Brax)",
        "launcher": "upstream",
        "environments": [
            ("ant_4x2",         "Ant (4x2)"),
            ("halfcheetah_6x1", "HalfCheetah (6x1)"),
            ("hopper_3x1",      "Hopper (3x1)"),
            ("humanoid_9|8",    "Humanoid (9|8)"),
            ("walker2d_2x3",    "Walker2D (2x3)"),
        ],
        "variants": [],
        "supported_algorithms": ["ippo"],
    },
    "overcooked": {
        "label": "Overcooked (Cooperative Cooking)",
        "launcher": "upstream",
        "environments": [
            ("overcooked",    "Overcooked v1"),
            ("overcooked_v2", "Overcooked v2"),
        ],
        "variants": [],
        "supported_algorithms": ["ippo"],
    },
    "hanabi": {
        "label": "Hanabi (Cooperative Card Game)",
        "launcher": "upstream",
        "environments": [
            ("hanabi", "Hanabi"),
        ],
        "variants": [],
        "supported_algorithms": ["mappo", "ippo"],
    },
}

_FALLBACK_ALGORITHMS_DICT: Dict[str, str] = {
    "mappo":   "MAPPO — global centralized critic",
    "ippo":    "IPPO  — independent per-agent critics",
    "mat":     "MAT — Multi-Agent Transformer",
    "commnet": "CommNet — mean-pooled hidden-state communication",
    "coma":    "COMA — counterfactual multi-agent policy gradient",
}

# Algorithm math hyperparameters (rate, coefs, clip, epochs, minibatches, hidden dim)
_FALLBACK_ALGORITHM_HYPERPARAMETERS: List[Dict[str, Any]] = [
    {"name": "lr",            "type": "float", "default": 3e-4, "min": 1e-6,  "max": 0.1,  "step": 0.0001, "decimals": 6, "label": "Learning Rate"},
    {"name": "gamma",         "type": "float", "default": 0.99, "min": 0.8,   "max": 1.0,  "step": 0.01,   "decimals": 4, "label": "Gamma"},
    {"name": "gae_lambda",    "type": "float", "default": 0.95, "min": 0.8,   "max": 1.0,  "step": 0.01,   "decimals": 4, "label": "GAE Lambda"},
    {"name": "clip_eps",      "type": "float", "default": 0.2,  "min": 0.0,   "max": 0.5,  "step": 0.01,   "decimals": 3, "label": "Clip Eps"},
    {"name": "vf_coef",       "type": "float", "default": 0.5,  "min": 0.0,   "max": 2.0,  "step": 0.05,   "decimals": 3, "label": "VF Coef"},
    {"name": "ent_coef",      "type": "float", "default": 0.01, "min": 0.0,   "max": 0.5,  "step": 0.001,  "decimals": 4, "label": "Ent Coef"},
    {"name": "n_epochs",      "type": "int",   "default": 4,    "min": 1,     "max": 32,   "step": 1,      "label": "N Epochs"},
    {"name": "n_minibatches", "type": "int",   "default": 4,    "min": 1,     "max": 64,   "step": 1,      "label": "N Minibatches"},
    {"name": "hidden_dim",    "type": "int",   "default": 256,  "min": 32,    "max": 1024, "step": 64,     "label": "Hidden Dim"},
]

# Training-loop / infrastructure parameters (steps, envs, seed, view, logging)
_FALLBACK_TRAINING_PARAMETERS: List[Dict[str, Any]] = [
    {"name": "total_updates", "type": "int", "default": 2000, "min": 100,  "max": 100000, "step": 100, "label": "Total Updates"},
    {"name": "n_envs",        "type": "int", "default": 512,  "min": 1,    "max": 8192,   "step": 64,  "label": "N Envs"},
    {"name": "n_steps",       "type": "int", "default": 128,  "min": 8,    "max": 1024,   "step": 16,  "label": "N Steps"},
    {"name": "seed",          "type": "int", "default": 1,    "min": 0,    "max": 9999,   "step": 1,   "label": "Seed"},
    {"name": "view_size",     "type": "int", "default": 7,    "min": 3,    "max": 15,     "step": 2,   "label": "View Size"},
    {"name": "log_every",     "type": "int", "default": 50,   "min": 1,    "max": 1000,   "step": 10,  "label": "Log Every"},
    {"name": "save_every",    "type": "int", "default": 500,  "min": 10,   "max": 10000,  "step": 100, "label": "Save Every"},
]


# ---------------------------------------------------------------------------
# Schema accessors (schema-or-fallback)
# ---------------------------------------------------------------------------

def _families_from_schema() -> Dict[str, Dict[str, Any]]:
    """Return {family_key: family_config_dict} from schema, or fallback.

    Each family_config has keys: label, environments (list of (key,label) tuples),
    variants (list, may be empty), supported_algorithms (list of algo_keys).
    """
    fams = _JAXMARL_SCHEMAS.get("environment_families")
    if not isinstance(fams, dict) or not fams:
        return _FALLBACK_FAMILIES

    result: Dict[str, Dict[str, Any]] = {}
    for fam_key, fam_cfg in fams.items():
        if not isinstance(fam_cfg, dict):
            continue
        envs_raw = fam_cfg.get("environments", [])
        variants_raw = fam_cfg.get("variants", [])
        envs = [
            (e["key"], e["label"])
            for e in envs_raw
            if isinstance(e, dict) and "key" in e and "label" in e
        ]
        variants = [
            (v["key"], v["label"])
            for v in variants_raw
            if isinstance(v, dict) and "key" in v and "label" in v
        ]
        result[fam_key] = {
            "label":                 fam_cfg.get("label", fam_key),
            "launcher":              fam_cfg.get("launcher", "native"),
            "environments":          envs,
            "variants":              variants,
            "supported_algorithms":  fam_cfg.get("supported_algorithms", []),
        }
    return result or _FALLBACK_FAMILIES


def _algorithms_from_schema() -> Dict[str, str]:
    """Return {algo_key: display_label} from schema, or fallback."""
    algos = _JAXMARL_SCHEMAS.get("algorithms")
    if isinstance(algos, dict) and algos:
        result = {}
        for key, cfg in algos.items():
            if isinstance(cfg, dict):
                result[key] = cfg.get("label", key)
            else:
                result[key] = str(cfg)
        if result:
            return result
    return _FALLBACK_ALGORITHMS_DICT


def _algorithm_hyperparameters_from_schema() -> List[Dict[str, Any]]:
    hp = _JAXMARL_SCHEMAS.get("algorithm_hyperparameters")
    if isinstance(hp, list) and hp:
        return hp
    return _FALLBACK_ALGORITHM_HYPERPARAMETERS


def _training_parameters_from_schema() -> List[Dict[str, Any]]:
    tp = _JAXMARL_SCHEMAS.get("training_parameters")
    if isinstance(tp, list) and tp:
        return tp
    return _FALLBACK_TRAINING_PARAMETERS


_FAMILIES = _families_from_schema()
_ALGORITHMS = _algorithms_from_schema()
_ALGORITHM_HYPERPARAMETERS = _algorithm_hyperparameters_from_schema()
_TRAINING_PARAMETERS = _training_parameters_from_schema()


def _default_family_key() -> str:
    return next(iter(_FAMILIES.keys())) if _FAMILIES else "mosaic_multigrid"


def _default_environment_key(family_key: str) -> str:
    envs = _FAMILIES.get(family_key, {}).get("environments", [])
    return envs[0][0] if envs else ""


def _default_algorithm_key(family_key: str) -> str:
    algos = _FAMILIES.get(family_key, {}).get("supported_algorithms", [])
    return algos[0] if algos else "mappo"


def _default_variant_key(family_key: str) -> str:
    variants = _FAMILIES.get(family_key, {}).get("variants", [])
    return variants[0][0] if variants else ""


def _generate_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"jaxmarl_{ts}_{short}"


# ---------------------------------------------------------------------------
# Form state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FormState:
    run_id: str = field(default_factory=_generate_run_id)
    algo: str = "mappo"
    env_family: str = "mosaic_multigrid"    # family: mosaic_multigrid | socialjax
    environment: str = "soccer"             # env key within the family (soccer | cleanup | ...)
    variant: str = "G-1v0"                  # variant key (only used by mosaic_multigrid; "" for socialjax)
    total_updates: int = 2000
    n_envs: int = 512
    n_steps: int = 128
    n_epochs: int = 4
    n_minibatches: int = 4
    hidden_dim: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    seed: int = 1
    view_size: int = 7
    log_every: int = 50
    save_every: int = 500
    track_tensorboard: bool = True
    notes: str = ""
    use_gpu: bool = True
    # Weights & Biases (Gap 12)
    wandb_enabled: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_api_key: str = ""
    wandb_http_proxy: str = ""
    wandb_https_proxy: str = ""


# ---------------------------------------------------------------------------
# Main form
# ---------------------------------------------------------------------------

class JaxMARLTrainForm(QtWidgets.QDialog, LogConstantMixin):
    """Training configuration dialog for JaxMARL workers."""

    config_ready = QtCore.Signal(dict)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        self._logger = _LOGGER
        self.setWindowTitle("JaxMARL Training Configuration")
        self.setMinimumWidth(900)
        self.setModal(True)
        self._last_config: Optional[Dict[str, Any]] = None
        self._init_ui()
        # Shrink dialog to its actual content, don't inherit the trainer window size.
        self.adjustSize()
        self.log_constant(
            LOG_UI_TRAIN_FORM_INFO,
            message="JaxMARLTrainForm initialized",
            extra={"worker_id": "jaxmarl_worker"},
        )

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _init_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(12, 10, 12, 10)
        # Align contents to the top so the layout doesn't distribute vertical
        # slack between widgets when the parent dialog is oversized.
        root.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # Header
        hdr = QtWidgets.QLabel("<b>JaxMARL Training</b> — GPU-accelerated JAX/Flax RL")
        hdr.setStyleSheet("font-size: 14px; color: #5c85d6;")
        root.addWidget(hdr)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("color: #ccc;")
        root.addWidget(sep)

        # Run Name
        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(QtWidgets.QLabel("Run name:"))
        self._run_id_edit = QtWidgets.QLineEdit()
        self._run_id_edit.setText(_generate_run_id())
        self._run_id_edit.setPlaceholderText("auto-generated")
        run_row.addWidget(self._run_id_edit, 1)
        root.addLayout(run_row)

        # Algorithm / Environment Family / Environment / Variant — cascading dropdowns.
        # Environment Family drives which environments, variants, and algorithms are valid.
        core_row = QtWidgets.QHBoxLayout()
        core_row.setSpacing(8)

        core_row.addWidget(QtWidgets.QLabel("Algorithm:"))
        self._algo_combo = QtWidgets.QComboBox()
        core_row.addWidget(self._algo_combo, 2)

        core_row.addSpacing(12)
        core_row.addWidget(QtWidgets.QLabel("Environment Family:"))
        self._family_combo = QtWidgets.QComboBox()
        for fam_key, fam_cfg in _FAMILIES.items():
            self._family_combo.addItem(fam_cfg.get("label", fam_key), fam_key)
        core_row.addWidget(self._family_combo, 2)

        root.addLayout(core_row)

        # Second row: Environment + Variant (variant hidden when family has none)
        env_row = QtWidgets.QHBoxLayout()
        env_row.setSpacing(8)
        env_row.addWidget(QtWidgets.QLabel("Environment:"))
        self._environment_combo = QtWidgets.QComboBox()
        env_row.addWidget(self._environment_combo, 2)

        env_row.addSpacing(12)
        self._variant_label = QtWidgets.QLabel("Variant:")
        env_row.addWidget(self._variant_label)
        self._variant_combo = QtWidgets.QComboBox()
        env_row.addWidget(self._variant_combo, 2)
        env_row.addStretch()
        root.addLayout(env_row)

        # Populate initial state (also sets algorithms + variants for default family)
        self._populate_algorithms()
        self._populate_environments()
        self._populate_variants()

        # Algorithm Hyper-parameters + Training Parameters (two separate groups)
        def _spin_int(lo, hi, val, step=1):
            w = QtWidgets.QSpinBox()
            w.setRange(int(lo), int(hi))
            w.setValue(int(val))
            w.setSingleStep(int(step))
            return w

        def _spin_dbl(lo, hi, val, decimals=6, step=0.0001):
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(float(lo), float(hi))
            w.setValue(float(val))
            w.setDecimals(int(decimals))
            w.setSingleStep(float(step))
            return w

        def _build_widget(spec: Dict[str, Any]) -> Optional[QtWidgets.QWidget]:
            typ = spec.get("type", "int")
            lo, hi = spec.get("min", 0), spec.get("max", 100)
            default, step = spec.get("default", 0), spec.get("step", 1)
            if typ == "int":
                return _spin_int(lo, hi, default, step)
            if typ == "float":
                return _spin_dbl(lo, hi, default, spec.get("decimals", 6), step)
            return None

        self._hp_widgets: Dict[str, QtWidgets.QWidget] = {}
        skipped: List[str] = []

        def _add_specs_to_grid(specs: List[Dict[str, Any]], grid: QtWidgets.QGridLayout, cols: int = 4) -> None:
            idx = 0
            for spec in specs:
                name = spec.get("name")
                label = spec.get("label", name)
                if not name:
                    skipped.append(str(spec))
                    continue
                widget = _build_widget(spec)
                if widget is None:
                    skipped.append(name)
                    continue
                help_text = spec.get("help")
                if help_text:
                    widget.setToolTip(help_text)
                r, c = divmod(idx, cols)
                cell = QtWidgets.QWidget()
                cl = QtWidgets.QVBoxLayout(cell)
                cl.setContentsMargins(0, 0, 0, 0)
                cl.setSpacing(2)
                lbl = QtWidgets.QLabel(label)
                lbl.setStyleSheet("font-size: 11px; font-weight: 600;")
                if help_text:
                    lbl.setToolTip(help_text)
                cl.addWidget(lbl)
                cl.addWidget(widget)
                grid.addWidget(cell, r, c)
                self._hp_widgets[name] = widget
                idx += 1

        algo_hp_grp = QtWidgets.QGroupBox("Algorithm Hyper-parameters")
        algo_hp_grid = QtWidgets.QGridLayout(algo_hp_grp)
        algo_hp_grid.setSpacing(6)
        _add_specs_to_grid(_ALGORITHM_HYPERPARAMETERS, algo_hp_grid)
        root.addWidget(algo_hp_grp)

        train_grp = QtWidgets.QGroupBox("Training Parameters")
        train_grid = QtWidgets.QGridLayout(train_grp)
        train_grid.setSpacing(6)
        _add_specs_to_grid(_TRAINING_PARAMETERS, train_grid)
        root.addWidget(train_grp)

        if skipped:
            self.log_constant(
                LOG_UI_TRAIN_FORM_WARNING,
                message=f"Skipped {len(skipped)} malformed hyperparameter specs",
                extra={"worker_id": "jaxmarl_worker", "skipped": ",".join(skipped)},
            )

        # Tracking — inline compact row (no GroupBox)
        tracking_row = QtWidgets.QHBoxLayout()
        tracking_row.setSpacing(16)
        tracking_lbl = QtWidgets.QLabel("<b>Tracking:</b>")
        tracking_row.addWidget(tracking_lbl)
        self._tb_check = QtWidgets.QCheckBox("TensorBoard")
        self._tb_check.setChecked(True)
        tracking_row.addWidget(self._tb_check)
        self._gpu_check = QtWidgets.QCheckBox("Use GPU (if available)")
        self._gpu_check.setChecked(True)
        tracking_row.addWidget(self._gpu_check)
        tracking_row.addStretch()
        root.addLayout(tracking_row)

        # Weights & Biases — full parity with xuance/cleanrl (Gap 12).
        # Fields are enabled/disabled by the master checkbox.
        wandb_head = QtWidgets.QHBoxLayout()
        wandb_head.setSpacing(16)
        wandb_head.addWidget(QtWidgets.QLabel("<b>Weights &amp; Biases:</b>"))
        self._wandb_enable_check = QtWidgets.QCheckBox("Enable WandB")
        wandb_head.addWidget(self._wandb_enable_check)
        wandb_head.addStretch()
        root.addLayout(wandb_head)

        wandb_grid = QtWidgets.QGridLayout()
        wandb_grid.setSpacing(6)
        wandb_grid.setContentsMargins(0, 0, 0, 0)

        def _make_wandb_edit(placeholder: str, password: bool = False) -> QtWidgets.QLineEdit:
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText(placeholder)
            if password:
                edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
            return edit

        self._wandb_project_edit    = _make_wandb_edit("jaxmarl-training")
        self._wandb_entity_edit     = _make_wandb_edit("your-team-or-user")
        self._wandb_run_label_edit  = _make_wandb_edit("optional; defaults to Run name above")
        self._wandb_api_key_edit    = _make_wandb_edit("or set WANDB_API_KEY externally", password=True)
        self._wandb_http_proxy_edit = _make_wandb_edit("e.g. http://127.0.0.1:7890")
        self._wandb_https_proxy_edit = _make_wandb_edit("e.g. http://127.0.0.1:7890")

        wandb_grid.addWidget(QtWidgets.QLabel("Project:"),     0, 0)
        wandb_grid.addWidget(self._wandb_project_edit,         0, 1)
        wandb_grid.addWidget(QtWidgets.QLabel("Entity:"),      0, 2)
        wandb_grid.addWidget(self._wandb_entity_edit,          0, 3)
        wandb_grid.addWidget(QtWidgets.QLabel("WandB run label:"), 1, 0)
        wandb_grid.addWidget(self._wandb_run_label_edit,           1, 1)
        wandb_grid.addWidget(QtWidgets.QLabel("API key:"),     1, 2)
        wandb_grid.addWidget(self._wandb_api_key_edit,         1, 3)
        wandb_grid.addWidget(QtWidgets.QLabel("HTTP proxy:"),  2, 0)
        wandb_grid.addWidget(self._wandb_http_proxy_edit,      2, 1)
        wandb_grid.addWidget(QtWidgets.QLabel("HTTPS proxy:"), 2, 2)
        wandb_grid.addWidget(self._wandb_https_proxy_edit,     2, 3)
        root.addLayout(wandb_grid)

        # Gate the wandb fields on the enable checkbox.
        self._wandb_field_widgets = [
            self._wandb_project_edit,
            self._wandb_entity_edit,
            self._wandb_run_label_edit,
            self._wandb_api_key_edit,
            self._wandb_http_proxy_edit,
            self._wandb_https_proxy_edit,
        ]
        self._wandb_enable_check.toggled.connect(self._on_wandb_toggled)
        self._on_wandb_toggled(False)  # start disabled

        # Notes — QPlainTextEdit so validation output can append cleanly.
        # Size policy fixed vertically so it doesn't grab layout stretch.
        notes_lbl = QtWidgets.QLabel("<b>Notes</b>")
        root.addWidget(notes_lbl)
        self._notes_edit = QtWidgets.QPlainTextEdit()
        self._notes_edit.setPlaceholderText(
            "Optional run notes. Dry-run validation output will be appended here."
        )
        self._notes_edit.setFixedHeight(110)
        self._notes_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        root.addWidget(self._notes_edit)

        # Absorb all remaining vertical slack here so nothing above stretches.
        root.addStretch(1)

        # Validation status + buttons (pinned to bottom because of the stretch above)
        self._validation_status_label = QtWidgets.QLabel(
            "Dry-run validation has not been executed yet."
        )
        self._validation_status_label.setStyleSheet("color: #666666;")
        root.addWidget(self._validation_status_label)
        self._last_validation_output: str = ""

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        self._validate_btn = QtWidgets.QPushButton("Validate")
        self._validate_btn.setToolTip(
            "Run jaxmarl_worker.cli --dry-run to validate this configuration "
            "without launching training."
        )
        self._validate_btn.clicked.connect(self._on_validate_clicked)
        btn_row.addWidget(self._validate_btn)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        ok_btn = btn_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if ok_btn:
            ok_btn.setText("Start Training")
        btn_row.addWidget(btn_box)
        root.addLayout(btn_row)

        # Connections — family drives everything else
        self._family_combo.currentIndexChanged.connect(self._on_family_changed)

    # -----------------------------------------------------------------------
    # Cascading dropdown population
    # -----------------------------------------------------------------------

    def _current_family_key(self) -> str:
        return self._family_combo.currentData() or _default_family_key()

    def _current_family_cfg(self) -> Dict[str, Any]:
        return _FAMILIES.get(self._current_family_key(), {})

    def _populate_algorithms(self) -> None:
        """Populate algorithm dropdown with algos supported by the current family."""
        self._algo_combo.clear()
        supported = self._current_family_cfg().get("supported_algorithms", [])
        for algo_key in supported:
            label = _ALGORITHMS.get(algo_key, algo_key)
            self._algo_combo.addItem(label, algo_key)

    def _populate_environments(self) -> None:
        """Populate environment dropdown with envs from the current family."""
        self._environment_combo.clear()
        envs = self._current_family_cfg().get("environments", [])
        for key, label in envs:
            self._environment_combo.addItem(label, key)

    def _populate_variants(self) -> None:
        """Populate variant dropdown, hide the row if family has no variants."""
        self._variant_combo.clear()
        variants = self._current_family_cfg().get("variants", [])
        for key, label in variants:
            self._variant_combo.addItem(label, key)
        has_variants = len(variants) > 0
        self._variant_combo.setVisible(has_variants)
        self._variant_label.setVisible(has_variants)

    def _on_family_changed(self) -> None:
        """Cascade: family change → repopulate algorithms, environments, variants."""
        self._populate_algorithms()
        self._populate_environments()
        self._populate_variants()
        self.adjustSize()

    # -----------------------------------------------------------------------
    # State collection
    # -----------------------------------------------------------------------

    def _collect_state(self) -> _FormState:
        family_key = self._family_combo.currentData() or _default_family_key()
        return _FormState(
            run_id            = self._run_id_edit.text().strip() or _generate_run_id(),
            algo              = self._algo_combo.currentData() or _default_algorithm_key(family_key),
            env_family        = family_key,
            environment       = self._environment_combo.currentData() or _default_environment_key(family_key),
            variant           = self._variant_combo.currentData() or "",
            total_updates     = self._hp_widgets["total_updates"].value(),
            n_envs            = self._hp_widgets["n_envs"].value(),
            n_steps           = self._hp_widgets["n_steps"].value(),
            n_epochs          = self._hp_widgets["n_epochs"].value(),
            n_minibatches     = self._hp_widgets["n_minibatches"].value(),
            hidden_dim        = self._hp_widgets["hidden_dim"].value(),
            lr                = self._hp_widgets["lr"].value(),
            gamma             = self._hp_widgets["gamma"].value(),
            gae_lambda        = self._hp_widgets["gae_lambda"].value(),
            clip_eps          = self._hp_widgets["clip_eps"].value(),
            vf_coef           = self._hp_widgets["vf_coef"].value(),
            ent_coef          = self._hp_widgets["ent_coef"].value(),
            seed              = self._hp_widgets["seed"].value(),
            view_size         = self._hp_widgets["view_size"].value(),
            log_every         = self._hp_widgets["log_every"].value(),
            save_every        = self._hp_widgets["save_every"].value(),
            track_tensorboard = self._tb_check.isChecked(),
            use_gpu           = self._gpu_check.isChecked(),
            notes             = self._notes_edit.toPlainText().strip(),
            wandb_enabled     = self._wandb_enable_check.isChecked(),
            wandb_project     = self._wandb_project_edit.text().strip(),
            wandb_entity      = self._wandb_entity_edit.text().strip(),
            wandb_run_name    = self._wandb_run_label_edit.text().strip(),
            wandb_api_key     = self._wandb_api_key_edit.text().strip(),
            wandb_http_proxy  = self._wandb_http_proxy_edit.text().strip(),
            wandb_https_proxy = self._wandb_https_proxy_edit.text().strip(),
        )

    def _on_wandb_toggled(self, checked: bool) -> None:
        """Enable/disable the wandb text fields based on the master checkbox."""
        for w in getattr(self, "_wandb_field_widgets", ()):
            w.setEnabled(checked)

    # -----------------------------------------------------------------------
    # Config building
    # -----------------------------------------------------------------------

    def _build_config(self, state: _FormState) -> Dict[str, Any]:
        run_id  = state.run_id
        run_dir = str(Path("var/trainer/runs") / run_id)

        worker_config: Dict[str, Any] = {
            "run_id":         run_id,
            "algo":           state.algo,
            "env_family":     state.env_family,     # family: mosaic_multigrid | socialjax
            "environment":    state.environment,    # env within family: soccer | cleanup | ...
            "variant":        state.variant,        # only used by mosaic_multigrid ("" for socialjax)
            "run_dir":        run_dir,
            "total_updates":  state.total_updates,
            "n_envs":         state.n_envs,
            "n_steps":        state.n_steps,
            "n_epochs":       state.n_epochs,
            "n_minibatches":  state.n_minibatches,
            "hidden_dim":     state.hidden_dim,
            "lr":             state.lr,
            "gamma":          state.gamma,
            "gae_lambda":     state.gae_lambda,
            "clip_eps":       state.clip_eps,
            "vf_coef":        state.vf_coef,
            "ent_coef":       state.ent_coef,
            "seed":           state.seed,
            "view_size":      state.view_size,
            "log_every":      state.log_every,
            "save_every":     state.save_every,
            "tensorboard":    state.track_tensorboard,
        }

        tb_relpath: Optional[str] = None
        if state.track_tensorboard:
            tb_relpath = f"{run_dir}/tensorboard"

        environment: Dict[str, str] = {
            "JAXMARL_RUN_ID": run_id,
        }
        if state.track_tensorboard:
            environment["JAXMARL_TENSORBOARD_DIR"] = tb_relpath or ""
        if state.use_gpu:
            # Let JAX pick the GPU — don't restrict CUDA_VISIBLE_DEVICES here
            pass

        # WandB env vars (consumed by jaxmarl_worker.wandb_helper). Only set
        # when explicitly enabled; the helper treats missing keys as no-op.
        if state.wandb_enabled:
            environment["JAXMARL_WANDB_ENABLED"] = "1"
            if state.wandb_project:
                environment["JAXMARL_WANDB_PROJECT"] = state.wandb_project
            if state.wandb_entity:
                environment["JAXMARL_WANDB_ENTITY"] = state.wandb_entity
            if state.wandb_run_name:
                environment["JAXMARL_WANDB_RUN_NAME"] = state.wandb_run_name
            if state.wandb_api_key:
                environment["WANDB_API_KEY"] = state.wandb_api_key
            if state.wandb_http_proxy:
                environment["HTTP_PROXY"] = state.wandb_http_proxy
            if state.wandb_https_proxy:
                environment["HTTPS_PROXY"] = state.wandb_https_proxy

        metadata: Dict[str, Any] = {
            "ui": {
                "worker_id":        "jaxmarl_worker",
                "algo":             state.algo,
                "env_family":       state.env_family,
                "environment":      state.environment,
                "variant":          state.variant,
                "track_tensorboard": state.track_tensorboard,
            },
            "worker": {
                "worker_id": "jaxmarl_worker",
                "module":    "jaxmarl_worker.cli",
                "use_grpc":  False,
                "config":    worker_config,
            },
        }

        config: Dict[str, Any] = {
            "run_name":   run_id,
            "entry_point": sys.executable,
            "arguments":  ["-m", "jaxmarl_worker.cli"],
            "environment": environment,
            "resources": {
                "cpus": 4,
                "memory_mb": 4096,
                "gpus": {
                    "requested": 1 if state.use_gpu else 0,
                    "mandatory": False,
                },
            },
            "metadata": metadata,
            "artifacts": {
                "tensorboard": {
                    "enabled":       state.track_tensorboard,
                    "relative_path": tb_relpath,
                },
                "notes": state.notes,
            },
        }
        return config

    # -----------------------------------------------------------------------
    # Accept handler
    # -----------------------------------------------------------------------

    def _on_accept(self) -> None:
        try:
            state = self._collect_state()
            # Refuse to launch upstream-only families from this form (no native
            # scan wrapper exists yet). Point users to Script Experiments.
            fam_cfg = _FAMILIES.get(state.env_family, {})
            if fam_cfg.get("launcher") == "upstream":
                self.log_constant(
                    LOG_UI_TRAIN_FORM_WARNING,
                    message="Refused Start Training: upstream family not integrated",
                    extra={"worker_id": "jaxmarl_worker", "env_family": state.env_family},
                )
                QtWidgets.QMessageBox.information(
                    self,
                    "Use Script Experiments",
                    (
                        f"'{fam_cfg.get('label', state.env_family)}' is an upstream "
                        f"JaxMARL family that does not yet have a MOSAIC-native scan "
                        f"wrapper. Please use the Script Experiments tab and launch "
                        f"the corresponding upstream baseline:\n\n"
                        f"  python -m JaxMARL.baselines.{state.algo.upper()}."
                        f"<baseline_for_{state.env_family}>\n\n"
                        f"(browse under JaxMARL/baselines/)"
                    ),
                )
                return
            if not state.run_id:
                self.log_constant(
                    LOG_UI_TRAIN_FORM_WARNING,
                    message="Empty run_id, regenerating",
                    extra={"worker_id": "jaxmarl_worker"},
                )
            config = self._build_config(state)
            self._last_config = config
            self.log_constant(
                LOG_UI_TRAIN_FORM_INFO,
                message="JaxMARL training config built",
                extra={
                    "worker_id":   "jaxmarl_worker",
                    "run_id":      state.run_id,
                    "algo":        state.algo,
                    "env_family":  state.env_family,
                    "environment": state.environment,
                    "variant":     state.variant,
                    "use_gpu":     state.use_gpu,
                },
            )
            self.config_ready.emit(config)
            self.accept()
        except Exception as exc:
            self.log_constant(
                LOG_UI_TRAIN_FORM_ERROR,
                message=f"Failed to build JaxMARL training config: {exc}",
                extra={"worker_id": "jaxmarl_worker"},
                exc_info=exc,
            )
            raise

    def get_config(self) -> Optional[Dict[str, Any]]:
        """Return the generated training configuration."""
        if self._last_config is not None:
            return copy.deepcopy(self._last_config)
        state = self._collect_state()
        return self._build_config(state)

    # -----------------------------------------------------------------------
    # Validation (dry-run)
    # -----------------------------------------------------------------------

    def _on_validate_clicked(self) -> None:
        """Handle Validate button: run dry-run without persisting config."""
        try:
            state = self._collect_state()
        except Exception as exc:
            self.log_constant(
                LOG_UI_TRAIN_FORM_WARNING,
                message=f"Cannot collect form state for validation: {exc}",
                extra={"worker_id": "jaxmarl_worker"},
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot Validate",
                f"Failed to collect form state: {exc}",
            )
            return
        self._run_validation(state, persist_config=False)

    def _run_validation(self, state: _FormState, *, persist_config: bool) -> bool:
        """Run dry-run validation via subprocess.

        Args:
            state: Current form state.
            persist_config: If True, cache the config into ``_last_config`` on success.

        Returns:
            True if the dry-run subprocess exited 0.
        """
        config = self._build_config(state)
        worker_config = config.get("metadata", {}).get("worker", {}).get("config", {})

        self._validation_status_label.setText("Running JaxMARL dry-run validation...")
        self._validation_status_label.setStyleSheet("color: #1565c0;")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            success, output = run_jaxmarl_dry_run(worker_config)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._set_validation_result(success, output)
        self._append_validation_notes(success, output)

        if persist_config and success:
            self._last_config = copy.deepcopy(config)
        elif not persist_config:
            self._last_config = None

        return success

    def _set_validation_result(self, success: bool, output: str) -> None:
        """Update the status label and emit a structured log event."""
        self._last_validation_output = output or ""
        if success:
            self._validation_status_label.setText("Dry-run validation succeeded.")
            self._validation_status_label.setStyleSheet("color: #2e7d32;")
            self.log_constant(
                LOG_UI_TRAIN_FORM_INFO,
                message="JaxMARL dry-run validation succeeded",
                extra={"worker_id": "jaxmarl_worker"},
            )
        else:
            self._validation_status_label.setText(
                "Dry-run validation failed. Check the details in Notes."
            )
            self._validation_status_label.setStyleSheet("color: #c62828;")
            snippet = (output or "").strip()
            self.log_constant(
                LOG_UI_TRAIN_FORM_ERROR,
                message="JaxMARL dry-run validation failed",
                extra={"worker_id": "jaxmarl_worker", "output": snippet[:1000]},
            )

    def _append_validation_notes(self, success: bool, output: str) -> None:
        """Append a timestamped dry-run result to the Notes field."""
        status = "SUCCESS" if success else "FAILED"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        details = output.strip() if output else (
            "Dry-run completed." if success else "Dry-run failed without output."
        )
        entry = (
            f"[Dry-Run {status} - {timestamp}]\n"
            f"{details}\n"
            f"{'-' * 40}\n"
        )
        self._notes_edit.appendPlainText(entry)


__all__ = ["JaxMARLTrainForm"]


# Register with form factory
try:
    from gym_gui.ui.forms.factory import get_worker_form_factory

    _factory = get_worker_form_factory()
    if not _factory.has_train_form("jaxmarl_worker"):
        _factory.register_train_form(
            "jaxmarl_worker",
            lambda parent=None, **kwargs: JaxMARLTrainForm(parent=parent, **kwargs),
        )
except ImportError:
    pass
