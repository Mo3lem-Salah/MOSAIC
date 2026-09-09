"""Dialog for loading and evaluating a JaxMARL .npz policy checkpoint.

Routes to `python -m jaxmarl_worker.cli --interactive` via the trainer daemon,
which spawns the InteractiveRuntime (JSON line protocol for step-by-step eval).

Supports both MAPPO/IPPO actors and MAT (Multi-Agent Transformer) actors, with
architecture-specific hyperparameters (n_embd/n_head/n_block) shown only when
MAT is selected.
"""

from __future__ import annotations

import copy
import logging
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QFileDialog

from gym_gui.config.paths import VAR_EVALS_DIR, VAR_TRAINER_DIR
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_POLICY_FORM_ERROR,
    LOG_UI_POLICY_FORM_INFO,
)

_LOGGER = logging.getLogger("gym_gui.ui.jaxmarl_policy_form")


_SPORTS: List[Tuple[str, str]] = [
    ("soccer", "Soccer"),
    ("af",     "American Football"),
    ("bb",     "Basketball"),
]

_VARIANTS: List[Tuple[str, str]] = [
    ("1v1",   "1 vs 1"),
    ("2v2",   "2 vs 2"),
    ("3v3",   "3 vs 3"),
    ("G-1v0", "Green solo (G-1v0)"),
    ("G-2v0", "Green 2-agent (G-2v0)"),
    ("G-3v0", "Green 3-agent (G-3v0)"),
    ("B-0v1", "Blue solo (B-0v1)"),
    ("B-0v2", "Blue 2-agent (B-0v2)"),
    ("B-0v3", "Blue 3-agent (B-0v3)"),
]

# Includes MAT here, unlike the train form (which only trains mappo/ippo today).
_ALGORITHMS: List[Tuple[str, str]] = [
    ("mappo", "MAPPO (centralized critic)"),
    ("ippo",  "IPPO (independent critics)"),
    ("mat",   "MAT (Multi-Agent Transformer)"),
]


def _generate_run_id(checkpoint_stem: str = "policy") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    slug = checkpoint_stem.replace("_", "-")[:24] or "policy"
    return f"jaxmarl_eval_{slug}_{ts}_{short}"


@dataclass(frozen=True)
class _PolicyState:
    run_id: str
    checkpoint_path: str
    sport: str
    variant: str
    algorithm: str
    view_size: int
    num_agents: Optional[int]
    n_embd: int
    n_head: int
    n_block: int


class JaxMARLPolicyForm(QtWidgets.QDialog, LogConstantMixin):
    """Evaluate a JaxMARL .npz policy checkpoint interactively."""

    config_ready = QtCore.Signal(dict)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        self._logger = _LOGGER
        self.setWindowTitle("Load JaxMARL Policy")
        self.setMinimumWidth(720)
        self.setModal(True)
        self._last_config: Optional[Dict[str, Any]] = None
        self._init_ui()
        self.adjustSize()
        self.log_constant(
            LOG_UI_POLICY_FORM_INFO,
            message="JaxMARLPolicyForm initialized",
            extra={"worker_id": "jaxmarl_worker"},
        )

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _init_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(12, 10, 12, 10)
        root.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # Header
        hdr = QtWidgets.QLabel("<b>Load JaxMARL Policy</b> — interactive .npz checkpoint evaluation")
        hdr.setStyleSheet("font-size: 13px; color: #5c85d6;")
        root.addWidget(hdr)
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("color: #ccc;")
        root.addWidget(sep)

        # Checkpoint path + browse
        ckpt_row = QtWidgets.QHBoxLayout()
        ckpt_row.addWidget(QtWidgets.QLabel("Checkpoint (.npz):"))
        self._ckpt_edit = QtWidgets.QLineEdit()
        self._ckpt_edit.setPlaceholderText("Path to jaxmarl .npz policy checkpoint")
        ckpt_row.addWidget(self._ckpt_edit, 1)
        self._browse_btn = QtWidgets.QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._on_browse_ckpt)
        ckpt_row.addWidget(self._browse_btn)
        root.addLayout(ckpt_row)

        # Sport / Variant / Algorithm on one row (mirrors train form density)
        core_row = QtWidgets.QHBoxLayout()
        core_row.setSpacing(8)
        core_row.addWidget(QtWidgets.QLabel("Sport:"))
        self._sport_combo = QtWidgets.QComboBox()
        for key, label in _SPORTS:
            self._sport_combo.addItem(label, key)
        core_row.addWidget(self._sport_combo, 1)

        core_row.addSpacing(12)
        core_row.addWidget(QtWidgets.QLabel("Variant:"))
        self._variant_combo = QtWidgets.QComboBox()
        for key, label in _VARIANTS:
            self._variant_combo.addItem(label, key)
        core_row.addWidget(self._variant_combo, 1)

        core_row.addSpacing(12)
        core_row.addWidget(QtWidgets.QLabel("Algorithm:"))
        self._algo_combo = QtWidgets.QComboBox()
        for key, label in _ALGORITHMS:
            self._algo_combo.addItem(label, key)
        self._algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        core_row.addWidget(self._algo_combo, 2)
        root.addLayout(core_row)

        # Env config: view_size + num_agents override
        env_row = QtWidgets.QHBoxLayout()
        env_row.setSpacing(8)
        env_row.addWidget(QtWidgets.QLabel("View size:"))
        self._view_size_spin = QtWidgets.QSpinBox()
        self._view_size_spin.setRange(3, 15)
        self._view_size_spin.setSingleStep(2)
        self._view_size_spin.setValue(7)
        env_row.addWidget(self._view_size_spin)

        env_row.addSpacing(16)
        env_row.addWidget(QtWidgets.QLabel("Num agents (override):"))
        self._num_agents_spin = QtWidgets.QSpinBox()
        self._num_agents_spin.setRange(0, 32)
        self._num_agents_spin.setValue(0)
        self._num_agents_spin.setSpecialValueText("(auto)")
        self._num_agents_spin.setToolTip("0 = infer from env; else force this agent count.")
        env_row.addWidget(self._num_agents_spin)
        env_row.addStretch()
        root.addLayout(env_row)

        # MAT-specific hyperparams (initially hidden; toggled by _on_algo_changed)
        self._mat_group = QtWidgets.QGroupBox("MAT Actor Hyperparameters")
        mat_grid = QtWidgets.QGridLayout(self._mat_group)
        mat_grid.setSpacing(6)

        self._n_embd_spin = QtWidgets.QSpinBox()
        self._n_embd_spin.setRange(16, 2048)
        self._n_embd_spin.setSingleStep(16)
        self._n_embd_spin.setValue(128)

        self._n_head_spin = QtWidgets.QSpinBox()
        self._n_head_spin.setRange(1, 32)
        self._n_head_spin.setValue(4)

        self._n_block_spin = QtWidgets.QSpinBox()
        self._n_block_spin.setRange(1, 24)
        self._n_block_spin.setValue(2)

        mat_grid.addWidget(QtWidgets.QLabel("n_embd"),  0, 0)
        mat_grid.addWidget(self._n_embd_spin,           0, 1)
        mat_grid.addWidget(QtWidgets.QLabel("n_head"),  0, 2)
        mat_grid.addWidget(self._n_head_spin,           0, 3)
        mat_grid.addWidget(QtWidgets.QLabel("n_block"), 0, 4)
        mat_grid.addWidget(self._n_block_spin,          0, 5)

        root.addWidget(self._mat_group)
        self._mat_group.setVisible(False)  # MAT is not the default

        # Absorb vertical slack before the button row
        root.addStretch(1)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        ok_btn = btn_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if ok_btn:
            ok_btn.setText("Load & Evaluate")
        btn_row.addWidget(btn_box)
        root.addLayout(btn_row)

    def _on_algo_changed(self) -> None:
        is_mat = self._algo_combo.currentData() == "mat"
        self._mat_group.setVisible(is_mat)
        self.adjustSize()

    def _on_browse_ckpt(self) -> None:
        start_dir = str(VAR_TRAINER_DIR) if VAR_TRAINER_DIR.exists() else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select JaxMARL .npz Checkpoint",
            start_dir,
            "NPZ Checkpoints (*.npz);;All Files (*)",
        )
        if path:
            self._ckpt_edit.setText(path)
            # Infer algorithm from filename if possible.
            lower = path.lower()
            if "/mat" in lower or "_mat" in lower or "mat_" in lower:
                self._select_combo_value(self._algo_combo, "mat")
            elif "/ippo" in lower or "_ippo" in lower or "ippo_" in lower:
                self._select_combo_value(self._algo_combo, "ippo")
            elif "/mappo" in lower or "_mappo" in lower or "mappo_" in lower:
                self._select_combo_value(self._algo_combo, "mappo")

    @staticmethod
    def _select_combo_value(combo: QtWidgets.QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # -----------------------------------------------------------------------
    # State collection & config building
    # -----------------------------------------------------------------------

    def _collect_state(self) -> _PolicyState:
        ckpt = self._ckpt_edit.text().strip()
        stem = Path(ckpt).stem if ckpt else "policy"
        num_agents_raw = self._num_agents_spin.value()
        return _PolicyState(
            run_id          = _generate_run_id(stem),
            checkpoint_path = ckpt,
            sport           = self._sport_combo.currentData() or "soccer",
            variant         = self._variant_combo.currentData() or "1v1",
            algorithm       = self._algo_combo.currentData() or "mappo",
            view_size       = self._view_size_spin.value(),
            num_agents      = None if num_agents_raw == 0 else num_agents_raw,
            n_embd          = self._n_embd_spin.value(),
            n_head          = self._n_head_spin.value(),
            n_block         = self._n_block_spin.value(),
        )

    def _build_config(self, state: _PolicyState) -> Dict[str, Any]:
        # env_id follows the InteractiveRuntime convention: "sport/variant"
        env_id = f"{state.sport}/{state.variant}"
        run_dir = str(VAR_EVALS_DIR / state.run_id)

        arguments = [
            "-m", "jaxmarl_worker.cli",
            "--interactive",
            "--env-id",      env_id,
            "--policy-path", state.checkpoint_path,
            "--view-size",   str(state.view_size),
            "--algorithm",   state.algorithm,
        ]
        if state.num_agents is not None:
            arguments += ["--num-agents", str(state.num_agents)]
        if state.algorithm == "mat":
            arguments += [
                "--n-embd",  str(state.n_embd),
                "--n-head",  str(state.n_head),
                "--n-block", str(state.n_block),
            ]

        worker_config = {
            "run_id":          state.run_id,
            "eval_only":       True,
            "policy_path":     state.checkpoint_path,
            "env_id":          env_id,
            "sport":           state.sport,
            "variant":         state.variant,
            "algorithm":       state.algorithm,
            "view_size":       state.view_size,
            "num_agents":      state.num_agents,
            "n_embd":          state.n_embd,
            "n_head":          state.n_head,
            "n_block":         state.n_block,
        }

        return {
            "run_name":   state.run_id,
            "entry_point": sys.executable,
            "arguments":  arguments,
            "environment": {
                "JAXMARL_RUN_ID":     state.run_id,
                "JAXMARL_EVAL_MODE":  "1",
            },
            "resources": {
                "cpus":       2,
                "memory_mb":  2048,
                "gpus":       {"requested": 1, "mandatory": False},
            },
            "metadata": {
                "ui": {
                    "worker_id":       "jaxmarl_worker",
                    "mode":            "policy_eval",
                    "env_id":          env_id,
                    "algorithm":       state.algorithm,
                    "policy_path":     state.checkpoint_path,
                },
                "worker": {
                    "worker_id": "jaxmarl_worker",
                    "module":    "jaxmarl_worker.cli",
                    "use_grpc":  False,
                    "config":    worker_config,
                },
            },
            "artifacts": {
                "output_prefix":    run_dir,
                "persist_logs":     True,
                "keep_checkpoints": False,
            },
        }

    def _on_accept(self) -> None:
        try:
            state = self._collect_state()
            if not state.checkpoint_path:
                self.log_constant(
                    LOG_UI_POLICY_FORM_ERROR,
                    message="Cannot load policy: no checkpoint path provided",
                    extra={"worker_id": "jaxmarl_worker"},
                )
                QtWidgets.QMessageBox.warning(
                    self,
                    "Checkpoint Required",
                    "Please browse to or paste a .npz checkpoint path first.",
                )
                return
            if not Path(state.checkpoint_path).exists():
                self.log_constant(
                    LOG_UI_POLICY_FORM_ERROR,
                    message=f"Checkpoint file not found: {state.checkpoint_path}",
                    extra={"worker_id": "jaxmarl_worker", "path": state.checkpoint_path},
                )
                QtWidgets.QMessageBox.warning(
                    self,
                    "Checkpoint Not Found",
                    f"File does not exist:\n{state.checkpoint_path}",
                )
                return
            config = self._build_config(state)
            self._last_config = config
            self.log_constant(
                LOG_UI_POLICY_FORM_INFO,
                message="JaxMARL policy eval config built",
                extra={
                    "worker_id": "jaxmarl_worker",
                    "run_id":    state.run_id,
                    "algorithm": state.algorithm,
                    "env_id":    f"{state.sport}/{state.variant}",
                },
            )
            self.config_ready.emit(config)
            self.accept()
        except Exception as exc:
            self.log_constant(
                LOG_UI_POLICY_FORM_ERROR,
                message=f"Failed to build JaxMARL policy config: {exc}",
                extra={"worker_id": "jaxmarl_worker"},
                exc_info=exc,
            )
            raise

    def get_config(self) -> Optional[Dict[str, Any]]:
        """Return the generated policy-eval configuration."""
        if self._last_config is not None:
            return copy.deepcopy(self._last_config)
        state = self._collect_state()
        return self._build_config(state)


__all__ = ["JaxMARLPolicyForm"]


# Register with form factory
try:
    from gym_gui.ui.forms.factory import get_worker_form_factory

    _factory = get_worker_form_factory()
    if not _factory.has_policy_form("jaxmarl_worker"):
        _factory.register_policy_form(
            "jaxmarl_worker",
            lambda parent=None, **kwargs: JaxMARLPolicyForm(parent=parent, **kwargs),
        )
except ImportError:
    pass
