"""JaxMARL Custom Script form.

Selects an existing training/eval script (.py or .sh) and launches it via
the trainer daemon as a subprocess. Unlike the train form (which builds
config for jaxmarl_worker.cli), this form runs the script directly with
minimal wrapping, so the script is the single source of truth for
hyperparameters, env config, and logging.

Typical script locations:
  - var/trainer/custom_scripts/     (user-authored scripts)
  - 3rd_party/workers/jaxmarl_worker/scripts/  (repo-shipped scripts, if any)
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

from gym_gui.config.paths import VAR_CUSTOM_SCRIPTS_DIR, VAR_TRAINER_DIR
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_TRAIN_FORM_ERROR,
    LOG_UI_TRAIN_FORM_INFO,
    LOG_UI_TRAIN_FORM_WARNING,
)

_LOGGER = logging.getLogger("gym_gui.ui.jaxmarl_script_form")

# Repo-shipped script roots to scan (best-effort; missing dirs are skipped)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPO_SCRIPT_ROOTS: List[Path] = [
    VAR_CUSTOM_SCRIPTS_DIR,
    _REPO_ROOT / "3rd_party" / "workers" / "jaxmarl_worker" / "scripts",
    _REPO_ROOT / "3rd_party" / "workers" / "mosaic" / "gar_worker" / "scripts",
    # Upstream JaxMARL training baselines (MAPPO / IPPO / QLearning) — used to
    # launch tier-2 families (MPE, MABrax, Overcooked, Hanabi) via the trainer.
    _REPO_ROOT / "3rd_party" / "workers" / "jaxmarl_worker" / "JaxMARL" / "baselines" / "MAPPO",
    _REPO_ROOT / "3rd_party" / "workers" / "jaxmarl_worker" / "JaxMARL" / "baselines" / "IPPO",
    _REPO_ROOT / "3rd_party" / "workers" / "jaxmarl_worker" / "JaxMARL" / "baselines" / "QLearning",
]


def _generate_run_id(script_stem: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    slug = script_stem.replace("_", "-")[:32] or "script"
    return f"jaxmarl_script_{slug}_{ts}_{short}"


def _discover_scripts() -> List[Tuple[str, Path]]:
    """Discover .py and .sh scripts under known jaxmarl script roots.

    Returns a list of (display_label, absolute_path) tuples, sorted by label.
    Silently skips roots that don't exist.
    """
    results: List[Tuple[str, Path]] = []
    for root in _REPO_SCRIPT_ROOTS:
        if not root.exists() or not root.is_dir():
            continue
        for ext in ("*.py", "*.sh"):
            for path in root.rglob(ext):
                if path.is_file():
                    rel = path.relative_to(_REPO_ROOT) if _REPO_ROOT in path.parents else path
                    results.append((str(rel), path))
    results.sort(key=lambda pair: pair[0])
    return results


@dataclass(frozen=True)
class _ScriptState:
    run_id: str
    script_path: str
    extra_args: str
    use_gpu: bool
    track_tensorboard: bool
    notes: str


class JaxMARLScriptForm(QtWidgets.QDialog, LogConstantMixin):
    """Launch a JaxMARL training or evaluation script."""

    config_ready = QtCore.Signal(dict)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        self._logger = _LOGGER
        self.setWindowTitle("JaxMARL Script Execution")
        self.setMinimumWidth(820)
        self.setModal(True)
        self._last_config: Optional[Dict[str, Any]] = None
        self._discovered: List[Tuple[str, Path]] = []
        self._init_ui()
        self._populate_scripts()
        self.adjustSize()
        self.log_constant(
            LOG_UI_TRAIN_FORM_INFO,
            message="JaxMARLScriptForm initialized",
            extra={"worker_id": "jaxmarl_worker"},
        )

    def _init_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(12, 10, 12, 10)
        root.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        hdr = QtWidgets.QLabel("<b>JaxMARL Script Execution</b> — run a .py or .sh script directly")
        hdr.setStyleSheet("font-size: 13px; color: #5c85d6;")
        root.addWidget(hdr)
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("color: #ccc;")
        root.addWidget(sep)

        # Script selection: discovered combo + browse button + resolved-path label
        script_row = QtWidgets.QHBoxLayout()
        script_row.addWidget(QtWidgets.QLabel("Script:"))
        self._script_combo = QtWidgets.QComboBox()
        self._script_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        script_row.addWidget(self._script_combo, 1)
        self._browse_btn = QtWidgets.QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._on_browse)
        script_row.addWidget(self._browse_btn)
        root.addLayout(script_row)

        # Resolved absolute path (read-only, updates as selection changes)
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(QtWidgets.QLabel("Absolute path:"))
        self._path_edit = QtWidgets.QLineEdit()
        self._path_edit.setReadOnly(True)
        self._path_edit.setStyleSheet("color: #555; background: #f4f4f4;")
        path_row.addWidget(self._path_edit, 1)
        root.addLayout(path_row)

        self._script_combo.currentIndexChanged.connect(self._on_script_changed)

        # Extra CLI args passed after the script
        args_row = QtWidgets.QHBoxLayout()
        args_row.addWidget(QtWidgets.QLabel("Extra args:"))
        self._args_edit = QtWidgets.QLineEdit()
        self._args_edit.setPlaceholderText("e.g. --seed 42 --n-envs 128  (optional, appended to script command)")
        args_row.addWidget(self._args_edit, 1)
        root.addLayout(args_row)

        # Tracking (inline)
        tracking_row = QtWidgets.QHBoxLayout()
        tracking_row.setSpacing(16)
        tracking_row.addWidget(QtWidgets.QLabel("<b>Tracking:</b>"))
        self._tb_check = QtWidgets.QCheckBox("TensorBoard (JAXMARL_TENSORBOARD_DIR env)")
        self._tb_check.setChecked(True)
        tracking_row.addWidget(self._tb_check)
        self._gpu_check = QtWidgets.QCheckBox("Use GPU (if available)")
        self._gpu_check.setChecked(True)
        tracking_row.addWidget(self._gpu_check)
        tracking_row.addStretch()
        root.addLayout(tracking_row)

        # Notes
        notes_lbl = QtWidgets.QLabel("<b>Notes</b>")
        root.addWidget(notes_lbl)
        self._notes_edit = QtWidgets.QPlainTextEdit()
        self._notes_edit.setPlaceholderText("Optional run notes")
        self._notes_edit.setFixedHeight(80)
        self._notes_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        root.addWidget(self._notes_edit)

        root.addStretch(1)

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
            ok_btn.setText("Launch Script")
        btn_row.addWidget(btn_box)
        root.addLayout(btn_row)

    def _populate_scripts(self) -> None:
        self._discovered = _discover_scripts()
        self._script_combo.clear()
        if not self._discovered:
            self._script_combo.addItem("(no scripts discovered - use Browse)")
            self._script_combo.setEnabled(False)
            self._path_edit.clear()
            self.log_constant(
                LOG_UI_TRAIN_FORM_WARNING,
                message="No jaxmarl scripts discovered under any known root",
                extra={
                    "worker_id": "jaxmarl_worker",
                    "roots": ",".join(str(r) for r in _REPO_SCRIPT_ROOTS),
                },
            )
            return
        self._script_combo.setEnabled(True)
        for label, path in self._discovered:
            self._script_combo.addItem(label, str(path))
        self._on_script_changed()

    def _on_script_changed(self) -> None:
        data = self._script_combo.currentData()
        self._path_edit.setText(str(data) if data else "")

    def _on_browse(self) -> None:
        start_dir = str(VAR_CUSTOM_SCRIPTS_DIR) if VAR_CUSTOM_SCRIPTS_DIR.exists() else str(_REPO_ROOT)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Script",
            start_dir,
            "Scripts (*.py *.sh);;Python (*.py);;Shell (*.sh);;All Files (*)",
        )
        if not path:
            return
        # Insert as an ad-hoc entry so get_config can find it.
        label = f"(custom) {path}"
        self._script_combo.addItem(label, path)
        self._script_combo.setCurrentIndex(self._script_combo.count() - 1)
        self._script_combo.setEnabled(True)

    # -----------------------------------------------------------------------
    # State collection & config building
    # -----------------------------------------------------------------------

    def _collect_state(self) -> _ScriptState:
        script_path = self._script_combo.currentData() or ""
        stem = Path(script_path).stem if script_path else "script"
        return _ScriptState(
            run_id            = _generate_run_id(stem),
            script_path       = str(script_path),
            extra_args        = self._args_edit.text().strip(),
            use_gpu           = self._gpu_check.isChecked(),
            track_tensorboard = self._tb_check.isChecked(),
            notes             = self._notes_edit.toPlainText().strip(),
        )

    def _build_config(self, state: _ScriptState) -> Dict[str, Any]:
        run_dir = str(VAR_TRAINER_DIR / "runs" / state.run_id)
        script = Path(state.script_path)
        # Extra args are space-separated; keep it simple, users can quote if needed.
        extra_argv = state.extra_args.split() if state.extra_args else []

        # Choose entry point + argv based on script extension.
        if script.suffix == ".sh":
            entry_point = "/bin/bash"
            arguments = [str(script), *extra_argv]
        else:
            entry_point = sys.executable
            arguments = [str(script), *extra_argv]

        environment: Dict[str, str] = {
            "JAXMARL_RUN_ID": state.run_id,
        }
        if state.track_tensorboard:
            environment["JAXMARL_TENSORBOARD_DIR"] = str(Path(run_dir) / "tensorboard")

        worker_config = {
            "run_id":       state.run_id,
            "script_path":  state.script_path,
            "extra_args":   state.extra_args,
            "tensorboard":  state.track_tensorboard,
            "notes":        state.notes,
        }

        return {
            "run_name":    state.run_id,
            "entry_point": entry_point,
            "arguments":   arguments,
            "environment": environment,
            "resources": {
                "cpus":      4,
                "memory_mb": 4096,
                "gpus":      {"requested": 1 if state.use_gpu else 0, "mandatory": False},
            },
            "metadata": {
                "ui": {
                    "worker_id":  "jaxmarl_worker",
                    "mode":       "custom_script",
                    "script":     Path(state.script_path).name,
                },
                "worker": {
                    "worker_id": "jaxmarl_worker",
                    "module":    "custom_script",
                    "use_grpc":  False,
                    "config":    worker_config,
                },
            },
            "artifacts": {
                "output_prefix":    run_dir,
                "persist_logs":     True,
                "keep_checkpoints": True,
                "notes":            state.notes,
            },
        }

    def _on_accept(self) -> None:
        try:
            state = self._collect_state()
            if not state.script_path:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Script Required",
                    "Please select a script from the dropdown or browse to one.",
                )
                return
            if not Path(state.script_path).exists():
                self.log_constant(
                    LOG_UI_TRAIN_FORM_ERROR,
                    message=f"Script file not found: {state.script_path}",
                    extra={"worker_id": "jaxmarl_worker", "path": state.script_path},
                )
                QtWidgets.QMessageBox.warning(
                    self,
                    "Script Not Found",
                    f"File does not exist:\n{state.script_path}",
                )
                return
            config = self._build_config(state)
            self._last_config = config
            self.log_constant(
                LOG_UI_TRAIN_FORM_INFO,
                message="JaxMARL script launch config built",
                extra={
                    "worker_id":  "jaxmarl_worker",
                    "run_id":     state.run_id,
                    "script":     Path(state.script_path).name,
                    "use_gpu":    state.use_gpu,
                },
            )
            self.config_ready.emit(config)
            self.accept()
        except Exception as exc:
            self.log_constant(
                LOG_UI_TRAIN_FORM_ERROR,
                message=f"Failed to build JaxMARL script config: {exc}",
                extra={"worker_id": "jaxmarl_worker"},
                exc_info=exc,
            )
            raise

    def get_config(self) -> Optional[Dict[str, Any]]:
        if self._last_config is not None:
            return copy.deepcopy(self._last_config)
        state = self._collect_state()
        return self._build_config(state)


__all__ = ["JaxMARLScriptForm"]


# Register with form factory
try:
    from gym_gui.ui.forms.factory import get_worker_form_factory

    _factory = get_worker_form_factory()
    if not _factory.has_script_form("jaxmarl_worker"):
        _factory.register_script_form(
            "jaxmarl_worker",
            lambda parent=None, **kwargs: JaxMARLScriptForm(parent=parent, **kwargs),
        )
except ImportError:
    pass
