"""Modal progress dialog shown while shared LinkGroup workers load their policies.

Shows one row per LinkGroup with a status label and a progress bar.
Auto-closes when all groups are ready, or stays open with an error
indicator if any group fails.

See docs/Development_Progress/1.0_DAY_73/TASK_4/PLAN_UPDATE_1.md.
"""

from __future__ import annotations

import logging
from typing import Dict

from qtpy import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal, pyqtSlot

_LOGGER = logging.getLogger(__name__)


class OperatorLaunchProgressDialog(QtWidgets.QDialog):
    """Dialog with a progress bar per LinkGroup.

    The dialog is modal (blocks the operator config UI) but does NOT block
    the Qt event loop, so the GUI stays responsive during the launch.

    Usage:
        dialog = OperatorLaunchProgressDialog(group_ids, parent=main_window)
        dialog.show()  # modeless show so signals can drive progress

        launcher.progress.connect(dialog.on_progress)
        launcher.ready.connect(dialog.on_ready)
        launcher.error.connect(dialog.on_error)

        dialog.exec()   # blocks until close_when_done is triggered
    """

    # User clicked the Cancel button.
    cancelled = pyqtSignal()

    def __init__(self, group_ids: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loading RL Policies...")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(120 + 60 * len(group_ids))

        self._group_ids = list(group_ids)
        self._status_labels: Dict[str, QtWidgets.QLabel] = {}
        self._progress_bars: Dict[str, QtWidgets.QProgressBar] = {}
        self._completed: set[str] = set()
        self._errored: Dict[str, str] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(
            "Spawning shared RL worker subprocesses and loading policies. "
            "This may take 30-120 seconds per group."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        for group_id in self._group_ids:
            group_box = QtWidgets.QGroupBox(f"LinkGroup: {group_id}", self)
            vbox = QtWidgets.QVBoxLayout(group_box)

            status_label = QtWidgets.QLabel("Waiting to start...", group_box)
            status_label.setStyleSheet("color: #888;")
            vbox.addWidget(status_label)

            progress_bar = QtWidgets.QProgressBar(group_box)
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            vbox.addWidget(progress_bar)

            self._status_labels[group_id] = status_label
            self._progress_bars[group_id] = progress_bar

            layout.addWidget(group_box)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        self._cancel_btn = QtWidgets.QPushButton("Cancel", self)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self._cancel_btn)
        layout.addLayout(btn_row)

    # ----- Slots (connected to LinkGroupLauncher signals) -----

    @pyqtSlot(str, str, int)
    def on_progress(self, group_id: str, status: str, pct: int) -> None:
        """Update the progress bar and status label for one group."""
        if group_id not in self._progress_bars:
            return
        self._status_labels[group_id].setText(status)
        self._status_labels[group_id].setStyleSheet("color: #333;")
        self._progress_bars[group_id].setValue(pct)

    @pyqtSlot(str, object)
    def on_ready(self, group_id: str, handle) -> None:
        """Mark one group as Ready and check if all are complete."""
        if group_id not in self._progress_bars:
            return
        self._status_labels[group_id].setText("Ready (policy loaded)")
        self._status_labels[group_id].setStyleSheet(
            "color: #2e7d32; font-weight: bold;"
        )
        self._progress_bars[group_id].setValue(100)
        self._completed.add(group_id)
        self._check_done()

    @pyqtSlot(str, str)
    def on_error(self, group_id: str, message: str) -> None:
        """Display an error for one group."""
        if group_id not in self._progress_bars:
            return
        self._status_labels[group_id].setText(f"ERROR: {message}")
        self._status_labels[group_id].setStyleSheet(
            "color: #c62828; font-weight: bold;"
        )
        self._progress_bars[group_id].setValue(0)
        self._errored[group_id] = message
        self._completed.add(group_id)
        self._cancel_btn.setText("Close")
        self._check_done()

    # ----- Internals -----

    def _check_done(self) -> None:
        """Auto-close when every group has finished (ready or errored)."""
        if len(self._completed) >= len(self._group_ids):
            if not self._errored:
                # All ready: close after a short delay so the user sees the
                # green "Ready" text briefly.
                QtCore.QTimer.singleShot(400, self.accept)
            # On error, keep the dialog open so the user can read the message

    def _on_cancel_clicked(self) -> None:
        if self._errored and len(self._completed) >= len(self._group_ids):
            # All done (with errors), the button is acting as "Close"
            self.reject()
        else:
            self.cancelled.emit()
            self._cancel_btn.setEnabled(False)
            self._cancel_btn.setText("Cancelling...")

    # ----- API for caller -----

    def has_errors(self) -> bool:
        return bool(self._errored)

    def error_messages(self) -> Dict[str, str]:
        return dict(self._errored)
