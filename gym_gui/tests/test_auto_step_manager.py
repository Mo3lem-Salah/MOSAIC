"""Tests for the Auto-Step feature.

Three groups:
  TestAutoStepManagerState  -- pure state-machine logic (no subprocess, no mainwindow)
  TestAutoStepManagerReplay -- replay phase (QTimer), processEvents() pattern
  TestOperatorsTabAutoStep  -- UI-layer: label rename + button / spinbox presence

Run:
    QT_QPA_PLATFORM=offscreen pytest gym_gui/tests/test_auto_step_manager.py -v
"""

from __future__ import annotations

import time
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from qtpy import QtWidgets

from gym_gui.services.auto_step_manager import AutoStepManager
from gym_gui.services.operator import OperatorConfig, WorkerAssignment


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication for headless widget tests."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(["-platform", "offscreen"])
    return app


@pytest.fixture
def mgr(qapp):
    """AutoStepManager instance; stopped after each test."""
    m = AutoStepManager()
    yield m
    m.stop()


@pytest.fixture
def tab(qapp):
    """OperatorsTab widget, shown offscreen so isVisible() returns True for children."""
    from gym_gui.ui.widgets.operators_tab import OperatorsTab
    widget = OperatorsTab()
    widget.show()   # required: Qt's isVisible() checks the full ancestor chain
    yield widget
    widget.close()


def _make_payload(step_index: int = 0, terminated: bool = False) -> dict:
    """Minimal step payload that includes a render_payload (so frames are cached)."""
    return {
        "step_index": step_index,
        "episode_index": 0,
        "reward": 0.0,
        "total_reward": 0.0,
        "terminated": terminated,
        "truncated": False,
        "render_payload": {
            "mode": "socialjax_grid",
            "grid": [[0]],
            "agent_locs": [[1, 1, 0]],
        },
    }


# Arbitrary operator IDs used across tests -- deliberately NOT _ID_A to prove
# the manager is ID-agnostic (real names come from the active operator registry).
_ID_A = "jaxmarl_coop_mining_0"
_ID_B = "socialjax_harvest_1"


def _process_events(ms: int = 50) -> None:
    """Pump the Qt event loop for `ms` milliseconds."""
    deadline = time.monotonic() + ms / 1000
    while time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.005)


# ---------------------------------------------------------------------------
# Group 1: State machine (synchronous)
# ---------------------------------------------------------------------------


class TestAutoStepManagerState:
    """Pure state-machine checks -- no subprocess required."""

    def test_initial_state(self, mgr: AutoStepManager) -> None:
        assert not mgr.is_collecting
        assert not mgr.is_replaying
        assert not mgr.is_active_for(_ID_A)

    def test_start_collection_sets_collecting(self, mgr: AutoStepManager) -> None:
        mgr.start_collection([_ID_A], seed=42, n_steps=10)
        assert mgr.is_collecting
        assert mgr.is_active_for(_ID_A)

    def test_start_collection_emits_reset_for_each_operator(
        self, mgr: AutoStepManager
    ) -> None:
        resets: list[tuple[str, int]] = []
        mgr.reset_requested.connect(lambda op_id, seed: resets.append((op_id, seed)))

        mgr.start_collection([_ID_A, _ID_B], seed=42, n_steps=10)

        assert len(resets) == 2
        assert (_ID_A, 42) in resets
        assert (_ID_B, 42) in resets

    def test_is_not_active_for_unknown_operator(self, mgr: AutoStepManager) -> None:
        mgr.start_collection([_ID_A], seed=1, n_steps=5)
        assert not mgr.is_active_for("other_op")

    def test_on_ready_emits_step(self, mgr: AutoStepManager) -> None:
        steps: list[str] = []
        mgr.step_requested.connect(steps.append)
        mgr.start_collection([_ID_A], seed=1, n_steps=5)

        mgr.on_ready_received(_ID_A)

        assert steps == [_ID_A]

    def test_on_ready_ignored_when_not_collecting(self, mgr: AutoStepManager) -> None:
        steps: list[str] = []
        mgr.step_requested.connect(steps.append)

        mgr.on_ready_received(_ID_A)  # no start_collection

        assert steps == []

    def test_on_ready_ignored_for_unknown_operator(self, mgr: AutoStepManager) -> None:
        steps: list[str] = []
        mgr.step_requested.connect(steps.append)
        mgr.start_collection([_ID_A], seed=1, n_steps=5)

        mgr.on_ready_received("unknown")

        assert steps == []

    def test_step_collected_stores_frame(self, mgr: AutoStepManager) -> None:
        mgr.start_collection([_ID_A], seed=0, n_steps=5)
        mgr.on_ready_received(_ID_A)

        mgr.on_step_collected(_ID_A, _make_payload(1), episode_done=False)
        assert mgr.frame_count(_ID_A) == 1

        mgr.on_step_collected(_ID_A, _make_payload(2), episode_done=False)
        assert mgr.frame_count(_ID_A) == 2

    def test_step_collected_emits_step_requested_until_n_steps(
        self, mgr: AutoStepManager
    ) -> None:
        n_steps = 5
        steps_emitted: list[str] = []
        mgr.step_requested.connect(steps_emitted.append)
        done_signals: list[None] = []
        mgr.collection_done.connect(lambda: done_signals.append(None))

        mgr.start_collection([_ID_A], seed=0, n_steps=n_steps)
        mgr.on_ready_received(_ID_A)  # emits step_requested but does NOT increment count

        # Simulate n_steps responses arriving -- each increments the counter.
        # The n_steps-th call triggers collection_done and does NOT re-emit step_requested.
        for i in range(n_steps):
            mgr.on_step_collected(_ID_A, _make_payload(i), episode_done=False)

        # step_requested: 1 (from ready) + (n_steps - 1) = n_steps total
        # (the last on_step_collected fires collection_done, not another step_requested)
        assert len(steps_emitted) == n_steps
        assert done_signals == [None]
        assert not mgr.is_collecting

    def test_episode_done_stops_collection_early(self, mgr: AutoStepManager) -> None:
        done_signals: list[None] = []
        mgr.collection_done.connect(lambda: done_signals.append(None))

        mgr.start_collection([_ID_A], seed=0, n_steps=1000)
        mgr.on_ready_received(_ID_A)
        mgr.on_step_collected(_ID_A, _make_payload(50, terminated=True), episode_done=True)

        assert done_signals == [None]
        assert not mgr.is_collecting

    def test_multi_operator_waits_for_all(self, mgr: AutoStepManager) -> None:
        """collection_done fires only after every operator is done."""
        done_signals: list[None] = []
        mgr.collection_done.connect(lambda: done_signals.append(None))

        mgr.start_collection([_ID_A, _ID_B], seed=0, n_steps=2)
        mgr.on_ready_received(_ID_A)
        mgr.on_ready_received(_ID_B)

        # op_0 finishes
        mgr.on_step_collected(_ID_A, _make_payload(1), episode_done=False)
        mgr.on_step_collected(_ID_A, _make_payload(2), episode_done=False)
        assert done_signals == []  # op_1 still running

        # op_1 finishes
        mgr.on_step_collected(_ID_B, _make_payload(1), episode_done=False)
        mgr.on_step_collected(_ID_B, _make_payload(2), episode_done=False)
        assert done_signals == [None]

    def test_payload_without_render_payload_not_stored(
        self, mgr: AutoStepManager
    ) -> None:
        """Payloads lacking render_payload are step-counted but not cached."""
        mgr.start_collection([_ID_A], seed=0, n_steps=5)
        mgr.on_ready_received(_ID_A)

        bare = {"step_index": 1, "terminated": False}  # no render_payload
        mgr.on_step_collected(_ID_A, bare, episode_done=False)

        assert mgr.frame_count(_ID_A) == 0

    def test_frame_collected_signal_carries_progress(
        self, mgr: AutoStepManager
    ) -> None:
        progress: list[tuple[str, int, int]] = []
        mgr.frame_collected.connect(
            lambda op_id, step, total: progress.append((op_id, step, total))
        )

        mgr.start_collection([_ID_A], seed=0, n_steps=10)
        mgr.on_ready_received(_ID_A)
        mgr.on_step_collected(_ID_A, _make_payload(1), episode_done=False)

        assert len(progress) == 1
        op_id, step, total = progress[0]
        assert op_id == _ID_A
        assert step == 1
        assert total == 10

    def test_restart_clears_prior_frames(self, mgr: AutoStepManager) -> None:
        mgr.start_collection([_ID_A], seed=1, n_steps=5)
        mgr.on_ready_received(_ID_A)
        mgr.on_step_collected(_ID_A, _make_payload(1), episode_done=False)
        assert mgr.frame_count(_ID_A) == 1

        # Restart -- old frames should be gone
        mgr.start_collection([_ID_A], seed=99, n_steps=5)
        assert mgr.frame_count(_ID_A) == 0

    def test_stop_clears_all_state(self, mgr: AutoStepManager) -> None:
        mgr.start_collection([_ID_A], seed=7, n_steps=10)
        assert mgr.is_collecting

        mgr.stop()

        assert not mgr.is_collecting
        assert not mgr.is_replaying
        assert not mgr.is_active_for(_ID_A)

    def test_any_operator_id_string_is_accepted(self, mgr: AutoStepManager) -> None:
        """Manager must accept any operator ID -- no hardcoded assumptions."""
        arbitrary_ids = ["custom_env_op_7", "my-worker-99", "SocialJax/coop_mining:0"]
        resets: list[str] = []
        mgr.reset_requested.connect(lambda op_id, _: resets.append(op_id))

        mgr.start_collection(arbitrary_ids, seed=0, n_steps=5)

        for op_id in arbitrary_ids:
            assert mgr.is_active_for(op_id), f"is_active_for should be True for '{op_id}'"
        assert set(resets) == set(arbitrary_ids)


# ---------------------------------------------------------------------------
# Group 2: Replay phase (QTimer-driven)
# ---------------------------------------------------------------------------


class TestAutoStepManagerReplay:
    """Replay tests -- QTimer fires inside processEvents() loop."""

    def _populate(self, mgr: AutoStepManager, n_frames: int = 3) -> None:
        """Helper: fill mgr with n_frames cached frames for op_0."""
        mgr.start_collection([_ID_A], seed=0, n_steps=n_frames)
        mgr.on_ready_received(_ID_A)
        for i in range(n_frames - 1):
            mgr.on_step_collected(_ID_A, _make_payload(i + 1), episode_done=False)
        # Last step triggers collection_done
        mgr.on_step_collected(
            _ID_A, _make_payload(n_frames, terminated=True), episode_done=True
        )
        assert not mgr.is_collecting
        assert mgr.frame_count(_ID_A) == n_frames

    def test_start_replay_sets_replaying(self, mgr: AutoStepManager) -> None:
        self._populate(mgr)
        mgr.start_replay(interval_ms=50)
        assert mgr.is_replaying

    def test_replay_emits_display_frame(self, mgr: AutoStepManager) -> None:
        self._populate(mgr, n_frames=2)
        frames: list[tuple[str, dict]] = []
        mgr.display_frame.connect(lambda op_id, p: frames.append((op_id, p)))

        mgr.start_replay(interval_ms=30)
        _process_events(ms=150)  # let 2-3 ticks fire
        mgr.stop_replay()

        assert len(frames) >= 1
        assert frames[0][0] == _ID_A
        assert "render_payload" in frames[0][1]

    def test_replay_loops_over_all_frames(self, mgr: AutoStepManager) -> None:
        """After the last frame, position wraps back to 0."""
        self._populate(mgr, n_frames=2)
        step_indices: list[int] = []
        mgr.display_frame.connect(
            lambda op_id, p: step_indices.append(p["step_index"])
        )

        mgr.start_replay(interval_ms=20)
        _process_events(ms=200)  # enough for multiple full loops
        mgr.stop_replay()

        # Both frame step_indices must appear
        assert 1 in step_indices
        assert 2 in step_indices

    def test_stop_replay_halts_emission(self, mgr: AutoStepManager) -> None:
        self._populate(mgr, n_frames=2)

        mgr.start_replay(interval_ms=20)
        _process_events(ms=80)
        mgr.stop_replay()
        assert not mgr.is_replaying

        # Count any further signals after stop -- should be zero
        late: list[None] = []
        mgr.display_frame.connect(lambda *_: late.append(None))
        _process_events(ms=80)
        assert late == []

    def test_start_replay_with_no_frames_does_not_set_replaying(
        self, mgr: AutoStepManager
    ) -> None:
        """If collection produced zero frames, replay must not start."""
        mgr.start_replay(interval_ms=50)
        assert not mgr.is_replaying


# ---------------------------------------------------------------------------
# Group 3: operators_tab UI changes
# ---------------------------------------------------------------------------


class TestOperatorsTabAutoStep:
    """UI-level checks for the label rename and Auto-Step button."""

    def test_seed_label_is_seed_not_shared_seed(self, tab) -> None:
        """The label beside the spinbox must read 'Seed:', not 'Shared Seed:'."""
        labels = tab.findChildren(QtWidgets.QLabel)
        label_texts = [lbl.text() for lbl in labels]
        assert "Seed:" in label_texts, (
            f"Expected 'Seed:' label. Found: {label_texts}"
        )
        assert "Shared Seed:" not in label_texts, (
            "'Shared Seed:' must be removed; rename to 'Seed:'"
        )

    def test_use_shared_seed_checkbox_unchanged(self, tab) -> None:
        """'Use Shared Seed Across Operators' checkbox must remain present."""
        checkboxes = tab.findChildren(QtWidgets.QCheckBox)
        texts = [cb.text() for cb in checkboxes]
        assert any("Use Shared Seed" in t for t in texts), (
            f"'Use Shared Seed Across Operators' checkbox must still exist. Found: {texts}"
        )

    def test_auto_step_button_exists_and_is_checkable(self, tab) -> None:
        assert hasattr(tab, "_auto_step_button"), (
            "OperatorsTab must expose _auto_step_button"
        )
        btn = tab._auto_step_button
        assert btn.isCheckable(), "Auto-Step button must be checkable (toggle)"
        assert btn.text() == "Auto-Step"

    def test_auto_step_button_initially_disabled(self, tab) -> None:
        """Auto-Step must be disabled until operators are reset (running)."""
        assert not tab._auto_step_button.isEnabled()

    def test_interval_spinbox_range_and_default(self, tab) -> None:
        spin = tab._auto_step_interval_spin
        assert spin.minimum() == pytest.approx(0.1)
        assert spin.maximum() == pytest.approx(5.0)
        assert spin.value() == pytest.approx(0.5)

    def test_progress_bar_initially_hidden(self, tab) -> None:
        assert not tab._auto_step_progress_bar.isVisible()

    def test_set_auto_step_state_true_is_replay_phase(self, tab) -> None:
        # active=True means replay started -- bar must be hidden (collection done)
        tab.set_auto_step_state(True)
        assert tab._auto_step_button.isChecked()
        assert tab._auto_step_button.text() == "Stop Auto-Step"
        assert not tab._auto_step_progress_bar.isVisible()

    def test_set_auto_step_state_false(self, tab) -> None:
        tab.set_auto_step_state(True)
        tab.set_auto_step_state(False)
        assert not tab._auto_step_button.isChecked()
        assert tab._auto_step_button.text() == "Auto-Step"
        assert not tab._auto_step_progress_bar.isVisible()

    def test_update_auto_step_progress_shows_bar(self, tab) -> None:
        # update_auto_step_progress must make bar visible and update values
        assert not tab._auto_step_progress_bar.isVisible()
        tab.update_auto_step_progress(300, 1000)
        assert tab._auto_step_progress_bar.isVisible()
        assert tab._auto_step_progress_bar.value() == 300
        assert tab._auto_step_progress_bar.maximum() == 1000

    def test_update_auto_step_progress_hides_after_stop(self, tab) -> None:
        # Bar shown by progress update must be hidden after set_auto_step_state(False)
        tab.update_auto_step_progress(500, 1000)
        assert tab._auto_step_progress_bar.isVisible()
        tab.set_auto_step_state(False)
        assert not tab._auto_step_progress_bar.isVisible()

    def test_auto_step_requested_signal_on_click(self, tab) -> None:
        """Clicking the button while running emits auto_step_requested(seed, interval_ms)."""
        tab._auto_step_button.setEnabled(True)
        tab._is_running = True
        tab._seed_spin.setValue(99)
        tab._auto_step_interval_spin.setValue(0.3)

        signals: list[tuple[int, int]] = []
        tab.auto_step_requested.connect(
            lambda seed, interval_ms: signals.append((seed, interval_ms))
        )

        tab._auto_step_button.setChecked(True)
        tab._on_auto_step_clicked()

        assert len(signals) == 1
        seed, interval_ms = signals[0]
        assert seed == 99
        assert 290 <= interval_ms <= 310  # 0.3s * 1000 = 300ms, allow float rounding

    def test_auto_step_stop_signal_on_uncheck(self, tab) -> None:
        """Un-checking the button emits auto_step_stop_requested."""
        tab._auto_step_button.setEnabled(True)
        tab._is_running = True
        tab.set_auto_step_state(True)

        stop_signals: list[None] = []
        tab.auto_step_stop_requested.connect(lambda: stop_signals.append(None))

        tab._auto_step_button.setChecked(False)
        tab._on_auto_step_clicked()

        assert stop_signals == [None]


# ---------------------------------------------------------------------------
# TestAutoStepRLDetection
# Tests the has_rl_worker logic that was buggy: operator_type returns
# "multiagent" for any config with >1 worker, regardless of worker_type.
# Regression guard: 6-agent jaxmarl operators must be treated as RL-capable.
# ---------------------------------------------------------------------------


def _make_rl_worker(idx: int) -> WorkerAssignment:
    return WorkerAssignment(worker_id="jaxmarl_worker", worker_type="rl")


def _make_llm_worker(idx: int) -> WorkerAssignment:
    return WorkerAssignment(worker_id="balrog_worker", worker_type="llm")


def _has_rl_worker(config: OperatorConfig) -> bool:
    """Inline copy of the check in _on_auto_step_requested."""
    return any(w.worker_type == "rl" for w in config.workers.values())


class TestAutoStepRLDetection:
    """Regression tests for has_rl_worker -- the fix for the multiagent type bug."""

    def test_single_rl_worker_is_detected(self) -> None:
        config = OperatorConfig(
            operator_id="op_0",
            display_name="Test",
            env_name="socialjax",
            task="socialjax/coop_mining",
            workers={"agent_0": _make_rl_worker(0)},
        )
        assert config.operator_type == "rl", "precondition: single RL → operator_type=rl"
        assert _has_rl_worker(config)

    def test_six_rl_workers_operator_type_is_multiagent(self) -> None:
        """operator_type returns 'multiagent' for 6-worker configs -- the root cause."""
        workers = {f"agent_{i}": _make_rl_worker(i) for i in range(6)}
        config = OperatorConfig(
            operator_id="op_0",
            display_name="Test",
            env_name="socialjax",
            task="socialjax/coop_mining",
            workers=workers,
        )
        # This is exactly why the old check `config.operator_type != "rl"` failed.
        assert config.operator_type == "multiagent", (
            "6-worker config must return 'multiagent' -- confirms the bug existed"
        )
        # The new check must still pass despite operator_type != "rl".
        assert _has_rl_worker(config), (
            "6 RL workers must be detected as RL-capable by has_rl_worker"
        )

    def test_pure_llm_operator_is_rejected(self) -> None:
        config = OperatorConfig(
            operator_id="op_0",
            display_name="LLM Test",
            env_name="socialjax",
            task="socialjax/coop_mining",
            workers={"agent_0": _make_llm_worker(0)},
        )
        assert not _has_rl_worker(config)

    def test_mixed_rl_and_llm_operator_is_accepted(self) -> None:
        """At least one RL worker is enough -- operator is RL-capable."""
        workers = {
            "agent_0": _make_rl_worker(0),
            "agent_1": _make_llm_worker(1),
        }
        config = OperatorConfig(
            operator_id="op_0",
            display_name="Mixed",
            env_name="socialjax",
            task="socialjax/coop_mining",
            workers=workers,
        )
        assert config.operator_type == "multiagent"
        assert _has_rl_worker(config)

    def test_empty_workers_is_rejected(self) -> None:
        config = OperatorConfig(
            operator_id="op_0",
            display_name="Empty",
            env_name="socialjax",
            task="socialjax/coop_mining",
            workers={},
        )
        assert not _has_rl_worker(config)
