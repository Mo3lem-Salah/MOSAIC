"""Unit tests for the GRF (Google Research Football) keyboard mapping table.

Verifies _STANDARD_GRF_ACTIONS and _GFOOTBALL_MAPPINGS without launching
a game or requiring actual gfootball to be installed.
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# GRF action-space reference (from gfootball.env.football_action_set)
# Indices are stable across all GRF scenarios that use the default action set.
# ---------------------------------------------------------------------------
GRF_ACTIONS = {
    0:  "idle",
    1:  "left",
    2:  "top_left",
    3:  "top",
    4:  "top_right",
    5:  "right",
    6:  "bottom_right",
    7:  "bottom",
    8:  "bottom_left",
    9:  "long_pass",
    10: "high_pass",
    11: "short_pass",
    12: "shot",
    13: "sprint",
    14: "release_direction",
    15: "release_sprint",
    16: "sliding",
    17: "dribble",
    18: "release_dribble",
}


class TestGRFActionTableStructure:
    """Verify _STANDARD_GRF_ACTIONS covers the full GRF action space."""

    @pytest.fixture(scope="class")
    def grf_actions(self):
        from gym_gui.controllers.human_input import _STANDARD_GRF_ACTIONS
        return _STANDARD_GRF_ACTIONS

    def test_all_19_actions_covered(self, grf_actions):
        """Every GRF action index 0-18 must appear at least once."""
        mapped = {m.action for m in grf_actions}
        missing = set(GRF_ACTIONS.keys()) - mapped
        assert not missing, (
            f"Missing action indices: {missing}. "
            f"GRF actions not reachable by keyboard: "
            f"{[GRF_ACTIONS[a] for a in sorted(missing)]}"
        )

    def test_no_duplicate_action_indices(self, grf_actions):
        """Each GRF action index must appear at most once."""
        seen: dict[int, int] = {}
        for i, m in enumerate(grf_actions):
            if m.action in seen:
                pytest.fail(
                    f"Action {m.action} ({GRF_ACTIONS.get(m.action, '?')}) "
                    f"appears at positions {seen[m.action]} and {i}."
                )
            seen[m.action] = i

    def test_no_out_of_range_actions(self, grf_actions):
        """No action index outside 0-18 (would silently be ignored by GRF)."""
        for m in grf_actions:
            assert 0 <= m.action <= 18, (
                f"Action {m.action} is outside GRF's valid range [0, 18]."
            )

    def test_table_has_exactly_19_entries(self, grf_actions):
        """Exactly one entry per GRF action (dense coverage, no gaps)."""
        assert len(grf_actions) == 19, (
            f"Expected 19 entries (one per GRF action), got {len(grf_actions)}."
        )


class TestGRFKeyAssignments:
    """Verify the specific key assignments match GRF conventions."""

    @pytest.fixture(scope="class")
    def action_to_keys(self):
        """Build {action_idx: [key_name, ...]} from _STANDARD_GRF_ACTIONS."""
        from gym_gui.controllers.human_input import _STANDARD_GRF_ACTIONS, _build_qt_key_enum_map
        enum_map = _build_qt_key_enum_map()
        result: dict[int, list[str]] = {}
        for m in _STANDARD_GRF_ACTIONS:
            names = []
            for seq in m.key_sequences:
                if len(seq) == 0:
                    continue
                combo = seq[0]
                key_int = combo.key() if hasattr(combo, "key") else int(combo)
                if hasattr(key_int, "value"):
                    key_int = key_int.value
                else:
                    key_int = int(key_int)
                name = enum_map.get(key_int, f"<unknown:{key_int}>")
                names.append(name)
            result[m.action] = names
        return result

    def test_shot_is_space(self, action_to_keys):
        """Action 12 (shot) must be mapped to Space — the most critical binding."""
        keys = action_to_keys.get(12, [])
        assert "Key_Space" in keys, (
            f"Shot (action 12) should be Space, got: {keys}"
        )

    def test_sprint_is_shift(self, action_to_keys):
        """Action 13 (sprint) should be Shift — hold-to-sprint paradigm."""
        keys = action_to_keys.get(13, [])
        assert "Key_Shift" in keys, (
            f"Sprint (action 13) should be Shift, got: {keys}"
        )

    def test_movement_uses_wasd_layout(self, action_to_keys):
        """Movement actions (1-8) should use the WASD+diagonal key cluster."""
        expected = {
            1: "Key_A",   # left
            3: "Key_W",   # top
            5: "Key_D",   # right
            7: "Key_S",   # bottom
        }
        for action, expected_key in expected.items():
            keys = action_to_keys.get(action, [])
            assert expected_key in keys, (
                f"Action {action} ({GRF_ACTIONS[action]}) should include "
                f"{expected_key}, got: {keys}"
            )

    def test_pass_keys_are_jkl(self, action_to_keys):
        """Pass actions (9=long, 10=high, 11=short) should be on J/K/L."""
        expected = {9: "Key_J", 10: "Key_K", 11: "Key_L"}
        for action, key in expected.items():
            keys = action_to_keys.get(action, [])
            assert key in keys, (
                f"Action {action} ({GRF_ACTIONS[action]}) should be {key}, got: {keys}"
            )


class TestGRFGameIdCoverage:
    """Verify _GFOOTBALL_MAPPINGS covers all GRF GameId entries in the enum."""

    def test_all_grf_game_ids_have_keyboard_mapping(self):
        """Every GameId.GRF_* entry must appear in _GFOOTBALL_MAPPINGS."""
        from gym_gui.controllers.human_input import _GFOOTBALL_MAPPINGS
        from gym_gui.core.enums import GameId

        grf_ids = [gid for gid in GameId if gid.name.startswith("GRF_")]
        unmapped = [gid for gid in grf_ids if gid not in _GFOOTBALL_MAPPINGS]
        assert not unmapped, (
            f"{len(unmapped)} GRF GameId(s) have no keyboard mapping: "
            f"{[gid.name for gid in unmapped]}. "
            f"Add them to _GFOOTBALL_MAPPINGS in human_input.py."
        )

    def test_no_stale_game_ids_in_mapping(self):
        """_GFOOTBALL_MAPPINGS must not reference GameId values that no longer exist."""
        from gym_gui.controllers.human_input import _GFOOTBALL_MAPPINGS
        from gym_gui.core.enums import GameId

        all_ids = set(GameId)
        stale = [gid for gid in _GFOOTBALL_MAPPINGS if gid not in all_ids]
        assert not stale, (
            f"_GFOOTBALL_MAPPINGS references removed GameId(s): {stale}"
        )
