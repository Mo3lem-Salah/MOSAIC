"""SMAC/SMACv2 unit icon lookup.

Maps SC2 units to portrait icons extracted from the game's own asset
archive. SMAC and SMACv2 each have their own extracted asset tree
(``gym_gui/assets/SMAC/mods/`` and ``gym_gui/assets/SMACv2/mods/``) --
these currently contain byte-identical texture files (same extraction,
same underlying game assets), but they are separate directories and must
be resolved independently per game family, not one hardcoded to the
other. See ``gym_gui/assets/SMAC/ATTRIBUTION.md`` and
``gym_gui/assets/SMACv2/ATTRIBUTION.md`` for the source and licensing
note -- these are Blizzard Entertainment's copyrighted assets, included
here for non-commercial SMAC/SMACv2 research tooling.

Resolution is by unit type **name substring**, not by raw ``unit_type``
ID. SMAC's custom ``.SC2Map`` files define modified "_RL" unit variants
(e.g. "Marine_RL", "Zealot_RL") with dynamically assigned runtime IDs that
differ per map and per process -- confirmed by direct measurement:
requesting the same unit (allied Marine) on different maps returned
``unit_type`` 1970 on one map and 1971 on another, and the IDs are not
pysc2's static catalog values (``pysc2.lib.units.Terran.Marine == 48``,
never observed in practice). The only reliable way to identify a unit is
resolving its name via ``RequestData``/``data_raw()`` at runtime (see
``resolve_unit_type_name``) and matching on that name, not hardcoding IDs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

_GYM_GUI_ROOT = Path(__file__).resolve().parent.parent.parent
_ASSET_FAMILIES = ("SMAC", "SMACv2")


def _assets_root(asset_family: str) -> Path:
    """Resolve the extracted-textures directory for a given game family.

    ``asset_family`` must be ``"SMAC"`` or ``"SMACv2"`` -- each has its
    own extracted asset tree under ``gym_gui/assets/<family>/``.
    """
    if asset_family not in _ASSET_FAMILIES:
        raise ValueError(
            f"Unknown SMAC asset family {asset_family!r}; expected one of "
            f"{_ASSET_FAMILIES}"
        )
    return (
        _GYM_GUI_ROOT
        / "assets"
        / asset_family
        / "mods"
        / "liberty.sc2mod"
        / "base.sc2assets"
        / "assets"
        / "textures"
    )

# Unit-name substring (lowercased, matched against the "_RL"-stripped SC2
# unit type name) -> icon filename. Covers every unit type SMAC's 23 v1
# maps and SMACv2's 3 procedural races can produce.
_NAME_TO_ICON: Dict[str, str] = {
    # Terran
    "marine": "btn-unit-terran-marine.png",
    "marauder": "btn-unit-terran-marauder.png",
    "medivac": "btn-unit-terran-medivac.png",
    # Protoss
    "zealot": "btn-unit-protoss-zealot.png",
    "stalker": "btn-unit-protoss-stalker.png",
    "colossus": "btn-unit-protoss-colossus.png",
    # Zerg
    "zergling": "btn-unit-zerg-zergling.png",
    "baneling": "btn-unit-zerg-baneling.png",
    "hydralisk": "btn-unit-zerg-hydralisk.png",
    # SpineCrawler (2s_vs_1sc) has no dedicated btn-unit icon in the
    # extracted asset set (only a wireframe silhouette) -- intentionally
    # omitted, falls back to None/placeholder.
}

_icon_path_cache: Dict[tuple[str, str], Optional[Path]] = {}
_unit_type_name_cache: Dict[int, str] = {}


def resolve_unit_type_name(smac_env: Any, unit_type: int) -> str:
    """Resolve a raw ``unit_type`` int to its SC2 unit name via ``data_raw()``.

    Cached process-wide by ``unit_type`` ID. This is safe within a single
    SC2 process/episode (IDs are stable for its lifetime), which covers
    this cache's practical usage in the GUI. Returns "" if lookup fails.
    """
    if unit_type in _unit_type_name_cache:
        return _unit_type_name_cache[unit_type]
    try:
        data = smac_env._controller.data_raw()
        for u in data.units:
            _unit_type_name_cache[u.unit_id] = u.name
        return _unit_type_name_cache.get(unit_type, "")
    except Exception:
        return ""


def get_unit_icon_path(unit_name: str, asset_family: str = "SMACv2") -> Optional[Path]:
    """Return the filesystem path to a unit's portrait icon, or None.

    Args:
        unit_name: Resolved SC2 unit type name (e.g. ``"Marine_RL"``,
            from :func:`resolve_unit_type_name`).
        asset_family: ``"SMAC"`` or ``"SMACv2"`` -- selects which
            family's extracted asset tree to load the icon from.

    Returns:
        Path to a 76x76 RGBA PNG icon if this unit name matches a known
        unit, otherwise None (caller should show a generic/placeholder
        icon).
    """
    if not unit_name:
        return None

    # SMAC's custom units are named e.g. "Marine_RL"; strip any trailing
    # "_RL" (or similar) suffix and lowercase for matching.
    normalized = unit_name.lower().split("_")[0]

    cache_key = (asset_family, normalized)
    if cache_key in _icon_path_cache:
        return _icon_path_cache[cache_key]

    filename = _NAME_TO_ICON.get(normalized)
    result: Optional[Path] = None
    if filename is not None:
        candidate = _assets_root(asset_family) / filename
        if candidate.is_file():
            result = candidate

    _icon_path_cache[cache_key] = result
    return result


__all__ = ["get_unit_icon_path", "resolve_unit_type_name", "_assets_root"]
