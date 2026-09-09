"""SMAC game documentation module.

SMAC (StarCraft Multi-Agent Challenge) is the standard cooperative multi-agent
RL benchmark using StarCraft II.  Agents control individual units in symmetric
and asymmetric micromanagement scenarios across 23 hand-designed maps.

SMAC v1 (hand-designed maps):
    3m, 8m, 25m, 2s3z, 3s5z, 5m_vs_6m, 8m_vs_9m, 10m_vs_11m, 27m_vs_30m,
    MMM, MMM2, 3s5z_vs_3s6z, 3s_vs_3z, 3s_vs_4z, 3s_vs_5z, 1c3s5z,
    2m_vs_1z, corridor, 6h_vs_8z, 2s_vs_1sc, so_many_baneling,
    bane_vs_bane, 2c_vs_64zg

Repository: https://github.com/oxwhirl/smac
"""
from __future__ import annotations

from .SMAC_1c3s5z import SMAC_1C3S5Z_HTML, get_smac_1c3s5z_html
from .SMAC_2c_vs_64zg import SMAC_2C_VS_64ZG_HTML, get_smac_2c_vs_64zg_html
from .SMAC_2m_vs_1z import SMAC_2M_VS_1Z_HTML, get_smac_2m_vs_1z_html
from .SMAC_2s3z import SMAC_2S3Z_HTML, get_smac_2s3z_html
from .SMAC_2s_vs_1sc import SMAC_2S_VS_1SC_HTML, get_smac_2s_vs_1sc_html
from .SMAC_3m import SMAC_3M_HTML, get_smac_3m_html
from .SMAC_3s5z import SMAC_3S5Z_HTML, get_smac_3s5z_html
from .SMAC_3s5z_vs_3s6z import (
    SMAC_3S5Z_VS_3S6Z_HTML,
    get_smac_3s5z_vs_3s6z_html,
)
from .SMAC_3s_vs_3z import SMAC_3S_VS_3Z_HTML, get_smac_3s_vs_3z_html
from .SMAC_3s_vs_4z import SMAC_3S_VS_4Z_HTML, get_smac_3s_vs_4z_html
from .SMAC_3s_vs_5z import SMAC_3S_VS_5Z_HTML, get_smac_3s_vs_5z_html
from .SMAC_5m_vs_6m import SMAC_5M_VS_6M_HTML, get_smac_5m_vs_6m_html
from .SMAC_6h_vs_8z import SMAC_6H_VS_8Z_HTML, get_smac_6h_vs_8z_html
from .SMAC_8m import SMAC_8M_HTML, get_smac_8m_html
from .SMAC_8m_vs_9m import SMAC_8M_VS_9M_HTML, get_smac_8m_vs_9m_html
from .SMAC_10m_vs_11m import SMAC_10M_VS_11M_HTML, get_smac_10m_vs_11m_html
from .SMAC_25m import SMAC_25M_HTML, get_smac_25m_html
from .SMAC_27m_vs_30m import SMAC_27M_VS_30M_HTML, get_smac_27m_vs_30m_html
from .SMAC_bane_vs_bane import (
    SMAC_BANE_VS_BANE_HTML,
    get_smac_bane_vs_bane_html,
)
from .SMAC_corridor import SMAC_CORRIDOR_HTML, get_smac_corridor_html
from .SMAC_MMM import SMAC_MMM_HTML, get_smac_mmm_html
from .SMAC_MMM2 import SMAC_MMM2_HTML, get_smac_mmm2_html
from .SMAC_so_many_baneling import (
    SMAC_SO_MANY_BANELING_HTML,
    get_smac_so_many_baneling_html,
)

__all__ = [
    "SMAC_3M_HTML",
    "SMAC_8M_HTML",
    "SMAC_25M_HTML",
    "SMAC_2S3Z_HTML",
    "SMAC_3S5Z_HTML",
    "SMAC_5M_VS_6M_HTML",
    "SMAC_8M_VS_9M_HTML",
    "SMAC_10M_VS_11M_HTML",
    "SMAC_27M_VS_30M_HTML",
    "SMAC_MMM_HTML",
    "SMAC_MMM2_HTML",
    "SMAC_3S5Z_VS_3S6Z_HTML",
    "SMAC_3S_VS_3Z_HTML",
    "SMAC_3S_VS_4Z_HTML",
    "SMAC_3S_VS_5Z_HTML",
    "SMAC_1C3S5Z_HTML",
    "SMAC_2M_VS_1Z_HTML",
    "SMAC_CORRIDOR_HTML",
    "SMAC_6H_VS_8Z_HTML",
    "SMAC_2S_VS_1SC_HTML",
    "SMAC_SO_MANY_BANELING_HTML",
    "SMAC_BANE_VS_BANE_HTML",
    "SMAC_2C_VS_64ZG_HTML",
    "get_smac_3m_html",
    "get_smac_8m_html",
    "get_smac_25m_html",
    "get_smac_2s3z_html",
    "get_smac_3s5z_html",
    "get_smac_5m_vs_6m_html",
    "get_smac_8m_vs_9m_html",
    "get_smac_10m_vs_11m_html",
    "get_smac_27m_vs_30m_html",
    "get_smac_mmm_html",
    "get_smac_mmm2_html",
    "get_smac_3s5z_vs_3s6z_html",
    "get_smac_3s_vs_3z_html",
    "get_smac_3s_vs_4z_html",
    "get_smac_3s_vs_5z_html",
    "get_smac_1c3s5z_html",
    "get_smac_2m_vs_1z_html",
    "get_smac_corridor_html",
    "get_smac_6h_vs_8z_html",
    "get_smac_2s_vs_1sc_html",
    "get_smac_so_many_baneling_html",
    "get_smac_bane_vs_bane_html",
    "get_smac_2c_vs_64zg_html",
]
