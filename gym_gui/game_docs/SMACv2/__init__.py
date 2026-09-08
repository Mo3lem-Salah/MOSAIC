"""SMACv2 game documentation module.

SMACv2 (StarCraft Multi-Agent Challenge v2) is the procedurally-generated
version of SMAC, featuring variable team compositions each episode.

SMACv2 base maps (procedural generation, default 10v10):
    - 10gen_terran, 10gen_protoss, 10gen_zerg

SMACv2 scenario presets (EPyMARL's published scenario table, 5 team sizes
x 3 races = 15 presets on top of the same 3 base maps above):
    - {Terran,Protoss,Zerg}_{5V5,10V10,20V20,10V11,20V23}

Repository: https://github.com/oxwhirl/smacv2
"""
from __future__ import annotations

from .SMACv2_Protoss import SMACV2_PROTOSS_HTML, get_smacv2_protoss_html
from .SMACv2_Protoss_5V5 import (
    SMACV2_PROTOSS_5V5_HTML,
    get_smacv2_protoss_5v5_html,
)
from .SMACv2_Protoss_10V10 import (
    SMACV2_PROTOSS_10V10_HTML,
    get_smacv2_protoss_10v10_html,
)
from .SMACv2_Protoss_10V11 import (
    SMACV2_PROTOSS_10V11_HTML,
    get_smacv2_protoss_10v11_html,
)
from .SMACv2_Protoss_20V20 import (
    SMACV2_PROTOSS_20V20_HTML,
    get_smacv2_protoss_20v20_html,
)
from .SMACv2_Protoss_20V23 import (
    SMACV2_PROTOSS_20V23_HTML,
    get_smacv2_protoss_20v23_html,
)
from .SMACv2_Terran import SMACV2_TERRAN_HTML, get_smacv2_terran_html
from .SMACv2_Terran_5V5 import (
    SMACV2_TERRAN_5V5_HTML,
    get_smacv2_terran_5v5_html,
)
from .SMACv2_Terran_10V10 import (
    SMACV2_TERRAN_10V10_HTML,
    get_smacv2_terran_10v10_html,
)
from .SMACv2_Terran_10V11 import (
    SMACV2_TERRAN_10V11_HTML,
    get_smacv2_terran_10v11_html,
)
from .SMACv2_Terran_20V20 import (
    SMACV2_TERRAN_20V20_HTML,
    get_smacv2_terran_20v20_html,
)
from .SMACv2_Terran_20V23 import (
    SMACV2_TERRAN_20V23_HTML,
    get_smacv2_terran_20v23_html,
)
from .SMACv2_Zerg import SMACV2_ZERG_HTML, get_smacv2_zerg_html
from .SMACv2_Zerg_5V5 import SMACV2_ZERG_5V5_HTML, get_smacv2_zerg_5v5_html
from .SMACv2_Zerg_10V10 import (
    SMACV2_ZERG_10V10_HTML,
    get_smacv2_zerg_10v10_html,
)
from .SMACv2_Zerg_10V11 import (
    SMACV2_ZERG_10V11_HTML,
    get_smacv2_zerg_10v11_html,
)
from .SMACv2_Zerg_20V20 import (
    SMACV2_ZERG_20V20_HTML,
    get_smacv2_zerg_20v20_html,
)
from .SMACv2_Zerg_20V23 import (
    SMACV2_ZERG_20V23_HTML,
    get_smacv2_zerg_20v23_html,
)

__all__ = [
    # SMACv2 base maps
    "SMACV2_TERRAN_HTML",
    "SMACV2_PROTOSS_HTML",
    "SMACV2_ZERG_HTML",
    "get_smacv2_terran_html",
    "get_smacv2_protoss_html",
    "get_smacv2_zerg_html",
    # SMACv2 scenario presets -- Terran
    "SMACV2_TERRAN_5V5_HTML",
    "SMACV2_TERRAN_10V10_HTML",
    "SMACV2_TERRAN_20V20_HTML",
    "SMACV2_TERRAN_10V11_HTML",
    "SMACV2_TERRAN_20V23_HTML",
    "get_smacv2_terran_5v5_html",
    "get_smacv2_terran_10v10_html",
    "get_smacv2_terran_20v20_html",
    "get_smacv2_terran_10v11_html",
    "get_smacv2_terran_20v23_html",
    # SMACv2 scenario presets -- Protoss
    "SMACV2_PROTOSS_5V5_HTML",
    "SMACV2_PROTOSS_10V10_HTML",
    "SMACV2_PROTOSS_20V20_HTML",
    "SMACV2_PROTOSS_10V11_HTML",
    "SMACV2_PROTOSS_20V23_HTML",
    "get_smacv2_protoss_5v5_html",
    "get_smacv2_protoss_10v10_html",
    "get_smacv2_protoss_20v20_html",
    "get_smacv2_protoss_10v11_html",
    "get_smacv2_protoss_20v23_html",
    # SMACv2 scenario presets -- Zerg
    "SMACV2_ZERG_5V5_HTML",
    "SMACV2_ZERG_10V10_HTML",
    "SMACV2_ZERG_20V20_HTML",
    "SMACV2_ZERG_10V11_HTML",
    "SMACV2_ZERG_20V23_HTML",
    "get_smacv2_zerg_5v5_html",
    "get_smacv2_zerg_10v10_html",
    "get_smacv2_zerg_20v20_html",
    "get_smacv2_zerg_10v11_html",
    "get_smacv2_zerg_20v23_html",
]
